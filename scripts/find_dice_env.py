from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import utils and config
sys.path.insert(0, str(Path(__file__).parent.parent))
import copy
import os
import json
import random
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from belief_structs import ContinuousBelief, DiscreteBelief, Action, Observation
from environment import TampuraEnv
from utils.franka_api import *
from utils.utils import *
from config.config import register_env
from unified_planning.model.problem import Problem
from utils.voxel_utils import VoxelGrid


DEFAULT_ARM_POS = [
    -0.0806406098426434,
    -1.6722951504174777,
    0.07069076842695393,
    -2.7449419709102822,
    0.08184716251979611,
    1.7516337599063168,
    0.7849295270972781,
    0.54,  # left finger
    0.54,  # right finger
]
NUM_PARTICLES = 10
KNOWN_POSE_SD = 0.01  # known-pose uncertainty threshold
INITIAL_SD = 0.02  # initial uncertainty for particles
GRID_RESOLUTION = 0.015
TARGET_OBJECT = "dice"
GRASP_MODE = "saved"
EXECUTE_ATTEMPTS = 100


@dataclass
class SceneWorld:
    franka: FRANKA


@dataclass
class SceneObservation(Observation):
    camera_image: Optional[CameraImage] = None
    camera_detected_ids: Dict[int, Any] = field(default_factory=dict)  # seg_id -> entity_idx from camera
    conf: List[float] = None
    grasp: Any = None  # object pose w.r.t ee frame [x, y, z, qx, qy, qz, qw]
    grasped_obj: int = None  # idx
    poses: Dict[str, Pose] = field(default_factory=lambda: {})  # object name <-> pose w.r.t world frame
    moved: Optional[str] = None
    possible_locations: List[Any] = field(default_factory=lambda: [])
    image_path: List[str] = None


class SceneBelief(ContinuousBelief):
    def __init__(self, world, visualize_grid=False):
        self.camera_intrinsics = None
        self.world = world
        self.grasp = None  # List [x, y, z, qx, qy, qz, qw]
        self.grasped_obj = None  # idx
        self.conf = DEFAULT_ARM_POS  # List
        self.possible_locations = None
        self.placed = False
        visibility_aabb = AABB(
            lower=[
                0.37, -0.7, 0.02,
            ],
            upper=[
                0.7, 0.7, 0.1,
            ],
        )
        self.moved = []
        self.object_poses = {}  # object name <-> pose w.r.t world frame
        self.visualize_grid = visualize_grid
        self.visibility_grid = self.setup_visibility_grid(visibility_aabb)

    def __repr__(self):
        """Detailed string representation for debugging"""
        lines = ["SceneBelief("]
        lines.append(f"  conf: {self.conf}")
        lines.append(f"  grasp: {self.grasp}")
        lines.append(f"  grasped_obj: {self.grasped_obj}")
        lines.append(f"  possible_locations: {self.possible_locations}")
        lines.append(f"  placed: {self.placed}")
        lines.append(f"  moved: {self.moved}")
        lines.append(f"  object_poses:")
        for obj, pose in self.object_poses.items():
            lines.append(f"    {obj}: {pose}")
        lines.append(")")
        return "\n".join(lines)

    def __str__(self):
        """Simple string representation"""
        return self.__repr__()

    def __deepcopy__(self, memo):
        new_belief = SceneBelief.__new__(SceneBelief)
        memo[id(self)] = new_belief
        
        new_belief.world = self.world
        
        new_belief.camera_intrinsics = copy.deepcopy(self.camera_intrinsics, memo)
        new_belief.conf = copy.deepcopy(self.conf, memo)
        new_belief.grasp = copy.deepcopy(self.grasp, memo)
        new_belief.grasped_obj = copy.deepcopy(self.grasped_obj, memo)
        new_belief.possible_locations = copy.deepcopy(self.possible_locations, memo)
        new_belief.placed = copy.deepcopy(self.placed, memo)
        new_belief.moved = copy.deepcopy(self.moved, memo)
        new_belief.object_poses = copy.deepcopy(self.object_poses, memo)
        new_belief.visualize_grid = self.visualize_grid
        new_belief.visibility_grid = copy.deepcopy(self.visibility_grid, memo)
        
        return new_belief

    def vectorize(self):
        obj_pose_vecs = []
        for obj in self.world.objects:
            if self.get_pose(obj) is None:
                obj_pose_vecs.append(unit_pose())  # [x, y, z, qx, qy, qz, qw]
            else:
                obj_pose_vecs.append(self.get_pose(obj))
        vectorized_vis_grid = np.array(
            [self.visibility_grid.is_occupied(voxel) for voxel in self.get_all_voxels()]
        )
        b_vec = np.concatenate([vectorized_vis_grid] + obj_pose_vecs)
        return b_vec
    
    def get_pose(self, obj):
        if obj in self.object_poses:
            return self.object_poses[obj]
        else:
            return None

    def update_objects(self, obs: SceneObservation):
        """Given a camera image, use the image.segmentationMaskBuffer to update
        the known objects."""

        for obj_name, pose in obs.poses.items(): # obs.poses에 들어온 물체들을 object_poses에 저장
            self.object_poses[obj_name] = pose
    
    def get_all_voxels(self):
        """Returns a list of all voxels in the visibility grid (Table)."""
        return list(self.visibility_grid.voxels_from_aabb(self.visibility_grid.aabb))
    
    def set_sim(self):
        for obj_name, pose in self.object_poses.items():
            if pose is None:
                pose = [100, 0, 0.1, 1, 0, 0, 0]  # Far away

            set_pose(self.world.franka.object_dict[obj_name], pose)

    def get_new_seen_voxels(
        self, camera_image: CameraImage, include_unseen=False
    ) -> List[Any]:
        voxels = []
        width, height = dimensions_from_camera_image(camera_image)

        # For each voxel in the grid, check whether it was seen in the image
        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):  # is_occupied = 아직 못 본 voxel
                center_world = self.visibility_grid.to_world(
                    self.visibility_grid.center_from_voxel(voxel)  # voxel 중심 위치 (grid 좌표)
                )
                center_camera = tform_point( # World 좌표를 카메라 좌표계로 변환
                    invert(camera_image.camera_pose), center_world
                )
                distance = center_camera[2]
                pixel = pixel_from_point(
                    camera_image.camera_matrix, center_camera, width, height
                )
                if pixel is not None:
                    depth = camera_image.depthPixels[pixel.row, pixel.column]

                    if distance <= depth: # voxel 중심이 카메라에서 보이는 물체보다 가까우면, 즉 voxel이 카메라 시야에 있으면
                        voxels.append(voxel)

                elif include_unseen:
                    voxels.append(voxel)

        return voxels
    
    def update_visibility(
        self,
        camera_image: CameraImage,
        possible_locations=None,
        include_unseen=False,
    ):
        new_voxels = self.get_new_seen_voxels(
            camera_image=camera_image, include_unseen=include_unseen
        )
        for voxel in new_voxels:
            self.visibility_grid.set_free(voxel=voxel)

        # Remove points that are in collision with an object
        distance_threshold = GRID_RESOLUTION / 2.0
        
        for voxel in self.get_all_voxels():
            if self.visibility_grid.is_occupied(voxel):
                point = self.visibility_grid.center_from_voxel(voxel)

                if possible_locations is not None:
                    distances = []
                    for possible_point in possible_locations:
                        transformed_pose = transformation_to_pose(possible_point)[0]
                        transformed_array = np.array(transformed_pose)
                        point_array = np.array(point)
                        distance = np.linalg.norm(point_array - transformed_array)
                        distances.append(distance)

                    min_distance = min(float(distance) for distance in distances)

                    if min_distance > 0.05:
                        self.visibility_grid.set_free(voxel=voxel)
                        continue

                collision_detected = False
                for obj_name in self.object_poses:
                    if obj_name not in self.world.franka.object_dict:
                        continue
                        
                    obj_entity_idx = self.world.franka.object_dict[obj_name]
                    obj_entity = self.world.franka.scene.entities[obj_entity_idx]
                        
                    aabb = obj_entity.get_AABB()
                    if torch.is_tensor(aabb):
                        aabb = aabb.detach().cpu().numpy()
                        
                    aabb_lower = aabb[0]
                    aabb_upper = aabb[1]
                        
                    point_array = np.array(point)
                        
                    distances_to_aabb = []
                    for i in range(3):
                        if point_array[i] < aabb_lower[i]:
                            distances_to_aabb.append(aabb_lower[i] - point_array[i])
                        elif point_array[i] > aabb_upper[i]:
                            distances_to_aabb.append(point_array[i] - aabb_upper[i])
                        else:
                            distances_to_aabb.append(0.0)
                        
                    min_distance_to_aabb = np.sqrt(sum(d**2 for d in distances_to_aabb))
                        
                    if min_distance_to_aabb <= distance_threshold:
                        collision_detected = True
                        break
                
                if collision_detected:
                    self.visibility_grid.set_free(voxel=voxel)
    
    def setup_visibility_grid(self, surface: AABB) -> VoxelGrid:
        """Creates a grid that represents the visibility of the robot."""
        resolutions = GRID_RESOLUTION * np.ones(3)
        surface_origin = Pose(Point(z=0.01))
        surface_aabb = AABB(
            lower=surface.lower,
            upper=[surface.upper[0], surface.upper[1], GRID_RESOLUTION * 2],
        )

        grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=(0, 0, 1, 1),
        )
        static_grid = VoxelGrid(
            resolutions,
            world_from_grid=surface_origin,
            aabb=surface_aabb,
            color=(0, 0, 0, 1),
        )
        for voxel in grid.voxels_from_aabb(surface_aabb):
            grid.set_occupied(voxel)
            static_grid.set_occupied(voxel)

        return grid

    def update(self, action_str: str, obs: SceneObservation) -> SceneBelief:
        new_belief = copy.deepcopy(self)
        action_name = action_str.split("(")[0]

        if "place" in action_name or "putdown" in action_name:
            new_belief.placed = True

        if obs.moved is not None:
            new_belief.moved = list(set(self.moved + [obs.moved]))

        new_belief.current_conf = obs.conf

        new_belief.update_objects(obs)
        new_belief.set_sim()

        if obs.possible_locations is not None:
            new_belief.possible_locations = obs.possible_locations

        if obs.camera_image is not None:
            new_belief.camera_intrinsics = obs.camera_image.camera_matrix
            new_belief.update_visibility(camera_image=obs.camera_image)

        if obs.grasp is not None:
            new_belief.grasp = obs.grasp
            new_belief.grasp_body = obs.grasp_body
        else:
            new_belief.grasp = None
            new_belief.grasp_body = None
        return new_belief
    
    def abstract(self, expected_db: DiscreteBelief, problem: Problem) -> tuple[bool, DiscreteBelief]:  # TODO
        """
        Abstract continuous belief to discrete facts.
        Returns grouped facts format: (predicate_name, ((arg1, arg2, ...), ...))
        for objects with known poses (low uncertainty).
        """
        # Actual known-pose objects based on current continuous belief
        known_pose_facts = []
        grouped_facts = None
        
        for obj_name in self.object_poses.keys():
                print(f"[abstract] {obj_name} is observed and known, treating as known pose.")
                if obj_name is not None:
                    known_pose_facts.append((obj_name,))
        
        if known_pose_facts:
            grouped_facts = ('known-pose', tuple(known_pose_facts))
        
        known_pose_objs = set(obj_name[0] for obj_name in grouped_facts[1]) if grouped_facts is not None else set()
        print(f"[abstract] Actual known-pose objects from continuous belief: {known_pose_objs}")

        # Expected known-pose objects from expected_db
        current_state = expected_db.state
        no_changes = True
        # print(f"[abstract] DEBUG: expected_db.state:", expected_db.state)
        
        expected_known_poses_in_state = set()
        for fluent_instance in current_state._values.keys():
            fluent_str = str(fluent_instance)
            # Extract predicate name before '('
            if '(' in fluent_str:
                pred_name = fluent_str.split('(')[0]
                if pred_name == 'known-pose':
                    value = current_state._values[fluent_instance]
                    is_true = value.constant_value() is True
                    # Extract object name from fluent_str: "known-pose(red_block)" -> "red_block"
                    obj_name_in_fluent = fluent_str.split('(')[1].rstrip(')')
                    # Add only TRUE ones to expected_known_poses_in_state
                    if is_true:
                        expected_known_poses_in_state.add(obj_name_in_fluent)
        
        print(f"[abstract] Expected known-pose objects from expected_db: {expected_known_poses_in_state}")
        
        # Update if different
        if known_pose_objs != expected_known_poses_in_state:      
            state_updates = {}
            true_value = problem.environment.expression_manager.TRUE()
            false_value = problem.environment.expression_manager.FALSE()
            
            for fluent_instance in current_state._values.keys():
                fluent_str = str(fluent_instance)
                if '(' in fluent_str:
                    pred_name = fluent_str.split('(')[0]
                    if pred_name == 'known-pose':
                        obj_name_in_fluent = fluent_str.split('(')[1].rstrip(')')
                        state_updates[fluent_instance] = true_value if obj_name_in_fluent in known_pose_objs else false_value
            
            if state_updates:
                current_state = current_state.make_child(state_updates)
                no_changes = False  # Mark that we updated the state
                print(f"[abstract] State updated. Changes made: {state_updates}")
        else:
            print(f"[abstract] No changes - both sets are identical")
        
        updated_db = DiscreteBelief(state=current_state)
        print(f"[abstract] Updated discrete belief:\n{updated_db}\n")
        return no_changes, updated_db


def obs_from_camera_image(
    world: SceneWorld,
    camera_image: CameraImage,
    conf
) -> SceneObservation:

    seg_buffer = camera_image.segmentationMaskBuffer
    
    # Handle both 2D (Genesis) and 3D (PyBullet) segmentation formats
    if seg_buffer.ndim == 3:
        # TAMPURA/PyBullet format: (height, width, channels)
        seg_bodies = seg_buffer[:, :, 0]
    else:
        # Genesis format: (height, width) - already 2D
        seg_bodies = seg_buffer

    # Get the unique values casted to int
    unique_seg_ids = np.unique(seg_bodies.astype(int))
    object_poses = {}
    
    # Map segmentation IDs to entity indices
    for seg_id in unique_seg_ids:
        seg_id = int(seg_id)
        # Find which object has this entity index in object_dict
        for entity_name, entity_idx in world.franka.object_dict.items():
            if entity_idx == seg_id:
                pose = get_pose(world.franka.scene.entities[entity_idx])
                object_poses[entity_name] = pose
    
    return SceneObservation(camera_image=camera_image, poses=object_poses, conf=conf, possible_locations=None)


def start_sim(json_path, method, prob_num, prob_idx, trial, repeat, num_distractor=NUM_PARTICLES, show_viewer=False, record_video=False):
    # load json file and bring entry
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for e in meta:
        if e.get("num") == prob_num and e.get("index") == prob_idx and e.get("trial") == trial:
            entry = e
            break
    objects_info = entry["objects"]

    # Initialize
    try:
        gs.init(backend=gs.gpu, precision="32", performance_mode=False)
    except Exception as e:
        if "already initialized" not in str(e).lower():
            raise
    franka = FRANKA(show_viewer=show_viewer, record_video=record_video)

    # Add cups and dice
    franka.object_dict = getattr(franka, "object_dict", {})  # add moveable objects

    for i, (name, info) in enumerate(objects_info.items()):
        pos = tuple(info["pose"]["position"])

        if name == "dice":
            dice = franka.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="/home/minseo/Genesis/genesis/assets/xml/062_dice/google_16k/textured.obj",
                    pos=pos,
                    scale=3.0,
                ),
            )
            franka.object_dict[name] = dice.idx
        else:
            cup = franka.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="/home/minseo/Genesis/genesis/assets/xml/065_b_cups_ud/google_16k/textured.obj",
                    pos=pos,
                    scale=2.0,
                ),
            )
            franka.object_dict[name] = cup.idx
    

    # Build the scene
    franka.scene.build()
    franka.scene.step()
    if record_video:
        franka.render_cameras()

    # Set control gains
    franka.set_control_gains()

    # attach mounted camera
    franka.cam_arm.attach(franka.robot.get_link(franka.EE_FRAMES['ee']), franka.T)

    # move to start config
    arm_joints = np.array(franka.DEFAULT_CONFIG['side'])
    gripper_joints = np.array([0.54, 0.54])
    init_qpos = np.zeros(9)
    init_qpos[franka.motors_dof] = arm_joints
    init_qpos[franka.fingers_dof] = gripper_joints
    franka.robot.control_dofs_position(init_qpos)

    for i in range(100):
        franka.scene.step()
        if record_video:
            franka.render_cameras()

    world = SceneWorld(franka=franka)

    screenshot_dir = Path(
        f"../experiments/tool_use/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = f"../experiments/tool_use/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}"
    file_path_list = franka.save_snapshot4(screenshot_dir, node_name="node0", world=world)

    return world, file_path_list


def start_tamp_sim(json_path, method, prob_num, prob_idx, trial, repeat, 
                   sampled_object_poses, robot_conf, show_viewer=False, record_video=True,
                   existing_tamp_world=None, held_object=None):
    """
    Initialize or reset Genesis simulator for TAMP planning with sampled continuous state.
    
    If existing_tamp_world is provided, reuses it (avoiding EGL context switching).
    Otherwise, creates a new planning world (first call only).
    
    Args:
        sampled_object_poses: Dict {object_name: pose}, where pose is [x,y,z,qx,qy,qz,qw]
        robot_conf: Robot joint configuration (9 DOFs) from continuous belief
        show_viewer: If True, display the planning world in viewer (only one viewer allowed at a time)
        existing_tamp_world: If provided, reuse this world instead of creating new one
        held_object: If provided, weld this object to the robot's end-effector
    Returns:
        HookWorld instance with franka simulator and empty ghosts dict
    """
    # If reusing existing world, just reset it
    if existing_tamp_world is not None:
        from scipy.spatial.transform import Rotation
        
        franka = existing_tamp_world.franka
        objects_info_dummy = {}  # Not needed for reset, just for reference
        
        # Reset object poses directly without recreating scene
        for obj_name, pose in sampled_object_poses.items():
            if obj_name in franka.object_dict:
                obj_idx = franka.object_dict[obj_name]
                entity = franka.scene.entities[obj_idx]
                pos = tuple(pose[0:3])
                quat = pose[3:7]  # [qx, qy, qz, qw]
                
                # Update entity pose directly
                entity.set_pos(pos)
                entity.set_quat(quat)
        
        # Reset robot configuration
        arm_joints = np.array(robot_conf[:7])
        gripper_joints = np.array(robot_conf[7:9])
        init_qpos = np.zeros(9)
        init_qpos[franka.motors_dof] = arm_joints
        init_qpos[franka.fingers_dof] = gripper_joints
        franka.robot.control_dofs_position(init_qpos)

        # delete constraint
        if held_object is not None:
            rigid_solver = franka.scene.sim.rigid_solver
            held_entity = franka.scene.entities[held_object]
            ee_link_idx = franka.robot.get_link(franka.EE_FRAMES['ee']).idx
            try:
                link_obj = held_entity.get_link("box_baselink").idx
            except Exception:
                try:
                    link_obj = held_entity.get_link("hook_link").idx
                except Exception:
                    link_obj = held_entity.get_root_link().idx
            rigid_solver.delete_weld_constraint(link_obj, ee_link_idx)
        
        for i in range(50):
            franka.scene.step()
            if record_video:
                franka.render_cameras()
        
        return existing_tamp_world
    
    # First call: create new planning world
    from scipy.spatial.transform import Rotation
    
    # Load problem metadata
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for e in meta:
        if e.get("num") == prob_num and e.get("index") == prob_idx and e.get("trial") == trial:
            entry = e
            break
    objects_info = entry["objects"]
    
    # Initialize Genesis
    try:
        gs.init(backend=gs.gpu, precision="32")
    except Exception as e:
        if "already initialized" not in str(e).lower():
            raise
    # Force headless mode for planning world to prevent EGL context switching
    # Original world has visualizer, planning world must not compete for global EGL context
    franka = FRANKA(show_viewer=False, record_video=record_video)
    
    # Add objects with sampled poses (single particle, no ghosts)
    franka.object_dict = {}
    
    for obj_name, pose in sampled_object_poses.items():
        info = objects_info.get(obj_name)
        if info is None:
            continue
        
        # Extract position and orientation from pose [x,y,z,qx,qy,qz,qw]
        pos = tuple(pose[0:3])
        quat = pose[3:7]  # [qx, qy, qz, qw]
        r = Rotation.from_quat(quat)
        euler = tuple(r.as_euler('xyz', degrees=True))
        
        if obj_name == "dice":
            dice = franka.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="/home/minseo/Genesis/genesis/assets/xml/062_dice/google_16k/textured.obj",
                    pos=pos,
                    euler=euler,
                    scale=3.0,
                ),
            )
            franka.object_dict[obj_name] = dice.idx
        else:
            cup = franka.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="/home/minseo/Genesis/genesis/assets/xml/065_b_cups_ud/google_16k/textured.obj",
                    pos=pos,
                    euler=euler,
                    scale=2.0,
                ),
            )
            franka.object_dict[obj_name] = cup.idx
    
    # Build the scene (only once on first call)
    franka.scene.build()
    franka.scene.step()
    if record_video:
        franka.render_cameras()

    # Set control gains
    franka.set_control_gains()

    # attach mounted camera
    franka.cam_arm.attach(franka.robot.get_link(franka.EE_FRAMES['ee']), franka.T)

    # move to start config
    arm_joints = np.array(robot_conf[:7])
    gripper_joints = np.array(robot_conf[7:9])
    init_qpos = np.zeros(9)
    init_qpos[franka.motors_dof] = arm_joints
    init_qpos[franka.fingers_dof] = gripper_joints
    
    franka.robot.control_dofs_position(init_qpos)
    for i in range(50):
        franka.scene.step()
        if record_video:
            franka.render_cameras()
    
    return SceneWorld(franka=franka)
    

class FindDiceEnv(TampuraEnv):
    def initialize(self, prob_num, prob_idx, trial, repeat) -> ContinuousBelief: 
        self.world, file_path_list = start_sim(
            self.config["json_path"], 
            self.config["planner"], 
            prob_num, 
            prob_idx, 
            trial, 
            repeat, 
            show_viewer=False, 
            record_video=self.config["record_video"]
        )

        possible_locations = None

        camera_image = self.world.franka.get_image(segment=True)

        obs = obs_from_camera_image(self.world, camera_image, DEFAULT_ARM_POS)
        
        # Create particle-based belief with proper initialization
        b0 = SceneBelief(self.world, visualize_grid=True)
        b0.update_visibility(
            camera_image=camera_image,
            possible_locations=possible_locations,
            include_unseen=True,
        )
        b0.visibility_grid.draw_intervals(b0.world.franka)
        b0 = b0.update("", obs)
        b0.set_sim()
        self.starting_belief = b0

        # Visualize the visibility grid in pybullet
        if self.vis:
            b0.visibility_grid.draw_intervals(b0.world.franka)

        return b0



register_env("find_dice", FindDiceEnv)


# if __name__ == '__main__':
#     from config import config as tconfig

#     config = tconfig.load_config(config_file="/home/minseo/develop/pomdp_llm/config/find_dice.yml")
#     env = tconfig.get_env(config["task"])(config=config)
#     b0 = env.initialize(1, 1, 1, 1)
#     print("b0: ", str(b0))