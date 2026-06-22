from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import utils and config
sys.path.insert(0, str(Path(__file__).parent.parent))
import copy
import os
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
from belief_structs import ContinuousBelief, DiscreteBelief, Action, Observation
from environment import TampuraEnv
from utils.franka_api import *
from utils.utils import *
from config.config import register_env
from unified_planning.model.problem import Problem


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
KNOWN_POSE_SD = 0.01
NUM_SIM_STEPS = 5
BLOCK_SIZE = 0.02
INITIAL_SD = 0.02


@dataclass
class HookWorld:
    franka: FRANKA
    ghosts: Dict[Any, Any] = field(default_factory=dict)


@dataclass
class HookObservation(Observation):
    conf: List[float] = None
    grasp: Any = None  # object pose w.r.t ee frame [x, y, z, qx, qy, qz, qw]
    grasped_obj: int = None  # idx
    hook_traj: List[Pose] = field(default_factory=lambda: [])
    poses: Dict[str, Pose] = field(default_factory=lambda: {})  # object name <-> pose w.r.t world frame
    image_path: List[str] = None


class HookBelief(ContinuousBelief):
    def __init__(self, world):
        self.world = world
        self.conf = DEFAULT_ARM_POS  # List
        self.grasp = None  # List [x, y, z, qx, qy, qz, qw]
        self.grasped_obj = None  # idx
        self.hook = self.world.franka.object_dict["hook"]  # index of genesis rigid entity

        # obj idx <-> corresponding ghost objects idxs
        self.ghost_dict = {
            v: self.world.ghosts[v] for k, v in self.world.franka.object_dict.items() if k != "hook"
        }
        # object idx <-> pose belief for each object idx ([x, y, z, qx, qy, qz, qw])
        self.object_dists = {
            v: [
                get_pose(self.world.franka.scene.entities[idx])
                for idx in self.world.ghosts[v]
            ]
            for k, v in self.world.franka.object_dict.items() if k != "hook"
        }
        # We know the hook pose
        self.object_dists[self.hook] = [
            get_pose(self.world.franka.scene.entities[self.hook])
        ]
        
        # Initialize particle weights (uniform distribution)
        self.object_weights = {}
        for obj_idx in self.object_dists:
            num_particles = len(self.object_dists[obj_idx])
            self.object_weights[obj_idx] = [1.0 / num_particles] * num_particles

    def __repr__(self):
        """Detailed string representation for debugging"""
        lines = ["HookBelief("]
        lines.append(f"  conf: {self.conf}")
        lines.append(f"  grasp: {self.grasp}")
        lines.append(f"  grasped_obj: {self.grasped_obj}")
        lines.append(f"  ghosts: {self.world.ghosts}")
        
        lines.append(f"  object_dists:")
        for obj_idx, poses in self.object_dists.items():
            obj_name = None
            for name, idx in self.world.franka.object_dict.items():
                if idx == obj_idx:
                    obj_name = name
                    break
            com = self.get_com(obj_idx)
            sd = self.get_sd(obj_idx)
            weights = self.object_weights[obj_idx]
            lines.append(f"    {obj_name} (idx={obj_idx}): COM={com}, num_particles={len(poses)}, sd={sd}, weights={[round(w, 4) for w in weights]}")
        
        lines.append(")")
        return "\n".join(lines)

    def __str__(self):
        """Simple string representation"""
        return self.__repr__()

    def __deepcopy__(self, memo):
        new_belief = HookBelief.__new__(HookBelief)
        memo[id(self)] = new_belief
        
        new_belief.world = self.world
        
        new_belief.conf = copy.deepcopy(self.conf, memo)
        new_belief.grasp = copy.deepcopy(self.grasp, memo)
        new_belief.grasped_obj = copy.deepcopy(self.grasped_obj, memo)
        new_belief.hook = self.hook 
        new_belief.ghost_dict = copy.deepcopy(self.ghost_dict, memo)
        new_belief.object_dists = copy.deepcopy(self.object_dists, memo)
        new_belief.object_weights = copy.deepcopy(self.object_weights, memo)
        
        return new_belief

    def vectorize(self):
        vecs = []
        for obj in self.world.franka.object_dict:
            vecs += [self.get_com(obj), self.get_sd(obj)]
        return np.concatenate(vecs)

    def simulate_hook_traj(self, hook_traj):
        # Get the new object particle poses after hook pulling
        for o, gos in self.ghost_dict.items():
            self.object_dists[o] = [
                get_pose(self.world.franka.scene.entities[idx]) for idx in gos
            ]
            # Update particle weights based on deviation from mean position
            self.update_weights_by_deviation(o)

    def update_weights_by_deviation(self, obj_idx):
        """
        Update particle weights based on deviation from mean position.
        Particles close to mean get higher weight (likely correct).
        Particles far from mean get lower weight (likely outliers).
        Uses Gaussian likelihood: exp(-0.5 * (distance / sigma)^2)
        """
        poses = np.array([pose[0:3] for pose in self.object_dists[obj_idx]])
        
        # Calculate mean position of all particles
        mean_pos = np.mean(poses, axis=0)
        
        # Distance of each particle from mean
        distances = np.linalg.norm(poses - mean_pos, axis=1)
        
        # Standard deviation of distances
        std_dist = np.std(distances)
        
        # Avoid division by zero
        if std_dist < 1e-6:
            std_dist = 0.01
        
        # Gaussian likelihood: particles close to mean get high probability
        likelihoods = np.exp(-0.5 * (distances / std_dist) ** 2)
        
        # Normalize to get valid probability distribution
        weights = likelihoods / np.sum(likelihoods)
        
        self.object_weights[obj_idx] = weights.tolist()

    def get_com(self, obj):
        # return np.mean(
        #     np.concatenate([np.array([pose[0:3]]) for pose in self.object_dists[obj]]),
        #     axis=0,
        # )
        """Get center of mass using weighted particle positions."""
        poses = np.array([pose[0:3] for pose in self.object_dists[obj]])
        weights = np.array(self.object_weights[obj])
        
        # Weighted mean
        weighted_mean = np.average(poses, axis=0, weights=weights)
        return weighted_mean

    def get_sd(self, obj):
        values = np.concatenate(
            [np.array([pose[0:3]]) for pose in self.object_dists[obj]]
        )
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        robust_sd = 1.4826 * mad  # Scaling factor for normal distribution
        return robust_sd
        # """Get standard deviation using weighted particles."""
        # poses = np.array([pose[0:3] for pose in self.object_dists[obj]])
        # weights = np.array(self.object_weights[obj])
        
        # # Weighted mean
        # weighted_mean = np.average(poses, axis=0, weights=weights)
        
        # # Weighted absolute deviations
        # abs_deviations = np.abs(poses - weighted_mean)
        
        # # Weighted median absolute deviation
        # weighted_mad = np.average(abs_deviations, axis=0, weights=weights)
        
        # # Normalized robust standard deviation (1.4826 is scaling factor for normal distribution)
        # robust_sd = 1.4826 * weighted_mad
        # return robust_sd

    def update(self, action_str: str, obs: HookObservation) -> HookBelief:
        new_belief = copy.deepcopy(self)

        if obs is None:
            return new_belief

        new_belief.conf = obs.conf
        new_belief.grasp = obs.grasp
        new_belief.grasped_obj = obs.grasped_obj

        # if obs.grasped_obj is not None:  # pickup, unstack
        #     # Find which main object this grasped_obj_idx (ghost particle) belongs to
        #     main_obj_idx = next(
        #         (obj_idx for obj_idx, ghost_indices in self.ghost_dict.items()
        #          if obs.grasped_obj in ghost_indices),
        #         None
        #     )
        #     if main_obj_idx is not None:
        #         new_belief.world.ghosts[main_obj_idx] = [obs.grasped_obj]  # Update belief with only the observed particle
        #         for ghost_idx in self.ghost_dict[main_obj_idx]:
        #             if ghost_idx != main_obj_idx:  # Hide all ghost particles except the main object
        #                 self.world.franka.scene.entities[ghost_idx].surface.opacity = 0

        for known_pose_idx, known_pose in obs.poses.items():  # place, stack
            # Find which main object this known_pose_idx (ghost particle) belongs to
            main_obj_idx = next(
                (obj_idx for obj_idx, ghost_indices in self.ghost_dict.items()
                 if known_pose_idx in ghost_indices),
                None
            )
            if main_obj_idx is not None:
                # Replace particles with observed pose
                new_belief.object_dists[main_obj_idx] = [known_pose]
                new_belief.object_weights[main_obj_idx] = [1.0]
                new_belief.world.ghosts[main_obj_idx] = [main_obj_idx]

        if len(obs.hook_traj) > 0:  # pull_towards
            # After pulling hook, update particle positions from physics
            # new_belief.simulate_hook_traj(obs.hook_traj)
            # After pulling hook, update ALL objects' particles from physics simulation
            # (since hook movement can affect other objects through collisions)
            for obj_idx in self.ghost_dict:
                new_belief.object_dists[obj_idx] = [
                    get_pose(self.world.franka.scene.entities[idx]) 
                    for idx in self.ghost_dict[obj_idx]
                ]
                # Update particle weights for this object
                new_belief.update_weights_by_deviation(obj_idx)

        return new_belief
    
    def abstract(self, expected_db: DiscreteBelief, problem: Problem) -> tuple[bool, DiscreteBelief]:
        """
        Abstract continuous belief to discrete facts.
        Returns grouped facts format: (predicate_name, ((arg1, arg2, ...), ...))
        for objects with known poses (low uncertainty).
        """
        # Actual known-pose objects based on current continuous belief
        known_pose_facts = []
        grouped_facts = None
        idx_to_name = {idx: name for name, idx in self.world.franka.object_dict.items()}
        
        for obj_idx in self.object_dists:
            if all(self.get_sd(obj_idx) < KNOWN_POSE_SD):
                obj_name = idx_to_name.get(obj_idx)
                print(f"[abstract] {obj_name} has low uncertainty (sd={self.get_sd(obj_idx)}) after execution, treating as known pose.")
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
    
    def sample_particle(self, obj):
        # """Sample a particle from the belief set."""
        # sampled_idx = random.choice(self.world.ghosts[obj])
        """Sample a particle from the belief set using weighted probability."""
        weights = self.object_weights[obj]
        idx = np.random.choice(len(self.world.ghosts[obj]), p=weights)
        sampled_idx = self.world.ghosts[obj][idx]
        return sampled_idx

    def compute_entropy(self, obj_idx):
        """Compute Shannon entropy of particle weights"""
        weights = np.array(self.object_weights[obj_idx])
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        return entropy

    def entropy_all(self):
        """Total entropy across all objects"""
        total_entropy = 0
        for obj_idx in self.object_dists:
            total_entropy += self.compute_entropy(obj_idx)
        return total_entropy

    def visualize_particles(self, save_path: str, step_name: str = ""):
        """Visualize all particles with weights using 3D scatter plots"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import FuncFormatter
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        num_objects = len(self.object_dists)
        fig = plt.figure(figsize=(6*num_objects, 5))
        
        for subplot_idx, (obj_idx, poses) in enumerate(self.object_dists.items()):
            ax = fig.add_subplot(1, num_objects, subplot_idx+1, projection='3d')
            
            # Get object name
            obj_name = None
            for name, idx in self.world.franka.object_dict.items():
                if idx == obj_idx:
                    obj_name = name
                    break
            
            # Extract positions and weights
            positions = np.array([pose[0:3] for pose in poses])
            weights = np.array(self.object_weights[obj_idx])
            
            # Visualize: constant size for all particles, color represents weight
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=weights, s=50, cmap='hot', alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Draw COM
            com = self.get_com(obj_idx)
            ax.scatter(*com, c='cyan', s=300, marker='*', label='COM', edgecolors='black', linewidth=1)
            
            # Format axes to 2 decimal places
            def format_func(value, tick_number):
                return f'{value:.2f}'
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{obj_name}\n(particles={len(poses)}, entropy={self.compute_entropy(obj_idx):.2f})')
            
            # Weight colorbar
            plt.colorbar(scatter, ax=ax, label='Weight', shrink=0.8)
            ax.legend(loc='upper right')
            
            # Set equal aspect ratio for better visualization
            ax.set_box_aspect([1,1,0.5])
        
        plt.suptitle(step_name, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_belief_trajectory(self, save_dir: str, belief_history: List['HookBelief'], action_names: List[str]):
        """Save series of belief updates as animation (GIF)"""
        from PIL import Image
        
        os.makedirs(save_dir, exist_ok=True)
        frames = []
        
        for i, (belief, action) in enumerate(zip(belief_history, action_names)):
            temp_path = f'{save_dir}/temp_{i:03d}.png'
            belief.visualize_particles(temp_path, f'Step {i}: {action}')
            frames.append(Image.open(temp_path))
        
        # Save as GIF
        frames[0].save(
            f'{save_dir}/belief_evolution.gif',
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        
        for i in range(len(frames)):
            try:
                os.remove(f'{save_dir}/temp_{i:03d}.png')
            except:
                pass
    
    def visualize_with_camera(self, save_dir: str, step_name: str = ""):
        """Save both camera image and particle visualization"""
        os.makedirs(save_dir, exist_ok=True)
        
        self.world.franka.cam_arm.render()
        camera_image = self.world.franka.cam_arm.rgb.cpu().numpy()
        
        img = Image.fromarray((camera_image * 255).astype(np.uint8))
        img.save(f'{save_dir}/{step_name}_camera.png')
        
        self.visualize_particles(f'{save_dir}/{step_name}_particles.png', step_name)


def add_gaussian_distributed_cubes(
    sim_wrapper, mean_pos, sd_dev, cube_color, cube_size=(0.04, 0.04, 0.04), num_cubes=100, alpha=0.3
):
    """Add small cubes distributed according to a Gaussian in the x and y
    directions, with random orientations.
    : sim_wrapper: Genesis wrapper instance.
    :param mean_pos: Tuple (x, y, z) indicating the mean pose.
    :param sd_dev: Standard deviation for Gaussian distribution.
    :param num_cubes: Number of cubes to generate.
    :param cube_size: Half-extent size of each cube (cubes are
        2*cube_size in each dimension).
    :return: List of cube IDs.
    """
    cubes = []

    for i in range(num_cubes):
        x_offset = np.random.normal(0, sd_dev)
        y_offset = np.random.normal(0, sd_dev)
        z_offset = 0  # Assuming no variation in z-direction

        cube_position = (
            mean_pos[0] + x_offset,
            mean_pos[1] + y_offset,
            mean_pos[2] + z_offset,
        )

        # Generate random orientation (Euler angles in degrees)
        yaw = np.random.uniform(-180, 180)
        random_euler = (0.0, 0.0, yaw)

        if i == 0:
            # main cube: collide with everything
            contype = MAIN
            conaffinity = PLANE | HOOK | ROBOT | MAIN
            rgbaColor = [cube_color[0], cube_color[1], cube_color[2], 1.0]
        else:
            # ghost cubes: dynamic, collide with plane and hook
            contype = GHOST
            conaffinity = PLANE | HOOK
            # With low alpha (transparency)
            rgbaColor = [cube_color[0], cube_color[1], cube_color[2], alpha]

        ent = sim_wrapper.scene.add_entity(
            morph=gs.morphs.Box(
                size=cube_size,
                pos=cube_position,
                euler=random_euler,
                fixed=False,
                contype=contype,
                conaffinity=conaffinity,
            ),
            surface=gs.surfaces.Rough(color=rgbaColor),
        )

        cubes.append(ent.idx)

    return cubes


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

    # Add blocks and hook
    ghosts = {}
    franka.object_dict = getattr(franka, "object_dict", {})  # add moveable objects

    for i, (name, info) in enumerate(objects_info.items()):
        if name=="hook":
            pos = tuple(info["pose"]["position"])
            hook = franka.scene.add_entity(
                morph=gs.morphs.URDF(
                    file="/home/minseo/Genesis/genesis/assets/urdf/hook/hook.urdf",
                    pos=pos,
                    euler=(0.0, 0.0, -30.0),
                ),
            )
            franka.object_dict[name] = hook.idx
        else:
            size = tuple(info["size"])
            pos = tuple(info["pose"]["position"])
            # Find color key in name, default to grey
            col = COLOR_MAP["grey"]
            for color_key in COLOR_MAP:
                if color_key in name:
                    col = COLOR_MAP[color_key]
                    break

            # ghost_cubes[0]: main cube idx
            # ghost_cubes[1:]: ghost particles idx
            ghost_cubes = add_gaussian_distributed_cubes(
                sim_wrapper=franka,
                mean_pos=pos,
                sd_dev=INITIAL_SD,
                cube_color=col,
                cube_size=size,
                num_cubes=num_distractor,
                alpha=2 / float(num_distractor),
            )

            ghosts[ghost_cubes[0]] = ghost_cubes[1:]
            franka.object_dict[name] = ghost_cubes[0]  # movable

    for geom in franka.robot.geoms:
        geom._contype = ROBOT
        geom._conaffinity = PLANE | MAIN | HOOK

    for geom in hook.geoms:
        geom._contype = HOOK
        geom._conaffinity = PLANE | MAIN | ROBOT

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
    # init_path = franka.motion_planning(init_qpos, planner="RRTConnect")
    # franka.move(init_path)
    franka.robot.control_dofs_position(init_qpos)
    for i in range(100):
        franka.scene.step()
        if record_video:
            franka.render_cameras()

    # check ee frame
    # init_pose = franka.current_ee_pose(draw=True)

    # move to a block (to check collision btw ghost cubes and robot)
    # init_pose = [0.48, 0.07, 0.04, 0, 1, 0, 0]  # [x y z qx qy qz qw]
    # ok1, traj1_or_err = franka.safe_plan(qpos_goal=franka.ik(init_pose, left=True), planner="RRTConnect", only_left=True)
    # franka.move(traj1_or_err, take_screenshot=False)

    # print("close gripper")
    # franka.close_gripper()
    # print("open gripper")
    # franka.open_gripper()

    # check ee frame
    # init_pose = franka.current_ee_pose(draw=True)

    # check camera frame
    # link_pos = franka.cam_arm._attached_link.get_pos().cpu().numpy()
    # link_quat = franka.cam_arm._attached_link.get_quat().cpu().numpy()
    # link_T = gs.utils.geom.trans_quat_to_T(link_pos, link_quat)
    # offset = franka.cam_arm._attached_offset_T.detach().cpu().numpy()
    # transform = link_T @ offset
    # mesh2 = franka.scene.draw_debug_frame(T=transform, axis_length=0.12, origin_size=0.01, axis_radius=0.005)

    world = HookWorld(franka=franka, ghosts=ghosts)


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
    hook = None
    
    for obj_name, pose in sampled_object_poses.items():
        info = objects_info.get(obj_name)
        if info is None:
            continue
        
        # Extract position and orientation from pose [x,y,z,qx,qy,qz,qw]
        pos = tuple(pose[0:3])
        quat = pose[3:7]  # [qx, qy, qz, qw]
        r = Rotation.from_quat(quat)
        euler = tuple(r.as_euler('xyz', degrees=True))
        
        if obj_name == "hook":
            # Add hook entity
            hook = franka.scene.add_entity(
                morph=gs.morphs.URDF(
                    file="/home/minseo/Genesis/genesis/assets/urdf/hook/hook.urdf",
                    pos=pos,
                    euler=euler,
                ),
            )
            franka.object_dict[obj_name] = hook.idx
        else:
            # Add block entity (main object only, no ghosts)
            size = tuple(info["size"])
            # Find color key in obj_name, default to grey
            col = COLOR_MAP["grey"]
            for color_key in COLOR_MAP:
                if color_key in obj_name:
                    col = COLOR_MAP[color_key]
                    break
            
            ent = franka.scene.add_entity(
                morph=gs.morphs.Box(
                    size=size,
                    pos=pos,
                    euler=euler,
                    fixed=False,
                    contype=MAIN,
                    conaffinity=PLANE | HOOK | ROBOT | MAIN,
                ),
                surface=gs.surfaces.Rough(color=col),
            )
            franka.object_dict[obj_name] = ent.idx
    
    # Set geometry contact masks
    for geom in franka.robot.geoms:
        geom._contype = ROBOT
        geom._conaffinity = PLANE | MAIN | HOOK
    
    if hook is not None:
        for geom in hook.geoms:
            geom._contype = HOOK
            geom._conaffinity = PLANE | MAIN | ROBOT
    
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
    
    return HookWorld(franka=franka, ghosts={})
    

def get_hook_traj(origin: Pose):
    """
    :param origin: center of mass of the particles
    : return pose trajectory of the hook w.r.t world frame
    """
    pre_hook_pose = multiply(Pose(Point(x=-0.0, y=-0.10, z=0.15)), origin)  # pre-grasp pose for the hook
    post_hook_pose_1 = multiply(Pose(Point(z=-0.15)), pre_hook_pose)  # move closer along z-axis
    post_hook_pose_2 = multiply(Pose(Point(x=-0.15, y=0.1)), post_hook_pose_1)  # pull towards the robot
    post_hook_pose_3 = multiply(Pose(Point(z=0.15)), post_hook_pose_2)  # lift

    intermediate_hook_poses = (
        [pre_hook_pose]
        + list(interpolate_poses(pre_hook_pose, post_hook_pose_1, pos_step_size=0.002))
        + list(interpolate_poses(post_hook_pose_1, post_hook_pose_2, pos_step_size=0.002))
        + list(interpolate_poses(post_hook_pose_2, post_hook_pose_3, pos_step_size=0.002))
    )
    return intermediate_hook_poses


class ToolUseEnv(TampuraEnv):
    def initialize(self, prob_num, prob_idx, trial, repeat) -> ContinuousBelief:
        self.world, file_path_list = start_sim(self.config["json_path"], self.config["planner"], prob_num, prob_idx, trial, repeat, show_viewer=False, record_video=self.config["record_video"])

        return HookBelief(self.world)


register_env("tool_use", ToolUseEnv)
