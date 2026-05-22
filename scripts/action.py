import sys
from pathlib import Path
import os
import logging
import random

logger = logging.getLogger("TAMP")

# Add parent directory to path so we can import utils and config
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import *


GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.10 + GRASP_LENGTH
TOOL_POSE = Pose(euler=Euler(pitch=np.pi/2))  # l_gripper_tool_frame (+x out of gripper arm)
MAX_GRASP_WIDTH = np.inf
SIDE_HEIGHT_OFFSET = 0.03  # z distance from top of object


def make_grasp(sim_wrapper, robot_name, object_name, direction_vec, center, extents, offset, grasp_type):
    """
    direction_vec: world-frame approach direction (normalized)
    approach_axis: which axis of the gripper frame should align with direction_vec (default: x)
    """
    grasps = []  # grasp and pre grasp
    # position = surface point + offset
    grasp_pos = center - (extents + offset) * direction_vec 
    grasp_pos = grasp_pos.tolist()

    # object pose
    object_quat = sim_wrapper.scene.entities[sim_wrapper.object_dict[object_name]].get_quat().cpu().detach().numpy().tolist()
    qw, qx, qy, qz = object_quat
    R_obj = R.from_quat([qx, qy, qz, qw]).as_matrix()
    ey = R_obj[:, 1]  # object y (world)

    # normalize
    u = np.asarray(direction_vec, dtype=float)
    u /= (np.linalg.norm(u) + 1e-12)

    # project ey onto plane ⟂ u so that y ⟂ x is guaranteed
    ey = np.asarray(ey, dtype=float)
    ey_proj = ey - np.dot(ey, u) * u
    n = np.linalg.norm(ey_proj)
    if n < 1e-9:
        # fallback: pick any vector perpendicular to u
        tmp = np.array([0.0, 0.0, 1.0])
        cand = np.cross(u, tmp)
        if np.linalg.norm(cand) < 1e-9:
            tmp = np.array([0.0, 1.0, 0.0])
            cand = np.cross(u, tmp)
        ey_proj = cand / (np.linalg.norm(cand) + 1e-12)
    else:
        ey_proj /= n

    # compute orientation: align gripper's x-axis to direction_vec, and gripper's y-axis to ey
    # default gripper frame: x forward, z down
    if robot_name == "dual_arm" or robot_name=="pr2":
        align1 = [1, 0, 0]
        align2 = [0, 1, 0]
    elif robot_name == "kuka" or robot_name=="franka":
        align1 = [0, 0, 1]
        align2 = [0, 1, 0]
    base_rot = R.align_vectors([u, ey_proj], [align1, align2])[0]

    if grasp_type == "top":
        if object_name == "hook":
            rotate = [0, np.pi]
        else:
            rotate = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    elif grasp_type == "side":
        rotate = [0, np.pi]

    for phi in rotate:
        # rotation about world-axis = direction_vec
        roll_rot = R.from_rotvec(u * phi)
        # compose: first align, then roll
        total_rot = roll_rot * base_rot

        # unpack to [qx, qy, qz, qw]
        qx, qy, qz, qw = total_rot.as_quat()
        grasp = [grasp_pos[0],grasp_pos[1],grasp_pos[2], qx, qy, qz, qw]
        pre_grasp = [
            grasp_pos[0] - direction_vec[0] * 0.06,
            grasp_pos[1] - direction_vec[1] * 0.06,
            grasp_pos[2] - direction_vec[2] * 0.06,
            qx, qy, qz, qw
        ]
        pre_grasp = [float(v) for v in pre_grasp]
        grasp = [float(v) for v in grasp]

        grasps.append([grasp, pre_grasp])

    return grasps


def get_grasp_pose(sim_wrapper, robot_name, object_name, grasp_type, belief, randomize=True):
    """
    Returns pre grasp pose of given object
    sim_wrapper : simulator wrapper (DualArm class type from dual_arm_api.py or PR2 class type fron pr2_api.py)
    object_name : Object name (string)
    grasp_type : 'top' or 'side'
    randomize : bool (default: True)
    return: array of [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]] (grasp and pre grasp)
    """
    obj_idx = sim_wrapper.object_dict[object_name]
    obj = sim_wrapper.scene.entities[obj_idx]
    aabb = obj.get_AABB().tolist()

    min_pt = np.array(aabb[0])
    max_pt = np.array(aabb[1])
    center = (min_pt + max_pt) / 2
    extents = (max_pt - min_pt) / 2  # half-sizes

    if robot_name == "pr2" or robot_name=="kuka" or robot_name=="franka":
        offset = 0.03  # 3cm away from surface
    elif robot_name == "dual_arm":
        offset = 0.00

    if 'top' in grasp_type:
        dir_vec = np.array([0, 0, -1])
        all_grasps = make_grasp(sim_wrapper, robot_name, object_name, dir_vec / np.linalg.norm(dir_vec), center, extents, offset, grasp_type)

    if 'side' in grasp_type:
        all_grasps = []
        directions = {
            'left': np.array([-1, 0, 0]),  # from +x side
            'right': np.array([1, 0, 0]),  # from -x side
            'front': np.array([0, -1, 0]),  # from +y side
            'back': np.array([0, 1, 0]),  # from -y side
        }
        for name, dir_vec in directions.items():
            grasps = make_grasp(sim_wrapper, robot_name, object_name, dir_vec / np.linalg.norm(dir_vec), center, extents, offset, grasp_type)
            all_grasps.extend(grasps)

    if randomize:
        random.shuffle(all_grasps)

    pre_grasp = all_grasps[0][1]
    approach = get_approach_vector_from_pose(pre_grasp, robot_name)

    return pre_grasp, approach, obj_idx


def get_grasp_ee(sim_wrapper, obj_idx):
    """Get a pose of the obj w.r.t to the end-effector frame"""

    # ee pose w.r.t world frame
    ee_link = sim_wrapper.robot.get_link(sim_wrapper.EE_FRAMES["ee"])
    ee_pos_w = ee_link.get_pos().cpu().numpy()
    ee_quat_w = ee_link.get_quat().cpu().numpy()
    world_T_ee = gs.utils.geom.trans_quat_to_T(ee_pos_w, ee_quat_w)

    # obj pose w.r.t world frame
    obj_pos_w = sim_wrapper.scene.entities[obj_idx].get_pos().cpu().numpy()
    obj_quat_w = sim_wrapper.scene.entities[obj_idx].get_quat().cpu().numpy()
    world_T_obj = gs.utils.geom.trans_quat_to_T(obj_pos_w, obj_quat_w)
    # Ensure it's on CPU as numpy array
    if isinstance(world_T_obj, torch.Tensor):
        world_T_obj = world_T_obj.cpu().numpy()
    if isinstance(world_T_ee, torch.Tensor):
        world_T_ee = world_T_ee.cpu().numpy()

    # Invert world_T_ee (work in numpy)
    R = world_T_ee[:3, :3]
    t = world_T_ee[:3, 3]
    ee_T_world = np.eye(4, dtype=np.float32)
    ee_T_world[:3, :3] = R.T
    ee_T_world[:3, 3] = -R.T @ t

    # ee_T_obj
    ee_T_obj = ee_T_world @ world_T_obj

    pos = ee_T_obj[:3, 3]
    # Convert to tensor for R_to_quat if needed
    R_obj = ee_T_obj[:3, :3]
    if isinstance(R_obj, np.ndarray):
        R_obj = torch.from_numpy(R_obj).float()
    quat = gs.utils.geom.R_to_quat(R_obj)  # w x y z
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def get_place_quat(sim_wrapper, robot_name, grasp_type, left=True):
    # random roll values for top grasp
    if grasp_type == 'top':
        approach = np.array([0.0, 0.0, -1.0])
        approach = approach / np.linalg.norm(approach)
        if robot_name == "dual_arm" or robot_name=="pr2":
            align_axis = [1, 0, 0]
        elif robot_name == "kuka" or robot_name=="franka":
            align_axis = [0, 0, 1]
        base_rot = R.align_vectors([approach], [align_axis])[0]

        phi = np.random.uniform(-np.pi, np.pi)
        roll_rot = R.from_rotvec(approach * phi)

        total = roll_rot * base_rot
        qx, qy, qz, qw = total.as_quat()  # [x,y,z,w]
        return [qx, qy, qz, qw]

    # random yaw values for side grasp
    elif grasp_type == 'side':
        link_name = sim_wrapper.EE_FRAMES['left'] if left else sim_wrapper.EE_FRAMES['right']
        cur_q = sim_wrapper.robot.get_link(link_name).get_quat().cpu().detach().numpy().tolist()
        qw, qx, qy, qz = cur_q

        rot = R.from_quat([qx, qy, qz, qw])  # [x,y,z,w]
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        yaw = np.random.uniform(-np.pi, np.pi)

        new_rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        qx, qy, qz, qw = new_rot.as_quat()
        return [qx, qy, qz, qw]


def get_place_pose(robot_name, object_name, region_name, sim_wrapper, grasp_type, belief, check_collisions=True, max_attempt=30):
    """
    Returns pre placement pose of given object on given region
    object_name : Object name (string)
    region_name : Region name (string or None)
    sim_wrapper : simulator wrapper (DualArm class type from dual_arm_api.py)
    grasp_type : 'top' or 'side'
    check_collisions: bool (default: True)
    max_attempt: int (default: 50)
    return: [[x, y, z, qx, qy, qz, qw], [x, y, z]] (pre pose and approach vector)
    """
    table_idx = sim_wrapper.object_dict.get('table')
    table = sim_wrapper.scene.entities[table_idx] if table_idx else None
    object_idx = sim_wrapper.object_dict.get(object_name)
    current_obj = sim_wrapper.scene.entities[object_idx] if object_idx else None 
    under_obj_idx = sim_wrapper.object_dict.get(region_name)
    under_obj = sim_wrapper.scene.entities[under_obj_idx] if under_obj_idx else None

    if under_obj is not None: # stack action
        bottom_aabb = under_obj.get_AABB().cpu().numpy()
        bottom_min_pt = bottom_aabb[0]
        bottom_max_pt = bottom_aabb[1]
        bottom_center = (bottom_min_pt + bottom_max_pt) / 2
        bottom_extents = (bottom_max_pt - bottom_min_pt) / 2  # half-sizes
        ignore_entities = [under_obj, table] if table is not None else [under_obj]
    else:
        ignore_entities = [table] if table else None

    upper_aabb = current_obj.get_AABB().cpu().numpy()
    upper_min_pt = upper_aabb[0]
    upper_max_pt = upper_aabb[1]
    upper_extents = (upper_max_pt - upper_min_pt) / 2  # half-sizes

    offset = 0.03
    collided_names = set()

    for attempt in range(max_attempt):
        if region_name in sim_wrapper.object_dict:
            # when stacking an obj on some obj
            if robot_name=="pr2" or robot_name=="franka":
                pos = bottom_center - (bottom_extents + 2 * upper_extents + offset + 0.06) * np.array([0, 0, -1])
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2])
            elif robot_name=="dual_arm":
                pos = bottom_center - (bottom_extents + 2 * upper_extents + 0.06) * np.array([0, 0, -1])
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2])
            elif robot_name=="kuka":
                pos = bottom_center - (bottom_extents + 2 * upper_extents + offset + 0.18) * np.array([0, 0, -1])
                z = float(pos[2])
                xy = [[bottom_min_pt[0]+0.02, bottom_max_pt[0]-0.02], [bottom_min_pt[1]+0.02, bottom_max_pt[1]-0.02]]
                x = random.uniform(*sorted(xy[0]))
                y = random.uniform(*sorted(xy[1]))

            # sample random orientation
            qx, qy, qz, qw = get_place_quat(sim_wrapper, robot_name, grasp_type)

        # putting an obj on the table
        else:
            if object_name == "hook":
                x = 0.1
                y = -0.35
                z = 0.075
            else:
                # sample random position
                rb = sim_wrapper.region_dict['table']
                x = random.uniform(*sorted(rb[0]))
                y = random.uniform(*sorted(rb[1]))
                z = random.uniform(*sorted(rb[2]))

            # sample random orientation
            qx, qy, qz, qw = get_place_quat(sim_wrapper, robot_name, grasp_type)

        place_pose = [x, y, z, qx, qy, qz, qw]

        approach = get_approach_vector_from_pose(place_pose, robot_name)
        logger.info(f"collision check attempt: {attempt}")

        if check_collisions:
            obj_target_pose = place_pose.copy()
            if robot_name=="pr2" or robot_name=="franka":
                obj_target_pose[2] -= upper_extents[2] + offset + 0.06
            elif robot_name=="dual_arm":
                obj_target_pose[2] -= upper_extents[2] + 0.06
            elif robot_name=="kuka":
                obj_target_pose[2] -= upper_extents[2] + offset + 0.15

            ok, pairs_all, pairs_kept = check_place_pose_collision(
                current_obj,
                obj_target_pose,
                ignore_entities=ignore_entities
            )
            pairs_kept = [] if pairs_kept is None else list(pairs_kept)

            if not ok:
                def _name_of_entity(ent):
                    for nm, obj_idx in sim_wrapper.object_dict.items():
                        if sim_wrapper.scene.entities[obj_idx] is ent:
                            return nm
                    return getattr(ent, "name", str(type(ent)))

                for item in pairs_kept:
                    other_ent = None
                    if isinstance(item, (tuple, list)) and len(item) >= 1:
                        cand = item[-1]
                        if hasattr(cand, "get_AABB") or hasattr(cand, "get_link"):
                            other_ent = cand
                    elif hasattr(item, "entity"):
                        other_ent = item.entity

                    if other_ent is not None:
                        collided_names.add(_name_of_entity(other_ent))

                if collided_names:
                    logger.info(f"[place] collision with: {sorted(collided_names)}")
                else:
                    logger.info(f"[place] collision detected (pairs: {pairs_kept})")

                continue
        return place_pose, approach, None, obj_target_pose

    return None, None, collided_names, None


def get_goal(sim_wrapper, robot, action, belief, left, grasp_type, domain_name='tool_use'):
    """
    Sample goal pose for given action
    sim_wrapper: simulator wrapper (DualArm class type from dual_arm_api.py)
    robot: type of a robot ('pr2' or 'dual_arm' or 'kuka')
    action: pddl action (string)
    left: left gripper (bool)
    grasp_type: grasp type ('top' or 'side')
    belief: belief state of the world (HookBelief, etc)
    domain_name: domain name (default: 'tool_use')
    return: [success (bool), [action_type (string), pre goal pose (list), approach vector (list)]]
    """
    tokens = action.strip("() ").split()
    act_type = tokens[0]
    params = tokens[1:]

    payload = {
        "ok": False,  # success
        "where": "goal",
        "action": action,  # grounded pddl action
        "act_type": act_type,  # action name
        "pose_traj": None,  # list of ee poses [[x,y,z,qw,qx,qy,qz], ...]
        "approach": None,  # [vx, vy, vz]
        "image_path": None,
        "error": None,
        "obj_target_pose": None,
    }

    try:
        if act_type in ["unstack", "pickup"]:
            object = params[0]
            grasp_pose_xyzw, approach, obj_idx = get_grasp_pose(sim_wrapper, robot, object, grasp_type, belief, randomize=True)

            if grasp_pose_xyzw is None or approach is None or len(grasp_pose_xyzw) != 7:
                payload["error"] = f"failed to find grasp pose for object {object}"
                logger.info("invalid_grasp_pose")
                return False, payload
            grasp_pose_wxyz = xyzw_to_wxyz(grasp_pose_xyzw)

            if robot == 'pr2' or robot == 'dual_arm':
                offset = 0.10
                move_height = 0.15
            elif robot == 'kuka':
                offset = 0.08
                move_height = 0.15
            elif robot == 'franka':
                offset = 0.095
                move_height = 0.095

            approach = np.asarray(approach, dtype=float)

            target_pose = np.array(grasp_pose_wxyz, dtype=float)
            target_pose[0:3] += offset * approach
            target_pose = target_pose.tolist()

            after_target_pose = np.array(grasp_pose_wxyz, dtype=float)
            after_target_pose[0:3] -= move_height * approach
            after_target_pose = after_target_pose.tolist()

            payload.update(ok=True, pose_traj=[grasp_pose_wxyz, target_pose, after_target_pose], approach=approach)

        elif act_type in ["putdown","stack","putdown_sink","putdown_stove","putdown_table"]:
            obj = params[0]
            region = params[1] if len(params) > 1 else None

            place_pose_xyzw, approach, collided_names, under_obj_certain_pose = get_place_pose(robot, obj, region, sim_wrapper, grasp_type, belief, check_collisions=True)
            if place_pose_xyzw is None or len(place_pose_xyzw) != 7:
                payload["error"] = f"failed to find collision-free placement pose for object {obj}: due to collision with {collided_names}"
                logger.info("invalid_place_pose")
                return False, payload
            place_pose_wxyz = xyzw_to_wxyz(place_pose_xyzw)

            if robot == 'pr2' or robot == 'dual_arm':
                offset = 0.10
                move_height = 0.15
            elif robot == 'kuka':
                offset = 0.20
                move_height = 0.15
            elif robot == 'franka':
                offset = 0.095
                move_height = 0.095

            approach_arr = np.array(approach)
            target_pose = place_pose_wxyz.copy()
            target_pose[0:3] = (np.array(target_pose[0:3]) + offset * approach_arr).tolist()
            after_target_pose = place_pose_wxyz.copy()
            after_target_pose[0:3] = (np.array(after_target_pose[0:3]) - move_height * approach_arr).tolist()

            payload.update(ok=True, pose_traj=[place_pose_wxyz, target_pose, after_target_pose], approach=approach, obj_target_pose=under_obj_certain_pose)

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"

    if not payload["ok"]:
        logger.info(f"[get_goal] fail: {payload['error']} for action '{action}'")
    return payload["ok"], payload


def get_path_pickup_unstack(method, prob_num, prob_idx, trial, repeat, subgoal_idx, node_name, sim_wrapper, action, goal_payload, domain_name, belief, left=True):
    """
    Handle pickup and unstack actions (3 trajectories, close gripper)
    """
    payload = {
        "ok": False,
        "where": "path",
        "action": action,
        "act_type": goal_payload.get("act_type"),
        "pose_traj": goal_payload.get("pose_traj"),
        "approach": goal_payload.get("approach"),
        "traj": None,
        "image_path": None,
        "error": None,
        "obj_target_pose": None,
    }

    if not goal_payload.get("ok"):
        payload["error"] = "goal_failed"
        return False, payload
    
    try:
        act_type = goal_payload["act_type"]
        ee_pose1 = np.array(goal_payload["pose_traj"][0])
        ee_pose2 = np.array(goal_payload["pose_traj"][1])
        ee_pose3 = np.array(goal_payload["pose_traj"][2])
        object_name = action.split()[1]
        object_idx = sim_wrapper.object_dict[object_name]
        object = sim_wrapper.scene.entities[object_idx]
        joint_angle = sim_wrapper.robot.get_dofs_position()

        # visualize initial state / goal state in frames
        init_pose = sim_wrapper.current_ee_pose()
        init_pose = [init_pose[0], init_pose[1], init_pose[2], init_pose[6], init_pose[3], init_pose[4], init_pose[5]]
        init_mat = pose7_to_T44(init_pose)
        pre_target_mat = pose7_to_T44(ee_pose1)
        after_target_mat = pose7_to_T44(ee_pose3)

        mesh1 = sim_wrapper.scene.draw_debug_frame(T=init_mat, axis_length=0.12, origin_size=0.02, axis_radius=0.005,
                                                origin_color=(255, 255, 0, 255))
        mesh1 = sim_wrapper.scene.draw_debug_frame(T=pre_target_mat, axis_length=0.12, origin_size=0.02, axis_radius=0.005,
                                                origin_color=(255, 0, 255, 255))
        mesh1 = sim_wrapper.scene.draw_debug_frame(T=after_target_mat, axis_length=0.12, origin_size=0.02,
                                                   axis_radius=0.005,
                                                   origin_color=(0, 0, 255, 255))

        # plan traj1: move to pre-grasp pose
        ok1, traj1_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose1, left=left),
            qpos_start=joint_angle,
            planner="RRTConnect",
            ignore_collision=False,
            only_left=True
        )
        if not ok1:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}", f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving to pre grasp pose: {traj1_or_err}"
            logger.info(f"{traj1_or_err}")
            return False, payload
        
        traj1 = traj1_or_err
        sim_wrapper.move(traj1, take_screenshot=False)

        # plan traj2: move closer to grasp pose
        ok2, traj2_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose2, left=left),
            planner="RRTConnect",
            ignore_collision=False,
            only_left=True
        )
        if not ok2:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving closer to grasp pose: {traj2_or_err}"
            logger.info(f"{traj2_or_err}")
            return False, payload
        
        traj2 = traj2_or_err
        sim_wrapper.move(traj2, take_screenshot=False)

        # close gripper for pickup/unstack
        sim_wrapper.close_gripper(object_name=object_name, attach=True)

        # plan traj3: move away after grasping
        ok3, traj3_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose3, left=left),
            planner="RRTConnect",
            ignore_collision=False,
            only_left=True,
            ee_link_name=sim_wrapper.EE_FRAMES['ee'],
            with_entity=object
        )
        if not ok3:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving away after grasping: {traj3_or_err}"
            logger.info(f"{traj3_or_err}")
            return False, payload
        
        traj3 = traj3_or_err
        sim_wrapper.move(traj3, take_screenshot=False)

        # capture the last state
        file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}", f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
        payload.update(ok=True, traj=[traj1, traj2, traj3], image_path=file_path_list)
        return True, payload

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        return False, payload


def get_path_putdown_stack(method, prob_num, prob_idx, trial, repeat, subgoal_idx, node_name, sim_wrapper, action, goal_payload, domain_name, belief, left=True):
    """
    Handle putdown, stack, putdown_sink, putdown_stove, putdown_table actions (3 trajectories, open gripper)
    """
    payload = {
        "ok": False,
        "where": "path",
        "action": action,
        "act_type": goal_payload.get("act_type"),
        "pose_traj": goal_payload.get("pose_traj"),
        "approach": goal_payload.get("approach"),
        "traj": None,
        "image_path": None,
        "error": None,
        "obj_target_pose": None,
    }

    if not goal_payload.get("ok"):
        payload["error"] = "goal_failed"
        return False, payload
    
    try:
        act_type = goal_payload["act_type"]
        ee_pose1 = np.array(goal_payload["pose_traj"][0])
        ee_pose2 = np.array(goal_payload["pose_traj"][1])
        ee_pose3 = np.array(goal_payload["pose_traj"][2])
        object_name = action.split()[1]
        object_idx = sim_wrapper.object_dict[object_name]
        object = sim_wrapper.scene.entities[object_idx]
        joint_angle = sim_wrapper.robot.get_dofs_position()

        # visualize initial state / goal state in frames
        init_pose = sim_wrapper.current_ee_pose()
        init_pose = [init_pose[0], init_pose[1], init_pose[2], init_pose[6], init_pose[3], init_pose[4], init_pose[5]]
        init_mat = pose7_to_T44(init_pose)
        pre_target_mat = pose7_to_T44(ee_pose1)
        after_target_mat = pose7_to_T44(ee_pose3)

        mesh1 = sim_wrapper.scene.draw_debug_frame(T=init_mat, axis_length=0.12, origin_size=0.02, axis_radius=0.005,
                                                origin_color=(255, 255, 0, 255))
        mesh1 = sim_wrapper.scene.draw_debug_frame(T=pre_target_mat, axis_length=0.12, origin_size=0.02, axis_radius=0.005,
                                                origin_color=(255, 0, 255, 255))
        mesh1 = sim_wrapper.scene.draw_debug_frame(T=after_target_mat, axis_length=0.12, origin_size=0.02,
                                                   axis_radius=0.005,
                                                   origin_color=(0, 0, 255, 255))

        # plan traj1: move to pre-placement pose
        with_entity = object
        ee_link_name = sim_wrapper.EE_FRAMES['ee']
        ok1, traj1_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose1, left=left),
            qpos_start=joint_angle,
            planner="RRTConnect",
            ignore_collision=False,
            only_left=True,
            ee_link_name=ee_link_name,
            with_entity=with_entity
        )
        if not ok1:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving to pre placement pose: {traj1_or_err}"
            logger.info(f"{traj1_or_err}")
            return False, payload
        
        traj1 = traj1_or_err
        sim_wrapper.move(traj1, take_screenshot=False)

        # plan traj2: move closer to placement pose
        ok2, traj2_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose2, left=left),
            planner="RRTConnect",
            ignore_collision=True,
            only_left=True,
            ee_link_name=ee_link_name,
            with_entity=with_entity
        )
        if not ok2:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving closer to placement pose: {traj2_or_err}"
            logger.info(f"{traj2_or_err}")
            return False, payload
        
        traj2 = traj2_or_err
        sim_wrapper.move(traj2, take_screenshot=False)

        # open gripper for putdown/stack actions
        sim_wrapper.open_gripper(object_name=object_name, attach=True)

        # plan traj3: move away after releasing
        ok3, traj3_or_err = sim_wrapper.safe_plan(
            qpos_goal=sim_wrapper.ik(ee_pose3, left=left),
            planner="RRTConnect",
            ignore_collision=False,
            only_left=True
        )
        if not ok3:
            file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
            payload["image_path"] = file_path_list
            payload["error"] = f"Action {action}: motion planning failed when moving away after releasing: {traj3_or_err}"
            logger.info(f"{traj3_or_err}")
            return False, payload
        
        traj3 = traj3_or_err
        sim_wrapper.move(traj3, take_screenshot=False)

        # capture the last state
        file_path_list = sim_wrapper.save_snapshot4(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, method, "screenshots", f"{prob_num}_{prob_idx}_{trial}_{repeat}",f"subgoal{subgoal_idx}")), node_name=node_name, world=belief.world if belief is not None else None)
        payload.update(ok=True, traj=[traj1, traj2, traj3], image_path=file_path_list, obj_target_pose=goal_payload.get("obj_target_pose"))
        return True, payload

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        return False, payload


def execute(method, prob_num, prob_idx, trial, repeat, subgoal_idx, node_name, domain_name, robot_name, sim_wrapper, action, belief=None, left=True, grasp_type='top'):
    """
    Main function to execute a high-level action: sample goal pose, plan path, then execute
    """
    ok_goal, goal_payload = get_goal(sim_wrapper, robot_name, action, belief=belief, left=left, grasp_type=grasp_type, domain_name=domain_name)
    if not ok_goal:
        return False, goal_payload
    
    act_type = goal_payload["act_type"]
    
    # Route to appropriate get_path function based on action type
    if act_type in ["pickup", "unstack"]:
        ok_path, path_payload = get_path_pickup_unstack(method, prob_num, prob_idx, trial, repeat, subgoal_idx, node_name, sim_wrapper, action, goal_payload, domain_name, belief=belief, left=left)
    elif act_type in ["putdown", "stack", "putdown_sink", "putdown_stove", "putdown_table"]:
        ok_path, path_payload = get_path_putdown_stack(method, prob_num, prob_idx, trial, repeat, subgoal_idx, node_name, sim_wrapper, action, goal_payload, domain_name, belief=belief, left=left)
    else:
        return False, {"ok": False, "error": f"Unknown action type: {act_type}"}
    
    if not ok_path:
        return False, path_payload

    return True, path_payload

