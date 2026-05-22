import time

import pr2_api as pr2, kuka_api as kuka
import os, json
import numpy as np


def execute_plan(problem_json_path, plan_json_path, robot_name, method, prob_num, prob_idx, trial, repeat, num_distractor=12):
    moveable_joints = np.array([])
    if robot_name=='pr2':
        sim_wrapper, root_image_paths = pr2.start_sim(problem_json_path, method, prob_num, prob_idx, trial, repeat)
        moveable_joints = sim_wrapper.left_arm
    elif robot_name=='kuka':
        sim_wrapper, root_image_paths = kuka.start_sim(problem_json_path, method, prob_num, prob_idx, trial, repeat, num_distractor=num_distractor)
        moveable_joints = np.array([0, 1, 2, 3, 4, 5, 6])
    init_scene = sim_wrapper.scene.sim.get_state()

    if not os.path.exists(plan_json_path):
        print(f"extract_traj: file not found: {plan_json_path}")
        return []

    with open(plan_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plan = data.get("plan", [])

    time.sleep(15)

    for i, act in enumerate(plan):
        cp = act.get("continuous_params", {})
        traj = cp.get("traj", None)
        action_type = cp.get("act_type", None)
        action_name = cp.get("action", None)
        if action_type == "":
            continue
        print(f"Executing action ({action_name})")

        if traj is None or not isinstance(traj, list) or len(traj) < 3:
            print(f"execute_plan: action {i} '{action_name}' has invalid traj; skip")
            return []

        # --- Move to pre pose ---
        print(f"Moving to pre pose")
        traj1 = traj[0]
        if not isinstance(traj1, list) or len(traj1) == 0:
            print("execute_plan: traj[0] invalid; skip action")
            return []
        # traj1_left = []
        # for wp in traj1:
        #     sel = _select_indices(wp, moveable_joints)
        #     if sel is None:
        #         print("execute_plan: traj1 index selection failed; skip action")
        #         return []
        #     traj1_left.append(sel)
        # if traj1_left:
        sim_wrapper.move(traj1, take_screenshot=False)

        # --- Move to pose ---
        print(f"Moving closer")
        traj2 = traj[1]
        # traj2_left = []
        # for wp in traj2:
        #     sel = _select_indices(wp, moveable_joints)
        #     if sel is None:
        #         print("execute_plan: traj2 index selection failed; skip action")
        #         traj2_left = []
        #         return []
        #     traj2_left.append(sel)
        # if traj2_left:
        sim_wrapper.move(traj2, take_screenshot=False)

        # --- Gripper action ---
        print("object: ", action_name.split()[1])
        if action_type == "unstack" or action_type == "pickup":
            print(f"Closing gripper")
            sim_wrapper.close_gripper(object=sim_wrapper.object_dict[action_name.split()[1]])
        elif action_type == "stack" or action_type == "putdown" or action_type == "putdown_sink" or action_type == "putdown_stove" or action_type == "putdown_table":
            print(f"Opening gripper")
            sim_wrapper.open_gripper(object=sim_wrapper.object_dict[action_name.split()[1]])

        # --- Move up / retreat ---
        print("Moving up")
        traj3 = traj[2]
        # traj3_left = []
        # for wp in traj3:
        #     sel = _select_indices(wp, moveable_joints)
        #     if sel is None:
        #         print("execute_plan: traj3 index selection failed; skip action")
        #         traj3_left = []
        #         return []
        #     traj3_left.append(sel)
        # if traj3_left:
        sim_wrapper.move(traj3, take_screenshot=False)


    print("Successfully finished executing the plan!")
