from pathlib import Path
import csv
import os
import re
import traceback
import torch
import time
from . import pr2_api as pr2, kuka_api as kuka
import json
import numpy as np


def append_csv_row(csv_path: Path, fieldnames, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def extract_plan_time(text: str):
    if not text:
        return None
    plan_times = re.findall(
        r"Plan:\s*\d+\s*\|\s*Cost:\s*[0-9.eE+-]+\s*\|\s*Time:\s*([0-9.]+)",
        text
    )
    if plan_times:
        try:
            return float(plan_times[-1])
        except Exception:
            pass
    summary_times = re.findall(
        r"Summary:\s*\{.*?run_time:\s*([0-9.]+)",
        text,
        flags=re.DOTALL
    )
    if summary_times:
        try:
            return float(summary_times[-1])
        except Exception:
            pass

    return None


def run_one_worker(func, kwargs, q):
    """Child process entry. Always puts a result or an error into q."""
    try:
        res = func(**kwargs)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", f"{e}\n{traceback.format_exc()}"))


def arm_is_left(pddl_args):
    if not pddl_args:
        return True
    if isinstance(pddl_args[0], str) and pddl_args[0].lower() in ("left", "right"):
        return pddl_args[0].lower() == "left"
    return "left" in [str(a).lower() for a in pddl_args]


def build_arm_traj_from_json(sim_wrapper, path_list, use_left, domain):
    traj = []
    current_qpos = sim_wrapper.robot.get_qpos().detach()
    device = current_qpos.device
    dtype = current_qpos.dtype

    if domain=="blocksworld_pr":
        move_idx = sim_wrapper.left_arm if use_left else sim_wrapper.right_arm
    elif domain=="kitchen":
        move_idx = [0, 1, 2, 3, 4, 5, 6]
    move_idx_t = torch.as_tensor(move_idx, dtype=torch.long, device=device)

    for wp in path_list:
        vals = torch.tensor(wp.get("values", []), dtype=dtype, device=device)
        if vals.numel() != move_idx_t.numel():
            raise ValueError(f"Waypoint dim mismatch: got {vals.numel()} values for {len(move_idx)} joints.")

        q = current_qpos.clone()
        q.index_copy_(0, move_idx_t, vals)
        traj.append(q)
    return traj


def execute_plan_genesis(problem_json_path, plan_json_path, robot_name, method, prob_num, prob_idx, trial, repeat, num_distractor=12):
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
            detached_object = sim_wrapper.scene.entities[sim_wrapper.object_dict[action_name.split()[1]]]
            sim_wrapper.close_gripper(object=detached_object)
        elif action_type == "stack" or action_type == "putdown" or action_type == "putdown_sink" or action_type == "putdown_stove" or action_type == "putdown_table":
            print(f"Opening gripper")
            attached_object = sim_wrapper.scene.entities[sim_wrapper.object_dict[action_name.split()[1]]]
            sim_wrapper.open_gripper(object=attached_object)

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