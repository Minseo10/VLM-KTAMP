import os
import json
import subprocess
from pathlib import Path
from utils import pr2_api as pr2, kuka_api as kuka
import genesis as gs
import torch
import time
import re
import multiprocessing as mp
import signal
from utils.experiment_utils import *


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PDDLSTREAM_ROOT = PROJECT_ROOT / "pddlstream"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def run_genesis(domain, problem_json_path, method, prob_num, prob_idx, trial, repeat, planning_time, record_video=True, num_distractor=0):
    export_dir = (EXPERIMENTS_DIR / domain / method / "plan").resolve()
    export_json = export_dir / f"{prob_num}_{prob_idx}_{trial}_{repeat}.json"
    with open(export_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # start genesis
    if domain=="blocksworld_pr":
        sim_wrapper, root_image_paths = pr2.start_sim(problem_json_path, method, prob_num, prob_idx, trial, repeat)
    elif domain=="kitchen":
        sim_wrapper, root_image_paths = kuka.start_sim(problem_json_path, method, prob_num, prob_idx, trial, repeat, num_distractor)
    start = time.time()

    for it in plan.get("items", []):
        action = it.get("pddl_action")
        args = it.get("pddl_args", [])
        cmds_group = it.get("commands", {})
        cmds = (cmds_group or {}).get("commands", [])

        if action == "move_base" or action=="clean" or action=="cook":
            print(f"[genesis] Skip base action: {it.get('idx')}")
            continue

        is_left = arm_is_left(args)

        for cmd in cmds:
            ctype = cmd.get("type")

            if ctype == "Trajectory":
                path_list = cmd.get("path", [])
                if not path_list:
                    continue
                traj = build_arm_traj_from_json(sim_wrapper, path_list, use_left=is_left, domain=domain)
                sim_wrapper.move(traj, take_screenshot=False)

            elif ctype == "Attach":  # TODO
                if domain=="kitchen":
                    attached_object = sim_wrapper.scene.entities[sim_wrapper.object_dict[cmd.get("body")]]
                    sim_wrapper.close_gripper(object=attached_object)
                elif domain=="blocksworld_pr":
                    sim_wrapper.close_gripper(left=bool(cmd.get("left", is_left)))

            elif ctype == "Detach":
                if domain == "kitchen":
                    detached_object = sim_wrapper.scene.entities[sim_wrapper.object_dict[cmd.get("body")]]
                    sim_wrapper.open_gripper(object=detached_object)
                elif domain == "blocksworld_pr":
                    sim_wrapper.open_gripper(left=bool(cmd.get("left", is_left)))

            elif ctype == "GripperCommand":
                tgt = cmd.get("target")
                if isinstance(tgt, (int, float)):
                    val = float(tgt)
                elif isinstance(tgt, (list, tuple)) and tgt:
                    try:
                        val = float(sum(tgt) / len(tgt))
                    except Exception:
                        val = None
                else:
                    val = None

                if val is not None:
                    if val <= 0.1:
                        sim_wrapper.close_gripper(left=bool(cmd.get("left", is_left)))
                    elif val >= 0.5:
                        sim_wrapper.open_gripper(left=bool(cmd.get("left", is_left)))

            elif ctype == "Group":
                sub_cmds = cmd.get("commands", [])
                for sub in sub_cmds:
                    cmds.append(sub)

            else:
                print(f"[genesis] Unknown cmd type: {ctype}")

    base_exp_dir = EXPERIMENTS_DIR / domain
    out_dir = base_exp_dir / method / "batch_outputs" / f"{prob_num}_{prob_idx}" / f"trial{trial}_repeat{repeat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    screenshot_dir = EXPERIMENTS_DIR / domain / method / "screenshots" / f"{prob_num}_{prob_idx}_{trial}_{repeat}"

    if record_video:
        sim_wrapper.cam_front.stop_recording(video_path)

    sim_wrapper.save_snapshot4(screenshot_dir, node_name="final")
    gs.destroy()
    end = time.time()

    res = {
        "prob_num": prob_num,
        "prob_idx": prob_idx,
        "trial": trial,
        "repeat": repeat,
        "planning_success": True,
        "planning_time_sec": planning_time,
        "sim_time_sec": end-start,
        "sim_success": None,
    }

    return res


def run_pddlstream(domain, problem_json_path, method, prob_num, prob_idx, trial, repeat, algorithm="adaptive", unit=True, enable=False, num_distractor=0, max_time=600):
    pddlstream_root = PDDLSTREAM_ROOT
    base_dir = EXPERIMENTS_DIR / domain / method
    export_dir = base_dir / "plan"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_json = export_dir / f"{prob_num}_{prob_idx}_{trial}_{repeat}.json"

    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{prob_num}_{prob_idx}_{trial}_{repeat}.log"

    # Build command
    if domain =="blocksworld_pr":
        cmd = [
            "python", "-m", "examples.pybullet.blocksworld_pr2.run",
            "-prob_num", str(prob_num),
            "-prob_idx", str(prob_idx),
            "-trial", str(trial),
            "--export_json", str(export_json),
            "-a", str(algorithm),
            "--json_path", str(problem_json_path)
        ]
    elif domain == "kitchen":
        cmd = [
            "python", "-m", "examples.pybullet.kuka.run",
            "--num_object", str(6),
            "--num_goal", str(prob_num),
            "--num_distractor", str(num_distractor),
            "--trial", str(trial),
            "--export_json", str(export_json),
            "-a", str(algorithm),
            "--json_path", str(problem_json_path)
        ]
    if unit:
        cmd.append("-u")
    if enable:
        cmd.append("-enable")
    if max_time is not None:
        cmd += ["-t", str(max_time)]

    print(f"[run] cwd = {pddlstream_root}")
    print(f"[run] cmd = {' '.join(cmd)}")
    print(f"[run] log -> {log_path}")

    # Change working directory and run
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(pddlstream_root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    end = time.time()

    try:
        combined = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        combined = ""

    plan_time = extract_plan_time(combined)
    if plan_time is None:
        plan_time = ""

    success = False
    m = re.search(r"Solved:\s*(True|False)", combined)
    if m:
        success = (m.group(1) == "True")

    if proc.returncode != 0:
        if proc.returncode < 0:
            sig = -proc.returncode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"SIG{sig}"
            print(f"\n[run] FAILED by signal {sig} ({sig_name})")
        else:
            print(f"\n[run] FAILED with exit code {proc.returncode}")
        return False, plan_time

    print(f"\n[run] DONE. See log: {log_path}")
    return success, plan_time


def run_pddlstream_batch(domain="blocksworld_pr",
                         problem_json_path="./experiments/kitchen/problem/problems_meta.json",
                         method="pddlstream",
                         algorithm="adaptive",
                         prob_num_range=range(3, 8),
                         prob_idx_range=range(1, 6),
                         trial_range=range(1, 3),
                         repeat_range=range(1, 4),
                         num_distractor=0,
                         timeout=600):

    summary_dir = Path(f"./experiments/{domain}/{method}/batch_outputs").resolve()
    summary_csv = summary_dir / "summary.csv"

    fieldnames = [
        "prob_num", "prob_idx", "trial", "repeat",
        "planning_success", "planning_time_sec",
        "sim_time_sec", "sim_success",
        "plan_json_path", "timed_out", "error",
    ]

    for prob_num in prob_num_range:
        for prob_idx in prob_idx_range:
            for trial in trial_range:
                for repeat in repeat_range:
                    print(f"\n=== RUN: prob_num={prob_num}, prob_idx={prob_idx}, trial={trial}, repeat={repeat}===")
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    plan_json_path = EXPERIMENTS_DIR / domain / method / "plan" / f"{prob_num}_{prob_idx}_{trial}_{repeat}.json"

                    # Planning with PDDLStream
                    try:
                        success, planning_time = run_pddlstream(
                            domain=domain,
                            problem_json_path=problem_json_path,
                            method=method,
                            prob_num=prob_num,
                            prob_idx=prob_idx,
                            trial=trial,
                            repeat=repeat,
                            algorithm=algorithm,
                            unit=True,
                            enable=False,
                            num_distractor=num_distractor,
                            max_time=timeout,
                        )
                    except Exception as e:
                        row = {
                            "prob_num": prob_num, "prob_idx": prob_idx, "trial": trial, "repeat": repeat,
                            "planning_success": False, "planning_time_sec": "",
                            "sim_time_sec": "", "sim_success": "",
                            "timed_out": "", "error": f"planning_error: {e}",
                        }
                        append_csv_row(summary_csv, fieldnames, row)
                        continue

                    # Run the plan in Genesis and record results (Optional)
                    res = {
                        "prob_num": prob_num, "prob_idx": prob_idx, "trial": trial, "repeat": repeat,
                        "planning_success": bool(success),
                        "planning_time_sec": planning_time,
                        "sim_time_sec": "",
                        "sim_success": "",
                        "plan_json_path": str(plan_json_path),
                        "timed_out": "",
                        "error": "",
                    }

                    """
                    if success:
                        try:
                            ctx = mp.get_context("spawn")
                            q = ctx.Queue()
                            wargs = dict(domain=domain, problem_json_path=problem_json_path, method=method, prob_num=prob_num, prob_idx=prob_idx,
                                         trial=trial, repeat=repeat, planning_time=planning_time, record_video=True, num_distractor=num_distractor)
                            p = ctx.Process(target=run_one_worker, args=(run_genesis, wargs, q))
                            p.start()
                            p.join(300)

                            if p.is_alive():
                                print(f"[TIMEOUT] Killing run (>{300}s).")
                                p.terminate()
                                p.join()
                                res.update({
                                    "sim_time_sec": "",
                                    "sim_success": False,
                                    "timed_out": True,
                                    "error": "timeout(>300s)",
                                })
                            else:
                                try:
                                    status, payload = q.get_nowait()
                                    if status == "ok":
                                        res.update({
                                            "sim_time_sec": payload.get("sim_time_sec", ""),
                                            "sim_success": payload.get("sim_success", ""),
                                            "timed_out": False,
                                        })
                                    else:
                                        res.update({"error": "child_error", "timed_out": False})
                                except Exception as e:
                                    res.update({"error": f"no child payload: {e}", "timed_out": False})
                        except Exception as e:
                            res.update({"error": f"sim_error: {e}", "timed_out": False})
                    """

                    append_csv_row(summary_csv, fieldnames, res)