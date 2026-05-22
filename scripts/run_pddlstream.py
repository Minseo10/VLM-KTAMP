import os
import json
import subprocess
from pathlib import Path
from utils import pr2_api as pr2, kuka_api as kuka
import genesis as gs
import torch
import time
import re, csv
import queue as pyqueue
import multiprocessing as mp
import traceback
import signal


def _run_one_worker(kwargs, q):
    """Child process entry. Always puts a result or an error into q."""
    try:
        res = run_genesis(**kwargs)  # float (sec)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", f"{e}\n{traceback.format_exc()}"))

def _extract_plan_time(text: str):
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

def run_pddlstream(domain, problem_json_path, method, prob_num, prob_idx, trial, repeat, algorithm="adaptive", unit=True, enable=False, num_distractor=0, max_time=900):
    # Paths
    pddlstream_root = Path(f"/home/minseo/develop/pddlstream").resolve()
    base_dir = Path(f"/home/minseo/robot_ws/src/tamp_llm/experiments/{domain}/{method}").resolve()
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
            stderr=subprocess.STDOUT,   # stderr도 같은 파일로
            text=True,
            check=False,
        )
    end = time.time()

    try:
        combined = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        combined = ""

    plan_time = _extract_plan_time(combined)
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

def _append_csv_row(csv_path: Path, fieldnames, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())

def run_pddlstream_batch(domain="blocksworld_pr",
                         problem_json_path="/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem/problems_meta.json",
                         method="pddlstream",
                         algorithm="adaptive",
                         prob_num_range=range(3, 8),
                         prob_idx_range=range(1, 6),
                         trial_range=range(1, 3),
                         repeat_range=range(1, 4),
                         num_distractor=0,
                         timeout=600):

    summary_dir = Path(f"../experiments/{domain}/{method}/batch_outputs").resolve()
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
                    plan_json_path = Path(
                        f"/home/minseo/robot_ws/src/tamp_llm/experiments/{domain}/{method}/plan/{prob_num}_{prob_idx}_{trial}_{repeat}.json"
                    )

                    # 1) 플래닝 (예외 발생해도 CSV에 실패 행 남기기)
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
                        _append_csv_row(summary_csv, fieldnames, row)
                        # 다음 케이스로
                        continue

                    # 2) (선택) 제네시스 시뮬레이션 – 결과를 즉시 CSV에 기록
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

                    if success:
                        try:
                            ctx = mp.get_context("spawn")
                            q = ctx.Queue()
                            wargs = dict(domain=domain, problem_json_path=problem_json_path, method=method, prob_num=prob_num, prob_idx=prob_idx,
                                         trial=trial, repeat=repeat, planning_time=planning_time, record_video=True, num_distractor=num_distractor)
                            p = ctx.Process(target=_run_one_worker, args=(wargs, q))
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
                                        # run_genesis가 돌려준 dict과 병합
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
                            # 시뮬레이터 자체 예외도 기록
                            res.update({"error": f"sim_error: {e}", "timed_out": False})

                    # 3) 이번 케이스 결과를 **즉시** CSV에 기록
                    _append_csv_row(summary_csv, fieldnames, res)

# def run_pddlstream_batch(domain="blocksworld_pr",
#                          algorithm="adaptive",
#                          prob_num_range=range(3, 8),
#                          prob_idx_range=range(1, 6),
#                          trial_range=range(1, 3),
#                          timeout=900):
#     # if sim_timeout is None:
#     #     sim_timeout = timeout
#     results = []
#
#     summary_dir = Path(f"../experiments/{domain}/pddlstream/batch_outputs").resolve()
#     summary_dir.mkdir(parents=True, exist_ok=True)
#     summary_csv = summary_dir / "summary.csv"
#     write_header = not summary_csv.exists()
#
#     with open(summary_csv, "a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow([
#                 "timestamp", "domain", "prob_num", "prob_idx", "trial",
#                 "planning_success", "planning_time_sec", "sim_time_sec",
#                 "plan_json_path"
#             ])
#
#         for prob_num in prob_num_range:
#             for prob_idx in prob_idx_range:
#                 for trial in trial_range:
#                     print(f"\n=== RUN: prob_num={prob_num}, prob_idx={prob_idx}, trial={trial} ===")
#
#                     # 1) 플래너는 메인 프로세스에서 실행
#                     success, planning_time = run_pddlstream(
#                         domain=domain,
#                         prob_num=prob_num,
#                         prob_idx=prob_idx,
#                         trial=trial,
#                         algorithm=algorithm,
#                         unit=True,
#                         enable=False,
#                         max_time=timeout
#                     )
#
#                     plan_json_path = Path(
#                         f"/home/minseo/robot_ws/src/tamp_llm/experiments/{domain}/pddlstream/plan/{prob_num}_{prob_idx}_{trial}.json"
#                     )
#
#                     # 2) 제네시스는 별도 프로세스(워커)로 실행
#                     sim_time = ""
#                     # if success:
#                     #     ctx = mp.get_context("spawn")
#                     #     q = ctx.Queue()
#                     #     wargs = dict(domain=domain, prob_num=prob_num, prob_idx=prob_idx, trial=trial,
#                     #                  record_video=True)
#                     #     start = time.time()
#                     #     p = ctx.Process(target=_run_one_worker, args=(wargs, q))
#                     #     p.start()
#                     #
#                     #     # 고정 시간대기가 아니라, "끝나면 바로" 넘어감
#                     #     p.join()  # 블록: 자식 종료까지 대기
#                     #     end = time.time()
#                     #     if p.exitcode == 0:
#                     #         try:
#                     #             status, payload = q.get(timeout=5.0)
#                     #             if status == "ok":
#                     #                 sim_time = float(payload)
#                     #             else:
#                     #                 print("[batch][genesis] child error:\n", payload)
#                     #                 sim_time = ""
#                     #         except Exception as e:
#                     #             print("[batch][genesis] child finished but no payload:", e)
#                     #             sim_time = ""
#                     #     else:
#                     #         print(f"[batch][genesis] child crashed (exitcode={p.exitcode})")
#                     #         sim_time = end-start
#                     # else:
#                     #     if not success:
#                     #         print("[batch] planning failed; skip genesis.")
#                     #     else:
#                     #         print("[batch] missing plan json; skip genesis.")
#                     res = {
#                         "prob_num": prob_num,
#                         "prob_idx": prob_idx,
#                         "trial": trial,
#                         "planning_success": success,
#                         "planning_time_sec": planning_time,
#                         "sim_time_sec": None,
#                         "sim_success": None,
#                     }
#
#                     if success:
#                         ctx = mp.get_context("spawn")  # native/lib 충돌 회피에 안전
#                         q = ctx.Queue()
#                         wargs = dict(domain=domain, prob_num=prob_num, prob_idx=prob_idx, trial=trial, planning_time=planning_time,
#                                                       record_video=True)
#                         p = ctx.Process(target=_run_one_worker, args=(wargs, q))
#                         p.start()
#                         p.join(300)
#
#                         if p.is_alive():
#                             print(f"[TIMEOUT] Killing run (>{300}s).")
#                             p.terminate()
#                             p.join()
#                             res = {
#                                 "prob_num": prob_num,
#                                 "prob_idx": prob_idx,
#                                 "trial": trial,
#                                 "planning_success": success,
#                                 "planning_time_sec": planning_time,
#                                 "sim_time_sec": None,
#                                 "sim_success": False,
#                                 "error": f"timeout(>300s)",
#                                 "timed_out": True,
#                             }
#                         else:
#                             try:
#                                 status, payload = q.get_nowait()
#                                 if status == "ok":
#                                     res = payload | {"timed_out": False}
#                                 else:
#                                     print("status not okay:" , status)
#                                     print("[ERROR] run failed in child process.")
#                                     res = {
#                                         "prob_num": prob_num,
#                                         "prob_idx": prob_idx,
#                                         "trial": trial,
#                                         "planning_success": True,
#                                         "planning_time_sec": planning_time,
#                                         "sim_time_sec": None,
#                                         "sim_success": None,
#                                         "timed_out": False,
#                                     }
#                             except Exception as e:
#                                 print("exception", e)
#                                 print("[ERROR] run failed in child process.")
#                                 res = {
#                                     "prob_num": prob_num,
#                                     "prob_idx": prob_idx,
#                                     "trial": trial,
#                                     "planning_success": True,
#                                     "planning_time_sec": planning_time,
#                                     "sim_time_sec": None,
#                                     "sim_success": None,
#                                     "error": f"no result from child: {e}",
#                                     "timed_out": False,
#                                 }
#                     results.append(res)
#
#     # CSV/JSON 저장 (기존 로직 그대로)
#     fieldnames = [
#         "prob_num","prob_idx","trial",
#         "planning_success", "planning_time_sec", "planning_time_sec", "sim_time_sec", "sim_success"
#     ]
#     with open(summary_csv, "a", newline="") as f:   # ← 덮어쓰기/헤더 1회
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in results:
#             w.writerow({k: r.get(k) for k in fieldnames})


def _arm_is_left(pddl_args):
    # pddl_args의 첫 인자가 'left'/'right'로 들어오는 구조를 가장 먼저 검사
    if not pddl_args:
        return True
    if isinstance(pddl_args[0], str) and pddl_args[0].lower() in ("left", "right"):
        return pddl_args[0].lower() == "left"
    # 혹시 안전하게 포함 여부로도 체크
    return "left" in [str(a).lower() for a in pddl_args]


def _build_arm_traj_from_json(sim_wrapper, path_list, use_left, domain):
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


def run_genesis(domain, problem_json_path, method, prob_num, prob_idx, trial, repeat, planning_time, record_video=True, num_distractor=0):
    export_dir = Path(f"/home/minseo/robot_ws/src/tamp_llm/experiments/{domain}/{method}/plan").resolve()
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

        is_left = _arm_is_left(args)

        for cmd in cmds:
            ctype = cmd.get("type")

            if ctype == "Trajectory":
                path_list = cmd.get("path", [])
                if not path_list:
                    continue
                traj = _build_arm_traj_from_json(sim_wrapper, path_list, use_left=is_left, domain=domain)
                sim_wrapper.move(traj, take_screenshot=False)

            elif ctype == "Attach":  # TODO
                if domain=="kitchen":
                    sim_wrapper.close_gripper(object=sim_wrapper.object_dict[cmd.get("body")])
                elif domain=="blocksworld_pr":
                    sim_wrapper.close_gripper(left=bool(cmd.get("left", is_left)))

            elif ctype == "Detach":
                if domain == "kitchen":
                    sim_wrapper.open_gripper(object=sim_wrapper.object_dict[cmd.get("body")])
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

    base_exp_dir = Path(f"../experiments/{domain}")
    out_dir = base_exp_dir / method / "batch_outputs" / f"{prob_num}_{prob_idx}" / f"trial{trial}_repeat{repeat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    screenshot_dir = f"../experiments/{domain}/{method}/screenshots/{prob_num}_{prob_idx}_{trial}_{repeat}"

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


if __name__ == "__main__":
    run_pddlstream_batch(domain="blocksworld_pr",
                         problem_json_path="/home/minseo/robot_ws/src/tamp_llm/experiments/blocksworld_pr/problem/problems_meta.json",
                         method = "pddlstream",
                         algorithm="adaptive",
                         prob_num_range=[3],  # 3..7
                         prob_idx_range=[1,],  # 1..5
                         trial_range=[1,],
                         repeat_range=[1,],
                         num_distractor=0,
                         timeout=600)

    # run_pddlstream_batch(domain="blocksworld_pr",
    #                      problem_json_path="/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem/problems_meta.json",
    #                      method = "pddlstream_ratio1_new_problem",
    #                      algorithm="adaptive",
    #                      prob_num_range=[3, 4, 5, 6],  # 3..7
    #                      prob_idx_range=[1, 2, 3, 4, 5],  # 1..5
    #                      trial_range=[1, 2],
    #                      repeat_range=[2,],
    #                      num_distractor=0,
    #                      timeout=600)
    # run_pddlstream_batch(domain="blocksworld_pr",
    #                      problem_json_path="/home/minseo/robot_ws/src/tamp_llm/experiments/kitchen/problem/problems_meta.json",
    #                      method = "pddlstream_ratio1_new_problem",
    #                      algorithm="adaptive",
    #                      prob_num_range=[6],  # 3..7
    #                      prob_idx_range=[2, 3, 4, 5],  # 1..5
    #                      trial_range=[1, 2],
    #                      repeat_range=[3,],
    #                      num_distractor=0,
    #                      timeout=600)


    # run_genesis(domain="kitchen", method="pddlstream_ratio2", prob_num=3, prob_idx=1, trial=4, repeat=1, planning_time=0, record_video=True,
    #                 num_distractor=0)


