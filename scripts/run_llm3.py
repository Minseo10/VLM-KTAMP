import subprocess
from pathlib import Path
import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LLM3_ROOT = PROJECT_ROOT / "LLM-TAMP"
LOG_DIR = LLM3_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _fmt_list_override(key, values):
    return f"{key}=[" + ",".join(map(str, values)) + "]"

def run_blocksworld(prob_num_range=[3, 4, 5, 6],
                    prob_idx_range=[1, 2, 3, 4, 5],
                    trial_range=[1, 2],
                    repeat_range=[1],
                    timeout_seconds=600):
    """
    python main.py --config-name=llm_tamp_blocksworld env=easy_box_small_basket \
        planner=llm_backtrack overwrite_instances=true play_traj=true use_gui=true \
        timeout_sec={timeout_seconds} prob_num=[...] prob_idx=[...] trial=[...] repeat=[...]
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"blocksworld_{timestamp}.log"

    cmd = [
        sys.executable, "main.py",
        "--config-name=llm_tamp_blocksworld",
        "env=easy_box_small_basket",
        "planner=llm_backtrack",
        _fmt_list_override("prob_num", prob_num_range),
        _fmt_list_override("prob_idx", prob_idx_range),
        _fmt_list_override("trial", trial_range),
        _fmt_list_override("repeat", repeat_range),
        f"timeout_sec={timeout_seconds}",
        "overwrite_instances=true",
        "play_traj=true",
        "use_gui=true",
    ]

    with open(log_file, "w") as f:
        f.write("# CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=LLM3_ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)

    if proc.returncode == 0:
        print(f"[Blocksworld] Run finished. Log saved at {log_file}")
    else:
        print(f"[Blocksworld] Run FAILED (code {proc.returncode}). See log: {log_file}")


def run_kitchen(prob_num_range=[3, 4, 5, 6],
                prob_idx_range=[1],
                trial_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                repeat_range=[1],
                timeout_seconds=600):
    """
    python main.py --config-name=llm_tamp_kitchen env=easy_box_small_basket \
        planner=llm_backtrack overwrite_instances=true play_traj=true use_gui=true \
        timeout_sec={timeout_seconds} prob_num=[...] prob_idx=[...] trial=[...] repeat=[...]
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"kitchen_{timestamp}.log"

    cmd = [
        sys.executable, "main.py",
        "--config-name=llm_tamp_kitchen",
        "env=easy_box_small_basket",
        "planner=llm_backtrack",
        _fmt_list_override("prob_num", prob_num_range),
        _fmt_list_override("prob_idx", prob_idx_range),
        _fmt_list_override("trial", trial_range),
        _fmt_list_override("repeat", repeat_range),
        f"timeout_sec={timeout_seconds}",
        "overwrite_instances=true",
        "play_traj=true",
        "use_gui=true",
    ]

    with open(log_file, "w") as f:
        f.write("# CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=LLM3_ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)

    if proc.returncode == 0:
        print(f"[Kitchen] Run finished. Log saved at {log_file}")
    else:
        print(f"[Kitchen] Run FAILED (code {proc.returncode}). See log: {log_file}")
