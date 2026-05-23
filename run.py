import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

sys.path.insert(0, str(PROJECT_ROOT))

# Import runners
from scripts.run_ours import run_ours_batch
from scripts.run_pddlstream import run_pddlstream_batch
from scripts.run_llm3 import run_blocksworld, run_kitchen


def run(domain="blocksworld_pr",
        method="ours",
        prob_num_range=None,
        prob_idx_range=None,
        trial_range=None,
        repeat_range=None,
        K=5,
        plan_number=30,
        timeout_seconds=600,
        model="gpt-4o",
        vis_sim=False):
    """Run experiments with specified method"""
    
    if prob_num_range is None:
        prob_num_range = range(3, 7)
    if prob_idx_range is None:
        prob_idx_range = range(1, 2)
    if trial_range is None:
        trial_range = range(1, 3)
    if repeat_range is None:
        repeat_range = range(1, 2)
    
    problems_meta_json = str(EXPERIMENTS_DIR / domain / "problem" / "problems_meta.json")
    
    if method in ["ours", "ours_ablation"]:
        method_name = "ours" if method == "ours" else "ours_ablation"
        run_ours_batch(
            domain=domain,
            method=method_name,
            problems_meta_json=problems_meta_json,
            prob_num_range=prob_num_range,
            prob_idx_range=prob_idx_range,
            trial_range=trial_range,
            K=K,
            plan_number=plan_number,
            timeout_seconds=timeout_seconds,
            model=model,
            vis_sim=vis_sim
        )
    elif method == "pddlstream":
        run_pddlstream_batch(
            domain=domain,
            problem_json_path=problems_meta_json,
            method=method,
            algorithm="adaptive",
            prob_num_range=prob_num_range,
            prob_idx_range=prob_idx_range,
            trial_range=trial_range,
            repeat_range=repeat_range,
            num_distractor= 12 if domain == "kitchen" else 0,
            timeout=timeout_seconds,        
        )
    elif method == "llm3":
        if domain == "blocksworld_pr":
            run_blocksworld(
                prob_num_range=prob_num_range,
                prob_idx_range=prob_idx_range,
                trial_range=trial_range,
                repeat_range=repeat_range,
                model=model,
            )
        elif domain == "kitchen":
            run_kitchen(
                prob_num_range=prob_num_range,
                prob_idx_range=prob_idx_range,
                trial_range=trial_range,
                repeat_range=repeat_range,
                model=model,
            )
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAMP experiments runner")
    
    parser.add_argument("--domain", type=str, default="blocksworld_pr",
                        choices=["blocksworld_pr", "kitchen"],
                        help="Domain name")
    
    parser.add_argument("--method", type=str, default="ours",
                        choices=["ours", "ours_ablation", "pddlstream", "llm3"],
                        help="Method name")
    
    parser.add_argument("--prob_complexity", type=int, nargs="+", default=list(range(3, 7)),
                        help="Problem complexity range (e.g., --prob_complexity 3 4 5 6)")
    
    parser.add_argument("--prob_idx", type=int, nargs="+", default=list(range(1, 2)),
                        help="Problem index range (e.g., --prob_idx 1 2 3 4 5)")
    
    parser.add_argument("--trial_range", type=int, nargs="+", default=list(range(1, 3)),
                        help="Trial range (e.g., --trial_range 1 2)")
    
    parser.add_argument("--K", type=int, default=5,
                        help="Max replanning attempts")
    
    parser.add_argument("--plan_number", type=int, default=30,
                        help="Number of candidate plans")
    
    parser.add_argument("--timeout_seconds", type=int, default=600,
                        help="Timeout per trial (seconds)")
    
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="VLM model")
    
    parser.add_argument("--vis_sim", type=bool, default=False,
                        help="Set to True to enable Genesis visualization. Only applicable for 'ours' and 'ours_ablation' methods.")
    
    args = parser.parse_args()
    
    prob_num_range = args.prob_complexity
    prob_idx_range = args.prob_idx
    trial_range = args.trial_range
    repeat_range = range(1, 2)
    
    run(
        domain=args.domain,
        method=args.method,
        prob_num_range=prob_num_range,
        prob_idx_range=prob_idx_range,
        trial_range=trial_range,
        repeat_range=repeat_range,
        K=args.K,
        plan_number=args.plan_number,
        timeout_seconds=args.timeout_seconds,
        model=args.model,
        vis_sim=args.vis_sim
    )
