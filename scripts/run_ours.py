import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified_planning.io import PDDLReader
import genesis as gs
from utils.experiment_utils import *
import multiprocessing as mp 
import argparse
from symbolic import upstate_to_grouped_facts


logger = logging.getLogger("TAMP")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

import hybrid_tree as ht
from hybrid_tree_generation import (
    diverse_planning,
    build_tree_from_dot,
    hybrid_tree_expansion,
    grouped_facts_to_str
)
import networkx as nx


def extract_final_plan_with_params(hybrid: ht.HybridTree):
    root = hybrid.root
    goal = hybrid.current_node
    if root is None or goal is None:
        return []

    try:
        node_path = nx.shortest_path(hybrid.G, source=root, target=goal)
    except nx.NetworkXNoPath:
        return []

    plan = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edata = hybrid.G.get_edge_data(u, v)
        chosen_key, chosen_attrs = None, None
        for k, attrs in edata.items():
            if 'continuous_params' in attrs:
                chosen_key, chosen_attrs = k, attrs
                break
        if chosen_attrs is None:
            chosen_key, chosen_attrs = next(iter(edata.items()))

        plan.append({
            "action": chosen_key,
            "feasible": bool(chosen_attrs.get("feasible", True)),
            "continuous_params": chosen_attrs.get("continuous_params", {}),
            "from_state": upstate_to_grouped_facts(u.discrete_belief.state),
            "to_state": upstate_to_grouped_facts(v.discrete_belief.state),
            "result_images": getattr(v, "image_path", None),
        })
    return plan


def run_one(domain: str,
            method: str,
            prob_num: int,
            prob_idx: int,
            trial: int,
            repeat: int,
            problems_meta_json: str,
            plan_number: int = 20,
            diverse_planner: str = "topk",
            K: int = 4,
            weight_threshold: float = 0.2,
            ablation: bool = False,
            num_distractor: int = 0,
            robot_name: str = 'pr2',
            model: str = 'gpt-4o',
            vis_sim: bool = False
            ):

    base_exp_dir = Path(f"./experiments/{domain}")
    log_dir = base_exp_dir / method / "batch_outputs" / f"{prob_num}_{prob_idx}" / f"trial{trial}_repeat{repeat}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting experiment: domain={domain}, method={method}, prob_num={prob_num}, prob_idx={prob_idx}, trial={trial}, repeat={repeat}")

    # Use absolute paths from project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    base_exp_dir = project_root / "experiments" / domain
    plan_dir = base_exp_dir / method / "plan"
    problem_pddl_path = base_exp_dir / "problem" / f"{domain}{prob_num}_{prob_idx}.pddl"
    domain_pddl_path = project_root / "domains" / f"domain_{domain}.pddl"
    out_dir = base_exp_dir / method / "batch_outputs" / f"{prob_num}_{prob_idx}" / f"trial{trial}_repeat{repeat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_img_path = out_dir / "hybrid_tree.png"
    final_plan_json = out_dir / "final_plan.json"
    meta_json = out_dir / "run_meta.json"
    video_path = out_dir / "video.mp4"

    start = time.time()
    plans, gv_path = diverse_planning(domain, method, problem_pddl_path, plan_number=plan_number, planner=diverse_planner)
    end = time.time()
    diverse_planning_time = end - start
    logger.info(f"Diverse planning time: {diverse_planning_time:.2f}s")

    start = time.time()
    reader = PDDLReader()
    domain_pddl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain}.pddl"))
    problem_template = reader.parse_problem(
            domain_pddl_path,
            str(problem_pddl_path)
        )
    discrete_tree = build_tree_from_dot(method, gv_path, domain, prob_num, prob_idx, problem_template)
    end = time.time()
    graph_building_time = end - start
    logger.info(f"Graph building time: {graph_building_time:.2f}s")

    from hybrid_tree_generation import vlm_selector, bfs_unvisited_selector, vlm_backtrack_selector
    from action import execute
    start = time.time()
    hybrid, tamp_success, hybrid_time, backtrack_count = hybrid_tree_expansion(
        method=method,
        json_path=problems_meta_json,
        problem_pddl_path=str(problem_pddl_path),
        prob_num=prob_num,
        prob_idx=prob_idx,
        trial=trial,
        repeat=repeat,
        model=model,
        vis_sim=vis_sim,
        domain_name=domain,
        robot_name=robot_name,
        grasp_type='top',
        discrete_tree=discrete_tree,
        problem_template=problem_template,
        sampler=execute,
        child_selector=vlm_selector,
        backtrack_selector= bfs_unvisited_selector if ablation else vlm_backtrack_selector,
        K=K,
        weight_threshold=weight_threshold,
        num_distractor=num_distractor,
    )
    end = time.time()
    gs.destroy()

    total_time = diverse_planning_time + graph_building_time + hybrid_time
    logger.info(f"Hybrid tree expansion time: {hybrid_time:.2f}s")
    logger.info(f"Total TAMP subgoal planning time: {total_time:.2f}s")
    logger.info(f"TAMP subgoal success: {tamp_success}")
    logger.info("Experiment finished")

    hybrid_img_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        hybrid.visualize_tree(str(hybrid_img_path))
    except Exception as e:
        logger.warning(f"hybrid.visualize_tree failed: {e}")

    final_plan = extract_final_plan_with_params(hybrid)
    with open(final_plan_json, "w") as f:
        json.dump({"plan": final_plan}, f, indent=2)
    with open(meta_json, "w") as f:
        json.dump({
            "prob_num": prob_num,
            "prob_idx": prob_idx,
            "trial": trial,
            "repeat": repeat,
            "diverse_planning_time": diverse_planning_time,
            "graph_building_time": graph_building_time,
            "hybrid_graph_planning_time": hybrid_time,
            "total_planning_time": total_time,
            "tamp_success": bool(tamp_success),
            "hybrid_tree_image": str(hybrid_img_path),
            "final_plan_json": str(final_plan_json),
            "gv_path": str(gv_path),
            "backtrack_count": backtrack_count,
        }, f, indent=2)

    logger.removeHandler(file_handler)
    file_handler.close()

    return {
        "prob_num": prob_num,
        "prob_idx": prob_idx,
        "trial": trial,
        "repeat": repeat,
        "diverse_planning_time": diverse_planning_time,
        "graph_building_time": graph_building_time,
        "hybrid_graph_planning_time": hybrid_time,
        "total_planning_time": total_time,
        "tamp_success": bool(tamp_success),
        "backtrack_count": backtrack_count,
    }


def run_ours_batch(domain="blocksworld_pr",
              method="ours",
              problems_meta_json=None,
              prob_num_range=None,
              prob_idx_range=None,
              trial_range=None,
              K=4,
              plan_number=30,
              timeout_seconds=10,
              model="gpt-4o",
              vis_sim=False
              ):
    
    if problems_meta_json is None:
        problems_meta_json = f"./experiments/{domain}/problem/problems_meta.json"
    
    ablation = (method == "ablation")
    robot_name = "pr2" if domain == "blocksworld_pr" else "kuka"
    num_distractor = 12 if domain == "kitchen" else 0
    
    repeat_range = range(1, 2)
    diverse_planner = "topk"
    weight_threshold = 0.2
    
    results_csv = Path(f"./experiments/{domain}/{method}/batch_outputs/summary.csv")
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prob_num","prob_idx","trial", "repeat",
        "diverse_planning_time","graph_building_time",
        "hybrid_graph_planning_time","total_planning_time",
        "tamp_success","backtrack_count", "timed_out", "sim_success", "error"
    ]

    for prob_num in prob_num_range:
        for prob_idx in prob_idx_range:
            for trial in trial_range:
                for repeat in repeat_range:
                    logger.info(f"\n=== RUN: prob_num={prob_num}, prob_idx={prob_idx}, trial={trial}, repeat={repeat} ===")
                    kwargs = dict(
                        domain=domain,
                        method=method,
                        prob_num=prob_num,
                        prob_idx=prob_idx,
                        trial=trial,
                        repeat=repeat,
                        problems_meta_json=problems_meta_json,
                        plan_number=plan_number,
                        diverse_planner=diverse_planner,
                        K=K,
                        weight_threshold=weight_threshold,
                        ablation=ablation,
                        num_distractor=num_distractor,
                        robot_name=robot_name,
                        model=model,
                        vis_sim=vis_sim
                    )

                    ctx = mp.get_context("spawn")
                    q = ctx.Queue()
                    p = ctx.Process(target=run_one_worker, args=(run_one, kwargs, q))
                    p.start()
                    p.join(timeout_seconds)

                    res = {
                        "prob_num": prob_num,
                        "prob_idx": prob_idx,
                        "trial": trial,
                        "repeat": repeat,
                        "diverse_planning_time": None,
                        "graph_building_time": None,
                        "hybrid_graph_planning_time": None,
                        "total_planning_time": None,
                        "symbolic_success": False,
                        "backtrack_count": 0,
                        "timed_out": False,
                        "sim_success": False,
                        "error": "",
                    }

                    if p.is_alive():
                        logger.info(f"[TIMEOUT] Killing run (>{timeout_seconds}s).")
                        p.terminate()
                        p.join()
                        res.update({
                            "timed_out": True,
                            "error": f"timeout(>{timeout_seconds}s)",
                        })
                    else:
                        try:
                            status, payload = q.get_nowait()
                            if status == "ok":
                                res.update(payload)
                                res.update({"timed_out": False, "sim_success": ""})
                            else:
                                logger.error("[ERROR] run failed in child process.")
                                logger.error(payload)
                                res.update({
                                    "error": payload,
                                })
                        except Exception as e:
                            # child crashed without posting a result
                            res.update({
                                "error": f"no result from child: {e}",
                            })

                    append_csv_row(results_csv, fieldnames, res)

    logger.info(f"\nBatch finished. CSV: {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch experiments for TAMP")
    parser.add_argument("--domain", type=str, default="blocksworld_pr", 
                        choices=["blocksworld_pr", "kitchen"],
                        help="Domain name")
    parser.add_argument("--method", type=str, default="ours",
                        choices=["ours", "ours_ablation"],
                        help="Method name")
    parser.add_argument("--prob_complexity", type=int, nargs="+", default=list(range(3, 7)),
                        help="Number of objects (e.g. --prob_complexity 3 4 5 6)")
    parser.add_argument("--prob_idx", type=int, nargs="+", default=list(range(1, 2)),
                        help="Problem index (e.g., --prob_idx 1 2 3 4 5)")
    parser.add_argument("--trial", type=int, nargs="+", default=list(range(1, 3)),
                        help="Trial number for each problem (e.g., --trial 1 2)")
    parser.add_argument("--K", type=int, default=5,
                        help="Maximum number of attempts for randomized replanning")
    parser.add_argument("--plan_number", type=int, default=30,
                        help="Number of plans to generate using a top-k planner for building the discrete state graph")
    parser.add_argument("--timeout_seconds", type=int, default=600,
                        help="Timeout per trial in seconds")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="VLM model to use")
    
    args = parser.parse_args()
    
    prob_num_range = range(args.prob_complexity[0], args.prob_complexity[1]) if args.prob_complexity else None
    prob_idx_range = range(args.prob_idx[0], args.prob_idx[1]) if args.prob_idx else None
    trial_range = range(args.trial[0], args.trial[1]) if args.trial else None
    
    run_ours_batch(
        domain=args.domain,
        method=args.method,
        prob_num_range=prob_num_range,
        prob_idx_range=prob_idx_range,
        trial_range=trial_range,
        K=args.K,
        plan_number=args.plan_number,
        timeout_seconds=args.timeout_seconds,
        model=args.model
    )
