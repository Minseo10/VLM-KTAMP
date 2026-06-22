from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path so we can import utils and config
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import List, Tuple, Optional, Any
from scripts import model
from belief_structs import *
from tool_use_env import start_tamp_sim
from find_dice_env import FindDiceEnv
from symbolic import Subgoal, generate_problem_pddl
from belief_tree import BeliefNode, BeliefTree
from hybrid_tree_generation import *
from utils.franka_api import get_pose, set_pose
from utils.prompts import vlm_validator
from utils.utils import _json_safe
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from scripts.action import execute, execute_trajectory
from config import config as tconfig
import json
import os
from datetime import datetime
import logging
import time
import signal
from contextlib import contextmanager
import gc
import re
MAX_RETRIES = 3


def solve_problem(domain_path: str, problem_path: str, config_path: str, prob_num: int, prob_idx: int, trial: int, repeat: int):
    config = tconfig.load_config(config_file=config_path)
    env = tconfig.get_env(config["task"])(config=config)
    prob_name = f"{prob_num}_{prob_idx}"

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_dir = os.path.join(project_root, "experiments", config["task"], config["planner"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{prob_name}_{timestamp}.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # Prevent Genesis logger from propagating to root logger (suppress duplicate logs)
    genesis_logger = logging.getLogger('genesis')
    genesis_logger.propagate = False
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initialize PDDL problem and simulator
    reader = PDDLReader()
    problem = reader.parse_problem(domain_path, problem_path)
    simulator = SequentialSimulator(problem)

    belief_tree, belief_nodes = policy(
        problem_template=PDDLReader().parse_problem(domain_path, problem_path),
        initial_state=simulator.get_initial_state(),
        goal_state=list(problem.goals),
        config=config,
        env=env,
        prob_num=prob_num,
        prob_idx=prob_idx,
        trial=trial,
        repeat=repeat,
        timestamp=timestamp,
    )

    # Save log to file
    log_path = os.path.join(log_dir, f"{prob_name}_{timestamp}.json")

    log = {
        "timestamp": timestamp,
        "prob_name": prob_name,
        "domain": config["task"],
        "planner": config["planner"],
        "robot": config["robot"],
        "num_nodes": len(belief_nodes),
        "nodes": [],
    }

    for node in belief_nodes:
        node_entry = {
            "name": node.name,
            "depth": node.depth,
            "action": str(node.action) if node.action is not None else None,
            "observation": str(node.observation) if node.observation is not None else None,
            "discrete_belief": str(node.discrete_belief),
            "continuous_belief": str(node.continuous_belief),
        }
        log["nodes"].append(node_entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    logging.info(f"Log saved to {log_path}")


def policy(problem_template: Any, initial_state: Any, goal_state: Any, config:Any, env: Any, prob_num: int, prob_idx: int, trial: int, repeat: int, timestamp: str = None) -> Tuple[BeliefTree, List[BeliefNode]]:
    logger = logging.getLogger(__name__)
    # Initialize beliefs
    cb0 = env.initialize(prob_num, prob_idx, trial, repeat)
    
    # Setup recording for original world
    prob_name = f"{prob_num}_{prob_idx}"
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", config["task"], config["planner"], "recordings", prob_name, timestamp))
    os.makedirs(video_dir, exist_ok=True)
    
    logger.info(f"[policy] Starting recording for original world. Video dir: {video_dir}")
    # env.world.franka.start_recording(save_dir=video_dir)
    
    # Initialize recording buffer with one step
    env.world.franka.scene.step()
    
    db0 = DiscreteBelief(state=initial_state)
    print("initial state:", initial_state)

    # Extract subgoals from symbolic plan
    subgoals = get_subgoals_from_plan(problem_template, initial_state, goal_state)

    # Process subgoals and build belief tree
    try:
        belief_tree, belief_nodes = process_subgoal_sequence(
            subgoals=subgoals,
            initial_continuous_belief=cb0,
            initial_discrete_belief=db0,
            problem=problem_template,
            config=config,
            env=env,
            prob_num=prob_num,
            prob_idx=prob_idx,
            trial=trial,
            repeat=repeat,
            timestamp=timestamp,
        )
    finally:
        # Stop recording for original world (whether success or failure)
        logger.info("[policy] Stopping recording for original world...")
        # env.world.franka.stop_recording(video_name=f"original_world_{prob_name}")
        logger.info("[policy] Original world recording stopped.")

    return belief_tree, belief_nodes


def get_subgoals_from_plan(problem_template: Any, initial_state: Any, goal_state: Any) -> List[Subgoal]:
    logger = logging.getLogger(__name__)
    """
    Extract subgoals by grouping consecutive Information/TAMP actions.
    
    Args:
        problem_template: unified_planning Problem template
        initial_state: unified_planning State used as initial state
        goal_state: Goal expressions (or list of expressions)
    
    Returns:
        List of subgoals from initial to goal state
    """
    problem = problem_template.clone()

    # Build a new planning problem from initial state and goal state.
    for fexp in list(problem.initial_values.keys()):
        problem.set_initial_value(fexp, initial_state.get_value(fexp))

    problem.clear_goals()
    if isinstance(goal_state, (list, tuple)):
        for g in goal_state:
            problem.add_goal(g)
    else:
        problem.add_goal(goal_state)
    
    # Generate plan using unified_planning
    planner = OneshotPlanner(problem_kind=problem.kind)
    plan_result = planner.solve(problem)
    logger.info(plan_result)
    
    if not plan_result.status == PlanGenerationResultStatus.SOLVED_SATISFICING and not plan_result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
        raise ValueError("No valid plan found")
    
    plan = plan_result.plan
    
    # Initialize simulator with initial state
    simulator = SequentialSimulator(problem)
    current_state = simulator.get_initial_state()
    
    subgoals = []
    previous_action_type = None
    current_actions = []
    
    # Process each action in the plan
    for action_instance in plan.actions:
        action_name = action_instance.action.name
        
        # Determine action type
        if action_name in ["pull_towards", "look"]:
            action_type = "Information"
        else:
            action_type = "TAMP"
        
        # When action type changes, save current state as subgoal with collected actions
        if previous_action_type is not None and action_type != previous_action_type:
            subgoals.append(Subgoal(goal_type=previous_action_type, state=current_state, actions=current_actions))
            current_actions = []
        
        # Add action to current group
        current_actions.append(action_instance)
        
        # Execute action and get next state
        current_state = simulator.apply(current_state, action_instance)
        previous_action_type = action_type
    
    # Add final subgoal after all actions
    if previous_action_type is not None:
        subgoals.append(Subgoal(goal_type=previous_action_type, state=current_state, actions=current_actions))

    for i, subgoal in enumerate(subgoals):
        logger.info(f"Subgoal {i+1}: Type={subgoal.goal_type}, Actions={[str(a) for a in subgoal.actions]}")
        logger.info(f"  State={subgoal.state}\n")
    
    return subgoals


def process_subgoal_sequence(
    subgoals: List[Subgoal],
    initial_continuous_belief: ContinuousBelief,
    initial_discrete_belief: DiscreteBelief,
    problem: Any,
    config: Any,
    env: Any,
    prob_num: int,
    prob_idx: int,
    trial: int,
    repeat: int,
    timestamp: str = None
) -> Tuple[BeliefTree, List[BeliefNode]]:
    logger = logging.getLogger(__name__)
    """
    Execute sequence of subgoals and maintain belief tree.
    
    Args:
        subgoals: List of Subgoal objects from symbolic planning
        initial_continuous_belief: Initial continuous belief state
        initial_discrete_belief: Initial discrete belief state
        problem: unified_planning Problem for discrete state tracking
        config: Configuration dictionary with "planner", "robot", "task", etc.
        env: Simulation environment for executing actions
        prob_num: Problem number for logging
        prob_idx: Problem index for logging
        timestamp: Shared timestamp for all recordings
    
    Returns:
        Tuple of (belief_tree, list_of_belief_nodes)
    """
    # Initialize root node with initial beliefs
    root_node = BeliefNode(
        cb=initial_continuous_belief,
        db=initial_discrete_belief
    )
    root_node.name = "root"
    
    belief_tree = BeliefTree(root_node)
    belief_nodes = [root_node]
    
    current_node = root_node
    simulator = SequentialSimulator(problem)
    
    # Planning world: created once on first TAMP subgoal, reused/reset for subsequent calls, destroyed at end
    planning_world = None
    last_tamp_subgoal_idx = -1  # Track last TAMP subgoal index for recording cleanup
    
    try:
        # Process each subgoal
        active_subgoals = list(subgoals)
        subgoal_idx = 0
        while subgoal_idx < len(active_subgoals):
            subgoal = active_subgoals[subgoal_idx]
            logger.info(f"\n\n=== Processing Subgoal {subgoal_idx + 1}: {subgoal.goal_type} ===")
            
            if subgoal.goal_type == "Information":
                # Before Information subgoal, stop planning world recording if it was started
                # if planning_world is not None and config.get("record_video", False):
                #     logger.info(f"[process_subgoal_sequence] Stopping planning world recording before Information subgoal {subgoal_idx}...")
                #     try:
                #         planning_world.franka.stop_recording(video_name=f"planning_world_subgoal{subgoal_idx}")
                #         logger.info(f"[process_subgoal_sequence] Planning world recording stopped before Information subgoal {subgoal_idx}.")
                #     except Exception as e:
                #         logger.debug(f"[process_subgoal_sequence] Could not stop planning world recording: {e}")

                # Execute information actions - tree updated inside function
                new_nodes, replanned_subgoals = execute_information_subgoal(
                    subgoal,
                    current_node,
                    belief_tree,
                    simulator,
                    problem,
                    list(problem.goals),
                    config,
                    env,
                    prob_num,
                    prob_idx,
                    subgoal_idx
                )
                if new_nodes:
                    current_node = new_nodes[-1]
                    belief_nodes.extend(new_nodes)
                if replanned_subgoals is not None:
                    active_subgoals = replanned_subgoals
                    subgoal_idx = 0
                    continue
            
            elif subgoal.goal_type == "TAMP":
                # Execute TAMP subgoal and manage planning world
                new_nodes, replanned_subgoals, planning_world = execute_tamp_subgoal(
                    subgoal,
                    current_node,
                    belief_tree,
                    simulator,
                    problem,
                    list(problem.goals),
                    config,
                    env,
                    prob_num,
                    prob_idx,
                    subgoal_idx,
                    trial=trial,
                    repeat=repeat,
                    planning_world=planning_world,  # Pass planning world for reuse
                    timestamp=timestamp,  # Pass shared timestamp
                )
                last_tamp_subgoal_idx = subgoal_idx  # Track last TAMP subgoal for recording cleanup
                
                # Explicit memory cleanup after TAMP subgoal to prevent accumulation
                logger.info(f"[process_subgoal_sequence] Cleaning up memory after TAMP subgoal {subgoal_idx}...")
                gc.collect()
                logger.info(f"[process_subgoal_sequence] Garbage collection completed.")
                
                if new_nodes:
                    current_node = new_nodes[-1]
                    belief_nodes.extend(new_nodes)
                if replanned_subgoals is not None:
                    active_subgoals = replanned_subgoals
                    subgoal_idx = 0
                    continue

            subgoal_idx += 1
        
        logger.info("\n=== All subgoals processed ===")
    finally:
        # solved all subgoals or error occurred.
        # Stop final recording and destroy planning world at the very end
        if planning_world is not None:
            # Stop the last recording before destroying
            # if config.get("record_video", False) and last_tamp_subgoal_idx >= 0:
            #     logger.info("[process_subgoal_sequence] Stopping final planning world recording...")
            #     try:
            #         planning_world.franka.stop_recording(video_name=f"planning_world_subgoal{last_tamp_subgoal_idx}")
            #         logger.info("[process_subgoal_sequence] Final recording stopped.")
            #     except Exception as e:
            #         logger.warning(f"[process_subgoal_sequence] Failed to stop final recording: {e}")
            
            logger.info("[process_subgoal_sequence] Destroying planning world at end of all subgoals...")
            destroy_planning_world(planning_world)
            logger.info("[process_subgoal_sequence] Planning world destroyed.")
    
    return belief_tree, belief_nodes


def validate_belief(domain_name, updated_cb, expected_db, observation, vlm_model, problem, subgoal_type): 
    logger = logging.getLogger(__name__)
    logger.info("[validate_belief] Validating belief update...")
    validation_details = ""
    
    # 1) Check known-pose and update if needed
    current_db = expected_db
    no_changes_section1 = True  # Track if section 1 made changes
    if domain_name == 'tool_use' and getattr(observation, 'hook_traj', None) is not None:
        no_changes_section1, current_db = updated_cb.abstract(expected_db, problem)
        logger.info(f"[validate_belief] Section 1 (known-pose): no_changes={no_changes_section1}")
    
    # 2) Validate with VLM and update state if any predicate is marked "No"
    final_db = current_db
    no_changes_section2 = True
    failed_predicates = []
    if subgoal_type == "TAMP":
        no_changes_section2, final_db, validation_details, failed_predicates = vlm_predicate_evaluator(domain_name, current_db, observation, vlm_model, problem)
        logger.info(f"[validate_belief] Section 2 (VLM validation): no_changes={no_changes_section2}")
    
    # Return True only if BOTH sections made no changes, False if any changes occurred
    all_valid = no_changes_section1 and no_changes_section2
    logger.info(f"[validate_belief] Final result: all_valid={all_valid}")
    return all_valid, final_db, validation_details, failed_predicates


def vlm_predicate_evaluator(domain_name, expected_db, observation, vlm_model, problem):
    current_state = expected_db.state
    no_changes = True  # Track if state was updated (True = no changes)
    failed_predicates = []

    # Ask without 'known-pose' and 'is-hook' predicates
    expected_db_str = str(expected_db.state)
    predicates_str = []
    pattern = r'([\w-]+\([^)]*\)):\s*(true|false)'
    matches = re.findall(pattern, expected_db_str)
    excluded_prefixes = ("known-pose(", "is-hook(")  
    
    for predicate, _ in matches:
        if predicate.startswith(excluded_prefixes):
            continue
        predicates_str.append(predicate)
    expected_db_predicates = '\n'.join(predicates_str)

    system_prompt = vlm_validator(domain_name, expected_db_predicates)
    system = [{
        "type": "input_text",
        "text": system_prompt
    }]

    # Current scene images
    message_content = []
    current_image_paths = observation.image_path
    print(f"Observation image paths: {current_image_paths}")
    file_id_list = []
    for image_path in current_image_paths:
        file_id = llm.create_file(client2, image_path)
        file_id_list.append(file_id)

    prompt = f"For current scene, images are:"
    message_content.append({"type": "input_text", "text": prompt})
    for file_id in file_id_list:
        message_content.append(
            {"type": "input_image", "file_id": file_id},
        )

    # Query vlm
    response = client2.responses.create(
        model=vlm_model,
        input=[{"role": "system", "content": system},
                  {"role": "user", "content": message_content}],
        max_output_tokens=2048,
        temperature=0.0,
    )
    answer_text = response.output_text
    logger = logging.getLogger(__name__)
    logger.info("VLM Response: %s", answer_text)

    # Return True only if all answers are "Yes"
    answers = re.findall(r'\[(Yes|No)\]', answer_text)    
    all_yes = all(ans == "Yes" for ans in answers)
    logger.info("Validation result: %s (Found %d answers: %s)", 
                "PASS" if all_yes else "FAIL", len(answers), answers)
    
    predicate_answers = list(zip(predicates_str, answers))
    failed_predicates = [pred for pred, ans in predicate_answers if ans == "No"]

    if failed_predicates:
        logger.warning("[validate_belief] Failed predicates [No]: %s", failed_predicates)
        
        # Prepare updates for state using UPState.make_child()
        state_updates = {}
        false_value = problem.environment.expression_manager.FALSE()
        
        for failed_pred in failed_predicates:
            # Parse predicate name and parameters
            # e.g., "predicate_name(arg1, arg2)" -> "predicate_name", [arg1, arg2]
            pred_name = failed_pred.split('(')[0]
            rest = failed_pred[len(pred_name)+1:-1]  # Remove "pred_name(" and ")"
            args = [arg.strip() for arg in rest.split(',')] if rest else []
            
            # Find matching fluent_instance in current state
            for fluent_instance in current_state._values.keys():
                fluent_str = str(fluent_instance)
                if '(' in fluent_str:
                    curr_pred_name = fluent_str.split('(')[0]
                    if curr_pred_name == pred_name:
                        # Extract arguments from fluent_str to match
                        curr_args_str = fluent_str.split('(')[1].rstrip(')')
                        curr_args = [arg.strip() for arg in curr_args_str.split(',')] if curr_args_str else []
                        
                        # If argument count matches, set to False
                        if len(curr_args) == len(args):
                            state_updates[fluent_instance] = false_value
                            logger.info("[validate_belief] Set %s to False", failed_pred)
        
        # Create new state with updated values using make_child()
        if state_updates:
            current_state = current_state.make_child(state_updates)
            no_changes = False  # Mark that we updated the state

    return no_changes, DiscreteBelief(state=current_state), answer_text, failed_predicates


def execute_information_subgoal(
    subgoal: Subgoal,
    parent_node: BeliefNode,
    belief_tree: BeliefTree,
    simulator: SequentialSimulator,
    original_problem_template: Any,
    goal_state: Any,
    config: Any,
    env: Any,
    prob_num: int,
    prob_idx: int,
    subgoal_idx: int
) -> Tuple[List[BeliefNode], Optional[List[Subgoal]]]:
    logger = logging.getLogger(__name__)
    """
    Execute information actions and add belief node for each action.
    Returns list of created nodes and optional replanned subgoals.
    """

    current_node = parent_node
    current_cb = parent_node.continuous_belief
    current_db = parent_node.discrete_belief
    created_nodes = []
    replanned_subgoals: Optional[List[Subgoal]] = None

    subgoal_pddl_path = generate_problem_pddl(
            domain_name=config["task"],
            problem_name=f"{prob_num}_{prob_idx}",
            original_problem=original_problem_template,
            initial_state=current_db.state,
            goal_state=subgoal.state,
            subgoal_idx=subgoal_idx,
        )

    reader = PDDLReader()
    domain_name = config["task"]
    domain_pddl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain_name}.pddl"))
    subgoal_problem_template = reader.parse_problem(
        domain_pddl_path,
        subgoal_pddl_path
    )
    simulator = SequentialSimulator(subgoal_problem_template)
    
    for action_idx, action_instance in enumerate(subgoal.actions):
        action_name = action_instance.action.name
        action_params = [str(obj) for obj in action_instance.actual_parameters]
        action_str = f"({action_name} {' '.join(action_params)})"
        logger.info(f"Executing Information action: {action_str}")
        
        retry_count = 0
        action_success = False
        
        while retry_count < MAX_RETRIES and not action_success:
            # Execute action and receive observation
            success, path_payload = execute(
                method=config["planner"],
                prob_num=prob_num,
                prob_idx=prob_idx,
                trial=trial,
                repeat=repeat,
                subgoal_idx=subgoal_idx,
                node_name=f"info_{action_name}_{action_idx}",
                domain_name=config["task"],
                robot_name=config["robot"],
                sim_wrapper=env.world.franka,
                action=action_str,
                belief=current_cb,
                left=True,
                grasp_type='top'
            )
                
            if success:
                observation = path_payload["obs"]

                # Update continuous belief
                updated_cb = current_cb.update(action_str, observation)
                    
                # Calculate expected discrete state using PDDL simulator
                expected_db_state = simulator.apply(current_db.state, action_instance)
                expected_db = DiscreteBelief(state=expected_db_state)
                logger.info(f"Expected discrete belief state after action:\n{expected_db_state}\n")

                # Validate belief with VLM - returns (validation_result, filtered_belief, validation_details, failed_predicates)
                valid, updated_db, validation_details, failed_predicates = validate_belief(config["task"], updated_cb, expected_db, observation, config["vlm_model"], original_problem_template, subgoal_type="Information")
                logger.info(f"Belief validation passed: {valid}")

                # Restart planning from the updated discrete state
                if not valid:
                    # Get VLM-refined predicates for failed predicates only
                    vlm_refined_db = current_db.update(
                        config=config,
                        action=action_str,
                        observation=observation,
                        continuous_belief=updated_cb,
                        vlm_observation=validation_details,
                        problem=original_problem_template,
                        subgoal_type="Information",
                        failed_predicates=failed_predicates
                    )
                    logger.info(f"\n\nUpdated continuous belief:\n{updated_cb}\n\nUpdated discrete belief:\n{updated_db}\n")
                    updated_db = vlm_refined_db
                    logger.info("[Replanning] from updated belief due to unexpected outcome...")
                    replanned_subgoals = get_subgoals_from_plan(
                        problem_template=original_problem_template,
                        initial_state=updated_db.state,
                        goal_state=goal_state,
                    )
                    
                # Create new belief node
                new_node = BeliefNode(
                    cb=updated_cb,
                    db=updated_db
                )
                new_node.name = f"info_{action_name}_{action_idx}"

                logger.info(f"\n\nUpdated continuous belief:\n{updated_cb}\n\nUpdated discrete belief:\n{updated_db}\n")
                    
                # Add to tree
                belief_tree.add_node(
                    parent_node=current_node,
                    child_node=new_node,
                    action=action_str,
                    observation=observation
                )
                    
                created_nodes.append(new_node)
                current_node = new_node
                current_cb = updated_cb
                current_db = updated_db
                logger.info(f"Node added to tree, discrete & continuous beliefs updated")
                action_success = True
                if replanned_subgoals is not None:
                    return created_nodes, replanned_subgoals
            else:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logger.info(f"Retry {retry_count}/{MAX_RETRIES}: {path_payload.get('error')}")
                else:
                    logger.warning(f"Information action {action_name} failed after {MAX_RETRIES} retries. Triggering replanning...")
                    replanned_subgoals = get_subgoals_from_plan(
                        problem_template=original_problem_template,
                        initial_state=current_db.state,
                        goal_state=goal_state,
                    )
                    return created_nodes, replanned_subgoals

    return created_nodes, replanned_subgoals

# TODO: Different Problem Type 
def sample_hybrid_state(belief_node: BeliefNode) -> Tuple[dict, Any, dict]:
    """
    Sample from belief distributions in belief node.
    """
    cb = belief_node.continuous_belief
    db = belief_node.discrete_belief

    # Sample particles for each object in continuous belief (exclude hook)
    sampled_particles = {}
    idx_to_name = {idx: name for name, idx in cb.world.franka.object_dict.items()}
    sampled_particle_indices = {}
    for obj_idx in cb.ghost_dict.keys():
        sampled_particle_idx = cb.sample_particle(obj_idx)
        sampled_pose = get_pose(cb.world.franka.scene.entities[sampled_particle_idx])
        obj_name = idx_to_name.get(obj_idx)
        if obj_name is not None:
            sampled_particles[obj_name] = sampled_pose
        sampled_particle_indices[obj_idx] = sampled_particle_idx
    
    # Hook has known pose, no particles to sample
    if cb.hook in cb.object_dists:
        hook_name = idx_to_name.get(cb.hook)
        if hook_name is not None:
            sampled_particles[hook_name] = get_pose(cb.world.franka.scene.entities[cb.hook])
        sampled_particle_indices[cb.hook] = cb.hook
    
    return sampled_particles, db.state, sampled_particle_indices


def destroy_planning_world(planning_world: Any):
    if planning_world is None:
        return
    try:
        planning_world.franka.scene.destroy()
        del planning_world
    except Exception:
        pass


@contextmanager
def time_limit(seconds: int):
    """POSIX-only: Raises TimeoutError after seconds have elapsed"""
    def _handler(signum, frame):
        raise TimeoutError(f"timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds), 0.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, old)


def solve_tamp(
    initial_sim_state: Any,
    sim_wrapper: Any,
    franka_original: Any,
    initial_pddl_state: Any,
    goal_pddl_state: Any,
    original_problem: Any,
    config: Any,
    prob_num: int,
    prob_idx: int,
    subgoal_idx: int,
    trial: int,
    repeat: int,
    timeout: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    logger = logging.getLogger(__name__)
    
    def _solve_tamp_internal():
        ablation = False
        domain_name = config["task"]
        problem_name = f"{prob_num}_{prob_idx}"

        # Define problem pddl for new TAMP subgoal
        subgoal_pddl_path = generate_problem_pddl(
            domain_name=domain_name,
            problem_name=f"{prob_num}_{prob_idx}",
            original_problem=original_problem,
            initial_state=initial_pddl_state,
            goal_state=goal_pddl_state,
            subgoal_idx=subgoal_idx,
        )

        start = time.time()
        plans, gv_path = diverse_planning(domain_name, config["planner"], subgoal_pddl_path, plan_number=5, planner="topk")
        end = time.time()
        diverse_planning_time = end - start
        logger.info(f"[hybrid_tree_expansion] Diverse planning time: {diverse_planning_time}")

        start = time.time()
        reader = PDDLReader()
        domain_pddl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domains", f"domain_{domain_name}.pddl"))
        subgoal_problem_template = reader.parse_problem(
            domain_pddl_path,
            subgoal_pddl_path
        )
        discrete_tree = build_tree_from_dot(config["planner"], str(gv_path), domain_name, prob_num, prob_idx, subgoal_problem_template)
        end = time.time()
        graph_building_time = end - start
        logger.info(f"[hybrid_tree_expansion] Graph building time: {graph_building_time}")

        hybrid, tamp_success, hybrid_time, backtrack_count = hybrid_tree_expansion(
            initial_sim_state=initial_sim_state,
            sim_wrapper=sim_wrapper,
            franka_original=franka_original,
            method=config["planner"],
            json_path=config["json_path"],
            problem_pddl_path=subgoal_pddl_path,
            prob_num=prob_num,
            prob_idx=prob_idx,
            subgoal_idx=subgoal_idx,
            trial=trial,
            repeat=repeat,
            model=config["vlm_model"],
            domain_name=domain_name,
            robot_name=config["robot"],
            grasp_type='top',
            discrete_tree=discrete_tree,
            problem_template=subgoal_problem_template,
            sampler=model.execute,
            child_selector=vlm_selector,
            backtrack_selector=bfs_unvisited_selector if ablation else vlm_backtrack_selector,
            K=5,
            weight_threshold=0.2,
            record_video=False,
            num_distractor=12,
        )

        logger.info(f"[hybrid_tree_expansion] Hybrid tree expansion time: {hybrid_time}")
        total_time = diverse_planning_time + graph_building_time + hybrid_time
        logger.info(f"[hybrid_tree_expansion] Total TAMP subgoal planning time: {total_time}")
        logger.info(f"[hybrid_tree_expansion] TAMP subgoal success: {tamp_success}")

        # save TAMP plan
        final_plan = extract_final_plan_with_params(hybrid)
        plan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", domain_name, "tamp_plans"))
        os.makedirs(plan_dir, exist_ok=True)
        final_plan_json = os.path.join(plan_dir, f"{problem_name}_subgoal{subgoal_idx}_plan.json")
        with open(final_plan_json, "w") as f:
            json.dump(_json_safe({"plan": final_plan}), f, indent=2)
        
        return bool(tamp_success), final_plan_json
    
    try:
        if timeout is not None:
            with time_limit(timeout):
                return _solve_tamp_internal()
        else:
            return _solve_tamp_internal()
    except TimeoutError:
        logger.warning(f"Solve TAMP timeout after {timeout}s")
        return False, None


def execute_tamp_subgoal(
    subgoal: Subgoal,
    parent_node: BeliefNode,
    belief_tree: BeliefTree,
    simulator: SequentialSimulator,
    problem: Any,
    goal_state: Any,
    config: Any,
    env: Any,
    prob_num: int,
    prob_idx: int,
    subgoal_idx: int,
    trial: int,
    repeat: int,
    planning_world: Any = None,
    timestamp: str = None
) -> Tuple[List[BeliefNode], Optional[List[Subgoal]], Any]:
    logger = logging.getLogger(__name__)
    """
    Execute TAMP actions and add belief node for each action.
    Returns list of created nodes and optional replanned subgoals.
    """
    current_node = parent_node
    current_cb = parent_node.continuous_belief
    current_db = parent_node.discrete_belief
    created_nodes = []
    replanned_subgoals: Optional[List[Subgoal]] = None

    scene_state = env.world.franka.scene.get_state()
    held_object = current_cb.grasped_obj

    # Sample hybrid state from current belief node
    sampled_object_poses, hyrid_discrete_state, sampled_particle_indices = sample_hybrid_state(current_node)
    logger.info(f"[execute_tamp_subgoal] Sampled object poses:")
    for obj_name, pose in sampled_object_poses.items():
        logger.info(f"  {obj_name}: {pose}")
    logger.info(f"[execute_tamp_subgoal] Sampled particle indices: {sampled_particle_indices}\n")

    tamp_success = False
    final_plan_json = None
    original_video_dir = getattr(env.world.franka, "recording_save_dir", None)
    
    try:
        logger.info("[execute_tamp_subgoal] Creating/resetting TAMP planning world...")
        is_first_tamp_call = planning_world is None
        
        # Close original world viewer to free OpenGL context for planning world
        if is_first_tamp_call:
            logger.info("[execute_tamp_subgoal] Closing original world viewer to avoid OpenGL context conflict...")
            try:
                if hasattr(env.world.franka.scene, '_visualizer') and env.world.franka.scene._visualizer is not None:
                    env.world.franka.scene._visualizer.close()
                    logger.info("[execute_tamp_subgoal] Original world viewer closed.")
            except Exception as e:
                logger.debug(f"[execute_tamp_subgoal] Could not close viewer: {e}")
        
        planning_world = start_tamp_sim(
            config["json_path"], 
            config["planner"], 
            prob_num, 
            prob_idx, 
            trial, 
            repeat, 
            sampled_object_poses, 
            current_cb.conf, 
            show_viewer=config["show_viewer"], 
            record_video=config["record_video"],
            existing_tamp_world=planning_world,  # Pass existing world for reuse
            held_object=held_object,
        )
        logger.info("[execute_tamp_subgoal] Planning world ready (created or reset).")

        # add weld constraint
        if held_object is not None:
            rigid_solver = planning_world.franka.scene.sim.rigid_solver
            held_entity = planning_world.franka.scene.entities[held_object]
            ee_link_idx = planning_world.franka.robot.get_link(planning_world.franka.EE_FRAMES['ee']).idx
            try:
                link_obj = held_entity.get_link("box_baselink").idx
            except Exception:
                try:
                    link_obj = held_entity.get_link("hook_link").idx
                except Exception:
                    link_obj = held_entity.get_root_link().idx
            rigid_solver.add_weld_constraint(link_obj, ee_link_idx)

        planning_initial_state = planning_world.franka.scene.sim.get_state()

        # # Setup recording for TAMP planning world
        # if config.get("record_video", False):
        #     prob_name = f"{prob_num}_{prob_idx}"
        #     video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", config["task"], config["planner"], "recordings", prob_name, timestamp))
        #     os.makedirs(video_dir, exist_ok=True)
            
        #     logger.info(f"[execute_tamp_subgoal] Starting recording for TAMP planning world. Video dir: {video_dir}")
        #     planning_world.franka.start_recording(save_dir=video_dir)

        # Solve TAMP
        tamp_success, final_plan_json = solve_tamp(
            initial_sim_state=planning_initial_state,
            sim_wrapper=planning_world.franka,
            franka_original=env.world.franka,
            initial_pddl_state=hyrid_discrete_state,
            goal_pddl_state=subgoal.state,
            original_problem=problem,
            config=config,
            prob_num=prob_num,
            prob_idx=prob_idx,
            subgoal_idx=subgoal_idx,
            trial=trial,
            repeat=repeat,
            timeout=200,
        )
    finally:
        if held_object is not None:
            rigid_solver = env.world.franka.scene.sim.rigid_solver
            held_entity = env.world.franka.scene.entities[held_object]
            ee_link_idx = env.world.franka.robot.get_link(env.world.franka.EE_FRAMES['ee']).idx

            try:
                link_obj = held_entity.get_link("box_baselink").idx
            except Exception:
                try:
                    link_obj = held_entity.get_link("hook_link").idx
                except Exception:
                    link_obj = held_entity.get_root_link().idx

            try:
                rigid_solver.add_weld_constraint(link_obj, ee_link_idx)
            except Exception:
                pass

    if tamp_success:
        # execute TAMP plan and observe the outcome for each action
        with open(final_plan_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        plan = data.get("plan", [])

        for i, act in enumerate(plan):
            cp = act.get("continuous_params", {})
            traj = cp.get("traj", None)
            action_type = cp.get("act_type", None)
            action_str = cp.get("action", None)
            parts = action_str.replace('(', '').replace(')', '').split()
            action_name = parts[0]
            action_params = parts[1:] if len(parts) > 1 else []

            if action_type == "":
                continue
            logger.info(f"Executing TAMP action: {action_str}")

            obj_name = action_params[0] if action_params else None
            
            retry_count = 0
            action_success = False
            success = False
            execute_payload = None
            
            while retry_count < MAX_RETRIES and not action_success:
                success, execute_payload = execute_trajectory(
                    method=config["planner"],
                    prob_num=prob_num,
                    prob_idx=prob_idx,
                    trial=trial,
                    repeat=repeat,
                    subgoal_idx=subgoal_idx,
                    node_name=f"execute_{action_name}_{i}",
                    sim_wrapper=env.world.franka,
                    traj=traj,
                    action_type=action_type,
                    obj_name=obj_name,
                    domain_name=config["task"],
                    belief=current_cb,
                    tamp_solution=cp,
                    sampled_particle_indices=sampled_particle_indices,
                )
                
                if success:
                    observation = execute_payload["obs"]

                    # Update continuous belief
                    updated_cb = current_cb.update(action_str, observation)

                    # Calculate expected discrete state
                    action = problem.action(action_name)
                    param_objects = [problem.object(p) for p in action_params]
                    action_instance = action(*param_objects)
                    expected_db_state = simulator.apply(current_db.state, action_instance)

                    expected_db = DiscreteBelief(state=expected_db_state)

                    # Validate belief with VLM - returns (validation_result, filtered_belief, validation_details)
                    valid, updated_db, validation_details, failed_predicates = validate_belief(config["task"], updated_cb, expected_db, observation, config["vlm_model"], problem, subgoal_type="TAMP")
                    logger.info(f"Belief validation passed: {valid}")

                    # Restart planning from the updated discrete state
                    if not valid:
                        # Get VLM-refined predicates for failed predicates only
                        vlm_refined_db = current_db.update(
                            config=config,
                            action=action_str,
                            observation=observation,
                            continuous_belief=updated_cb,
                            vlm_observation=validation_details,
                            problem=problem,
                            subgoal_type="TAMP",
                            failed_predicates=failed_predicates,
                        )
                        updated_db = vlm_refined_db
                        logger.info(f"\n\nUpdated continuous belief:\n{updated_cb}\n\nUpdated discrete belief:\n{updated_db}\n")
                        logger.info("[Replanning] from updated belief due to unexpected outcome...")
                        replanned_subgoals = get_subgoals_from_plan(
                            problem_template=problem,
                            initial_state=updated_db.state,
                            goal_state=goal_state,
                        )
                    
                    logger.info(f"\n\nUpdated continuous belief:\n{updated_cb}\n\nUpdated discrete belief:\n{updated_db}\n")

                    # Create new belief node
                    new_node = BeliefNode(
                        cb=updated_cb,
                        db=updated_db
                    )
                    new_node.name = f"tamp_{action_type}_{i}"

                    # Add to tree
                    belief_tree.add_node(
                        parent_node=current_node,
                        child_node=new_node,
                        action=action_str,
                        observation=observation
                    )
                        
                    created_nodes.append(new_node)
                    current_node = new_node
                    current_cb = updated_cb
                    current_db = updated_db
                    logger.info(f"Node added to tree, discrete & continuous beliefs updated")
                    action_success = True
                    if replanned_subgoals is not None:
                        return created_nodes, replanned_subgoals, planning_world
                else:
                    # TAMP solving succedd but execution failed - very unlikely to happen
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        logger.info(f"Retry {retry_count}/{MAX_RETRIES}: {execute_payload.get('error')}")
                    else:
                        logger.warning(f"Action {action_name} failed after {MAX_RETRIES} retries. Triggering replanning due to execution failure...")
                        # Execution failed multiple times - need information gathering
                        replanned_subgoals = get_subgoals_from_plan(
                            problem_template=problem,
                            initial_state=current_db.state,
                            goal_state=goal_state,
                        )
                        return created_nodes, replanned_subgoals, planning_world

    else:
        # TAMP solving failed
        logger.warning(f"TAMP solving failed for subgoal {subgoal_idx + 1}")
        logger.info("Triggering information gathering to refine belief...")

        replanned_subgoals = get_subgoals_from_plan(
            problem_template=problem,
            initial_state=current_db.state,
            goal_state=goal_state,
        )
        return created_nodes, replanned_subgoals, planning_world
        
    # Explicit memory cleanup before returning
    logger.info(f"[execute_tamp_subgoal] Cleaning up local variables...")
    if final_plan_json and os.path.exists(final_plan_json):
        try:
            os.remove(final_plan_json)
            logger.debug(f"[execute_tamp_subgoal] Removed temporary plan file: {final_plan_json}")
        except Exception as e:
            logger.debug(f"[execute_tamp_subgoal] Could not remove plan file: {e}")
    gc.collect()
    
    return created_nodes, replanned_subgoals, planning_world


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    domain_path = "/home/minseo/develop/pomdp_llm/domains/domain_tool_use.pddl"
    problem_path = "/home/minseo/develop/pomdp_llm/experiments/tool_use/problem/tool_use3_1.pddl"
    config_path = "/home/minseo/develop/pomdp_llm/config/tool_use.yml"
    prob_num = 3
    prob_idx = 1
    trial = 1
    repeat = 1

    solve_problem(domain_path, problem_path, config_path=config_path, prob_num=prob_num, prob_idx=prob_idx, trial=trial, repeat=repeat)