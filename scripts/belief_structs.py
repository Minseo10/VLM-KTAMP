from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Tuple, List
from utils.utils import *
from symbolic import upstate_to_grouped_facts
from unified_planning.model import State as UPState
from unified_planning.model import Problem
import logging, os, json
from openai import OpenAI
from utils.prompts import vlm_predicate_proposer


config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
api_key = config["OPENAI_API_KEY"]
client2 = OpenAI(api_key=api_key)


@dataclass
class State:
    pass


@dataclass
class Observation:
    pass


@dataclass
class Action:
    name: str = "default-action"
    args: List[Any] = field(default_factory=lambda: [])

    def __hash__(self):
        return hash(tuple([self.name] + list(self.args)))

    def __eq__(self, b):
        return hash(self) == hash(b)

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(self.args))


@dataclass
class ContinuousBelief:
    def update(
        self,
        action: Action,
        observation: Observation,
    ) -> ContinuousBelief:
        raise NotImplementedError


@dataclass
class DiscreteBelief:
    """
    Discrete belief state supporting multiple representations:
    - grouped facts: (('pred', (('arg1', 'arg2'), ...)), ...) for pddlpy
    - UPState: unified_planning fluent-based state
    """
    
    state: Any = field(default=None)  # Can be grouped facts tuple or UPState
    
    def __repr__(self) -> str:
        return f"DiscreteBelief({self.state})"
    
    def __hash__(self) -> int:
        return hash(str(self.state))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiscreteBelief):
            return False
        return self.state == other.state
    
    def is_grouped_facts(self) -> bool:
        """Check if state is grouped facts format (tuple of tuples)."""
        return isinstance(self.state, tuple) and len(self.state) > 0 and \
               isinstance(self.state[0], tuple) and len(self.state[0]) == 2
    
    def is_upstate(self) -> bool:
        """Check if state is unified_planning UPState."""
        return isinstance(self.state, UPState)
    
    def get_grouped_facts(self) -> Any:
        """
        Get state as grouped facts format.
        If already in grouped facts format, return as-is.
        If UPState, return self (caller must handle conversion).
        """
        if self.is_grouped_facts():
            return self.state
        elif self.is_upstate():
            return upstate_to_grouped_facts(self.state)
        return self.state
    
    def _parse_fluent_string(self, fluent_str: str) -> Tuple[str, str]:
        """Parse fluent string like 'on-table(red_block)' into ('on-table', 'red_block')."""
        if '(' in fluent_str:
            pred_name = fluent_str.split('(')[0].strip()
            args_str = fluent_str.split('(')[1].rstrip(')').strip()
            return pred_name, args_str
        return fluent_str, ''
    
    def _fluents_match(self, pred_name: str, args_str: str, ext_pred_name: str, ext_args: str) -> bool:
        """Check if two fluents match (handling spaces and dash/underscore variants)."""
        if pred_name != ext_pred_name:
            return False
        
        ext_args_normalized = ', '.join(arg.strip() for arg in ext_args.split(',')) if ext_args else ''
        args_normalized = ', '.join(arg.strip() for arg in args_str.split(',')) if args_str else ''
        
        return (ext_args_normalized == args_normalized or 
                ext_args.replace(' ', '') == args_str.replace(' ', ''))
    
    def update(self, config, action: Action, observation: Observation, continuous_belief: ContinuousBelief, vlm_observation: str, problem: Problem, subgoal_type: str, failed_predicates: List[str]) -> DiscreteBelief:
        """
        Update discrete belief based on action and observation.
        
        Parse VLM response to extract predicates for failed predicate refinement.
        Only updates predicates related to objects in failed_predicates.
        Preserves existing state predicates and merges VLM proposed predicates.
        Format: "1. predicate(args): description"
        """
        current_state = self.state
        logger = logging.getLogger(__name__)

        if subgoal_type == "TAMP" and failed_predicates:
            # Get object names
            objects = []
            for obj_idx, _ in continuous_belief.object_dists.items():
                for name, idx in continuous_belief.world.franka.object_dict.items():
                    if idx == obj_idx:
                        objects.append(name)

            system_prompt = vlm_predicate_proposer(config["task"], vlm_observation, failed_predicates, str(objects))  # vlm-observation에선 known-pose, hook 관련은 제외되어 있음
            system = [{
                "type": "input_text",
                "text": system_prompt
            }]
            logger.info(f"[DiscreteBelief update] VLM observation:{vlm_observation}")
            # logger.info(f"[DiscreteBelief update] Failed predicates: {failed_predicates}")

            # Current scene images from observation
            message_content = []
            current_image_paths = observation.image_path
            logger.info(f"[DiscreteBelief update] Observation image paths: {current_image_paths}")
            file_id_list = []
            for image_path in current_image_paths:
                file_id = create_file(client2, image_path)
                file_id_list.append(file_id)

            prompt = f"For current scene, images are:"
            message_content.append({"type": "input_text", "text": prompt})
            for file_id in file_id_list:
                message_content.append(
                    {"type": "input_image", "file_id": file_id},
                )

            # Query vlm
            response = client2.responses.create(
                model=config["vlm_model"],
                input=[{"role": "system", "content": system},
                        {"role": "user", "content": message_content}],
                max_output_tokens=2048,
                temperature=0.0,
            )
            answer_text = response.output_text
            logger.info("[DiscreteBelief update] VLM Output:\n%s", answer_text)

            # Extract predicates from answer_text
            import re
            pattern = r'(\w+[\w\-]*)\(([^)]*)\):\s*([^\n]*)'
            matches = re.findall(pattern, answer_text)
            vlm_proposed_predicates = {}
            
            for pred_name, args_str, description in matches:
                vlm_proposed_predicates[(pred_name, args_str)] = True

            # Build fluent lookup dictionary indexed by name (handling - and _ variants)
            fluent_lookup = {}
            for fluent_def in problem.fluents:
                fluent_name = str(fluent_def.name)
                fluent_lookup[fluent_name] = fluent_def
                fluent_lookup[fluent_name.replace('-', '_')] = fluent_def
                fluent_lookup[fluent_name.replace('_', '-')] = fluent_def

            # Strategy: Replace predicates with VLM proposed ones, but preserve known-pose and is-hook
            state_updates = {}
            true_value = problem.environment.expression_manager.TRUE()
            false_value = problem.environment.expression_manager.FALSE()
            
            # Process current state: replace with VLM proposed, but keep known-pose and is-hook
            processed_fluents = set()
            for fluent_instance in current_state._values.keys():
                fluent_str = str(fluent_instance)
                pred_name, args_str = self._parse_fluent_string(fluent_str)
                processed_fluents.add((pred_name, args_str))
                
                # Check if this predicate should be preserved (known-pose or is-hook)
                is_preserved = pred_name == "known-pose" or pred_name == "is-hook"
                
                if is_preserved:
                    # Keep existing value for known-pose and is-hook
                    state_updates[fluent_instance] = current_state._values[fluent_instance]
                    # logger.info(f"[DiscreteBelief update] Preserved (protected): {fluent_str}")
                else:
                    # Find matching predicate in VLM proposed list
                    matched = False
                    for (ext_pred_name, ext_args), is_true in vlm_proposed_predicates.items():
                        if self._fluents_match(pred_name, args_str, ext_pred_name, ext_args):
                            state_updates[fluent_instance] = true_value
                            matched = True
                            # logger.info(f"[DiscreteBelief update] Updated from VLM: {fluent_str} = True")
                            break
                    
                    if not matched:
                        # Not in VLM proposed, so set to FALSE (closed-world assumption)
                        state_updates[fluent_instance] = false_value
                        # logger.info(f"[DiscreteBelief update] Not in VLM proposed: {fluent_str} = False")

            # Handle VLM proposed predicates not in current_state (skip known-pose and is-hook)
            for (ext_pred_name, ext_args), is_true in vlm_proposed_predicates.items():
                is_protected = ext_pred_name == "known-pose" or ext_pred_name == "is-hook"
                
                if (ext_pred_name, ext_args) not in processed_fluents and is_true and not is_protected:
                    fluent_def = fluent_lookup.get(ext_pred_name)
                    if fluent_def:
                        try:
                            if ext_args:
                                arg_names = [arg.strip() for arg in ext_args.split(',')]
                                # Convert string names to problem objects
                                args = [problem.object(arg_name) for arg_name in arg_names]
                                fluent_instance = fluent_def(*args)
                            else:
                                fluent_instance = fluent_def()
                            state_updates[fluent_instance] = true_value
                            # logger.info(f"[DiscreteBelief update] Added new fluent from VLM: {fluent_instance} = True")
                        except Exception as e:
                            logger.warning(f"[DiscreteBelief update] Failed to create {ext_pred_name}({ext_args}): {e}")
                    else:
                        logger.warning(f"[DiscreteBelief update] No fluent definition found for: {ext_pred_name}")

            if state_updates:
                current_state = current_state.make_child(state_updates)
                logger.info(f"[DiscreteBelief update] VLM updated state:\n{current_state}")
        else:
            if not failed_predicates:
                logger.info(f"[DiscreteBelief update] No VLM update applied. No failed predicates.")
                print(f"[DiscreteBelief update] No VLM update applied. No failed predicates.")
            else:
                logger.info(f"[DiscreteBelief update] No VLM update applied. Subgoal type: {subgoal_type}")
                print(f"[DiscreteBelief update] No VLM update applied. Subgoal type: {subgoal_type}")

        return DiscreteBelief(state=current_state)
