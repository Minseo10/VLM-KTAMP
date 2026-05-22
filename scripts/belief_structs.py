from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Tuple, List
from utils.utils import *
from symbolic import upstate_to_grouped_facts
from unified_planning.model import State as UPState
from unified_planning.model import Problem
import logging, os, json
from openai import OpenAI


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
        raise NotImplementedError
