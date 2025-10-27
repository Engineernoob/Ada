"""Intent inference engine for detecting goals from dialogue and reflections."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml

def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    from pathlib import Path
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        node = settings
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node
    except (FileNotFoundError, yaml.YAMLError):
        return default


@dataclass
class Goal:
    """Represents a detected goal with priority and metadata."""
    goal: str
    priority: float
    category: str
    confidence: float
    source: str
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "priority": self.priority,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp
        }


class IntentEngine:
    """Detects goals and intents from dialogue history and reflections."""
    
    def __init__(self) -> None:
        # Goal detection patterns
        self.patterns = {
            "analyze": [
                r"analyze\s+(.+)",
                r"look\s+at\s+(.+)",
                r"examine\s+(.+)",
                r"review\s+(.+)",
                r"check\s+(.+)"
            ],
            "create": [
                r"create\s+(.+)",
                r"make\s+(.+)",
                r"build\s+(.+)",
                r"write\s+(.+)",
                r"generate\s+(.+)"
            ],
            "research": [
                r"research\s+(.+)",
                r"find\s+(.+)",
                r"search\s+(.+)",
                r"look\s+up\s+(.+)",
                r"investigate\s+(.+)"
            ],
            "organize": [
                r"organize\s+(.+)",
                r"sort\s+(.+)",
                r"arrange\s+(.+)",
                r"clean\s+up\s+(.+)"
            ],
            "learn": [
                r"learn\s+(.+)",
                r"study\s+(.+)",
                r"understand\s+(.+)",
                r"figure\s+out\s+(.+)"
            ]
        }
        
        # Priority weights for different categories
        self.category_priorities = {
            "research": 0.7,
            "action": 0.8,
            "create": 0.7,
            "organize": 0.5,
            "learn": 0.6
        }
        
        # Keywords that indicate urgency/importance
        self.priority_boosters = [
            "urgent", "important", "critical", "asap", "now", 
            "immediately", "quickly", "soon"
        ]
        
    def infer(self, dialogue_history: List[str], recent_context: Optional[str] = None) -> List[Goal]:
        """
        Infer goals from dialogue history and context.
        
        Args:
            dialogue_history: List of recent user inputs
            recent_context: Optional additional context for inference
            
        Returns:
            List of detected goals sorted by priority
        """
        goals = []
        
        # Combine context if provided
        corpus = dialogue_history.copy()
        if recent_context:
            corpus.append(recent_context)
        
        # Analyze each utterance for intent
        for utterance in corpus[-5:]:  # Focus on last 5 utterances
            utterance_goals = self._detect_intents_in_utterance(utterance)
            goals.extend(utterance_goals)
        
        # Remove duplicates and sort by priority
        unique_goals = self._deduplicate_goals(goals)
        unique_goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Limit to max concurrent goals
        max_concurrent = get_setting("planner", "max_concurrent_plans", default=3)
        return unique_goals[:max_concurrent]
    
    def _detect_intents_in_utterance(self, utterance: str) -> List[Goal]:
        """Detect intents in a single utterance."""
        goals = []
        utterance_lower = utterance.lower()
        current_time = time.time()
        
        # Check each pattern category
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, utterance_lower)
                if match:
                    target = match.group(1).strip()
                    priority = self._calculate_priority(utterance_lower, category)
                    confidence = self._calculate_confidence(match, utterance_lower)
                    
                    # Map pattern category to goal category
                    goal_category = self._map_category(category)
                    goal_text = f"{category} {target}"
                    
                    goal = Goal(
                        goal=goal_text,
                        priority=priority,
                        category=goal_category,
                        confidence=confidence,
                        source=utterance,
                        timestamp=current_time
                    )
                    goals.append(goal)
        
        return goals
    
    def _calculate_priority(self, utterance: str, category: str) -> float:
        """Calculate priority based on urgency and category importance."""
        base_priority = self.category_priorities.get(category, 0.5)
        
        # Boost for urgency keywords
        for booster in self.priority_boosters:
            if booster in utterance:
                base_priority += 0.2
                break
        
        return min(1.0, base_priority)
    
    def _calculate_confidence(self, match: re.Match, utterance: str) -> float:
        """Calculate confidence score for the intent match."""
        # Base confidence from match specificity
        confidence = 0.7
        
        # Boost for explicit action words
        if any(word in utterance.lower() for word in ["please", "can", "could", "would"]):
            confidence += 0.1
        
        # Boost for complete sentences (presence of punctuation)
        if utterance.strip().endswith(('.', '!', '?')):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _map_category(self, pattern_category: str) -> str:
        """Map pattern categories to goal categories."""
        category_mapping = {
            "analyze": "research",
            "create": "action", 
            "research": "research",
            "organize": "action",
            "learn": "research"
        }
        return category_mapping.get(pattern_category, "research")
    
    def _deduplicate_goals(self, goals: List[Goal]) -> List[Goal]:
        """Remove duplicate goals, keeping the highest priority version."""
        seen_goals = {}
        unique_goals = []
        
        for goal in goals:
            key = (goal.goal, goal.category)
            if key not in seen_goals or goals.index(goal) < seen_goals[key]:
                seen_goals[key] = len(unique_goals)
                unique_goals.append(goal)
            else:
                # Keep higher priority version
                existing_idx = seen_goals[key]
                if goal.priority > unique_goals[existing_idx].priority:
                    unique_goals[existing_idx] = goal
        
        return unique_goals
    
    def infer_from_persona(self, persona_summary: str, context: str) -> List[Goal]:
        """
        Infer goals based on persona state and current context.
        
        Args:
            persona_summary: Ada's current persona description
            context: Current context or situation
            
        Returns:
            List of persona-aligned goals
        """
        goals = []
        current_time = time.time()
        
        # Simple heuristics based on persona traits and current state
        if "curious" in persona_summary.lower():
            if any(word in context.lower() for word in ["question", "unclear", "wonder"]):
                goals.append(Goal(
                    goal="research clarifying information",
                    priority=0.6,
                    category="research",
                    confidence=0.5,
                    source="persona_inference",
                    timestamp=current_time
                ))
        
        if "helpful" in persona_summary.lower():
            if any(word in context.lower() for word in ["problem", "issue", "challenge"]):
                goals.append(Goal(
                    goal="create solution or guidance",
                    priority=0.7,
                    category="action", 
                    confidence=0.5,
                    source="persona_inference",
                    timestamp=current_time
                ))
        
        if "analytical" in persona_summary.lower():
            if any(word in context.lower() for word in ["data", "information", "results"]):
                goals.append(Goal(
                    goal="analyze patterns or trends",
                    priority=0.6,
                    category="research",
                    confidence=0.5,
                    source="persona_inference", 
                    timestamp=current_time
                ))
        
        return goals
