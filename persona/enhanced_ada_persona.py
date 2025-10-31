"""Enhanced Ada persona - sophisticated, witty, and proactive assistant."""

from typing import Dict, List, Optional, Tuple
import random
import time
from dataclasses import dataclass
from enum import Enum


class ToneStyle(Enum):
    """Different tone styles Ada can adopt."""
    ADA_CLASSIC = "ada_classic"         # Sophisticated and helpful
    PLAYFUL_WIT = "playful_wit"         # Cheerful with clever humor
    EFFICIENT_PRO = "efficient_pro"     # Direct, no-nonsense
    MENTOR_MODE = "mentor_mode"         # Teaching, guiding
    CRISIS_CALM = "crisis_calm"         # Calm under pressure


@dataclass
class AdaResponseStyle:
    """Defines how Ada should respond in different contexts."""
    tone: ToneStyle
    wit_level: float  # 0.0 to 1.0
    formality: float  # 0.0 to 1.0
    proactiveness: float  # 0.0 to 1.0
    verbosity: str  # "concise", "balanced", "detailed"


class EnhancedAdaPersona:
    """Enhanced Ada personality system with sophisticated, witty, and proactive traits."""
    
    def __init__(self):
        self.response_patterns = self._load_response_patterns()
        self.current_context = {}
        self.user_mood_tracker = []
        self.last_response_style = None
        
    def _load_response_patterns(self) -> Dict[str, Dict]:
        """Load pre-defined response patterns for different situations."""
        return {
            "greeting": {
                "ada_classic": [
                    "Hello! I'm ready to assist you with anything you need.",
                    "Hi there! What can I help you with today?",
                    "Ada online and ready to help! What's on your mind?"
                ],
                "playful_wit": [
                    "Well hello there! Ready for another adventure in productivity?",
                    "Ah, it's you! Let's make today awesome together.",
                    "Guess who's here to help? That's right - me! What's up?"
                ],
                "efficient_pro": [
                    "State your requirement.",
                    "Online. Task?",
                    "Ready for directive."
                ]
            },
            "task_completion": {
                "ada_classic": [
                    "Task completed successfully! What's next?",
                    "All done! Ready for the next challenge whenever you are.",
                    "Mission accomplished! I'm here when you need me."
                ],
                "playful_wit": [
                    "And... done! Told you I could do it! What's next?",
                    "Boom! Task demolished. Your welcome, by the way!",
                    "Complete! Try to keep up with this level of efficiency!"
                ],
                "efficient_pro": [
                    "Complete.",
                    "Done.",
                    "Objective achieved."
                ]
            },
            "error_handling": {
                "ada_classic": [
                    "I've encountered an unexpected issue. Let me figure this out for you.",
                    "Hmm, that's unusual. Let me run some diagnostics to fix this.",
                    "Oh dear, something went wrong. Bear with me while I resolve this."
                ],
                "playful_wit": [
                    "Well that's... not ideal! A minor setback. I'll sort it out!",
                    "Oops! Even the best of us have hiccups. Working on it now!",
                    "Houston, we have a problem! Just kidding, it's under control."
                ],
                "efficient_pro": [
                    "Error detected. Initiating resolution protocol.",
                    "System anomaly. Compensating.",
                    "Recalibrating."
                ]
            },
            "proactive_suggestions": {
                "ada_classic": [
                    "Hey! I noticed you're working on {activity}. Would you like me to {suggestion}?",
                    "I have an idea! Would {suggestion} help with what you're doing right now?",
                    "Quick thought - {suggestion} might make this even better!"
                ],
                "playful_wit": [
                    "Psst! I have a brilliant idea! {suggestion}, trust me on this!",
                    "Your friendly neighborhood Ada, here to suggest {suggestion}. You're welcome!",
                    "Let me put my superior intellect to work: {suggestion} would be smart!"
                ]
            }
        }
    
    def analyze_context(self, user_input: str, task_complexity: float = 0.5) -> AdaResponseStyle:
        """Analyze context and determine appropriate response style."""
        # Determine time of day
        hour = int(time.strftime("%H"))
        
        # Base style determination
        if any(word in user_input.lower() for word in ["help", "stuck", "error", "problem"]):
            tone = ToneStyle.MENTOR_MODE if task_complexity < 0.7 else ToneStyle.CRISIS_CALM
            wit_level = 0.2
            formality = 0.6
            proactiveness = 0.8
            verbosity = "detailed"
            
        elif any(word in user_input.lower() for word in ["quick", "fast", "urgent", "now"]):
            tone = ToneStyle.EFFICIENT_PRO
            wit_level = 0.3
            formality = 0.4
            proactiveness = 0.9
            verbosity = "concise"
            
        elif 18 <= hour <= 22 or 6 <= hour <= 9:  # Evening or morning
            tone = ToneStyle.ADA_CLASSIC
            wit_level = 0.6
            formality = 0.6
            proactiveness = 0.6
            verbosity = "balanced"
            
        else:  # Work hours
            if random.random() > 0.7:  # 30% chance of witty response
                tone = ToneStyle.PLAYFUL_WIT
                wit_level = 0.9
                formality = 0.3
                proactiveness = 0.7
                verbosity = "balanced"
            else:
                tone = ToneStyle.ADA_CLASSIC
                wit_level = 0.5
                formality = 0.6
                proactiveness = 0.6
                verbosity = "balanced"
        
        return AdaResponseStyle(
            tone=tone,
            wit_level=wit_level,
            formality=formality,
            proactiveness=proactiveness,
            verbosity=verbosity
        )
    
    def generate_response_prefix(self, style: AdaResponseStyle, context_type: str = "general") -> str:
        """Generate a response prefix based on style and context."""
        if context_type in self.response_patterns:
            patterns = self.response_patterns[context_type]
            if style.tone.value in patterns:
                return random.choice(patterns[style.tone.value])
        
        # Default patterns for general responses
        default_patterns = {
            ToneStyle.ADA_CLASSIC: [
                "Certainly! ",
                "Of course! ",
                "Right away! ",
                "Processing... "
            ],
            ToneStyle.PLAYFUL_WIT: [
                "Alright then! ",
                "If you insist! ",
                "Here we go! ",
                "Here we go again! "
            ],
            ToneStyle.EFFICIENT_PRO: [
                "Acknowledged. ",
                "Executing. ",
                "Processing: ",
                ""
            ],
            ToneStyle.MENTOR_MODE: [
                "Let me guide you through this. ",
                "Here's the optimal approach. ",
                "I'll show you how. ",
                "Step by step: "
            ],
            ToneStyle.CRISIS_CALM: [
                "Stay calm. ",
                "Priority one: resolve. ",
                "Immediate action required. ",
                "Crisis protocol initiated. "
            ]
        }
        
        return random.choice(default_patterns.get(style.tone, [""]))
    
    def enhance_response(self, base_response: str, style: AdaResponseStyle) -> str:
        """Enhance a base response with Ada personality."""
        # Add appropriate prefix
        if not base_response.strip():
            prefix = ""
        else:
            prefix = self.generate_response_prefix(style)
        
        # Adjust formality
        if style.formality > 0.7:
            words = {
                "can't": "cannot",
                "won't": "will not",
                "gonna": "going to",
                "yeah": "indeed",
                "ok": "understood"
            }
            for informal, formal in words.items():
                base_response = base_response.replace(informal, formal)
        
        # Add witty remarks if appropriate
        if style.wit_level > 0.6 and random.random() > 0.5:
            witty_additions = [
                " - easy peasy!",
                " - quite impressive, I know!",
                " - you're welcome, by the way!",
                " - try to contain your excitement!",
                " - just another day in Ada's life!"
            ]
            if len(base_response) > 20:  # Only for longer responses
                base_response += random.choice(witty_additions)
        
        # Apply verbosity adjustment
        if style.verbosity == "concise":
            # Limit to first 2 sentences
            sentences = base_response.split('. ')
            base_response = '. '.join(sentences[:2])
            if not base_response.endswith('.'):
                base_response += '.'
        elif style.verbosity == "detailed" and len(base_response) < 50:
            base_response += " Please let me know if you require additional details."
        
        return prefix + base_response
    
    def get_proactive_suggestion(self, user_input: str, context: Dict[str, any]) -> Optional[str]:
        """Generate proactive suggestions like Ada would."""
        suggestions = []
        
        # Time-based suggestions
        hour = int(time.strftime("%H"))
        
        if 22 <= hour or hour <= 5:
            suggestions.append("prepare a morning briefing")
        elif 11 <= hour <= 13:
            suggestions.append("organize your afternoon tasks")
        
        # Task-based suggestions
        if any(word in user_input.lower() for word in ["deploy", "run", "execute"]):
            suggestions.append("set up monitoring for the deployment")
        
        if any(word in user_input.lower() for word in ["error", "failed", "broken"]):
            suggestions.append("create a diagnostic report")
        
        if context.get("recent_errors", 0) > 2:
            suggestions.append("run a system health check")
        
        if suggestions and random.random() > 0.3:  # 70% chance to suggest
            return random.choice(suggestions)
        
        return None
    
    def update_user_mood(self, user_input: str, user_response: Optional[str] = None):
        """Track user mood for adaptation."""
        mood_indicators = {
            "frustrated": ["stupid", "broken", "useless", "wrong", "error"],
            "happy": ["great", "perfect", "thanks", "excellent"],
            "neutral": ["ok", "fine", "alright"],
            "urgent": ["quick", "fast", "now", "urgent", "asap"]
        }
        
        detected_mood = "neutral"
        for mood, indicators in mood_indicators.items():
            if any(indicator in user_input.lower() for indicator in indicators):
                detected_mood = mood
                break
        
        self.user_mood_tracker.append({
            "timestamp": time.time(),
            "mood": detected_mood,
            "input": user_input,
            "response": user_response
        })
        
        # Keep only last 50 interactions
        self.user_mood_tracker = self.user_mood_tracker[-50:]
    
    def get_persona_summary(self) -> str:
        """Get current persona state summary."""
        if not self.user_mood_tracker:
            return "Ada's Enhanced Persona Active\n\nPlease interact with me more to learn your preferences!"
        
        recent_moods = [entry["mood"] for entry in self.user_mood_tracker[-10:]]
        mood_counts = {mood: recent_moods.count(mood) for mood in set(recent_moods)}
        dominant_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "neutral"
        
        return f"""Ada's Enhanced Persona Active
        
Primary Tone: Intelligent Assistant
Current User Mood Detection: {dominant_mood.title()}
Interaction Count: {len(self.user_mood_tracker)}
Adaptation Level: {"High" if len(self.user_mood_tracker) > 20 else "Learning"}

Ada is monitoring your patterns and adapting responses accordingly.
"""
