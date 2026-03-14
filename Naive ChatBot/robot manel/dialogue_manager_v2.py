import json
import time
import random
import numpy as np
from utils_v2 import *


# Chitchat responses with keyword matching.
# Improvement over v1: same structure, but the intent classifier
# is now more accurate so fewer commands land here incorrectly.
CHITCHAT_RESPONSES = {
    "greeting": {
        "triggers": ["hello", "hi", "hey", "good morning", "good afternoon", "howdy", "greetings", "good day",
                      "morning", "afternoon", "evening", "welcome", "yo"],
        "responses": ["Hello! I'm Manel, your carpentry assistant. How can I help?",
                       "Hi there! Ready to work. What do you need?",
                       "Good day! What task shall we do?"]
    },
    "farewell": {
        "triggers": ["bye", "goodbye", "see you", "leaving", "good night", "catch you", "take care"],
        "responses": ["Goodbye! Good work today.",
                       "See you next time! Stay safe.",
                       "Bye! Don't forget to clean the workshop."]
    },
    "thanks": {
        "triggers": ["thank", "thanks", "appreciated", "cheers", "great job", "well done", "nice work", "good work"],
        "responses": ["You're welcome! Happy to help.",
                       "No problem! That's what I'm here for.",
                       "Glad I could help!"]
    },
    "identity": {
        "triggers": ["who are you", "what are you", "your name", "about yourself", "capabilities", "what can you do",
                      "your purpose", "built for", "help me", "how can you help", "what do you know", "describe yourself",
                      "what tasks", "how do you work"],
        "responses": ["I'm Manel, a carpentry assistant robot. I can help with cutting, drilling, sanding, assembly, "
                       "measurements, and fetching tools.",
                       "I'm your collaborative robot assistant for carpentry tasks. Just tell me what you need!",
                       "I'm Manel! I can cut, drill, sand, assemble, glue, clamp, measure, fetch tools, and navigate "
                       "around the workshop."]
    },
    "smalltalk": {
        "triggers": ["weather", "tired", "hungry", "coffee", "break", "lunch", "bored", "cold", "hot",
                      "hard work", "long day", "joke", "funny", "music", "time is it", "day is it",
                      "how old", "dream", "feelings", "happy", "favorite", "do you like", "smart",
                      "interesting", "how much longer"],
        "responses": ["Ha, I'm just a robot - I stick to carpentry! What do you need me to do?",
                       "I wish I could help with that! But I'm better with wood. Need anything cut or drilled?",
                       "That's above my pay grade! But I can sand, cut, drill, or assemble something for you."]
    },
    "confirmation": {
        "triggers": ["yes", "sure", "of course", "absolutely", "definitely", "ok", "okay", "alright",
                      "fine", "correct", "right", "exactly", "agree", "got it", "understood", "copy",
                      "roger", "affirmative", "sounds good", "makes sense"],
        "responses": ["Got it! What's next?",
                       "Understood. Ready for the next task!",
                       "Alright! Tell me what to do."]
    },
    "negation": {
        "triggers": ["no", "not really", "negative", "nope", "nah", "never mind", "forget it",
                      "does not matter", "cancel", "abort", "disregard", "scratch that", "ignore",
                      "undo", "take that back", "not what i meant", "mistake", "oops", "my bad",
                      "wrong"],
        "responses": ["No problem! Let me know when you need something.",
                       "Okay, standing by for your next command.",
                       "Understood, ignoring that. What would you like instead?"]
    },
    "wait": {
        "triggers": ["wait", "hold on", "moment", "second", "pause", "stop", "freeze", "halt",
                      "stand by", "hang on", "not now", "give me a minute", "give me a moment",
                      "give me a sec", "let me think", "slow down"],
        "responses": ["Standing by... Let me know when you're ready!",
                       "Paused. Take your time!",
                       "Waiting for your command."]
    },
    "repeat": {
        "triggers": ["repeat", "say again", "did not understand", "what did you say", "come again",
                      "pardon", "excuse me", "sorry what", "missed that", "what was that", "catch that",
                      "louder"],
        "responses": ["Sorry about that! Could you rephrase your request?",
                       "I'll try to be clearer. What do you need?",
                       "My apologies. Please tell me again what you'd like me to do."]
    },
    "safety": {
        "triggers": ["careful", "watch out", "heads up", "look out", "stay safe", "safety first",
                      "be alert", "pay attention", "focus", "danger", "emergency"],
        "responses": ["Safety acknowledged! I'm always careful. What's the concern?",
                       "Understood, staying alert. Safety is my top priority!",
                       "Got it, proceeding with caution!"]
    },
    "default": {
        "triggers": [],
        "responses": ["I'm not sure I understand. Could you rephrase that?",
                       "I didn't quite get that. Can you say it differently?",
                       "Sorry, I don't understand. Try asking me to do a carpentry task!"]
    }
}


def get_chitchat_response(text):
    """Returns a chitchat response based on keyword matching."""
    text_lower = text.lower()
    for category, data in CHITCHAT_RESPONSES.items():
        if category == "default":
            continue
        for trigger in data["triggers"]:
            if trigger in text_lower:
                return random.choice(data["responses"])
    return random.choice(CHITCHAT_RESPONSES["default"]["responses"])


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])

        # Action plans:
        with open(paths['ACTION_PLANS'], 'r') as f:
            self.action_plans = json.load(f)

        # Task log for status/summary:
        self.task_log = []

        print("Resources loaded successfully!")

    def generate_answer(self, question):
        """Combines robot command and chitchat parts using intent recognition.

        Improvement over v1: uses prediction probability to handle
        ambiguous cases more gracefully instead of a hard binary decision.
        """

        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform(np.array([prepared_question]))

        # Get intent prediction with probabilities
        intent = self.intent_recognizer.predict(features)[0]
        intent_proba = self.intent_recognizer.predict_proba(features)
        intent_confidence = intent_proba.max()

        # If intent confidence is low (ambiguous), check both paths
        # This helps with borderline cases like "give me a summary"
        if intent == 'chitchat' and intent_confidence < 0.65:
            # Low-confidence chitchat - try classifying as command too
            tag = self.tag_classifier.predict(features)[0]
            tag_confidence = self.tag_classifier.predict_proba(features).max()

            # If the tag classifier is reasonably confident, treat as command
            if tag_confidence > 0.45:
                return self._build_command_response(question, tag, tag_confidence)

        # Chit-chat part:
        if intent == 'chitchat':
            response = get_chitchat_response(question)
            return response

        # Robot command part:
        else:
            tag = self.tag_classifier.predict(features)[0]
            confidence = self.tag_classifier.predict_proba(features).max()

            if confidence < 0.4:
                return "I'm not confident I understood that command. Could you rephrase?"

            return self._build_command_response(question, tag, confidence)

    def _build_command_response(self, question, tag, confidence):
        """Builds a response for a recognized robot command."""
        plan = self.action_plans.get(tag, {})
        steps = plan.get("steps", ["unknown_action"])
        description = plan.get("description", "Unknown action")
        safety = plan.get("safety_notes", "None")

        # Log the task
        self.task_log.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "question": question,
            "action": tag,
            "confidence": float(confidence)
        })

        # Build response
        response = f"Action: {tag} - {description}\n"
        response += f"Confidence: {confidence:.2f}\n"
        response += f"Steps: {' -> '.join(steps)}\n"
        response += f"Safety: {safety}"

        return response

    def get_summary(self):
        """Returns a summary of all tasks performed."""
        if not self.task_log:
            return "No tasks performed yet."

        lines = ["=== Task Summary ==="]
        for i, task in enumerate(self.task_log, 1):
            lines.append(f"{i}. [{task['timestamp']}] {task['action']} "
                         f"(confidence={task['confidence']:.2f}) - \"{task['question']}\"")
        lines.append(f"\nTotal tasks: {len(self.task_log)}")
        return "\n".join(lines)
