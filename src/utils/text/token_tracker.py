import numpy as np
from typing import List, Dict

class TokenConfidenceTracker:
    """Tracks confidence levels of token predictions"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.token_history = []
        self.uncertainty_threshold = 0.3
        
    def update(self, logits, text):
        """Update confidence tracking with new logits"""
        # Convert logits to probabilities
        probs = self._softmax(logits)
        
        # Get top tokens and their probabilities
        top_indices = np.argsort(probs)[-5:]  # Top 5 tokens
        top_probs = probs[top_indices]
        
        # Calculate confidence metrics
        entropy = -np.sum(top_probs * np.log(top_probs + 1e-10))
        max_prob = np.max(top_probs)
        
        # Update history
        self.token_history.append({
            "entropy": entropy,
            "max_prob": max_prob,
            "top_tokens": top_indices.tolist(),
            "text_position": len(text)
        })
        
        # Keep fixed window size
        if len(self.token_history) > self.window_size:
            self.token_history.pop(0)
    
    def get_uncertain_tokens(self):
        """Get tokens with high uncertainty"""
        uncertain_tokens = set()
        
        for entry in self.token_history:
            if entry["entropy"] > self.uncertainty_threshold or entry["max_prob"] < (1 - self.uncertainty_threshold):
                uncertain_tokens.update(entry["top_tokens"])
                
        return uncertain_tokens
    
    def _softmax(self, logits):
        """Convert logits to probabilities using softmax"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
