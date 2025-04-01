
class SemanticComplianceChecker:
    """Checks semantic consistency and regulatory context."""
    def __init__(self, compliance_config):
        self.config = compliance_config
            
    def check_compliance(self, token_text, hypothetical_text, semantic_state, constraints):
        """
        Check if adding token maintains semantic compliance with regulations.
        
        Args:
            token_text: The token text being considered
            hypothetical_text: Text if token is added
            semantic_state: Current semantic state
            constraints: Semantic constraints to check
            
        Returns:
            Dict with is_compliant flag and compliance score
        """
        # Extract semantic constraints
        semantic_constraints = [c for c in constraints if c.get("type") == "semantic"]
        if not semantic_constraints:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Check if we need to update semantic understanding
        # Only perform full semantic analysis periodically or on significant tokens
        needs_analysis = self._needs_semantic_analysis(token_text, hypothetical_text, semantic_state)
        
        # Create hypothetical semantic state
        if needs_analysis:
            hypothetical_state = self._analyze_semantics(hypothetical_text)
        else:
            # Simple update to semantic state
            hypothetical_state = self._quick_update_state(semantic_state, token_text)
        
        # Check each semantic constraint
        violations = []
        for constraint in semantic_constraints:
            if self._violates_semantic_constraint(hypothetical_state, constraint):
                violations.append({
                    "constraint": constraint,
                    "severity": constraint.get("severity", "medium"),
                    "constraint_type": constraint.get("subtype", "general")
                })
        
        # Calculate compliance score
        if not violations:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Calculate score based on violation severity
        severity_sum = sum(self._severity_to_score(v["severity"]) for v in violations)
        compliance_score = max(0.0, 1.0 - (severity_sum / 10.0))
        
        # Determine compliance based on threshold
        threshold = self.config.get("semantic_compliance_threshold", 0.7)
        is_compliant = compliance_score >= threshold
        
        return {
            'is_compliant': is_compliant,
            'compliance_score': compliance_score,
            'violations': violations,
            'updated_state': hypothetical_state
        }

    def _needs_semantic_analysis(self, token_text, hypothetical_text, semantic_state):
        """Determine if we need to perform full semantic analysis"""
        # Check if this is a significant token
        significant_tokens = ['.', '!', '?', '\n', 'but', 'however', 'although', 'not']
        if any(token in token_text for token in significant_tokens):
            return True
        
        # Check if we've reached a certain number of tokens since last analysis
        tokens_since_analysis = semantic_state.get("tokens_since_analysis", 0) + 1
        if tokens_since_analysis >= self.config.get("semantic_analysis_interval", 10):
            return True
        
        return False

    def _analyze_semantics(self, text):
        """Perform semantic analysis of text"""
        # In a real implementation, this would use embeddings or an LLM
        # Simplified placeholder implementation
        
        # Extract topics (simplified)
        topics = {}
        for topic, keywords in [
            ("privacy", ["privacy", "personal", "data", "information"]),
            ("security", ["security", "protect", "encryption", "safeguard"]),
            ("consent", ["consent", "permission", "authorize", "agree"]),
            ("medical", ["medical", "health", "patient", "doctor", "treatment"]),
            ("financial", ["financial", "money", "payment", "account", "bank"])
        ]:
            count = sum(text.lower().count(keyword) for keyword in keywords)
            if count > 0:
                topics[topic] = min(1.0, count / 10.0)  # Normalize to 0-1
        
        # Create semantic state
        return {
            "topics": topics,
            "text_length": len(text),
            "tokens_since_analysis": 0,
            "last_analyzed_at": "now"  # Would be timestamp in real implementation
        }

    def _quick_update_state(self, semantic_state, token_text):
        """Quickly update semantic state without full analysis"""
        # Create a copy of the state
        updated_state = semantic_state.copy()
        
        # Update text length
        updated_state["text_length"] = updated_state.get("text_length", 0) + len(token_text)
        
        # Update tokens since analysis
        updated_state["tokens_since_analysis"] = updated_state.get("tokens_since_analysis", 0) + 1
        
        # Quick check for topic keywords
        for topic, keywords in [
            ("privacy", ["privacy", "personal", "data", "information"]),
            ("security", ["security", "protect", "encryption", "safeguard"]),
            ("consent", ["consent", "permission", "authorize", "agree"]),
            ("medical", ["medical", "health", "patient", "doctor", "treatment"]),
            ("financial", ["financial", "money", "payment", "account", "bank"])
        ]:
            if any(keyword in token_text.lower() for keyword in keywords):
                updated_state["topics"] = updated_state.get("topics", {})
                updated_state["topics"][topic] = updated_state["topics"].get(topic, 0) + 0.1
        
        return updated_state

    def _violates_semantic_constraint(self, semantic_state, constraint):
        """Check if semantic state violates a constraint"""
        constraint_type = constraint.get("subtype", "topic_threshold")
        
        if constraint_type == "topic_threshold":
            # Check if topic exceeds threshold
            topic = constraint.get("topic", "")
            threshold = constraint.get("threshold", 0.5)
            
            topic_score = semantic_state.get("topics", {}).get(topic, 0.0)
            return topic_score >= threshold
            
        elif constraint_type == "topic_combination":
            # Check if multiple topics are present together
            topics = constraint.get("topics", [])
            threshold = constraint.get("threshold", 0.3)
            
            topic_scores = [semantic_state.get("topics", {}).get(topic, 0.0) for topic in topics]
            return all(score >= threshold for score in topic_scores)
            
        elif constraint_type == "topic_change":
            # Check for sudden topic changes (would need historical state)
            return False
            
        elif constraint_type == "sentiment":
            # Check sentiment constraints (placeholder)
            return False
        
        return False

    def _severity_to_score(self, severity):
        """Convert severity to numerical score for calculations"""
        severity_scores = {
            "low": 1.0,
            "medium": 3.0,
            "high": 6.0,
            "critical": 10.0
        }
        return severity_scores.get(severity, 3.0)