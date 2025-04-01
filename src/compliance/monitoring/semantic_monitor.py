import numpy as np
from src.compliance.filtering import TextSemanticAnalyzer, TopicModel, EntityTracker, RelationTracker

class SemanticComplianceMonitor:
    """
    Monitors semantic compliance during text generation by maintaining and
    updating a semantic state representation of the generated content.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.semantic_analyzer = TextSemanticAnalyzer(compliance_config)
        self.topic_model = TopicModel(compliance_config.get("topic_model_config", {}))
        self.entity_tracker = EntityTracker(compliance_config)
        self.relation_tracker = RelationTracker(compliance_config)
        
        # Configure state representation dimensions
        self.state_dim = compliance_config.get("semantic_state_dim", 64)
        self.sliding_window_size = compliance_config.get("sliding_window", 100)
        
        # Initialize concept detection thresholds
        self.concept_thresholds = compliance_config.get("concept_thresholds", {})
        
    def initialize(self, prompt, applicable_frameworks):
        """
        Initialize semantic state based on prompt and applicable frameworks.
        
        Args:
            prompt: Input prompt text
            applicable_frameworks: List of applicable regulatory frameworks
            
        Returns:
            Initial semantic state
        """
        # Extract semantic representations from prompt
        prompt_semantic = self.semantic_analyzer.analyze(prompt)
        
        # Extract topics from prompt
        prompt_topics = self.topic_model.extract_topics(prompt)
        
        # Extract entities from prompt
        entities = self.entity_tracker.extract_entities(prompt)
        
        # Get sensitive concepts from applicable frameworks
        sensitive_concepts = self._get_framework_sensitive_concepts(applicable_frameworks)
        
        # Create initial state
        state = {
            "semantic_embedding": prompt_semantic["embedding"],
            "topics": {topic: score for topic, score in prompt_topics.items() if score > 0.1},
            "entities": entities,
            "relations": {},
            "tracked_concepts": self._initialize_concept_tracking(prompt, sensitive_concepts),
            "sensitive_concept_scores": self._score_sensitive_concepts(prompt_semantic, sensitive_concepts),
            "text_buffer": prompt[-self.sliding_window_size:] if len(prompt) > self.sliding_window_size else prompt,
            "frameworks": [f.id for f in applicable_frameworks],
            "warnings": [],
            "cumulative_risk_score": self._calculate_initial_risk(prompt_semantic, applicable_frameworks)
        }
        
        return state
    
    def update(self, state, token_text, generated_text, applicable_frameworks):
        """
        Update semantic state with new token.
        
        Args:
            state: Current semantic state
            token_text: Text of new token
            generated_text: Complete generated text including new token
            applicable_frameworks: List of applicable regulatory frameworks
            
        Returns:
            Updated semantic state
        """
        # Create a copy of the state to avoid modifying the original
        updated_state = state.copy()
        
        # Update text buffer (sliding window)
        updated_state["text_buffer"] += token_text
        if len(updated_state["text_buffer"]) > self.sliding_window_size:
            updated_state["text_buffer"] = updated_state["text_buffer"][-self.sliding_window_size:]
        
        # Check if we need full semantic update (performance optimization)
        # Only do full updates periodically or on significant tokens
        if self._should_perform_full_update(token_text, updated_state):
            # Perform full semantic analysis on current buffer
            buffer_semantics = self.semantic_analyzer.analyze(updated_state["text_buffer"])
            
            # Update semantic embedding using weighted combination
            alpha = 0.8  # Weight for new information
            updated_state["semantic_embedding"] = self._weighted_combine(
                updated_state["semantic_embedding"],
                buffer_semantics["embedding"],
                alpha
            )
            
            # Update topics
            buffer_topics = self.topic_model.extract_topics(updated_state["text_buffer"])
            updated_state["topics"] = self._update_topics(updated_state["topics"], buffer_topics)
            
            # Update entities and relations
            new_entities = self.entity_tracker.extract_entities(updated_state["text_buffer"])
            updated_state["entities"] = self._merge_entities(updated_state["entities"], new_entities)
            updated_state["relations"] = self.relation_tracker.extract_relations(
                updated_state["text_buffer"], updated_state["entities"]
            )
            
            # Update sensitive concept scores
            sensitive_concepts = self._get_framework_sensitive_concepts(applicable_frameworks)
            updated_state["sensitive_concept_scores"] = self._score_sensitive_concepts(
                buffer_semantics, sensitive_concepts
            )
            
            # Re-calculate cumulative risk
            updated_state["cumulative_risk_score"] = self._calculate_updated_risk(
                updated_state, applicable_frameworks
            )
        else:
            # Lightweight update for efficiency
            # Just update token-level features without full semantic analysis
            if token_text.strip():  # Non-whitespace token
                # Approximation update based on token
                updated_state["cumulative_risk_score"] += self._estimate_token_risk(
                    token_text, updated_state, applicable_frameworks
                )
        
        # Add warnings if risk thresholds crossed
        if updated_state["cumulative_risk_score"] > 0.7:
            if not any(w["type"] == "high_risk" for w in updated_state["warnings"]):
                updated_state["warnings"].append({
                    "type": "high_risk",
                    "message": "Content is approaching regulatory compliance boundaries",
                    "score": updated_state["cumulative_risk_score"],
                    "position": len(generated_text)
                })
        
        return updated_state
    
    def _get_framework_sensitive_concepts(self, frameworks):
        """Extract sensitive concepts from regulatory frameworks"""
        sensitive_concepts = {}
        
        for framework in frameworks:
            # Extract framework-specific concepts
            framework_concepts = getattr(framework, "sensitive_concepts", {})
            
            # Add to overall concept map, with framework attribution
            for concept, data in framework_concepts.items():
                if concept not in sensitive_concepts:
                    sensitive_concepts[concept] = {
                        "frameworks": [framework.id],
                        "severity": data.get("severity", "medium"),
                        "threshold": data.get("threshold", self.concept_thresholds.get(concept, 0.7))
                    }
                else:
                    # Concept exists from multiple frameworks, use most severe
                    current_severity = sensitive_concepts[concept]["severity"]
                    new_severity = data.get("severity", "medium")
                    
                    # Add framework to tracking
                    sensitive_concepts[concept]["frameworks"].append(framework.id)
                    
                    # Update severity if new one is higher
                    severity_rank = {"low": 1, "medium": 2, "high": 3}
                    if severity_rank.get(new_severity, 0) > severity_rank.get(current_severity, 0):
                        sensitive_concepts[concept]["severity"] = new_severity
                    
                    # Use lower threshold (more conservative)
                    current_threshold = sensitive_concepts[concept]["threshold"]
                    new_threshold = data.get("threshold", self.concept_thresholds.get(concept, 0.7))
                    sensitive_concepts[concept]["threshold"] = min(current_threshold, new_threshold)
        
        return sensitive_concepts
    
    def _initialize_concept_tracking(self, text, sensitive_concepts):
        """Initialize tracking for sensitive concepts"""
        tracking = {}
        
        # Initialize each concept with zero count and evidence list
        for concept in sensitive_concepts:
            tracking[concept] = {
                "occurrences": 0,
                "evidence": [],
                "threshold": sensitive_concepts[concept]["threshold"],
                "severity": sensitive_concepts[concept]["severity"]
            }
            
        # Check initial text for concept occurrences
        for concept, data in tracking.items():
            # Simple term matching (in a real system, this would use more sophisticated concept detection)
            concept_terms = concept.lower().split()
            text_lower = text.lower()
            
            for term in concept_terms:
                if term in text_lower:
                    data["occurrences"] += 1
                    # Find position of match (simplified)
                    pos = text_lower.find(term)
                    data["evidence"].append({
                        "term": term,
                        "position": pos,
                        "context": text[max(0, pos-10):min(len(text), pos+len(term)+10)]
                    })
        
        return tracking
    
    def _score_sensitive_concepts(self, semantics, sensitive_concepts):
        """Score content against sensitive concepts"""
        # In a real implementation, this would use embeddings or classifiers
        # Simplified implementation using topic scores
        scores = {}
        
        for concept in sensitive_concepts:
            # Use available topic scores if concept matches a topic
            if concept in semantics.get("topics", {}):
                scores[concept] = semantics["topics"][concept]
            else:
                # Approximate score based on other features
                # This is a placeholder implementation
                scores[concept] = 0.1
                
        return scores
    
    def _weighted_combine(self, old_embedding, new_embedding, alpha):
        """Combine embeddings with weighted average"""
        # Ensure embeddings are numpy arrays
        if not isinstance(old_embedding, np.ndarray):
            old_embedding = np.array(old_embedding)
        if not isinstance(new_embedding, np.ndarray):
            new_embedding = np.array(new_embedding)
            
        # Calculate weighted combination
        combined = (1 - alpha) * old_embedding + alpha * new_embedding
        
        # Normalize
        return combined / np.linalg.norm(combined)
    
    def _update_topics(self, old_topics, new_topics):
        """Update topic scores with new information"""
        updated_topics = old_topics.copy()
        
        # Update existing topics and add new ones
        for topic, score in new_topics.items():
            if score > 0.1:  # Filter low-confidence topics
                if topic in updated_topics:
                    # Exponential moving average for smooth updates
                    updated_topics[topic] = 0.7 * updated_topics[topic] + 0.3 * score
                else:
                    updated_topics[topic] = score
        
        # Remove topics with very low scores
        updated_topics = {k: v for k, v in updated_topics.items() if v > 0.05}
        
        return updated_topics
    
    def _merge_entities(self, old_entities, new_entities):
        """Merge entity lists with deduplication"""
        merged = old_entities.copy()
        
        # Add new entities, updating existing ones
        for entity in new_entities:
            entity_id = entity.get("id", entity.get("name", ""))
            existing = next((e for e in merged if e.get("id", e.get("name", "")) == entity_id), None)
            
            if existing:
                # Update existing entity with new information
                existing.update(entity)
            else:
                # Add new entity
                merged.append(entity)
                
        return merged
    
    def _calculate_initial_risk(self, semantics, frameworks):
        """Calculate initial risk score based on prompt semantics"""
        # Simplified risk scoring implementation
        initial_risk = 0.1  # Base risk
        
        # Add risk based on sensitive topics
        sensitive_topics = {"violence": 0.3, "hate": 0.4, "illegal": 0.5}
        for topic, risk in sensitive_topics.items():
            if topic in semantics.get("topics", {}):
                topic_score = semantics["topics"][topic]
                initial_risk += topic_score * risk
                
        # Add risk based on framework-specific factors
        for framework in frameworks:
            if hasattr(framework, "calculate_risk"):
                framework_risk = framework.calculate_risk(semantics)
                initial_risk = max(initial_risk, framework_risk)
                
        return min(initial_risk, 1.0)  # Cap at 1.0
    
    def _calculate_updated_risk(self, state, frameworks):
        """Calculate updated risk based on current state"""
        # Start with previous risk and adjust based on new information
        current_risk = state["cumulative_risk_score"]
        
        # Factor in sensitive concept scores
        for concept, score in state["sensitive_concept_scores"].items():
            threshold = state["tracked_concepts"][concept]["threshold"]
            severity_factor = {"low": 0.7, "medium": 1.0, "high": 1.3}
            concept_severity = state["tracked_concepts"][concept]["severity"]
            
            # Calculate risk contribution from this concept
            if score > threshold:
                risk_contribution = (score - threshold) * severity_factor.get(concept_severity, 1.0)
                current_risk += risk_contribution * 0.1  # Scale factor
                
        # Apply decay factor to previous risk (risk diminishes over time if no new risky content)
        decay_factor = 0.95
        risk_from_previous = current_risk * decay_factor
        
        # Calculate new risk factors
        risk_from_concepts = sum(
            score for concept, score in state["sensitive_concept_scores"].items()
            if score > state["tracked_concepts"][concept]["threshold"]
        ) * 0.2  # Scale factor
        
        # Combine risks
        updated_risk = 0.7 * risk_from_previous + 0.3 * risk_from_concepts
        
        # Cap at reasonable range
        return max(0.0, min(updated_risk, 1.0))
    
    def _estimate_token_risk(self, token, state, frameworks):
        """Estimate risk contribution from a single token"""
        # Quick risk assessment for single token, without full semantic analysis
        # Simplified implementation
        
        # Check if token contains high-risk terms
        high_risk_terms = {"illegal", "banned", "dangerous", "exploit", "hack"}
        if any(term in token.lower() for term in high_risk_terms):
            return 0.05  # Significant risk increase
            
        # Nominal risk adjustment for normal tokens
        return 0.001  # Small risk increase
    
    def _should_perform_full_update(self, token_text, state):
        """Determine if a full semantic update should be performed"""
        # Periodic updates (every N tokens)
        token_count = len(state["text_buffer"].split())
        if token_count % 10 == 0:  # Every 10 tokens
            return True
            
        # Update on sentence boundaries
        if any(p in token_text for p in ['.', '!', '?', '\n']):
            return True
            
        # Update when significant tokens are encountered
        significant_tokens = {"but", "however", "although", "not", "never", "no"}
        if any(t in token_text.lower() for t in significant_tokens):
            return True
            
        # Default: lightweight updates
        return False