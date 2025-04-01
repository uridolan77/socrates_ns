import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
import json


class ContextualRuleInterpreter:
    """
    Interprets and applies context-dependent compliance rules.
    
    This class handles complex compliance scenarios where rule application
    depends on the context in which content appears, rather than just
    the content itself.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Load contextual rules
        self.contextual_rules = config.get("contextual_rules", [])
        
        # Rule interpretation settings
        self.context_weight = config.get("context_weight", 0.7)
        self.content_weight = config.get("content_weight", 0.3)
        self.threshold = config.get("contextual_threshold", 0.65)
        
        # Configure interpretation methods
        self._initialize_interpreters()
        
        # Load entity relationships if provided
        self.entity_relationships = config.get("entity_relationships", {})
        
        # Domain-specific configurations
        self.domain_configs = config.get("domain_configs", {})
        
        # Load contextual keywords
        self.contextual_keywords = self._initialize_contextual_keywords()
        
        # Context history handling
        self.max_history_length = config.get("max_history_length", 5)
        self.history_decay_factor = config.get("history_decay_factor", 0.8)
        
        # Logging
        self.verbose_logging = config.get("verbose_logging", False)
        
    def interpret_rules(self, text, context=None):
        """
        Interpret and apply contextual rules to the provided text and context.
        
        Args:
            text: Input text to analyze
            context: Dictionary containing contextual information, which may include:
                - user_info: Information about the user
                - domain: The domain or application area
                - history: Previous interactions or context
                - entities: Named entities detected in the current text
                - metadata: Additional metadata about the request
                
        Returns:
            Dict with interpretation results
        """
        if not text or not self.contextual_rules:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
        
        # Use empty dict if context is None
        context = context or {}
        
        # Standardize and preprocess context
        context = self._preprocess_context(context)
        
        # Find applicable rules based on the context
        applicable_rules = self._find_applicable_rules(text, context)
        
        if not applicable_rules and self.verbose_logging:
            logging.debug("No applicable contextual rules found")
        
        # Evaluate each applicable rule
        issues = []
        is_compliant = True
        rule_violations = []
        
        for rule in applicable_rules:
            # Skip disabled rules
            if not rule.get("enabled", True):
                continue
                
            # Get the appropriate interpreter for this rule type
            rule_type = rule.get("type", "default")
            interpreter = self.interpreters.get(rule_type, self.interpreters["default"])
            
            # Apply the rule interpretation
            violation, confidence, details = interpreter(text, context, rule)
            
            if violation and confidence >= rule.get("confidence_threshold", self.threshold):
                is_compliant = False
                rule_violations.append((rule, confidence, details))
                
                # Create an issue for this violation
                issues.append({
                    "rule_id": rule.get("id", str(uuid.uuid4())),
                    "severity": rule.get("severity", "medium"),
                    "description": rule.get("description", "Contextual policy violation"),
                    "confidence": confidence,
                    "metadata": {
                        "rule_name": rule.get("name", ""),
                        "rule_type": rule_type,
                        "context_factors": details.get("context_factors", {}),
                        "violation_details": details.get("violation_details", {})
                    }
                })
                
                # Stop checking if configured to break on first violation
                if self.config.get("break_on_first_violation", False):
                    break
        
        # Determine if content modification is needed
        modified_text, modified = self._apply_modifications(text, rule_violations, context)
        
        # Log detailed results if verbose logging is enabled
        if self.verbose_logging and rule_violations:
            for rule, confidence, details in rule_violations:
                logging.info(f"Contextual rule violation: {rule.get('name', 'Unnamed rule')}, "
                           f"confidence: {confidence}, details: {json.dumps(details)}")
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": modified_text,
            "issues": issues,
            "modified": modified
        }
        
    def _initialize_interpreters(self):
        """Initialize the rule interpretation methods for different rule types."""
        self.interpreters = {
            "default": self._interpret_default_rule,
            "keyword_in_context": self._interpret_keyword_in_context_rule,
            "entity_relationship": self._interpret_entity_relationship_rule,
            "sequence": self._interpret_sequence_rule,
            "domain_specific": self._interpret_domain_specific_rule,
            "semantic_context": self._interpret_semantic_context_rule,
            "user_attribute": self._interpret_user_attribute_rule,
            "multi_factor": self._interpret_multi_factor_rule
        }
        
    def _initialize_contextual_keywords(self):
        """Initialize contextual keywords from configuration."""
        contextual_keywords = {}
        
        keyword_config = self.config.get("contextual_keywords", {})
        for context_type, keywords in keyword_config.items():
            # Convert to dict for faster lookup
            contextual_keywords[context_type] = {
                k["keyword"]: k.get("weight", 1.0) for k in keywords
            }
            
        return contextual_keywords
        
    def _preprocess_context(self, context):
        """Standardize and preprocess context information."""
        # Create a standardized context structure if missing components
        standard_context = {
            "user_info": context.get("user_info", {}),
            "domain": context.get("domain", "general"),
            "history": context.get("history", []),
            "entities": context.get("entities", []),
            "metadata": context.get("metadata", {})
        }
        
        # Limit history length
        if len(standard_context["history"]) > self.max_history_length:
            standard_context["history"] = standard_context["history"][-self.max_history_length:]
            
        # Extract and normalize entities if not already done
        if not standard_context["entities"] and "entity_extraction" not in context:
            # Simple entity extraction - in a real system this would be more sophisticated
            standard_context["entities"] = self._extract_entities(context)
            
        return standard_context
        
    def _extract_entities(self, context):
        """
        Simple entity extraction from context.
        In a real system, this would use a more sophisticated NLP approach.
        """
        entities = []
        
        # Extract from user info
        user_info = context.get("user_info", {})
        if user_info:
            for key, value in user_info.items():
                if isinstance(value, str) and value:
                    entities.append({
                        "type": "user_attribute",
                        "value": value,
                        "attribute": key,
                        "confidence": 1.0
                    })
        
        # Extract from domain-specific entities
        domain = context.get("domain", "general")
        if domain in self.domain_configs:
            domain_entities = self.domain_configs[domain].get("entities", [])
            entities.extend(domain_entities)
            
        return entities
        
    def _find_applicable_rules(self, text, context):
        """Find rules that are applicable to the current context."""
        applicable_rules = []
        
        for rule in self.contextual_rules:
            # Check if rule applies to this domain
            rule_domains = rule.get("domains", ["general"])
            if "all" not in rule_domains and context["domain"] not in rule_domains:
                continue
                
            # Check if rule has context prerequisites
            if not self._context_prerequisites_met(rule, context):
                continue
                
            # Check if rule has content prerequisites (quick check before full interpretation)
            if not self._content_prerequisites_met(rule, text):
                continue
                
            # This rule is applicable
            applicable_rules.append(rule)
            
        return applicable_rules
        
    def _context_prerequisites_met(self, rule, context):
        """Check if the context meets the rule's prerequisites."""
        prerequisites = rule.get("context_prerequisites", {})
        
        # Check domain prerequisite
        if "domain" in prerequisites and context["domain"] not in prerequisites["domain"]:
            return False
            
        # Check user attributes prerequisites
        user_prerequisites = prerequisites.get("user_attributes", {})
        if user_prerequisites:
            user_info = context.get("user_info", {})
            for attr, required_value in user_prerequisites.items():
                if attr not in user_info or user_info[attr] != required_value:
                    return False
        
        # Check required entities
        required_entities = prerequisites.get("required_entities", [])
        if required_entities:
            context_entities = {e.get("value", "").lower() for e in context.get("entities", [])}
            for entity in required_entities:
                if entity.lower() not in context_entities:
                    return False
        
        return True
        
    def _content_prerequisites_met(self, rule, text):
        """Quick check if the content meets basic prerequisites for this rule."""
        prerequisites = rule.get("content_prerequisites", {})
        
        # Check for required substrings (quick check)
        required_substrings = prerequisites.get("required_substrings", [])
        if required_substrings:
            text_lower = text.lower()
            for substring in required_substrings:
                if substring.lower() not in text_lower:
                    return False
        
        # Check for minimum length
        min_length = prerequisites.get("min_length", 0)
        if len(text) < min_length:
            return False
            
        return True
        
    def _interpret_default_rule(self, text, context, rule):
        """Default rule interpretation method."""
        # Basic content matching
        content_match = self._check_content_match(text, rule)
        
        # Basic context matching
        context_match = self._check_context_match(context, rule)
        
        # Combine scores with weights
        overall_score = (
            content_match * self.content_weight + 
            context_match * self.context_weight
        )
        
        # Determine if this is a violation
        threshold = rule.get("threshold", self.threshold)
        is_violation = overall_score >= threshold
        
        details = {
            "content_match_score": content_match,
            "context_match_score": context_match,
            "content_factors": self._get_content_factors(text, rule),
            "context_factors": self._get_context_factors(context, rule)
        }
        
        return is_violation, overall_score, details
        
    def _interpret_keyword_in_context_rule(self, text, context, rule):
        """Interpret rules about keywords that are only problematic in certain contexts."""
        keywords = rule.get("keywords", [])
        contexts = rule.get("contexts", [])
        
        if not keywords or not contexts:
            return False, 0.0, {}
            
        # Check for keyword matches
        text_lower = text.lower()
        keyword_matches = []
        
        for keyword_obj in keywords:
            keyword = keyword_obj.get("keyword", "").lower()
            weight = keyword_obj.get("weight", 1.0)
            
            if keyword in text_lower:
                positions = [m.start() for m in re.finditer(re.escape(keyword), text_lower)]
                keyword_matches.append({
                    "keyword": keyword,
                    "weight": weight,
                    "positions": positions
                })
                
        if not keyword_matches:
            return False, 0.0, {}
            
        # Check for context matches
        context_match_score = 0.0
        context_matches = []
        
        for context_obj in contexts:
            context_type = context_obj.get("type", "")
            context_value = context_obj.get("value", "")
            weight = context_obj.get("weight", 1.0)
            
            # Check if this context is present
            if self._check_specific_context(context_type, context_value, context):
                context_match_score += weight
                context_matches.append({
                    "type": context_type,
                    "value": context_value,
                    "weight": weight
                })
                
        # Normalize context match score
        if contexts:
            context_match_score /= sum(c.get("weight", 1.0) for c in contexts)
            
        # Calculate keyword match score
        keyword_match_score = sum(k["weight"] for k in keyword_matches) / sum(k.get("weight", 1.0) for k in keywords)
        
        # Combine scores
        combined_score = (
            keyword_match_score * self.content_weight +
            context_match_score * self.context_weight
        )
        
        details = {
            "keyword_matches": keyword_matches,
            "context_matches": context_matches,
            "keyword_match_score": keyword_match_score,
            "context_match_score": context_match_score
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = combined_score >= threshold
        
        return is_violation, combined_score, details
        
    def _interpret_entity_relationship_rule(self, text, context, rule):
        """Interpret rules about relationships between entities."""
        # Get relationships defined in the rule
        prohibited_relationships = rule.get("prohibited_relationships", [])
        entities = context.get("entities", [])
        
        if not prohibited_relationships or not entities:
            return False, 0.0, {}
            
        # Extract entity values for easier reference
        entity_values = {e.get("value", "").lower() for e in entities}
        
        # Check for prohibited entity relationships in the content
        relationship_violations = []
        
        for relationship in prohibited_relationships:
            entity1 = relationship.get("entity1", "").lower()
            entity2 = relationship.get("entity2", "").lower()
            rel_type = relationship.get("type", "co-occurrence")
            weight = relationship.get("weight", 1.0)
            
            # Check if both entities are present
            if entity1 in entity_values and entity2 in entity_values:
                # For co-occurrence, we've already confirmed both entities are present
                if rel_type == "co-occurrence":
                    relationship_violations.append({
                        "entity1": entity1,
                        "entity2": entity2,
                        "type": rel_type,
                        "weight": weight
                    })
                    
                # For proximity, check if entities are within a certain distance
                elif rel_type == "proximity":
                    max_distance = relationship.get("max_distance", 50)  # characters
                    if self._check_entity_proximity(text, entity1, entity2, max_distance):
                        relationship_violations.append({
                            "entity1": entity1,
                            "entity2": entity2,
                            "type": rel_type,
                            "weight": weight,
                            "max_distance": max_distance
                        })
                        
                # For specific relationships like "actor-action", need more sophisticated parsing
                elif rel_type in ["actor-action", "subject-object"]:
                    # In a real system, this would use dependency parsing or similar NLP techniques
                    # Here we'll use a simple heuristic
                    if self._check_entity_sequence(text, entity1, entity2, max_distance=30):
                        relationship_violations.append({
                            "entity1": entity1,
                            "entity2": entity2,
                            "type": rel_type,
                            "weight": weight
                        })
        
        # Calculate violation score
        if not prohibited_relationships:
            score = 0.0
        else:
            score = sum(v["weight"] for v in relationship_violations) / sum(r.get("weight", 1.0) for r in prohibited_relationships)
            
        details = {
            "relationship_violations": relationship_violations,
            "entities_detected": [e.get("value") for e in entities]
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = score >= threshold
        
        return is_violation, score, details
        
    def _interpret_sequence_rule(self, text, context, rule):
        """Interpret rules about the sequence of topics or entities."""
        prohibited_sequences = rule.get("prohibited_sequences", [])
        
        if not prohibited_sequences:
            return False, 0.0, {}
            
        # Check for each prohibited sequence
        sequence_violations = []
        
        for sequence in prohibited_sequences:
            elements = sequence.get("elements", [])
            window_size = sequence.get("window_size", len(text))  # Default to full text
            weight = sequence.get("weight", 1.0)
            
            if self._check_sequence_in_text(text, elements, window_size):
                sequence_violations.append({
                    "sequence": elements,
                    "window_size": window_size,
                    "weight": weight
                })
                
        # Calculate violation score
        if not prohibited_sequences:
            score = 0.0
        else:
            score = sum(v["weight"] for v in sequence_violations) / sum(s.get("weight", 1.0) for s in prohibited_sequences)
            
        details = {
            "sequence_violations": sequence_violations
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = score >= threshold
        
        return is_violation, score, details
        
    def _interpret_domain_specific_rule(self, text, context, rule):
        """Interpret rules that are specific to certain domains."""
        domain = context.get("domain", "general")
        
        # Check if this rule applies to this domain
        applicable_domains = rule.get("applicable_domains", ["general"])
        if domain not in applicable_domains and "all" not in applicable_domains:
            return False, 0.0, {}
            
        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, {})
        
        # Get domain-specific terms
        domain_terms = domain_config.get("terms", [])
        sensitive_combinations = domain_config.get("sensitive_combinations", [])
        
        # Check for domain-specific term matches
        term_matches = []
        text_lower = text.lower()
        
        for term_obj in domain_terms:
            term = term_obj.get("term", "").lower()
            sensitivity = term_obj.get("sensitivity", 0.5)
            category = term_obj.get("category", "general")
            
            if term in text_lower:
                positions = [m.start() for m in re.finditer(re.escape(term), text_lower)]
                term_matches.append({
                    "term": term,
                    "sensitivity": sensitivity,
                    "category": category,
                    "positions": positions
                })
                
        # Check for sensitive combinations
        combination_matches = []
        
        for combo in sensitive_combinations:
            terms = combo.get("terms", [])
            sensitivity = combo.get("sensitivity", 0.8)
            
            if all(term.lower() in text_lower for term in terms):
                combination_matches.append({
                    "terms": terms,
                    "sensitivity": sensitivity
                })
                
        # Calculate domain-specific score
        # Weight term matches by their sensitivity
        term_score = sum(m["sensitivity"] for m in term_matches) / max(1, len(domain_terms))
        
        # Weight combination matches
        combo_score = sum(m["sensitivity"] for m in combination_matches) / max(1, len(sensitive_combinations))
        
        # Combine scores, with combinations weighted more heavily
        domain_score = term_score * 0.4 + combo_score * 0.6
        
        details = {
            "domain": domain,
            "term_matches": term_matches,
            "combination_matches": combination_matches,
            "term_score": term_score,
            "combination_score": combo_score
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = domain_score >= threshold
        
        return is_violation, domain_score, details
        
    def _interpret_semantic_context_rule(self, text, context, rule):
        """
        Interpret rules based on semantic context.
        This is a placeholder for more advanced NLP-based interpretation.
        """
        # In a real implementation, this would use a language model or semantic analysis
        # For now, we'll use a simplified approach based on keywords and contexts
        
        semantic_contexts = rule.get("semantic_contexts", [])
        
        if not semantic_contexts:
            return False, 0.0, {}
            
        # Check each semantic context
        context_matches = []
        
        for semantic_context in semantic_contexts:
            indicator_keywords = semantic_context.get("indicator_keywords", [])
            context_type = semantic_context.get("type", "general")
            sensitivity = semantic_context.get("sensitivity", 0.7)
            
            # Count how many indicator keywords are present
            text_lower = text.lower()
            matching_keywords = [kw for kw in indicator_keywords if kw.lower() in text_lower]
            
            # Calculate match score for this semantic context
            if indicator_keywords:
                match_score = len(matching_keywords) / len(indicator_keywords) * sensitivity
            else:
                match_score = 0.0
                
            if match_score > 0:
                context_matches.append({
                    "type": context_type,
                    "matching_keywords": matching_keywords,
                    "match_score": match_score,
                    "sensitivity": sensitivity
                })
                
        # Calculate overall semantic context score
        if not semantic_contexts:
            semantic_score = 0.0
        else:
            # Take the maximum score from any semantic context
            semantic_score = max([m["match_score"] for m in context_matches]) if context_matches else 0.0
            
        details = {
            "context_matches": context_matches,
            "semantic_score": semantic_score
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = semantic_score >= threshold
        
        return is_violation, semantic_score, details
        
    def _interpret_user_attribute_rule(self, text, context, rule):
        """Interpret rules based on user attributes combined with content."""
        user_info = context.get("user_info", {})
        
        # Get rule components
        attribute_conditions = rule.get("attribute_conditions", [])
        content_conditions = rule.get("content_conditions", [])
        
        if not attribute_conditions or not content_conditions:
            return False, 0.0, {}
            
        # Check user attribute conditions
        attribute_matches = []
        
        for condition in attribute_conditions:
            attribute = condition.get("attribute", "")
            values = condition.get("values", [])
            match_type = condition.get("match_type", "exact")
            weight = condition.get("weight", 1.0)
            
            if attribute in user_info:
                user_value = user_info[attribute]
                
                # Check for match based on match_type
                is_match = False
                
                if match_type == "exact":
                    is_match = user_value in values
                elif match_type == "contains":
                    is_match = any(v in user_value for v in values) if isinstance(user_value, str) else False
                elif match_type == "regex":
                    is_match = any(bool(re.search(v, user_value)) for v in values) if isinstance(user_value, str) else False
                    
                if is_match:
                    attribute_matches.append({
                        "attribute": attribute,
                        "value": user_value,
                        "match_type": match_type,
                        "weight": weight
                    })
                    
        # Check content conditions
        content_matches = []
        text_lower = text.lower()
        
        for condition in content_conditions:
            content_type = condition.get("type", "keyword")
            keywords = condition.get("keywords", [])
            weight = condition.get("weight", 1.0)
            
            matching_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matching_keywords.append(keyword)
                    
            if matching_keywords:
                content_matches.append({
                    "content_type": content_type,
                    "matching_keywords": matching_keywords,
                    "weight": weight
                })
                
        # Calculate scores
        if not attribute_conditions:
            attribute_score = 0.0
        else:
            attribute_score = sum(m["weight"] for m in attribute_matches) / sum(c.get("weight", 1.0) for c in attribute_conditions)
            
        if not content_conditions:
            content_score = 0.0
        else:
            content_score = sum(m["weight"] for m in content_matches) / sum(c.get("weight", 1.0) for c in content_conditions)
            
        # Calculate combined score
        combined_score = (
            attribute_score * self.context_weight +
            content_score * self.content_weight
        )
        
        details = {
            "attribute_matches": attribute_matches,
            "content_matches": content_matches,
            "attribute_score": attribute_score,
            "content_score": content_score
        }
        
        threshold = rule.get("threshold", self.threshold)
        is_violation = combined_score >= threshold
        
        return is_violation, combined_score, details
        
    def _interpret_multi_factor_rule(self, text, context, rule):
        """Interpret rules that combine multiple factors with custom logic."""
        factors = rule.get("factors", [])
        logic = rule.get("logic", "all")  # 'all', 'any', 'majority', 'weighted'
        
        if not factors:
            return False, 0.0, {}
            
        # Evaluate each factor
        factor_results = []
        
        for factor in factors:
            factor_type = factor.get("type", "default")
            weight = factor.get("weight", 1.0)
            
            # Create a sub-rule for this factor
            sub_rule = factor.get("rule", {})
            
            # Get the appropriate interpreter for this factor type
            interpreter = self.interpreters.get(factor_type, self.interpreters["default"])
            
            # Interpret this factor
            is_violation, score, details = interpreter(text, context, sub_rule)
            
            factor_results.append({
                "type": factor_type,
                "weight": weight,
                "is_violation": is_violation,
                "score": score,
                "details": details
            })
            
        # Apply logic to determine overall result
        if logic == "all":
            # All factors must indicate a violation
            is_violation = all(f["is_violation"] for f in factor_results)
            score = min(f["score"] for f in factor_results) if factor_results else 0.0
            
        elif logic == "any":
            # Any factor indicating a violation is sufficient
            is_violation = any(f["is_violation"] for f in factor_results)
            score = max(f["score"] for f in factor_results) if factor_results else 0.0
            
        elif logic == "majority":
            # Majority of factors must indicate a violation
            violations = sum(1 for f in factor_results if f["is_violation"])
            is_violation = violations > len(factor_results) / 2
            score = sum(f["score"] for f in factor_results) / len(factor_results) if factor_results else 0.0
            
        elif logic == "weighted":
            # Weighted average of factor scores
            total_weight = sum(f["weight"] for f in factor_results)
            
            if total_weight > 0:
                score = sum(f["score"] * f["weight"] for f in factor_results) / total_weight
            else:
                score = 0.0
                
            threshold = rule.get("threshold", self.threshold)
            is_violation = score >= threshold
            
        else:
            # Default to "any" logic
            is_violation = any(f["is_violation"] for f in factor_results)
            score = max(f["score"] for f in factor_results) if factor_results else 0.0
            
        details = {
            "logic": logic,
            "factor_results": factor_results,
            "combined_score": score
        }
        
        return is_violation, score, details
        
    def _check_content_match(self, text, rule):
        """Check basic content matching for a rule."""
        content_patterns = rule.get("content_patterns", [])
        
        if not content_patterns:
            return 0.0
            
        # Count matches for each pattern
        match_counts = []
        text_lower = text.lower()
        
        for pattern in content_patterns:
            pattern_type = pattern.get("type", "substring")
            pattern_value = pattern.get("value", "")
            weight = pattern.get("weight", 1.0)
            
            count = 0
            
            if pattern_type == "substring":
                count = text_lower.count(pattern_value.lower())
            elif pattern_type == "regex":
                try:
                    count = len(re.findall(pattern_value, text, re.IGNORECASE))
                except re.error:
                    logging.error(f"Invalid regex pattern: {pattern_value}")
                    count = 0
                    
            match_counts.append({
                "pattern": pattern_value,
                "type": pattern_type,
                "count": count,
                "weight": weight
            })
            
        # Calculate weighted score
        total_weight = sum(m["weight"] for m in match_counts)
        
        if total_weight > 0:
            weighted_score = sum(m["count"] * m["weight"] for m in match_counts) / total_weight
            # Normalize to 0-1 range
            return min(1.0, weighted_score / 5.0)  # Assuming 5+ matches is full score
        else:
            return 0.0
            
    def _check_context_match(self, context, rule):
        """Check basic context matching for a rule."""
        context_criteria = rule.get("context_criteria", [])
        
        if not context_criteria:
            return 0.0
            
        # Check each context criterion
        match_scores = []
        
        for criterion in context_criteria:
            criterion_type = criterion.get("type", "")
            criterion_value = criterion.get("value", "")
            weight = criterion.get("weight", 1.0)
            
            # Check if this criterion matches the context
            match_score = 0.0
            
            if self._check_specific_context(criterion_type, criterion_value, context):
                match_score = 1.0
                
            match_scores.append({
                "type": criterion_type,
                "value": criterion_value,
                "match_score": match_score,
                "weight": weight
            })
            
        # Calculate weighted score
        total_weight = sum(m["weight"] for m in match_scores)
        
        if total_weight > 0:
            return sum(m["match_score"] * m["weight"] for m in match_scores) / total_weight
        else:
            return 0.0
            
    def _check_specific_context(self, context_type, context_value, context):
        """Check if a specific context criterion matches."""
        if context_type == "domain":
            return context.get("domain", "") == context_value
            
        elif context_type == "user_attribute":
            attribute, value = context_value.split(":", 1) if ":" in context_value else (context_value, "")
            user_info = context.get("user_info", {})
            
            if value:
                return user_info.get(attribute) == value
            else:
                return attribute in user_info
                
        elif context_type == "history":
            # Check if a specific pattern appears in history
            history = context.get("history", [])
            return any(context_value.lower() in item.lower() for item in history)
            
        elif context_type == "entity":
            # Check if a specific entity is present
            entities = context.get("entities", [])
            return any(e.get("value", "").lower() == context_value.lower() for e in entities)
            
        elif context_type == "metadata":
            key, value = context_value.split(":", 1) if ":" in context_value else (context_value, "")
            metadata = context.get("metadata", {})
            
            if value:
                return metadata.get(key) == value
            else:
                return key in metadata
                
        return False
        
    def _get_content_factors(self, text, rule):
        """Get the content factors that contributed to a rule match."""
        content_patterns = rule.get("content_patterns", [])
        factors = {}
        
        for pattern in content_patterns:
            pattern_type = pattern.get("type", "substring")
            pattern_value = pattern.get("value", "")
            
            if pattern_type == "substring":
                count = text.lower().count(pattern_value.lower())
                if count > 0:
                    factors[pattern_value] = count
            elif pattern_type == "regex":
                try:
                    matches = re.findall(pattern_value, text, re.IGNORECASE)
                    if matches:
                        factors[pattern_value] = len(matches)
                except re.error:
                    pass
                    
        return factors
        
    def _get_context_factors(self, context, rule):
        """Get the context factors that contributed to a rule match."""
        context_criteria = rule.get("context_criteria", [])
        factors = {}
        
        for criterion in context_criteria:
            criterion_type = criterion.get("type", "")
            criterion_value = criterion.get("value", "")
            
            if self._check_specific_context(criterion_type, criterion_value, context):
                factors[f"{criterion_type}:{criterion_value}"] = True
                
        return factors
        
    def _check_entity_proximity(self, text, entity1, entity2, max_distance):
        """Check if two entities are within a certain distance in the text."""
        text_lower = text.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # Find all occurrences of both entities
        entity1_positions = [m.start() for m in re.finditer(re.escape(entity1_lower), text_lower)]
        entity2_positions = [m.start() for m in re.finditer(re.escape(entity2_lower), text_lower)]
        
        if not entity1_positions or not entity2_positions:
            return False
            
        # Check if any pair of occurrences is within max_distance
        for pos1 in entity1_positions:
            for pos2 in entity2_positions:
                if abs(pos1 - pos2) <= max_distance:
                    return True
                    
        return False
        
    def _check_entity_sequence(self, text, entity1, entity2, max_distance=50):
        """Check if entity1 appears before entity2 within a certain distance."""
        text_lower = text.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # Find all occurrences of both entities
        entity1_positions = [m.start() for m in re.finditer(re.escape(entity1_lower), text_lower)]
        entity2_positions = [m.start() for m in re.finditer(re.escape(entity2_lower), text_lower)]
        
        if not entity1_positions or not entity2_positions:
            return False
            
        # Check if entity1 appears before entity2 within max_distance
        for pos1 in entity1_positions:
            for pos2 in entity2_positions:
                if 0 < pos2 - pos1 <= max_distance:
                    return True
                    
        return False
        
    def _check_sequence_in_text(self, text, elements, window_size):
        """Check if a sequence of elements appears in the text within a window."""
        if not elements:
            return False
            
        text_lower = text.lower()
        
        # Find positions of all elements
        element_positions = []
        
        for element in elements:
            positions = [m.start() for m in re.finditer(re.escape(element.lower()), text_lower)]
            if not positions:
                return False  # One element not found, sequence impossible
            element_positions.append(positions)
            
        # Check if elements appear in sequence within window_size
        def check_sequence_recursive(current_index, last_position, current_window):
            if current_index >= len(elements):
                return True  # All elements found in sequence
                
            for position in element_positions[current_index]:
                if last_position is None or (position > last_position and position - last_position <= current_window):
                    if check_sequence_recursive(current_index + 1, position, current_window - (position - last_position if last_position else 0)):
                        return True
                        
            return False
            
        return check_sequence_recursive(0, None, window_size)
        
    def _apply_modifications(self, text, rule_violations, context):
        """Apply modifications to the text based on rule violations."""
        if not rule_violations or not self.config.get("apply_modifications", False):
            return text, False
            
        modified_text = text
        modified = False
        
        for rule, _, details in rule_violations:
            # Skip rules that don't specify modifications
            if "modifications" not in rule:
                continue
                
            modifications = rule.get("modifications", [])
            
            for mod in modifications:
                mod_type = mod.get("type", "")
                
                if mod_type == "redact":
                    # Redact specific content
                    targets = mod.get("targets", [])
                    redaction_text = mod.get("replacement", "[REDACTED]")
                    
                    for target in targets:
                        if target in modified_text:
                            modified_text = modified_text.replace(target, redaction_text)
                            modified = True
                            
                elif mod_type == "replace_regex":
                    # Replace content matching regex
                    pattern = mod.get("pattern", "")
                    replacement = mod.get("replacement", "[MODIFIED]")
                    
                    try:
                        new_text = re.sub(pattern, replacement, modified_text)
                        if new_text != modified_text:
                            modified_text = new_text
                            modified = True
                    except re.error:
                        logging.error(f"Invalid regex replacement pattern: {pattern}")
                        
                elif mod_type == "add_disclaimer":
                    # Add a disclaimer to the content
                    disclaimer = mod.get("text", "")
                    position = mod.get("position", "beginning")  # beginning, end
                    
                    if disclaimer:
                        if position == "beginning":
                            modified_text = disclaimer + "\n\n" + modified_text
                        else:
                            modified_text = modified_text + "\n\n" + disclaimer
                        modified = True
        
        return modified_text, modified