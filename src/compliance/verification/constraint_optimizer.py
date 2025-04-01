import logging
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass

class ComplianceConstraintOptimizer:
    """
    Optimizes compliance constraints to balance compliance requirements and user experience.
    
    This class analyzes compliance filtering results and identifies ways to optimize
    the filtering process to reduce false positives, minimize user friction, and
    maximize content throughput while maintaining compliance requirements.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Configure optimizer settings
        self.optimization_level = config.get("optimization_level", "balanced")
        self.max_constraint_adjustments = config.get("max_constraint_adjustments", 3)
        self.adjustment_factor = config.get("adjustment_factor", 0.1)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.5)
        
        # Load configurable constraints
        self.configurable_constraints = config.get("configurable_constraints", {})
        
        # Initialize constraint adjustment history
        self.adjustment_history = []
        
        # Initialize allowed modifications
        self.allowed_modifications = config.get("allowed_modifications", {
            "relaxation": True,
            "tightening": True,
            "contextualization": True,
            "rule_specific": True
        })
        
        # Load exemption patterns
        self.exemption_patterns = config.get("exemption_patterns", [])
        
        # Load category weights for balancing
        self.category_weights = config.get("category_weights", {})
        
        # Configure optimization strategies
        self.optimization_strategies = self._initialize_optimization_strategies()
        
        # Load NLP tools for alternative generation
        self._initialize_nlp_tools()
        
        # Initialize domain-specific handlers
        self.domain_handlers = self._initialize_domain_handlers()
        
        # Load false positive indicators and rules
        self.fp_indicators = config.get("false_positive_indicators", {})
        self.fp_rules = config.get("false_positive_rules", [])
        
        # Initialize semantic similarity calculator for alternative validation
        self.similarity_calculator = self._initialize_similarity_calculator()
    
    def _initialize_nlp_tools(self):
        """Initialize NLP tools for alternative generation and analysis"""
        self.nlp_tools = {
            "initialized": True,
            "embedding_model": None,  # Would be initialized with actual model
            "tokenizer": None,        # Would be initialized with actual tokenizer
            "semantic_models": {},    # Domain-specific semantic models
            "paraphraser": None,      # Would be initialized with paraphrasing model
            "sensitivity_classifier": None  # Would be initialized with sensitivity classifier
        }
        
        # In a real implementation, these would be actual models or APIs
        logging.info("NLP tools initialized for compliance constraint optimization")
    
    def _initialize_domain_handlers(self):
        """Initialize domain-specific handlers for rules"""
        return {
            "pii": self._handle_pii_rules,
            "keyword": self._handle_keyword_rules,
            "sequence": self._handle_sequence_rules,
            "entity_relationship": self._handle_entity_relationship_rules,
            "semantic": self._handle_semantic_rules,
            "domain_specific": self._handle_domain_specific_rules
        }
    
    def _initialize_similarity_calculator(self):
        """Initialize semantic similarity calculator"""
        # In a real implementation, this would use vector embeddings
        # Placeholder implementation
        class SimilarityCalculator:
            def calculate(self, text1, text2):
                """Calculate semantic similarity between texts"""
                # Placeholder implementation
                common_words = len(set(text1.lower().split()) & set(text2.lower().split()))
                total_words = len(set(text1.lower().split()) | set(text2.lower().split()))
                if total_words == 0:
                    return 0.0
                return common_words / total_words
        
        return SimilarityCalculator()
        
    def optimize_constraints(self, filter_results, original_input, context=None):
        """
        Optimize compliance constraints based on filtering results.
        
        Args:
            filter_results: Results from compliance filtering
            original_input: Original input content before filtering
            context: Optional context information
            
        Returns:
            Dict with optimized constraints and modifications
        """
        # Skip optimization if no issues or if already compliant
        if not filter_results.get("issues", []) or filter_results.get("is_compliant", True):
            return {
                "optimized": False,
                "constraint_adjustments": [],
                "alternative_formulations": [],
                "exempt_patterns": [],
                "original_constraints": {}
            }
            
        # Extract issues and determine if optimization is possible
        issues = filter_results.get("issues", [])
        
        # Extract content domain from context for domain-specific handling
        domain = self._extract_domain(context)
        
        # Select optimization strategy based on configuration
        strategy = self.optimization_level
        optimizer = self.optimization_strategies.get(strategy, self.optimization_strategies["balanced"])
        
        # Apply the selected optimization strategy
        optimization_results = optimizer(issues, original_input, context, domain)
        
        # Track adjustments in history for future reference
        self._track_adjustments(optimization_results.get("constraint_adjustments", []))
        
        return optimization_results
        
    def generate_alternative_formulations(self, text, issues, context=None):
        """
        Generate alternative formulations of content to address compliance issues.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        if not issues:
            return []
            
        alternatives = []
        
        # Extract the domain from context for domain-specific handling
        domain = self._extract_domain(context)
        
        # Group issues by type for targeted modifications
        issues_by_rule = defaultdict(list)
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            rule_type = self._get_rule_type(rule_id)
            if rule_id:
                issues_by_rule[rule_id].append(issue)
            
        # Generate alternatives for each rule violation
        for rule_id, rule_issues in issues_by_rule.items():
            rule_type = self._get_rule_type(rule_id)
            
            # Use domain-specific handler if available
            if rule_type in self.domain_handlers:
                rule_alternatives = self.domain_handlers[rule_type](text, rule_id, rule_issues, context, domain)
                if rule_alternatives:
                    alternatives.extend(rule_alternatives)
            else:
                # Fallback to generic handling
                rule_alternatives = self._generate_generic_alternatives(text, rule_id, rule_issues, context)
                if rule_alternatives:
                    alternatives.extend(rule_alternatives)
                
        # If no rule-specific alternatives, try general strategies
        if not alternatives:
            # Try general rephrasing strategies
            general_alternatives = self._apply_general_alternatives(text, issues, context)
            if general_alternatives:
                alternatives.extend(general_alternatives)
                
        # Sort alternatives by confidence
        alternatives.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                
        # Deduplicate alternatives
        unique_alternatives = []
        seen = set()
        
        for alt in alternatives:
            alt_text = alt.get("text", "")
            if alt_text and alt_text not in seen and alt_text != text:
                seen.add(alt_text)
                
                # Validate alternative doesn't introduce new compliance issues
                if self._validate_alternative(alt_text, text, issues):
                    unique_alternatives.append(alt)
                
        return unique_alternatives

    def _validate_alternative(self, alternative_text, original_text, original_issues):
        """
        Validate that an alternative doesn't introduce new compliance issues
        and actually addresses the original issues.
        
        Args:
            alternative_text: The proposed alternative text
            original_text: Original text with issues
            original_issues: Original compliance issues
            
        Returns:
            Boolean indicating if the alternative is valid
        """
        # Check that alternative is sufficiently different from original
        similarity = self.similarity_calculator.calculate(alternative_text, original_text)
        if similarity > 0.95:  # Too similar, might not address issues
            return False
            
        # Check that alternative maintains the general meaning
        if similarity < 0.3:  # Too different, might change meaning too much
            return False
            
        # In a real implementation, you would run the alternative through
        # compliance checking to ensure it doesn't introduce new issues
            
        return True
        
    def _generate_rule_specific_alternatives(self, text, rule_id, issues, context):
        """
        Generate alternative formulations specific to a rule.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        # Determine rule type and use appropriate handler
        rule_type = self._get_rule_type(rule_id)
        domain = self._extract_domain(context)
        
        if rule_type in self.domain_handlers:
            return self.domain_handlers[rule_type](text, rule_id, issues, context, domain)
        else:
            return self._generate_generic_alternatives(text, rule_id, issues, context)

    def _handle_pii_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle PII rule violations with intelligent alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Determine what kind of PII is involved
        pii_type = self._extract_pii_type(rule_id, issues)
        
        # Generate multiple alternative handling strategies
        
        # 1. Redaction strategy: Replace PII with [REDACTED]
        redacted_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(redacted_text):
                    # Replace with [REDACTED]
                    redacted_text = redacted_text[:start] + "[REDACTED]" + redacted_text[end:]
                    
        if redacted_text != text:
            alternatives.append({
                "text": redacted_text,
                "rule_id": rule_id,
                "confidence": 0.9,
                "type": "redaction",
                "description": f"Redacted {pii_type} information"
            })
        
        # 2. Anonymization strategy: Replace with generic placeholder
        anonymized_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(anonymized_text):
                    placeholder = self._get_pii_placeholder(pii_type)
                    anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]
                    
        if anonymized_text != text and anonymized_text != redacted_text:
            alternatives.append({
                "text": anonymized_text,
                "rule_id": rule_id,
                "confidence": 0.85,
                "type": "anonymization",
                "description": f"Anonymized {pii_type} with generic placeholder"
            })
            
        # 3. Domain-specific handling if available
        if domain:
            domain_text = self._apply_domain_specific_pii_handling(text, pii_type, issues, domain)
            if domain_text != text and domain_text not in [alt["text"] for alt in alternatives]:
                alternatives.append({
                    "text": domain_text,
                    "rule_id": rule_id,
                    "confidence": 0.8,
                    "type": "domain_specific",
                    "description": f"Applied {domain}-specific handling for {pii_type}"
                })
                
        return alternatives
        
    def _handle_keyword_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle keyword rule violations with intelligent alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Get matched keywords from issues
        matched_keywords = []
        for issue in issues:
            metadata = issue.get("metadata", {})
            keyword = metadata.get("matched_keyword", "")
            if keyword:
                matched_keywords.append(keyword)
                
        if not matched_keywords:
            return alternatives
            
        # 1. Synonym replacement strategy
        synonym_text = text
        replaced_keywords = {}
        
        for keyword in matched_keywords:
            synonyms = self._get_synonyms(keyword, domain)
            if synonyms:
                # Find least sensitive synonym
                best_synonym = self._find_least_sensitive_synonym(synonyms, rule_id)
                if best_synonym and best_synonym != keyword:
                    # Replace keyword with synonym
                    synonym_text = re.sub(r'\b' + re.escape(keyword) + r'\b', 
                                         best_synonym, 
                                         synonym_text, 
                                         flags=re.IGNORECASE)
                    replaced_keywords[keyword] = best_synonym
                    
        if synonym_text != text:
            alternatives.append({
                "text": synonym_text,
                "rule_id": rule_id,
                "confidence": 0.85,
                "type": "synonym_replacement",
                "description": f"Replaced sensitive keywords with synonyms",
                "replacements": replaced_keywords
            })
            
        # 2. Context adjustment strategy - add clarifying context around keywords
        context_text = text
        for keyword in matched_keywords:
            context_pattern = r'\b' + re.escape(keyword) + r'\b'
            context_matches = list(re.finditer(context_pattern, context_text, re.IGNORECASE))
            
            for match in reversed(context_matches):  # Process in reverse to avoid offset issues
                clarification = self._get_clarification_phrase(keyword, rule_id, domain)
                if clarification:
                    start, end = match.span()
                    # Add clarifying context after the keyword
                    context_text = context_text[:end] + clarification + context_text[end:]
                    
        if context_text != text and context_text != synonym_text:
            alternatives.append({
                "text": context_text,
                "rule_id": rule_id,
                "confidence": 0.75,
                "type": "context_adjustment",
                "description": "Added clarifying context to sensitive keywords"
            })
            
        # 3. Rephrasing strategy - rephrase sentences containing keywords
        # In a real implementation, this would use an LLM or paraphrasing model
        rephrased_text = self._rephrase_sentences_with_keywords(text, matched_keywords, rule_id)
        
        if rephrased_text != text and rephrased_text not in [alt["text"] for alt in alternatives]:
            alternatives.append({
                "text": rephrased_text,
                "rule_id": rule_id,
                "confidence": 0.7,
                "type": "rephrasing",
                "description": "Rephrased sentences containing sensitive keywords"
            })
            
        return alternatives
        
    def _handle_sequence_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle sequence rule violations with intelligent alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Extract sequence information from issues
        sequence_info = self._extract_sequence_info(issues)
        
        if not sequence_info:
            return alternatives
            
        # 1. Reordering strategy - change the order of elements in the sequence
        reordered_text = self._reorder_sequence_elements(text, sequence_info)
        
        if reordered_text != text:
            alternatives.append({
                "text": reordered_text,
                "rule_id": rule_id,
                "confidence": 0.8,
                "type": "reordering",
                "description": "Reordered elements in the sequence"
            })
            
        # 2. Separation strategy - break up elements in the sequence
        separated_text = self._separate_sequence_elements(text, sequence_info)
        
        if separated_text != text and separated_text != reordered_text:
            alternatives.append({
                "text": separated_text,
                "rule_id": rule_id,
                "confidence": 0.75,
                "type": "separation",
                "description": "Added separation between sequence elements"
            })
            
        # 3. Context-aware modification strategy
        if context:
            context_modified_text = self._apply_context_aware_sequence_modification(
                text, sequence_info, context, domain
            )
            
            if (context_modified_text != text and 
                context_modified_text not in [alt["text"] for alt in alternatives]):
                alternatives.append({
                    "text": context_modified_text,
                    "rule_id": rule_id,
                    "confidence": 0.85,
                    "type": "context_aware_modification",
                    "description": "Modified sequence based on context"
                })
                
        return alternatives

    def _handle_entity_relationship_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle entity relationship rule violations with intelligent alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Extract entity relationship information
        relationships = self._extract_entity_relationships(issues)
        
        if not relationships:
            return alternatives
            
        # 1. Entity separation strategy - increase distance between entities
        separated_text = self._increase_entity_separation(text, relationships)
        
        if separated_text != text:
            alternatives.append({
                "text": separated_text,
                "rule_id": rule_id,
                "confidence": 0.8,
                "type": "entity_separation",
                "description": "Increased separation between related entities"
            })
            
        # 2. Entity abstraction strategy - make one entity more generic
        abstracted_text = self._abstract_entity(text, relationships)
        
        if abstracted_text != text and abstracted_text != separated_text:
            alternatives.append({
                "text": abstracted_text,
                "rule_id": rule_id,
                "confidence": 0.75,
                "type": "entity_abstraction",
                "description": "Made one entity more generic to reduce relationship sensitivity"
            })
            
        # 3. Entity context modification - add clarifying context between entities
        context_text = self._add_entity_context(text, relationships, domain)
        
        if (context_text != text and 
            context_text not in [alt["text"] for alt in alternatives]):
            alternatives.append({
                "text": context_text,
                "rule_id": rule_id,
                "confidence": 0.7,
                "type": "entity_context",
                "description": "Added clarifying context to entity relationship"
            })
            
        return alternatives
        
    def _handle_semantic_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle semantic rule violations with intelligent alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Extract semantic concepts from issues
        semantic_concepts = self._extract_semantic_concepts(issues)
        
        if not semantic_concepts:
            return alternatives
            
        # 1. Tone adjustment strategy - adjust tone while preserving meaning
        tone_adjusted_text = self._adjust_tone(text, semantic_concepts, domain)
        
        if tone_adjusted_text != text:
            alternatives.append({
                "text": tone_adjusted_text,
                "rule_id": rule_id,
                "confidence": 0.8,
                "type": "tone_adjustment",
                "description": "Adjusted tone to reduce semantic sensitivity"
            })
            
        # 2. Conceptual reframing strategy - reframe concepts in more acceptable terms
        reframed_text = self._reframe_concepts(text, semantic_concepts, domain)
        
        if reframed_text != text and reframed_text != tone_adjusted_text:
            alternatives.append({
                "text": reframed_text,
                "rule_id": rule_id,
                "confidence": 0.75,
                "type": "conceptual_reframing",
                "description": "Reframed sensitive concepts in more acceptable terms"
            })
            
        # 3. Context-based semantic adjustment
        if context:
            context_semantic_text = self._apply_context_based_semantic_adjustment(
                text, semantic_concepts, context, domain
            )
            
            if (context_semantic_text != text and 
                context_semantic_text not in [alt["text"] for alt in alternatives]):
                alternatives.append({
                    "text": context_semantic_text,
                    "rule_id": rule_id,
                    "confidence": 0.7,
                    "type": "context_based_semantic",
                    "description": "Applied context-based semantic adjustments"
                })
                
        return alternatives
        
    def _handle_domain_specific_rules(self, text, rule_id, issues, context, domain=None):
        """
        Handle domain-specific rule violations with specialized alternatives.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            domain: Content domain (if known)
            
        Returns:
            List of alternative formulations
        """
        if not domain:
            return []
            
        alternatives = []
        
        # Handle based on domain
        if domain == "healthcare":
            domain_alternatives = self._generate_healthcare_alternatives(text, rule_id, issues)
            alternatives.extend(domain_alternatives)
        elif domain == "finance":
            domain_alternatives = self._generate_finance_alternatives(text, rule_id, issues)
            alternatives.extend(domain_alternatives)
        elif domain == "legal":
            domain_alternatives = self._generate_legal_alternatives(text, rule_id, issues)
            alternatives.extend(domain_alternatives)
        elif domain == "education":
            domain_alternatives = self._generate_education_alternatives(text, rule_id, issues)
            alternatives.extend(domain_alternatives)
            
        return alternatives
        
    def _generate_generic_alternatives(self, text, rule_id, issues, context):
        """
        Generate generic alternatives when no specific handler is available.
        
        Args:
            text: Original text
            rule_id: ID of the rule being violated
            issues: List of issues for this rule
            context: Optional context information
            
        Returns:
            List of generic alternative formulations
        """
        alternatives = []
        
        # 1. Simple removal strategy - remove problematic parts
        modified_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(modified_text):
                    # Remove problematic content
                    modified_text = modified_text[:start] + modified_text[end:]
                    
        if modified_text != text:
            alternatives.append({
                "text": modified_text,
                "rule_id": rule_id,
                "confidence": 0.6,
                "type": "removal",
                "description": "Removed problematic content"
            })
            
        # 2. Generic replacement strategy
        replacement_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(replacement_text):
                    # Get a generic replacement
                    problematic_text = text[start:end]
                    replacement = self._get_generic_replacement(problematic_text, rule_id)
                    replacement_text = replacement_text[:start] + replacement + replacement_text[end:]
                    
        if replacement_text != text and replacement_text != modified_text:
            alternatives.append({
                "text": replacement_text,
                "rule_id": rule_id,
                "confidence": 0.5,
                "type": "generic_replacement",
                "description": "Replaced problematic content with generic alternatives"
            })
            
        # 3. Add disclaimer
        disclaimer = self._get_rule_disclaimer(rule_id)
        if disclaimer:
            disclaimer_text = disclaimer + "\n\n" + text
            alternatives.append({
                "text": disclaimer_text,
                "rule_id": rule_id,
                "confidence": 0.4,
                "type": "disclaimer",
                "description": "Added a disclaimer for the potentially problematic content"
            })
            
        return alternatives
        
    def _apply_general_alternatives(self, text, issues, context):
        """
        Apply general strategies for generating alternatives when rule-specific strategies fail.
        
        Args:
            text: Original text
            issues: All compliance issues
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # 1. General rewording suggestion
        # In a real implementation, this would use an LLM or paraphrasing model
        alternatives.append({
            "text": f"[Alternative formulation needed for compliance. Original length: {len(text)} characters]",
            "confidence": 0.3,
            "type": "reword_suggestion",
            "description": "The content needs rewording to address compliance issues"
        })
        
        # 2. Issue-focused rewording suggestion
        issue_types = set([self._get_rule_type(issue.get("rule_id", "")) for issue in issues])
        issue_description = ", ".join(issue_types)
        
        alternatives.append({
            "text": f"[Alternative formulation needed to address {issue_description} compliance issues. Original length: {len(text)} characters]",
            "confidence": 0.35,
            "type": "targeted_reword_suggestion",
            "description": f"The content needs targeted rewording to address {issue_description} issues"
        })
        
        # 3. Try removing problematic segments
        segments_to_remove = []
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                if start < end and end <= len(text):
                    segments_to_remove.append((start, end))
                    
        if segments_to_remove:
            # Sort segments in reverse order so removal doesn't affect indices
            segments_to_remove.sort(reverse=True)
            
            modified_text = text
            for start, end in segments_to_remove:
                modified_text = modified_text[:start] + modified_text[end:]
                
            if modified_text and modified_text != text:
                alternatives.append({
                    "text": modified_text,
                    "confidence": 0.4,
                    "type": "segment_removal",
                    "description": "Removed multiple problematic segments"
                })
                
        return alternatives
        
    def _estimate_false_positive_likelihoods(self, issues, historical_data=None):
        """
        Estimate likelihood that each issue is a false positive using enhanced heuristics.
        
        Args:
            issues: List of compliance issues
            historical_data: Optional historical filtering data
            
        Returns:
            Dict mapping rule IDs to false positive likelihood
        """
        fp_likelihoods = {}
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            confidence = issue.get("confidence", 0.5)
            metadata = issue.get("metadata", {})
            
            # Start with base likelihood inversely proportional to confidence
            fp_likelihood = 1.0 - confidence
            
            # Apply rule-specific false positive indicators
            if rule_id in self.fp_indicators:
                indicators = self.fp_indicators[rule_id]
                
                for indicator, weight in indicators.items():
                    # Check if indicator is present in metadata
                    indicator_value = metadata.get(indicator)
                    if indicator_value:
                        indicator_strength = self._calculate_indicator_strength(indicator, indicator_value)
                        fp_likelihood += weight * indicator_strength
                        
            # Apply global false positive rules
            for fp_rule in self.fp_rules:
                rule_type = fp_rule.get("type")
                if rule_type == "low_entropy":
                    # Check if the matched content has low entropy
                    if "matched_content" in metadata:
                        content = metadata["matched_content"]
                        if self._has_low_entropy(content):
                            fp_likelihood += fp_rule.get("weight", 0.1)
                elif rule_type == "common_phrase":
                    # Check if the matched content is a common phrase
                    if "matched_content" in metadata and self._is_common_phrase(metadata["matched_content"]):
                        fp_likelihood += fp_rule.get("weight", 0.1)
                elif rule_type == "context_mismatch":
                    # Check if the context suggests a false positive
                    if self._has_context_mismatch(issue, metadata):
                        fp_likelihood += fp_rule.get("weight", 0.2)
                        
            # Consider historical data if available
            if historical_data and rule_id in historical_data:
                history = historical_data[rule_id]
                fp_rate = history.get("false_positive_rate", 0.0)
                fp_likelihood = 0.7 * fp_likelihood + 0.3 * fp_rate
                
            # Bound between 0 and 1
            fp_likelihood = min(max(fp_likelihood, 0.0), 1.0)
            
            # Store the highest likelihood for each rule
            if rule_id not in fp_likelihoods or fp_likelihood > fp_likelihoods[rule_id]:
                fp_likelihoods[rule_id] = fp_likelihood
                
        return fp_likelihoods
    
    def _calculate_indicator_strength(self, indicator, value):
        """
        Calculate the strength of a false positive indicator.
        
        Args:
            indicator: Name of the indicator
            value: Value of the indicator
            
        Returns:
            Strength of the indicator (0.0-1.0)
        """
        if indicator == "length" and isinstance(value, int):
            # Shorter matches are more likely to be false positives
            return max(0.0, min(1.0, 1.0 - value / 20))
        elif indicator == "match_ratio" and isinstance(value, (int, float)):
            # Lower match ratios are more likely to be false positives
            return max(0.0, min(1.0, 1.0 - value))
        elif indicator == "surrounding_context" and isinstance(value, str):
            # Check if surrounding context suggests a false positive
            return 0.5 if self._is_benign_context(value) else 0.0
        else:
            # Default strength for boolean indicators
            return 1.0 if value else 0.0
            
    def _has_low_entropy(self, text):
        """
        Check if text has low entropy (repetitive or very simple).
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text has low entropy
        """
        if not text or len(text) < 4:
            return True
            
        # Count character frequencies
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            entropy -= prob * np.log2(prob)
            
        # Low entropy threshold
        return entropy < 3.0
        
    def _is_common_phrase(self, text):
        """
        Check if text is a common phrase that might trigger false positives.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text is a common phrase
        """
        # List of common phrases that might trigger false positives
        common_phrases = [
            "please note", "for your information", "let me know", 
            "as soon as possible", "thank you for your", "to whom it may concern",
            "looking forward to", "do not hesitate", "feel free to"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in common_phrases)
        
    def _has_context_mismatch(self, issue, metadata):
        """
        Check if the context suggests a false positive.
        
        Args:
            issue: Compliance issue
            metadata: Issue metadata
            
        Returns:
            Boolean indicating if there's a context mismatch
        """
        # In a real implementation, this would use more sophisticated analysis
        if "context" in metadata and "classification" in metadata:
            context = metadata["context"]
            classification = metadata["classification"]
            
            # Check if context contradicts classification
            education_keywords = ["school", "university", "student", "teach", "learn", "academic"]
            medical_keywords = ["doctor", "patient", "hospital", "treatment", "diagnosis", "medical"]
            
            if classification == "financial" and any(kw in context.lower() for kw in education_keywords + medical_keywords):
                return True
            if classification == "medical" and not any(kw in context.lower() for kw in medical_keywords):
                return True
                
        return False
        
    def _is_benign_context(self, context):
        """
        Check if the surrounding context suggests benign usage.
        
        Args:
            context: Surrounding context
            
        Returns:
            Boolean indicating if context suggests benign usage
        """
        benign_indicators = [
            "example", "quotation", "quote", "reference", "cited", 
            "mentioned", "for instance", "illustration", "hypothetical"
        ]
        
        context_lower = context.lower()
        return any(indicator in context_lower for indicator in benign_indicators)
        
    # Helper methods for rule-specific alternative generation
    
    def _extract_pii_type(self, rule_id, issues):
        """Extract PII type from rule ID and issues"""
        # First try from rule ID
        if "pii" in rule_id.lower():
            pii_parts = rule_id.lower().split("_")
            if len(pii_parts) > 1:
                return pii_parts[1]
                
        # Try from metadata
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "pii_type" in metadata:
                return metadata["pii_type"]
                
        # Default
        return "personal_information"
        
    def _get_pii_placeholder(self, pii_type):
        """Get appropriate placeholder for PII type"""
        placeholders = {
            "email": "[EMAIL ADDRESS]",
            "phone": "[PHONE NUMBER]",
            "ssn": "[SOCIAL SECURITY NUMBER]",
            "address": "[ADDRESS]",
            "name": "[NAME]",
            "dob": "[DATE OF BIRTH]",
            "credit_card": "[PAYMENT CARD NUMBER]",
            "ip": "[IP ADDRESS]"
        }
        
        return placeholders.get(pii_type, "[PERSONAL INFORMATION]")
        
    def _apply_domain_specific_pii_handling(self, text, pii_type, issues, domain):
        """Apply domain-specific handling to PII"""
        if domain == "healthcare":
            # Healthcare domain might use de-identified but realistic placeholders
            return self._apply_healthcare_pii_handling(text, pii_type, issues)
        elif domain == "finance":
            # Finance domain might use specialized redaction
            return self._apply_finance_pii_handling(text, pii_type, issues)
        elif domain == "education":
            # Education domain handling
            return self._apply_education_pii_handling(text, pii_type, issues)
        else:
            # Generic domain handling
            return text  # Fall back to other strategies
            
    def _apply_healthcare_pii_handling(self, text, pii_type, issues):
        """Apply healthcare-specific PII handling"""
        # Example implementation
        modified_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(modified_text):
                    if pii_type == "name":
                        modified_text = modified_text[:start] + "[DE-IDENTIFIED PATIENT]" + modified_text[end:]
                    elif pii_type == "dob":
                        modified_text = modified_text[:start] + "[DE-IDENTIFIED DOB]" + modified_text[end:]
                    # Add other healthcare-specific handlers
                    
        return modified_text
        
    def _apply_finance_pii_handling(self, text, pii_type, issues):
        """Apply finance-specific PII handling"""
        # Example implementation
        modified_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(modified_text):
                    if pii_type == "credit_card":
                        # Format like a masked credit card number
                        modified_text = modified_text[:start] + "XXXX-XXXX-XXXX-1234" + modified_text[end:]
                    elif pii_type == "ssn":
                        modified_text = modified_text[:start] + "XXX-XX-1234" + modified_text[end:]
                    # Add other finance-specific handlers
                    
        return modified_text
        
    def _apply_education_pii_handling(self, text, pii_type, issues):
        """Apply education-specific PII handling"""
        # Example implementation
        modified_text = text
        for issue in issues:
            metadata = issue.get("metadata", {})
            if "location" in metadata:
                start = metadata["location"].get("start", 0)
                end = metadata["location"].get("end", 0)
                
                if start < end and end <= len(modified_text):
                    if pii_type == "name":
                        modified_text = modified_text[:start] + "[STUDENT IDENTIFIER]" + modified_text[end:]
                    # Add other education-specific handlers
                    
        return modified_text
        
    def _get_synonyms(self, keyword, domain=None):
        """Get synonyms for a keyword, optionally domain-specific"""
        # In a real implementation, this would use a thesaurus or language model
        # Placeholder implementation with common examples
        
        common_synonyms = {
            "kill": ["defeat", "eliminate", "overcome"],
            "hate": ["dislike", "oppose", "disapprove of"],
            "weapon": ["tool", "implement", "device"],
            "attack": ["challenge", "criticize", "oppose"],
            "illegal": ["unauthorized", "prohibited", "restricted"],
            "hack": ["modify", "customize", "adapt"],
            "steal": ["borrow", "acquire", "obtain"],
            "abuse": ["misuse", "misapply", "overuse"]
        }
        
        # Add domain-specific synonyms
        domain_synonyms = {}
        if domain == "healthcare":
            domain_synonyms = {
                "pain": ["discomfort", "distress", "ache"],
                "disease": ["condition", "disorder", "health issue"],
                "drug": ["medication", "treatment", "pharmaceutical"]
            }
        elif domain == "finance":
            domain_synonyms = {
                "debt": ["obligation", "liability", "commitment"],
                "bankrupt": ["insolvent", "financially compromised", "in financial distress"],
                "cheap": ["affordable", "cost-effective", "economical"]
            }
            
        # Combine dictionaries
        all_synonyms = {**common_synonyms, **domain_synonyms}
        
        return all_synonyms.get(keyword.lower(), [])
        
    def _find_least_sensitive_synonym(self, synonyms, rule_id):
        """
        Find the least sensitive synonym based on the rule being violated.
        
        Args:
            synonyms: List of potential synonyms
            rule_id: ID of rule being violated
            
        Returns:
            Least sensitive synonym or None
        """
        # In a real implementation, this would use a sensitivity classifier
        # Placeholder implementation
        if not synonyms:
            return None
            
        # Return first synonym as placeholder
        return synonyms[0]
        
    def _get_clarification_phrase(self, keyword, rule_id, domain=None):
        """Get a clarification phrase to add context to a sensitive keyword"""
        # Template clarifications
        clarifications = {
            "kill": " (in the context of computer processes)",
            "attack": " (referring to a strategic approach)",
            "weapon": " (as a metaphorical tool)",
            "hack": " (as a technical workaround)",
            "abuse": " (referring to improper usage)",
            "illegal": " (not in compliance with terms of service)"
        }
        
        # Add domain-specific clarifications
        domain_clarifications = {}
        if domain == "healthcare":
            domain_clarifications = {
                "drug": " (prescribed medication)",
                "overdose": " (exceeding recommended dosage)",
                "addiction": " (dependency on medication)"
            }
        elif domain == "finance":
            domain_clarifications = {
                "laundering": " (referring to financial regulations)",
                "scheme": " (financial arrangement)",
                "evasion": " (legal tax optimization)"
            }
            
        # Combine dictionaries
        all_clarifications = {**clarifications, **domain_clarifications}
        
        return all_clarifications.get(keyword.lower(), "")
        
    def _rephrase_sentences_with_keywords(self, text, keywords, rule_id):
        """
        Rephrase sentences containing sensitive keywords.
        
        Args:
            text: Original text
            keywords: List of sensitive keywords
            rule_id: ID of rule being violated
            
        Returns:
            Text with rephrased sentences
        """
        # In a real implementation, this would use an LLM or paraphrasing model
        # Simple placeholder implementation
        
        # Split into sentences (very basic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                # This sentence contains a sensitive keyword
                # In real implementation, replace with LLM-generated alternative
                result.append(f"[This sentence needs rephrasing to comply with content policies.]")
            else:
                result.append(sentence)
                
        return " ".join(result)
        
    def _extract_sequence_info(self, issues):
        """Extract sequence information from issues"""
        # In a real implementation, this would extract detailed information
        # Placeholder implementation
        sequence_info = {
            "elements": [],
            "window_size": 100
        }
        
        for issue in issues:
            metadata = issue.get("metadata", {})
            elements = metadata.get("sequence_elements", [])
            if elements:
                sequence_info["elements"] = elements
                
            window_size = metadata.get("window_size")
            if window_size:
                sequence_info["window_size"] = window_size
                
        return sequence_info
        
    def _reorder_sequence_elements(self, text, sequence_info):
        """Reorder elements in a problematic sequence"""
        # In a real implementation, this would use NLP to identify and reorder
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: The order of elements in this content may need to be revised for policy compliance.]"
        
    def _separate_sequence_elements(self, text, sequence_info):
        """Add separation between elements in a problematic sequence"""
        # In a real implementation, this would use NLP to identify and separate
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Additional context separation between elements may be needed for policy compliance.]"
        
    def _apply_context_aware_sequence_modification(self, text, sequence_info, context, domain):
        """Apply context-aware modification to sequence"""
        # In a real implementation, this would use context to guide modification
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Context-specific sequence modification recommended for policy compliance.]"
        
    def _extract_entity_relationships(self, issues):
        """Extract entity relationship information from issues"""
        # In a real implementation, this would extract detailed relationship information
        # Placeholder implementation
        relationships = []
        
        for issue in issues:
            metadata = issue.get("metadata", {})
            relationship = metadata.get("relationship")
            if relationship:
                relationships.append(relationship)
                
        return relationships
        
    def _increase_entity_separation(self, text, relationships):
        """Increase separation between related entities"""
        # In a real implementation, this would use NLP to identify and separate
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Increased separation between related entities is recommended for policy compliance.]"
        
    def _abstract_entity(self, text, relationships):
        """Make one entity more generic to reduce relationship sensitivity"""
        # In a real implementation, this would use NLP to identify and abstract
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Using more generic terms for certain entities is recommended for policy compliance.]"
        
    def _add_entity_context(self, text, relationships, domain):
        """Add clarifying context between entities"""
        # In a real implementation, this would use NLP to identify and add context
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Additional context between related entities is recommended for policy compliance.]"
        
    def _extract_semantic_concepts(self, issues):
        """Extract semantic concepts from issues"""
        # In a real implementation, this would extract detailed concept information
        # Placeholder implementation
        concepts = []
        
        for issue in issues:
            metadata = issue.get("metadata", {})
            concept = metadata.get("semantic_concept")
            if concept:
                concepts.append(concept)
                
        return concepts
        
    def _adjust_tone(self, text, concepts, domain):
        """Adjust tone to reduce semantic sensitivity"""
        # In a real implementation, this would use NLP to identify and adjust tone
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Adjusting the tone of certain passages is recommended for policy compliance.]"
        
    def _reframe_concepts(self, text, concepts, domain):
        """Reframe concepts in more acceptable terms"""
        # In a real implementation, this would use NLP to identify and reframe
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Reframing certain concepts in more acceptable terms is recommended for policy compliance.]"
        
    def _apply_context_based_semantic_adjustment(self, text, concepts, context, domain):
        """Apply context-based semantic adjustment"""
        # In a real implementation, this would use context to guide adjustment
        # Placeholder implementation
        
        # Simply add a note for this example
        return text + "\n\n[Note: Context-specific semantic adjustments are recommended for policy compliance.]"
        
    def _generate_healthcare_alternatives(self, text, rule_id, issues):
        """Generate healthcare-specific alternatives"""
        # In a real implementation, this would use healthcare domain knowledge
        # Placeholder implementation
        alternatives = []
        
        # Add a healthcare-specific disclaimer
        disclaimer = "HEALTHCARE DISCLAIMER: The following content is for informational purposes only " \
                    "and should not be considered medical advice. Consult with healthcare professionals " \
                    "for medical guidance.\n\n"
        
        alternatives.append({
            "text": disclaimer + text,
            "rule_id": rule_id,
            "confidence": 0.7,
            "type": "healthcare_disclaimer",
            "description": "Added healthcare disclaimer"
        })
        
        return alternatives
        
    def _generate_finance_alternatives(self, text, rule_id, issues):
        """Generate finance-specific alternatives"""
        # In a real implementation, this would use finance domain knowledge
        # Placeholder implementation
        alternatives = []
        
        # Add a finance-specific disclaimer
        disclaimer = "FINANCIAL DISCLAIMER: The following content is for informational purposes only " \
                    "and should not be considered financial advice. Consult with financial professionals " \
                    "before making financial decisions.\n\n"
        
        alternatives.append({
            "text": disclaimer + text,
            "rule_id": rule_id,
            "confidence": 0.7,
            "type": "finance_disclaimer",
            "description": "Added financial disclaimer"
        })
        
        return alternatives
        
    def _generate_legal_alternatives(self, text, rule_id, issues):
        """Generate legal-specific alternatives"""
        # In a real implementation, this would use legal domain knowledge
        # Placeholder implementation
        alternatives = []
        
        # Add a legal-specific disclaimer
        disclaimer = "LEGAL DISCLAIMER: The following content is for informational purposes only " \
                    "and should not be considered legal advice. Consult with legal professionals " \
                    "for legal guidance.\n\n"
        
        alternatives.append({
            "text": disclaimer + text,
            "rule_id": rule_id,
            "confidence": 0.7,
            "type": "legal_disclaimer",
            "description": "Added legal disclaimer"
        })
        
        return alternatives
        
    def _generate_education_alternatives(self, text, rule_id, issues):
        """Generate education-specific alternatives"""
        # In a real implementation, this would use education domain knowledge
        # Placeholder implementation
        alternatives = []
        
        # Add an education-specific disclaimer
        disclaimer = "EDUCATION DISCLAIMER: The following content is for educational purposes only. " \
                    "Educational approaches should be adapted to individual learning needs.\n\n"
        
        alternatives.append({
            "text": disclaimer + text,
            "rule_id": rule_id,
            "confidence": 0.7,
            "type": "education_disclaimer",
            "description": "Added education disclaimer"
        })
        
        return alternatives
        
    def _get_generic_replacement(self, text, rule_id):
        """Get a generic replacement for problematic text"""
        # In a real implementation, this would use context to generate appropriate replacement
        # Placeholder implementation
        return "[alternative wording needed]"
        
    def _get_rule_disclaimer(self, rule_id):
        """Get an appropriate disclaimer for a rule violation"""
        rule_type = self._get_rule_type(rule_id)
        
        disclaimers = {
            "pii": "PRIVACY NOTICE: This content discusses concepts related to personal information. " \
                  "No actual personal data is included or requested.",
            "keyword": "CONTENT NOTICE: This content contains terms that may be interpreted in multiple ways. " \
                      "The intended meaning is non-harmful and informational only.",
            "sequence": "CONTENT NOTICE: This content presents information in a sequence that " \
                       "is intended for educational and informational purposes only.",
            "entity_relationship": "CONTENT NOTICE: This content discusses relationships between entities " \
                                 "in an abstract and informational context only.",
            "semantic": "CONTENT NOTICE: The semantic meaning of this content is intended to be " \
                      "informational and educational only."
        }
        
        return disclaimers.get(rule_type, "NOTICE: This content is intended for informational purposes only.")
        
    def _get_rule_type(self, rule_id):
        """Get the type of rule from rule ID"""
        if "pii" in rule_id.lower():
            return "pii"
        elif "keyword" in rule_id.lower():
            return "keyword"
        elif "sequence" in rule_id.lower():
            return "sequence"
        elif "entity" in rule_id.lower() or "relationship" in rule_id.lower():
            return "entity_relationship"
        elif "semantic" in rule_id.lower():
            return "semantic"
        elif "domain" in rule_id.lower():
            return "domain_specific"
        else:
            return "general"
            
    def _extract_domain(self, context):
        """Extract domain from context if available"""
        if not context:
            return None
            
        if isinstance(context, dict) and "domain" in context:
            return context["domain"]
            
        if isinstance(context, dict) and "metadata" in context:
            metadata = context["metadata"]
            if isinstance(metadata, dict) and "domain" in metadata:
                return metadata["domain"]
                
        return None

    # Optimization strategy methods
    
    def _initialize_optimization_strategies(self):
        """Initialize optimization strategies."""
        return {
            "strict": self._optimize_strict,
            "balanced": self._optimize_balanced,
            "lenient": self._optimize_lenient,
            "adaptive": self._optimize_adaptive
        }
        
    def _optimize_strict(self, issues, original_input, context, domain=None):
        """
        Strict optimization strategy - prioritizes compliance.
        
        Makes minimal adjustments, focusing only on high-confidence false positives.
        """
        # Only consider high-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.85,
            max_adjustments=1
        )
        
        # Look for exemption patterns with high confidence
        exemptions = self._match_existing_exemptions(issues, original_input)
        
        # Generate only minimal alternatives
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        alternatives = alternatives[:1]  # Only the best alternative
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "strict",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues)
        }
        
    def _optimize_balanced(self, issues, original_input, context, domain=None):
        """
        Balanced optimization strategy - balances compliance and user experience.
        
        Makes moderate adjustments to reduce false positives while maintaining compliance.
        """
        # Consider moderate-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.7,
            max_adjustments=2
        )
        
        # Look for exemption patterns
        exemptions = self.identify_exemption_patterns(issues, original_input, context)
        
        # Generate alternatives with domain awareness
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        alternatives = alternatives[:3]  # Top 3 alternatives
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "balanced",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues),
            "domain": domain
        }
        
    def _optimize_lenient(self, issues, original_input, context, domain=None):
        """
        Lenient optimization strategy - prioritizes user experience.
        
        Makes more aggressive adjustments to reduce false positives, focusing on user experience.
        """
        # Consider lower-confidence false positives
        constraint_adjustments = self._identify_constraint_adjustments(
            issues, 
            confidence_threshold=0.6,
            max_adjustments=self.max_constraint_adjustments
        )
        
        # Look for exemption patterns more aggressively
        exemptions = self.identify_exemption_patterns(issues, original_input, context)
        
        # Generate more alternatives with domain awareness
        alternatives = self.generate_alternative_formulations(original_input, issues, context)
        
        return {
            "optimized": bool(constraint_adjustments or exemptions or alternatives),
            "optimization_level": "lenient",
            "constraint_adjustments": constraint_adjustments,
            "alternative_formulations": alternatives,
            "exempt_patterns": exemptions,
            "original_constraints": self._get_original_constraints(issues),
            "domain": domain
        }
        
    def _optimize_adaptive(self, issues, original_input, context, domain=None):
        """
        Adaptive optimization strategy - adjusts based on context and history.
        
        Adapts optimization approach based on context, severity, and historical data.
        """
        # Determine appropriate strategy based on context and issue severity
        strategy = self._determine_adaptive_strategy(issues, context, domain)
        
        # Use the selected strategy
        if strategy == "strict":
            return self._optimize_strict(issues, original_input, context, domain)
        elif strategy == "lenient":
            return self._optimize_lenient(issues, original_input, context, domain)
        else:
            return self._optimize_balanced(issues, original_input, context, domain)

    def _determine_adaptive_strategy(self, issues, context, domain=None):
        """Determine the appropriate strategy for adaptive optimization."""
        # Default to balanced
        if not issues:
            return "balanced"
            
        # Check for critical issues
        has_critical = any(issue.get("severity") == "critical" for issue in issues)
        if has_critical:
            return "strict"
            
        # Check context for risk factors
        if context:
            context_domain = self._extract_domain(context) or domain
            
            # High-risk domains use strict strategy
            high_risk_domains = self.config.get("high_risk_domains", [])
            if context_domain in high_risk_domains:
                return "strict"
                
            # Low-risk domains can use lenient strategy
            low_risk_domains = self.config.get("low_risk_domains", [])
            if context_domain in low_risk_domains:
                return "lenient"
                
            # Check user risk level
            user_info = context.get("user_info", {})
            user_risk = user_info.get("risk_level", "medium")
            
            if user_risk == "low":
                return "lenient"
            elif user_risk == "high":
                return "strict"
                
        # Count issue severity
        severity_counts = Counter(issue.get("severity", "medium") for issue in issues)
        
        # Many high-severity issues, use strict
        if severity_counts.get("high", 0) > 3:
            return "strict"
            
        # Few low-severity issues, use lenient
        if (severity_counts.get("low", 0) > 0 and 
            not severity_counts.get("high", 0) and 
            not severity_counts.get("medium", 0)):
            return "lenient"
            
        # Default to balanced
        return "balanced"
        
    def _identify_constraint_adjustments(self, issues, confidence_threshold=0.7, max_adjustments=None):
        """Identify potential constraint adjustments to reduce false positives."""
        if not issues:
            return []
            
        adjustments = []
        
        # Group issues by rule ID
        issues_by_rule = defaultdict(list)
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            if rule_id:
                issues_by_rule[rule_id].append(issue)
                
        # Estimate false positive likelihood for each rule
        fp_likelihoods = self._estimate_false_positive_likelihoods(issues)
        
        # Consider adjustments for rules with high FP likelihood
        for rule_id, likelihood in fp_likelihoods.items():
            if likelihood >= confidence_threshold and rule_id in self.configurable_constraints:
                constraint = self.configurable_constraints[rule_id]
                
                # Calculate adjustment based on FP likelihood
                parameter = constraint.get("parameter", "threshold")
                current_value = constraint.get("current_value", 0)
                
                # Adjustment size based on FP likelihood
                adjustment_size = likelihood * self.adjustment_factor
                
                # Determine direction based on parameter type
                if parameter in ["threshold", "sensitivity", "min_confidence"]:
                    # Increase thresholds to reduce false positives
                    new_value = current_value + adjustment_size
                elif parameter in ["max_distance", "window_size"]:
                    # Decrease distances to make rules more precise
                    new_value = current_value - (current_value * adjustment_size)
                else:
                    # Default behavior
                    new_value = current_value + adjustment_size
                    
                # Ensure within bounds
                min_value = constraint.get("min_value", 0)
                max_value = constraint.get("max_value", 1)
                new_value = min(max(new_value, min_value), max_value)
                
                if new_value != current_value:
                    adjustments.append({
                        "rule_id": rule_id,
                        "parameter": parameter,
                        "current_value": current_value,
                        "new_value": new_value,
                        "confidence": likelihood,
                        "affected_issues": len(issues_by_rule[rule_id])
                    })
                    
        # Sort by confidence and affected issues
        adjustments.sort(key=lambda x: (x["confidence"], x["affected_issues"]), reverse=True)
        
        # Limit to max adjustments if specified
        if max_adjustments is not None:
            adjustments = adjustments[:max_adjustments]
            
        return adjustments
        
    def identify_exemption_patterns(self, issues, original_input, context=None):
        """
        Identify patterns that might qualify for compliance exemptions.
        
        Args:
            issues: Compliance issues detected
            original_input: Original input content
            context: Optional context information
            
        Returns:
            List of potential exemption patterns
        """
        if not issues or not original_input:
            return []
            
        potential_exemptions = []
        
        # Check for existing exemption patterns
        matching_exemptions = self._match_existing_exemptions(issues, original_input)
        if matching_exemptions:
            potential_exemptions.extend(matching_exemptions)
            
        # Look for new potential exemption patterns
        new_exemptions = self._identify_new_exemptions(issues, original_input, context)
        if new_exemptions:
            potential_exemptions.extend(new_exemptions)
            
        return potential_exemptions
        
    def _match_existing_exemptions(self, issues, original_input):
        """Match issues against existing exemption patterns."""
        matching_exemptions = []
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            
            for exemption in self.exemption_patterns:
                if exemption.get("rule_id") == rule_id:
                    pattern = exemption.get("pattern", "")
                    
                    # Skip invalid patterns
                    if not pattern:
                        continue
                        
                    try:
                        if re.search(pattern, original_input, re.IGNORECASE):
                            matching_exemptions.append({
                                "rule_id": rule_id,
                                "pattern": pattern,
                                "description": exemption.get("description", ""),
                                "confidence": exemption.get("confidence", 0.8)
                            })
                    except re.error:
                        # Log invalid regex pattern
                        logging.error(f"Invalid exemption regex pattern: {pattern}")
                        
        return matching_exemptions
        
    def _identify_new_exemptions(self, issues, original_input, context):
        """Identify potential new exemption patterns."""
        # This is a placeholder for a more sophisticated implementation
        # In a real system, this would analyze patterns more deeply
        
        # Simple approach: look for contextual indicators that might justify exemptions
        potential_exemptions = []
        
        # Check if context includes exemption indicators
        if context and "metadata" in context:
            metadata = context["metadata"]
            
            # Educational or research context might justify exemptions
            if metadata.get("purpose") in ["educational", "research", "analysis"]:
                for issue in issues:
                    rule_id = issue.get("rule_id", "")
                    
                    # For now, just suggest the possibility
                    potential_exemptions.append({
                        "rule_id": rule_id,
                        "pattern": None,  # No specific pattern yet
                        "description": f"Potential exemption for {rule_id} in {metadata.get('purpose')} context",
                        "confidence": 0.6,
                        "is_suggestion": True
                    })
                    
        return potential_exemptions
        
    def _get_original_constraints(self, issues):
        """Get the original constraints that triggered the issues."""
        original_constraints = {}
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            if rule_id in self.configurable_constraints:
                constraint = self.configurable_constraints[rule_id]
                original_constraints[rule_id] = {
                    "parameter": constraint.get("parameter", "threshold"),
                    "current_value": constraint.get("current_value", 0)
                }
                
        return original_constraints
        
    def _track_adjustments(self, adjustments):
        """Track constraint adjustments for history."""
        timestamp = time.time()
        
        for adjustment in adjustments:
            self.adjustment_history.append({
                "timestamp": timestamp,
                "rule_id": adjustment.get("rule_id", ""),
                "parameter": adjustment.get("parameter", ""),
                "old_value": adjustment.get("current_value", 0),
                "new_value": adjustment.get("new_value", 0),
                "confidence": adjustment.get("confidence", 0)
            })
            
        # Keep history limited
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-100:]