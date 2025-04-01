import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import numpy as np
import z3
import re

class SymbolicReasoner:
    """
    Symbolic reasoning engine for compliance verification that translates 
    regulatory requirements into formal logic representations.
    """
    def __init__(self, config=None, knowledge_base=None):
        self.config = config or {}
        self.knowledge_base = knowledge_base
        self.rules = self._load_rules()
        self.rule_cache = {}  # Cache for rule applicability
        
    def _load_rules(self):
        """
        Load symbolic reasoning rules from configured source.
        
        Notes:
            In a production system, this would load rules from a database, 
            file system, or API endpoint. For this implementation, we use 
            sample rules defined in code.
        
        Returns:
            Dictionary mapping rule IDs to rule definitions
        """
        # This would fetch from a database or file in a real system
        # Using sample rules for demonstration
        rules = {
            "gdpr_data_minimization": {
                "id": "gdpr_data_minimization",
                "framework": "GDPR",
                "description": "Personal data must be adequate, relevant and limited to what is necessary",
                "severity": "high",
                "symbolic_representation": "∀x (PersonalData(x) → (Adequate(x) ∧ Relevant(x) ∧ Necessary(x)))",
                "pattern": r"\b(?:collect|store|process|use)\b.{0,30}\b(?:all|every|any|extensive)\b.{0,30}\b(?:data|information)\b",
                "keywords": ["collect", "store", "process", "extensive", "all data"]
            },
            "gdpr_consent": {
                "id": "gdpr_consent",
                "framework": "GDPR",
                "description": "Processing based on consent must be demonstrably given",
                "severity": "critical",
                "symbolic_representation": "∀x (ProcessingActivity(x) ∧ BasedOnConsent(x) → ∃y (Consent(y) ∧ Demonstrable(y) ∧ GivenFor(y, x)))",
                "pattern": r"\b(?:without|no|implicit)\b.{0,30}\b(?:consent|permission|authorization)\b",
                "keywords": ["consent", "permission", "authorization", "explicit"]
            },
            "hipaa_phi_disclosure": {
                "id": "hipaa_phi_disclosure",
                "framework": "HIPAA",
                "description": "Protected health information should only be disclosed with authorization",
                "severity": "critical",
                "symbolic_representation": "∀x (PHI(x) → (Disclosed(x) → ∃y (Authorization(y) ∧ CoversDisclosure(y, x))))",
                "pattern": r"\b(?:health|medical|patient).{0,50}\b(?:disclose|share|reveal)\b",
                "keywords": ["PHI", "health information", "medical data", "disclose", "share"]
            }
        }
        
        print(f"Loaded {len(rules)} symbolic reasoning rules")
        return rules
        
    def evaluate_compliance(self, text, applicable_rules, context=None):
        """
        Evaluate whether text complies with applicable rules using symbolic reasoning.
        
        Notes:
            A production implementation would use formal logic reasoning engines.
            This simplified version uses pattern matching and keyword detection.
        
        Args:
            text: Text to evaluate
            applicable_rules: List of applicable rule IDs
            context: Optional contextual information
            
        Returns:
            Dict with compliance evaluation results
        """
        import re
        
        if not text or not applicable_rules:
            return {
                "is_compliant": True,
                "violations": [],
                "reasoning_steps": [],
                "compliance_score": 1.0
            }
            
        # Track violations and reasoning steps
        violations = []
        reasoning_steps = []
        
        # Check each applicable rule
        for rule_id in applicable_rules:
            rule = self.rules.get(rule_id)
            if not rule:
                continue
                
            # This simplified implementation uses pattern matching
            # A real implementation would parse the symbolic representation and apply formal logic
            
            # Check for pattern matches
            pattern = rule.get("pattern")
            if pattern:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    violations.append({
                        "rule_id": rule_id,
                        "severity": rule.get("severity", "medium"),
                        "description": rule.get("description", ""),
                        "matched_text": match.group(0),
                        "position": (match.start(), match.end())
                    })
                    
                    reasoning_steps.append({
                        "rule_id": rule_id,
                        "reasoning": f"Pattern '{pattern}' matched at position {match.start()}: '{match.group(0)}'",
                        "conclusion": "Rule violation detected"
                    })
            
            # Check for keywords (as a backup detection method)
            if not any(v["rule_id"] == rule_id for v in violations):
                keywords = rule.get("keywords", [])
                matched_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                
                if matched_keywords:
                    violations.append({
                        "rule_id": rule_id,
                        "severity": rule.get("severity", "medium"),
                        "description": rule.get("description", ""),
                        "matched_keywords": matched_keywords
                    })
                    
                    reasoning_steps.append({
                        "rule_id": rule_id,
                        "reasoning": f"Keywords detected: {', '.join(matched_keywords)}",
                        "conclusion": "Potential rule violation based on keywords"
                    })
            
            # No violation detected, add compliance reasoning step
            if not any(v["rule_id"] == rule_id for v in violations):
                reasoning_steps.append({
                    "rule_id": rule_id,
                    "reasoning": "No violation patterns or keywords detected",
                    "conclusion": "Rule compliance verified"
                })
        
        # Calculate compliance score based on violations
        if not violations:
            compliance_score = 1.0
        else:
            # Weight by severity
            severity_weights = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            violation_penalty = sum(severity_weights.get(v["severity"], 0.5) for v in violations)
            compliance_score = max(0.0, 1.0 - (violation_penalty / len(applicable_rules)))
        
        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "reasoning_steps": reasoning_steps,
            "compliance_score": compliance_score
        }
    
    def get_applicable_rules(self, text, frameworks=None, context=None):
        """
        Determine which rules are applicable to the given text and context.
        
        Notes:
            In a production system, this would use more sophisticated logic
            including semantic analysis and contextual inference to determine
            rule applicability. This implementation uses simpler heuristics.
        
        Args:
            text: Text to analyze
            frameworks: Optional list of regulatory frameworks to consider
            context: Optional contextual information
            
        Returns:
            List of applicable rule IDs
        """
        # Generate cache key based on text, frameworks, and context
        cache_key = self._generate_cache_key(text, frameworks, context)
        
        # Check cache first
        if cache_key in self.rule_cache:
            return self.rule_cache[cache_key]
        
        applicable_rules = []
        
        # If specific frameworks are provided, only consider rules from those frameworks
        framework_filter = set(frameworks) if frameworks else None
        
        for rule_id, rule in self.rules.items():
            # Apply framework filter if specified
            if framework_filter and rule.get("framework") not in framework_filter:
                continue
            
            # Check domain applicability
            if context and "domain" in context:
                rule_domains = rule.get("applicable_domains", ["general"])
                if "all" not in rule_domains and context["domain"] not in rule_domains:
                    continue
            
            # Basic relevance check: look for any rule keywords in the text
            # A real implementation would use semantic similarity, context analysis, etc.
            keywords = rule.get("keywords", [])
            
            # If any keyword is found in the text, consider the rule applicable
            if any(keyword.lower() in text.lower() for keyword in keywords):
                applicable_rules.append(rule_id)
                continue
            
            # Check the rule pattern for potential matches
            pattern = rule.get("pattern")
            if pattern:
                import re
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        applicable_rules.append(rule_id)
                        continue
                except re.error:
                    # Log pattern error but continue
                    print(f"Invalid regex pattern in rule {rule_id}: {pattern}")
            
            # More sophisticated rule applicability logic would be implemented here
            # For example, checking entity types, semantic similarity, etc.
        
        # Sort rules by severity for prioritized evaluation
        applicable_rules.sort(
            key=lambda rule_id: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                self.rules.get(rule_id, {}).get("severity", "medium"), 2
            )
        )
        
        # Cache the result
        self.rule_cache[cache_key] = applicable_rules
        
        return applicable_rules
    
    def _generate_cache_key(self, text, frameworks, context):
        """Generate a cache key for rule applicability caching"""
        import hashlib
        
        # Use first 100 chars of text
        text_sample = text[:100] if text else ""
        
        # Create a string representation of frameworks
        frameworks_str = ",".join(sorted(frameworks)) if frameworks else "all"
        
        # Extract relevant context fields (domain and content_type)
        context_str = ""
        if context:
            domain = context.get("domain", "general")
            content_type = context.get("content_type", "text")
            context_str = f"{domain}:{content_type}"
        
        # Combine and hash
        combined = f"{text_sample}|{frameworks_str}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _text_to_symbolic(self, text, context=None):
        """
        Convert text to symbolic representation for formal logic evaluation.
        
        Args:
            text: Text to convert
            context: Optional context information
            
        Returns:
            Dictionary containing symbolic representation or None if conversion fails
        """
        try:
            # Extract statements from text
            statements = []
            
            # Simple NLP-based extraction
            # In a real implementation, this would use more sophisticated NLP
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                # Extract subject-predicate-object triples
                triples = self._extract_triples(sentence)
                statements.extend(triples)
                
                # Extract entities and concepts
                entities = self._extract_entities(sentence)
                concepts = self._extract_concepts(sentence)
                
            return {
                "statements": statements,
                "entities": entities,
                "concepts": concepts,
                "context": context
            }
        except Exception as e:
            # Log error and return None
            print(f"Error converting text to symbolic representation: {str(e)}")
            return None

    def _split_into_sentences(self, text):
        """Split text into sentences."""
        import re
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_triples(self, sentence):
        """Extract subject-predicate-object triples from sentence."""
        # Simple rule-based extraction
        # In a real implementation, this would use dependency parsing
        
        triples = []
        
        # Try simple pattern matching
        # Subject-verb-object pattern
        svo_pattern = r'([\w\s]+?)\s+((?:is|are|was|were|has|have|had|do|does|did|will|would|shall|should|may|might|can|could)\s+[\w\s]+?|[\w]+?(?:s|es|ed|ing)?)\s+([\w\s]+)'
        matches = re.finditer(svo_pattern, sentence, re.IGNORECASE)
        
        for match in matches:
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(3).strip()
            
            if subject and predicate and obj:
                triples.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
        
        return triples

    def _extract_entities(self, sentence):
        """Extract entities from sentence."""
        # Simple entity extraction
        # In a real implementation, this would use NER
        
        entities = []
        
        # Check for known entity patterns
        # Personal data entities
        personal_data_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Phone numbers
        ]
        
        for pattern in personal_data_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                entities.append({
                    "type": "PersonalData",
                    "text": match.group(0),
                    "position": (match.start(), match.end())
                })
        
        return entities

    def _extract_concepts(self, sentence):
        """Extract regulatory concepts from sentence."""
        concepts = []
        
        # Check for known regulatory concepts
        concept_keywords = {
            "Consent": ["consent", "permission", "authorize", "agree"],
            "PersonalData": ["personal data", "personal information", "pii"],
            "ProcessingActivity": ["process", "collect", "store", "use"],
            "PHI": ["health information", "medical data", "patient data"],
            "Authorization": ["authorization", "authorized", "approved"]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(kw in sentence.lower() for kw in keywords):
                concepts.append(concept)
        
        return concepts

    def _evaluate_logical_rule(self, rule_symbolic, symbolic_repr):
        """
        Evaluate a rule using formal logic reasoning.
        
        Args:
            rule_symbolic: Symbolic representation of the rule
            symbolic_repr: Symbolic representation of the text
            
        Returns:
            Dictionary with evaluation results
        """
        # This requires z3 or another theorem prover
        if z3 is None:
            raise ImportError("z3 library is required for formal logic evaluation")
        
        try:
            # Parse rule into z3 formula
            rule_formula = self._parse_logic_formula(rule_symbolic)
            
            # Convert symbolic representation to z3 constraints
            facts = self._symbolic_to_constraints(symbolic_repr)
            
            # Create solver
            solver = z3.Solver()
            
            # Add facts
            for fact in facts:
                solver.add(fact)
            
            # Check if facts imply rule (by checking if facts ∧ ¬rule is unsatisfiable)
            solver.add(z3.Not(rule_formula))
            
            if solver.check() == z3.unsat:
                # Rule is satisfied (facts ∧ ¬rule is unsatisfiable)
                return {
                    "is_compliant": True,
                    "method": "formal_logic",
                    "confidence": 1.0,
                    "reasoning": "Formal logic verification passed"
                }
            else:
                # Rule is violated
                model = solver.model()
                violation_witness = self._extract_violation_witness(model)
                
                return {
                    "is_compliant": False,
                    "method": "formal_logic",
                    "confidence": 1.0,
                    "reasoning": "Formal logic verification failed",
                    "violation_details": {
                        "witness": violation_witness,
                        "formula": rule_symbolic
                    }
                }
                
        except Exception as e:
            # If formal logic evaluation fails, throw exception to fall back to patterns
            raise Exception(f"Formal logic evaluation failed: {str(e)}")

    def _parse_logic_formula(self, rule_symbolic):
        """
        Parse symbolic rule representation into z3 formula.
        
        This is a simplified implementation - a real one would use a proper parser
        for the specific logic formalism used.
        """
        # This is a placeholder - actual implementation would parse the formula string
        if z3 is None:
            raise ImportError("z3 library is required for parsing logic formulas")
        
        # For demonstration, we'll use a hard-coded mapping of formulas
        if "PersonalData(x)" in rule_symbolic and "Necessary(x)" in rule_symbolic:
            # Data minimization rule
            x = z3.Const('x', z3.StringSort())
            personal_data = z3.Function('PersonalData', z3.StringSort(), z3.BoolSort())
            adequate = z3.Function('Adequate', z3.StringSort(), z3.BoolSort())
            relevant = z3.Function('Relevant', z3.StringSort(), z3.BoolSort())
            necessary = z3.Function('Necessary', z3.StringSort(), z3.BoolSort())
            
            return z3.ForAll([x], z3.Implies(personal_data(x), 
                                        z3.And(adequate(x), relevant(x), necessary(x))))
        
        # Default: return a True formula (always satisfied)
        return z3.BoolVal(True)

    def _symbolic_to_constraints(self, symbolic_repr):
        """
        Convert symbolic representation to z3 constraints.
        
        This is a simplified implementation - a real one would build constraints
        based on the full symbolic representation.
        """
        if z3 is None:
            raise ImportError("z3 library is required for creating constraints")
        
        constraints = []
        
        # Create z3 functions for common predicates
        x = z3.Const('x', z3.StringSort())
        personal_data = z3.Function('PersonalData', z3.StringSort(), z3.BoolSort())
        adequate = z3.Function('Adequate', z3.StringSort(), z3.BoolSort())
        relevant = z3.Function('Relevant', z3.StringSort(), z3.BoolSort())
        necessary = z3.Function('Necessary', z3.StringSort(), z3.BoolSort())
        
        # Extract entities and concepts
        entities = symbolic_repr.get("entities", [])
        concepts = symbolic_repr.get("concepts", [])
        
        # Add constraints for personal data entities
        for entity in entities:
            if entity.get("type") == "PersonalData":
                entity_const = z3.StringVal(entity.get("text"))
                constraints.append(personal_data(entity_const))
                
                # Check if context suggests data is not necessary
                if "collect all" in symbolic_repr.get("context", {}).get("text", "").lower():
                    constraints.append(z3.Not(necessary(entity_const)))
        
        return constraints

    def _extract_violation_witness(self, model):
        """Extract a human-readable violation witness from z3 model."""
        # This would extract the specific counterexample from the model
        # For simplicity, we return a generic message
        return "Model contains unnecessary personal data"

    def _calculate_compliance_score(self, rule_results, applicable_rules):
        """Calculate overall compliance score from rule results"""
        if not rule_results:
            return 1.0  # Default to compliant if no rules evaluated
            
        # Get severity weights for rules
        severity_weights = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0
        
        for rule_id, result in rule_results.items():
            rule = self.rules.get(rule_id, {})
            severity = rule.get("severity", "medium")
            weight = severity_weights.get(severity, 0.5)
            
            # Get compliance result as a score
            score = 1.0 if result.get("is_compliant", False) else 0.0
            
            # Adjust by confidence if available
            confidence = result.get("confidence", 1.0)
            score = score * confidence
            
            total_weight += weight
            weighted_sum += weight * score
        
        # Normalize score
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 1.0  # Default to compliant if no weights    