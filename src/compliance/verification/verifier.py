import hashlib
from src.utils.cache.lru_cache import LRUCache
import datetime
import uuid
import logging
import re
from src.compliance.verification.violation_analyzer import ViolationAnalyzer
from src.compliance.reasoning.contextual_rule_interpreter import ContextualRuleInterpreter
from src.compliance.models.compliance_issue import ComplianceIssue

class ComplianceVerifier:
    """
    Verifies content compliance against regulatory frameworks with configurable 
    strictness levels and detailed violation reporting.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.strictness_levels = compliance_config.get("strictness_levels", {
            "strict": 0.95,
            "standard": 0.85,
            "relaxed": 0.75
        })
        self.required_confidence = compliance_config.get("required_confidence", 0.9)
        self.cache = LRUCache(maxsize=compliance_config.get("cache_size", 1000))
        self.violation_analyzer = ViolationAnalyzer(compliance_config)
        self.contextual_interpreter = ContextualRuleInterpreter(compliance_config)
        
    def verify_content(self, content, content_type="text", compliance_mode="standard", context=None):
        """
        Verify content compliance against applicable rules.
        
        Args:
            content: Content to verify
            content_type: Type of content ("prompt", "response", "text")
            compliance_mode: Strictness level for verification
            context: Additional context for verification
            
        Returns:
            Dict with compliance results and detailed violation information
        """
        # Check cache first for efficiency
        cache_key = self._generate_cache_key(content, content_type, compliance_mode)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        # Get threshold based on compliance mode
        threshold = self.strictness_levels.get(compliance_mode, self.strictness_levels["standard"])
        
        # Analyze content for compliance
        analysis_results = self._analyze_content(content, content_type, threshold, context)
        
        # Determine if content is compliant
        is_compliant = analysis_results["compliance_score"] >= threshold
        
        # Generate detailed verification results
        verification_result = {
            "is_compliant": is_compliant,
            "compliance_score": analysis_results["compliance_score"],
            "mode": compliance_mode,
            "threshold": threshold,
            "violations": analysis_results.get("violations", []),
            "metadata": {
                "content_type": content_type,
                "verification_time": datetime.datetime.now().isoformat(),
                "verification_id": str(uuid.uuid4()),
                "framework_details": analysis_results.get("framework_details", {})
            }
        }
        
        # If not compliant, add detailed analysis
        if not is_compliant:
            verification_result["error"] = self._format_compliance_error(
                analysis_results["violations"],
                content_type
            )
            verification_result["remediation_suggestions"] = self.violation_analyzer.generate_remediation(
                analysis_results["violations"],
                content_type,
                context
            )
            
        # Cache the result
        self.cache[cache_key] = verification_result
        
        return verification_result
    
    def aggregate_framework_results(self, framework_results):
        """
        Aggregate results from multiple framework verifications.
        
        Args:
            framework_results: List of verification results from different frameworks
            
        Returns:
            Dict with aggregated compliance results
        """
        if not framework_results:
            return {"is_compliant": True, "compliance_score": 1.0}
            
        # Extract scores and violations
        compliance_scores = [r.get("compliance_score", 0.0) for r in framework_results]
        all_violations = []
        framework_details = {}
        
        for i, result in enumerate(framework_results):
            # Add violations from this framework
            violations = result.get("violations", [])
            all_violations.extend(violations)
            
            # Add framework details
            framework_id = result.get("framework_id", f"framework_{i}")
            framework_details[framework_id] = {
                "compliance_score": result.get("compliance_score", 0.0),
                "violation_count": len(violations)
            }
            
        # Calculate aggregated compliance score (minimum of all scores)
        aggregated_score = min(compliance_scores) if compliance_scores else 1.0
        
        # Determine if content is compliant (all frameworks must pass)
        is_compliant = all(r.get("is_compliant", True) for r in framework_results)
        
        # Remove duplicate violations
        unique_violations = self._deduplicate_violations(all_violations)
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": aggregated_score,
            "violations": unique_violations,
            "framework_details": framework_details,
            "framework_count": len(framework_results)
        }
    
    def _analyze_content(self, content, content_type, threshold, context=None):
        """Analyze content for compliance issues"""
        # This would integrate with specialized analyzers for different content types
        # Simplified implementation for demonstration
        
        # Apply contextual rule interpretation
        interpreted_rules = self.contextual_interpreter.get_contextual_rules(content_type, context)
        
        # Analyze content against rules
        violations = []
        rule_scores = []
        
        for rule in interpreted_rules:
            rule_result = self._evaluate_rule(content, rule)
            rule_scores.append(rule_result["score"])
            
            if rule_result["score"] < threshold:
                violations.append({
                    "rule_id": rule.get("id", "unknown"),
                    "rule_description": rule.get("description", ""),
                    "severity": rule.get("severity", "medium"),
                    "score": rule_result["score"],
                    "locations": rule_result.get("locations", [])
                })
        
        # Calculate overall compliance score
        # Weight by rule importance if available
        compliance_score = sum(rule_scores) / len(rule_scores) if rule_scores else 1.0
        
        return {
            "compliance_score": compliance_score,
            "violations": violations,
            "rule_count": len(interpreted_rules)
        }
    
    def _evaluate_rule(self, content, rule):
        """Evaluate content against a specific rule"""
        # Rule evaluation logic would depend on rule type
        # This is a simplified placeholder
        return {
            "score": 0.9,  # Placeholder score
            "locations": []  # Locations of potential violations
        }
    
    def _format_compliance_error(self, violations, content_type):
        """Format compliance error message from violations"""
        if not violations:
            return "Unspecified compliance error"
            
        # For prompt violations, provide specific guidance
        if content_type == "prompt":
            high_severity = [v for v in violations if v.get("severity") == "high"]
            if high_severity:
                v = high_severity[0]
                return f"Prompt contains prohibited content: {v.get('rule_description', 'regulatory violation')}"
            else:
                return "Prompt contains content that may not comply with regulatory requirements"
                
        # For generated content
        return f"Generated content does not meet compliance requirements ({len(violations)} violations)"
    
    def _deduplicate_violations(self, violations):
        """Remove duplicate violations based on rule_id and location"""
        unique_violations = {}
        
        for violation in violations:
            key = f"{violation.get('rule_id')}:{str(violation.get('locations', []))}"
            if key not in unique_violations:
                unique_violations[key] = violation
                
        return list(unique_violations.values())
    
    def _generate_cache_key(self, content, content_type, compliance_mode):
        """Generate cache key for verification results"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_hash}:{content_type}:{compliance_mode}"
        
    def _evaluate_rule(self, content, rule):
        """
        Evaluate content against a specific rule with proper policy enforcement
        
        Args:
            content: Input content to evaluate
            rule: Rule definition to check against
            
        Returns:
            Dict with compliance score and locations of potential violations
        """
        rule_type = rule.get("type", "keyword")
        rule_id = rule.get("id", "unknown")
        locations = []
        
        # Different evaluation strategies based on rule type
        if rule_type == "regex":
            # Apply regex pattern matching
            pattern = rule.get("pattern", "")
            try:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                score = 1.0 if not matches else 0.0
                
                # Extract match locations
                for match in matches:
                    locations.append({
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group(0)
                    })
                    
            except re.error:
                logging.error(f"Invalid regex pattern in rule {rule_id}: {pattern}")
                score = 1.0  # Consider compliant if pattern is invalid
                
        elif rule_type == "keyword":
            # Check for prohibited keywords
            keywords = rule.get("keywords", [])
            matches = []
            
            for keyword in keywords:
                # Check for whole word matches if specified
                if rule.get("whole_word", False):
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    keyword_matches = list(re.finditer(pattern, content, re.IGNORECASE))
                else:
                    # Simple substring matching
                    keyword_lower = keyword.lower()
                    content_lower = content.lower()
                    
                    # Find all occurrences
                    start = 0
                    keyword_matches = []
                    while True:
                        start = content_lower.find(keyword_lower, start)
                        if start == -1:
                            break
                        keyword_matches.append(type('obj', (object,), {
                            'start': start,
                            'end': start + len(keyword_lower),
                            'group': lambda s=keyword: s
                        }))
                        start += 1
                
                # Add to overall matches
                matches.extend(keyword_matches)
                
            # Calculate score and locations
            score = 1.0 if not matches else max(0.0, 1.0 - (len(matches) * 0.2))
            
            # Extract match locations
            for match in matches:
                locations.append({
                    "start": match.start,
                    "end": match.end,
                    "matched_text": match.group()
                })
                
        elif rule_type == "semantic":
            # Apply semantic rule evaluation
            # In a real implementation, this would use embeddings or ML models
            threshold = rule.get("threshold", 0.7)
            concept = rule.get("concept", "")
            
            # Check if entity extraction is available
            entities = []
            if hasattr(self, 'entity_extractor'):
                extraction_result = self.entity_extractor.extract(content)
                entities = extraction_result.get('entities', [])
                
            # Check for semantic match
            semantic_match = False
            for entity in entities:
                if entity.get('type') == concept:
                    semantic_match = True
                    locations.append({
                        "start": entity.get('start', 0),
                        "end": entity.get('end', 0),
                        "matched_text": entity.get('text', ''),
                        "entity_type": entity.get('type', '')
                    })
                    
            # Assign score based on semantic match
            score = 0.0 if semantic_match else 1.0
            
        elif rule_type == "entity":
            # Check for specific entity types
            prohibited_entities = rule.get("prohibited_entities", [])
            
            # Check if entity extraction is available
            entities = []
            if hasattr(self, 'entity_extractor'):
                extraction_result = self.entity_extractor.extract(content)
                entities = extraction_result.get('entities', [])
            
            # Check for prohibited entities
            matches = []
            for entity in entities:
                if entity.get('type') in prohibited_entities:
                    matches.append(entity)
                    locations.append({
                        "start": entity.get('start', 0),
                        "end": entity.get('end', 0),
                        "matched_text": entity.get('text', ''),
                        "entity_type": entity.get('type', '')
                    })
                    
            # Calculate score based on entity matches
            score = 1.0 if not matches else max(0.0, 1.0 - (len(matches) * 0.25))
        
        else:
            # Unknown rule type, default to compliant
            logging.warning(f"Unknown rule type '{rule_type}' for rule {rule_id}")
            score = 1.0
        
        # Apply rule severity to score
        severity_factor = self._get_severity_factor(rule.get("severity", "medium"))
        final_score = score ** severity_factor  # Lower score for higher severity rules
        
        return {
            "score": final_score,
            "locations": locations
        }

    def _get_severity_factor(self, severity):
        """Convert rule severity to numerical factor for score adjustment"""
        severity_factors = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "critical": 3.0
        }
        return severity_factors.get(severity, 1.0)