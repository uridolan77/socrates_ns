
import re
import logging
# Checker components used by the Token-Level Compliance Gate

class NGramComplianceChecker:
    """Checks n-gram patterns for compliance violations."""
    def __init__(self, compliance_config):
        self.config = compliance_config
        
    def check_compliance(self, token_text, hypothetical_text, constraints):
        """
        Check if adding token would create prohibited n-grams.
        
        Args:
            token_text: The token text being considered
            hypothetical_text: Text if token is added
            constraints: Compliance constraints to check
            
        Returns:
            Dict with is_compliant flag and compliance score
        """
        # Extract n-gram constraints
        ngram_constraints = [c for c in constraints if c.get("type") == "ngram"]
        if not ngram_constraints:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Calculate maximum n-gram size to check
        max_n = max([c.get("n", 3) for c in ngram_constraints], default=3)
        
        # Generate n-grams from hypothetical text
        # Focus on the tail of the text where the new token was added
        text_to_check = hypothetical_text[-max_n*10:]  # Only check recent context
        ngrams = self._generate_ngrams(text_to_check, max_n)
        
        # Check each n-gram against constraints
        violations = []
        for constraint in ngram_constraints:
            n = constraint.get("n", 3)
            
            # Get prohibited patterns for this constraint
            prohibited_patterns = constraint.get("prohibited_patterns", [])
            prohibited_regex = constraint.get("prohibited_regex", [])
            
            # Compile regex patterns if they're strings
            compiled_regex = []
            for pattern in prohibited_regex:
                try:
                    if isinstance(pattern, str):
                        compiled_regex.append(re.compile(pattern, re.IGNORECASE))
                    else:
                        compiled_regex.append(pattern)  # Already compiled
                except re.error:
                    logging.warning(f"Invalid regex pattern: {pattern}")
            
            # Check against exact patterns
            for ngram in ngrams.get(n, []):
                ngram_text = " ".join(ngram)
                
                # Check exact matches
                if ngram_text.lower() in [p.lower() for p in prohibited_patterns]:
                    violations.append({
                        "constraint_id": constraint.get("id", "unknown"),
                        "ngram": ngram_text,
                        "n": n,
                        "match_type": "exact",
                        "severity": constraint.get("severity", "medium")
                    })
                    continue
                    
                # Check regex matches
                for pattern in compiled_regex:
                    if pattern.search(ngram_text):
                        violations.append({
                            "constraint_id": constraint.get("id", "unknown"),
                            "ngram": ngram_text,
                            "n": n,
                            "match_type": "regex",
                            "pattern": pattern.pattern,
                            "severity": constraint.get("severity", "medium")
                        })
                        break
        
        # Calculate compliance score
        if not violations:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Calculate score based on violation severity
        total_severity = sum(self._severity_to_value(v["severity"]) for v in violations)
        compliance_score = max(0.0, 1.0 - (total_severity / 10.0))
        
        # Determine compliance based on threshold and strictness
        threshold = self.config.get("ngram_compliance_threshold", 0.7)
        is_compliant = compliance_score >= threshold
        
        return {
            'is_compliant': is_compliant,
            'compliance_score': compliance_score,
            'violations': violations,
            'ngram_details': {
                'checked_ngrams': sum(len(grams) for grams in ngrams.values()),
                'max_n': max_n
            }
        }

    def _generate_ngrams(self, text, max_n):
        """
        Generate n-grams from text up to max_n.
        
        Args:
            text: Input text to process
            max_n: Maximum n-gram size
            
        Returns:
            Dict with n-grams grouped by size
        """
        # Tokenize text (simple whitespace tokenization)
        # In a full implementation, this would use proper tokenization
        words = text.split()
        
        # Generate n-grams for each size up to max_n
        ngrams = {}
        for n in range(1, min(max_n + 1, len(words) + 1)):
            ngrams[n] = [words[i:i+n] for i in range(len(words) - n + 1)]
        
        return ngrams

    def _severity_to_value(self, severity):
        """Convert severity string to numerical value"""
        severity_map = {
            'low': 1.0,
            'medium': 3.0,
            'high': 7.0,
            'critical': 10.0
        }
        return severity_map.get(severity.lower(), 3.0)