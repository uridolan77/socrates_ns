from src.compliance.models.compliance_issue import ComplianceIssue
import re
import logging

class ContentComplianceDetector:
    """
    Detects content that violates compliance policies.
    """
    def __init__(self, config):
        self.config = config
        self.policy_rules = config.get("policy_rules", {})
        self.compliance_threshold = config.get("compliance_threshold", 0.7)
        self.rules_enabled = config.get("enabled_rules", [])
        
    def check_compliance(self, text, context=None):
        """
        Check if the provided text complies with all defined policy rules.
        
        Args:
            text: Input text to check
            context: Optional context information
            
        Returns:
            Dict with compliance results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        # Process each enabled rule
        for rule_id in self.rules_enabled:
            if rule_id not in self.policy_rules:
                continue
                
            rule = self.policy_rules[rule_id]
            violation_found = self._check_rule(text, rule, context)
            
            if violation_found:
                is_compliant = False
                issues.append(ComplianceIssue(
                    rule_id=rule_id,
                    severity=rule.get("severity", "medium"),
                    description=rule.get("description", "Policy violation detected"),
                    metadata={"rule_name": rule.get("name", "")}
                ))
                
                # Stop checking if configured to break on first violation
                if self.config.get("break_on_first_violation", False):
                    break
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }
        
    def _check_rule(self, text, rule, context=None):
        """Check if text violates a specific rule."""
        rule_type = rule.get("type", "keyword")
        
        if rule_type == "keyword":
            keywords = rule.get("keywords", [])
            return any(keyword.lower() in text.lower() for keyword in keywords)
            
        elif rule_type == "regex":
            pattern = rule.get("pattern", "")
            try:
                return bool(re.search(pattern, text, re.IGNORECASE))
            except re.error:
                logging.error(f"Invalid regex pattern: {pattern}")
                return False
                
        elif rule_type == "custom":
            # Placeholder for custom rule implementation
            # In a real system, this might call an external API or model
            return False
            
        return False

