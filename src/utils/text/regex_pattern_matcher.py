import re
import logging
import uuid
from compliance.models.compliance_issue import ComplianceIssue

class RegexPatternMatcher:
    """
    Matches text against predefined regex patterns for policy enforcement.
    """
    def __init__(self, patterns):
        self.patterns = []
        
        # Compile regex patterns
        for pattern_def in patterns:
            try:
                compiled = re.compile(pattern_def["pattern"], re.IGNORECASE)
                pattern_def["compiled"] = compiled
                self.patterns.append(pattern_def)
            except re.error as e:
                logging.error(f"Failed to compile regex pattern '{pattern_def.get('name')}': {str(e)}")
    
    def check_patterns(self, text, context=None):
        """
        Check text against configured regex patterns.
        
        Args:
            text: Input text to check
            context: Optional context information
            
        Returns:
            Dict with pattern matching results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        for pattern_def in self.patterns:
            compiled = pattern_def.get("compiled")
            if not compiled:
                continue
                
            matches = list(compiled.finditer(text))
            if matches:
                is_compliant = False
                for match in matches:
                    issues.append(ComplianceIssue(
                        rule_id=pattern_def.get("id", str(uuid.uuid4())),
                        severity=pattern_def.get("severity", "medium"),
                        description=pattern_def.get("description", "Pattern match detected"),
                        location={"start": match.start(), "end": match.end()},
                        metadata={"pattern_name": pattern_def.get("name", "")}
                    ))
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }
