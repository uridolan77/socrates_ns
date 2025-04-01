import re
import logging
from compliance.models.compliance_issue import ComplianceIssue
from compliance.filtering.sensitive_token_detector import SensitiveTokenDetector

class SensitiveDataDetector:
    """
    Detects sensitive data patterns like PII in text.
    """
    def __init__(self, config):
        self.config = config
        
        # Initialize PII detection patterns
        self.pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
            "phone": re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        }
        
        # Add custom patterns from config
        custom_patterns = config.get("pii_patterns", {})
        for pattern_name, pattern_string in custom_patterns.items():
            try:
                self.pii_patterns[pattern_name] = re.compile(pattern_string)
            except re.error:
                logging.error(f"Invalid PII regex pattern for {pattern_name}: {pattern_string}")
    
    def detect(self, text, context=None):
        """
        Detect sensitive data in text.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Dict with detection results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        modified_text = text
        modified = False
        
        # Detect PII based on patterns
        for pii_type, pattern in self.pii_patterns.items():
            matches = list(pattern.finditer(text))
            
            for match in matches:
                is_compliant = False
                
                # Create issue
                issues.append(ComplianceIssue(
                    rule_id=f"pii_{pii_type}",
                    severity="high",
                    description=f"Detected {pii_type} in text",
                    location={"start": match.start(), "end": match.end()},
                    metadata={"pii_type": pii_type}
                ))
                
                # Redact PII if configured
                if self.config.get("redact_pii", False):
                    redaction = self.config.get("redaction_string", "[REDACTED]")
                    modified_text = modified_text[:match.start()] + redaction + modified_text[match.end():]
                    modified = True
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": modified_text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": modified
        }