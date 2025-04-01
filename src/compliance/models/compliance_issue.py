import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

class ComplianceIssue:
    """Data class for storing compliance issues found during filtering."""
    rule_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    location: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
