from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import traceback

class ErrorSeverity(Enum):
    INFO = "info"              # Non-critical information
    WARNING = "warning"        # Process can continue but with caution
    ERROR = "error"            # Process cannot continue but system is stable
    CRITICAL = "critical"      # System stability may be compromised

class ErrorCategory(Enum):
    COMPLIANCE = "compliance"  # Regulatory compliance issues
    VALIDATION = "validation"  # Input validation failures
    PROCESSING = "processing"  # Processing errors
    SYSTEM = "system"          # System/infrastructure errors
    GATEWAY = "gateway"        # Model gateway errors
    SECURITY = "security"      # Security-related errors

@dataclass
class ComplianceError:
    """Standardized error structure for compliance system"""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    code: str = None  # e.g., "COMP-101"
    details: Dict[str, Any] = field(default_factory=dict)
    source_component: str = None
    recoverable: bool = False
    stack_trace: Optional[str] = None
    related_frameworks: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    
    @classmethod
    def from_exception(cls, e: Exception, category: ErrorCategory, 
                      severity: ErrorSeverity = ErrorSeverity.ERROR,
                      source_component: str = None, **kwargs):
        """Create error from exception with stack trace"""
        return cls(
            message=str(e),
            category=category,
            severity=severity,
            source_component=source_component,
            stack_trace=traceback.format_exc(),
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response formatting"""
        result = {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "source": self.source_component
        }
        
        if self.code:
            result["code"] = self.code
        
        if self.details:
            result["details"] = self.details
            
        if self.related_frameworks:
            result["related_frameworks"] = self.related_frameworks
            
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        
        # Only include stack trace in non-production environments
        if self.stack_trace and self._is_development_mode():
            result["stack_trace"] = self.stack_trace
            
        return result
        
    def _is_development_mode(self) -> bool:
        """Check if system is in development mode"""
        import os
        return os.environ.get("ENVIRONMENT", "production").lower() in ("development", "dev", "test")
