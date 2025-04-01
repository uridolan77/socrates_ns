from src.utils.error.compliance_error import ComplianceError, ErrorCategory, ErrorSeverity

class ComplianceErrorHandler:
    """Centralized error handling with recovery mechanisms"""
    
    def __init__(self, logger, config=None):
        self.logger = logger
        self.config = config or {}
        self.retry_counts = {}
        
    def handle_error(self, error, context=None, retry_key=None, 
                     max_retries=3, can_recover=False):
        """
        Handle errors with retry logic and appropriate logging
        
        Args:
            error: Error object or exception
            context: Processing context
            retry_key: Key for tracking retries
            max_retries: Maximum retry attempts
            can_recover: Whether recovery is possible
            
        Returns:
            Tuple of (should_retry, error_response)
        """
        # Convert exception to standardized error if needed
        if isinstance(error, Exception) and not isinstance(error, ComplianceError):
            error = ComplianceError.from_exception(
                error, 
                category=ErrorCategory.PROCESSING,
                source_component=context.get("component") if context else None
            )
            
        # Log based on severity
        self._log_error(error)
        
        # Check if we should retry
        should_retry = False
        if retry_key and self._is_retriable(error):
            current_retries = self.retry_counts.get(retry_key, 0)
            if current_retries < max_retries:
                should_retry = True
                self.retry_counts[retry_key] = current_retries + 1
                self.logger.info(f"Retrying operation {retry_key}, attempt {current_retries + 1}/{max_retries}")
        
        # Create appropriate response
        error_response = self._create_error_response(error, context, can_recover)
        
        return should_retry, error_response
    
    def _log_error(self, error):
        """Log error with appropriate level based on severity"""
        message = f"{error.category.value.upper()} ERROR: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message, extra={"error_details": error.to_dict()})
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(message, extra={"error_details": error.to_dict()})
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(message, extra={"error_details": error.to_dict()})
        else:
            self.logger.info(message, extra={"error_details": error.to_dict()})
    
    def _is_retriable(self, error):
        """Determine if error is retriable"""
        # Network and transient errors are retriable
        retriable_categories = [
            ErrorCategory.GATEWAY, 
            ErrorCategory.SYSTEM
        ]
        
        if error.category in retriable_categories:
            return True
            
        # Some validation errors might be retriable with different parameters
        if error.recoverable:
            return True
            
        return False
        
    def _create_error_response(self, error, context, can_recover):
        """Create standardized error response"""
        response = {
            "is_compliant": False,
            "error": error.to_dict(),
            "metadata": context or {}
        }
        
        # For recoverable errors, add partial results if available
        if can_recover and context and "partial_results" in context:
            response["partial_results"] = context["partial_results"]
            
        return response
    
    def reset_retries(self, retry_key=None):
        """Reset retry counters"""
        if retry_key:
            if retry_key in self.retry_counts:
                del self.retry_counts[retry_key]
        else:
            self.retry_counts = {}
