import logging

class LogViolationHandler:
    """Handler for logging violations"""
    
    def handle(self, content, violations):
        """Handle log violations by logging them"""
        # Log violations
        for violation in violations:
            logging.warning(f"Compliance violation detected: {violation.get('description', 'No description')}")
            
        return {
            'content': content,
            'is_modified': False,
            'handler': 'log',
            'violation_count': len(violations)
        }
