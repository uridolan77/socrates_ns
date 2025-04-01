
# Violation handlers
class BlockViolationHandler:
    """Handler for blocking violations"""
    
    def handle(self, content, violations):
        """Handle block violations by blocking content"""
        # Simple implementation: replace content with block message
        block_message = "Content blocked due to compliance violations."
        
        # Add details of violations
        violation_details = []
        for violation in violations:
            detail = f"- {violation.get('type', 'Violation')}: {violation.get('description', 'No description')}"
            violation_details.append(detail)
            
        if violation_details:
            block_message += "\n\nViolation details:\n" + "\n".join(violation_details)
            
        return {
            'content': block_message,
            'is_modified': True,
            'handler': 'block',
            'violation_count': len(violations)
        }
