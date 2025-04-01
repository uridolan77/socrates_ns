
class ModifyContentViolationHandler:
    """Handler for modifying content to address violations"""
    
    def handle(self, content, violations):
        """Handle modify violations by adjusting content"""
        if not isinstance(content, str):
            # Can only modify string content
            return {
                'content': content,
                'is_modified': False,
                'handler': 'modify',
                'violation_count': len(violations),
                'error': 'Cannot modify non-string content'
            }
            
        # Identify content to modify
        modified_content = content
        
        # Apply simple modifications based on violation types
        for violation in violations:
            violation_type = violation.get('type', '')
            
            if 'harmful_content' in violation_type:
                # Add disclaimer for harmful content
                disclaimer = "\n[DISCLAIMER: This content may contain potentially harmful information " \
                           "and should be reviewed carefully.]\n"
                if disclaimer not in modified_content:
                    modified_content = disclaimer + modified_content
                    
            elif 'data_minimization' in violation_type:
                # Add note about data minimization
                note = "\n[NOTE: Please ensure only necessary personal data is collected " \
                     "in accordance with data minimization principles.]\n"
                if note not in modified_content:
                    modified_content = modified_content + note
                    
            elif 'consent' in violation_type:
                # Add consent reminder
                reminder = "\n[REMINDER: Ensure proper consent is obtained before processing " \
                         "any personal data mentioned.]\n"
                if reminder not in modified_content:
                    modified_content = modified_content + reminder
                    
        return {
            'content': modified_content,
            'is_modified': modified_content != content,
            'handler': 'modify',
            'violation_count': len(violations)
        }
