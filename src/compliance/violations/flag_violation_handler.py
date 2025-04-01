
class FlagViolationHandler:
    """Handler for flagging violations"""
    
    def handle(self, content, violations):
        """Handle flag violations by adding warning"""
        # Simple implementation: add warning prefix to content
        if isinstance(content, str):
            warning = "[WARNING: This content has been flagged for potential compliance issues. "\
                     "Please review carefully.]\n\n"
            modified_content = warning + content
            
            return {
                'content': modified_content,
                'is_modified': True,
                'handler': 'flag',
                'violation_count': len(violations),
                'warnings': [v.get('description', 'Compliance issue') for v in violations]
            }
        else:
            # Can't modify non-string content, just return metadata
            return {
                'content': content,
                'is_modified': False,
                'handler': 'flag',
                'violation_count': len(violations),
                'warnings': [v.get('description', 'Compliance issue') for v in violations]
            }
