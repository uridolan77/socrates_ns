
import re

class AnonymizeViolationHandler:
    """Handler for anonymizing sensitive information"""
    
    def handle(self, content, violations):
        """Handle anonymize violations by anonymizing sensitive information"""
        if not isinstance(content, str):
            # Can only anonymize string content
            return {
                'content': content,
                'is_modified': False,
                'handler': 'anonymize',
                'violation_count': len(violations),
                'error': 'Cannot anonymize non-string content'
            }
            
        # Collect entity types to anonymize
        entity_types = set()
        for violation in violations:
            if 'entity_type' in violation:
                entity_types.add(violation['entity_type'])
            elif violation.get('source') == 'rule' and 'entity_types' in violation:
                entity_types.update(violation['entity_types'])
                
        if not entity_types:
            # Default to PII and PHI
            entity_types = {'PII', 'PHI'}
            
        # Anonymize entities of specified types
        modified_content = content
        replacements = []
        
        # Detect and replace PII patterns
        pii_patterns = {
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            'ssn': (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN]'),
            'phone': (r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
            'date': (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]'),
            'credit_card': (r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', '[CREDIT_CARD]')
        }
        
        for pii_type, (pattern, replacement) in pii_patterns.items():
            matches = re.finditer(pattern, modified_content)
            for match in matches:
                replacements.append((match.start(), match.end(), replacement))
                
        # Apply replacements in reverse order to avoid index shifting
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            modified_content = modified_content[:start] + replacement + modified_content[end:]
            
        return {
            'content': modified_content,
            'is_modified': modified_content != content,
            'handler': 'anonymize',
            'violation_count': len(violations),
            'anonymized_entities': len(replacements)
        }
