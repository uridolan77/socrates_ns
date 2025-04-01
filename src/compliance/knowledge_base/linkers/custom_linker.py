import logging

class CustomKnowledgeBaseLinker:
    """Entity linker using custom knowledge base"""
    
    def __init__(self, config):
        self.config = config
        self.kb_path = config.get('custom_kb_path')
        self.kb = self._load_knowledge_base()
        
    def _load_knowledge_base(self):
        """Load custom knowledge base"""
        if not self.kb_path:
            return {}
            
        try:
            import json
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except:
            logging.warning(f"Could not load custom knowledge base from {self.kb_path}")
            return {}
    
    def link(self, entity, text):
        """Link entity to custom KB entry"""
        if not self.kb:
            return None
            
        entity_text = entity['text'].lower()
        entity_type = entity.get('type', entity.get('ner_type', 'UNKNOWN'))
        
        # Search for direct match
        if entity_text in self.kb:
            kb_entry = self.kb[entity_text]
            return {
                'kb_id': kb_entry.get('id'),
                'kb_source': 'custom',
                'kb_type': kb_entry.get('type'),
                'kb_confidence': 0.95
            }
        
        # Search for partial match
        for kb_entity, kb_entry in self.kb.items():
            if (entity_text in kb_entity.lower() or kb_entity.lower() in entity_text) and \
               (kb_entry.get('type') == entity_type):
                return {
                    'kb_id': kb_entry.get('id'),
                    'kb_source': 'custom',
                    'kb_type': kb_entry.get('type'),
                    'kb_confidence': 0.7
                }
        
        return None

