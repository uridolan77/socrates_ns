
class RuleBasedEntityClassifier:
    """Rule-based classifier for entity types"""
    
    def __init__(self, config):
        self.config = config
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize classification rules"""
        # Default rules
        rules = {
            'PII': {
                'keywords': ['email', 'phone', 'address', 'name', 'birth', 'ssn', 'social security',
                           'passport', 'license', 'credit card', 'bank account'],
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
                    r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN
                    r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # phone
                ]
            },
            'PHI': {
                'keywords': ['patient', 'diagnosis', 'treatment', 'medication', 'medical record',
                           'health', 'doctor', 'hospital', 'clinic', 'prescription'],
                'patterns': [
                    r'\b(?:MRN|Medical Record Number)[:\s]*\d{4,10}\b',  # MRN
                    r'\b(?:Patient ID|PID)[:\s]*\d{4,10}\b',  # Patient ID
                ]
            },
            'ORGANIZATION': {
                'keywords': ['company', 'corporation', 'organization', 'enterprise', 'firm', 
                           'inc', 'llc', 'corp', 'ltd'],
                'ner_types': ['ORG', 'ORGANIZATION']
            },
            'PERSON': {
                'keywords': ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof'],
                'ner_types': ['PERSON', 'PER']
            },
            'LOCATION': {
                'keywords': ['street', 'avenue', 'boulevard', 'road', 'lane', 'drive', 
                           'place', 'court', 'plaza', 'square', 'city', 'state', 'country'],
                'ner_types': ['LOC', 'LOCATION', 'GPE']
            }
        }
        
        # Add custom rules from config
        if 'entity_classification_rules' in self.config:
            for entity_type, rule_def in self.config['entity_classification_rules'].items():
                if entity_type in rules:
                    # Update existing rule
                    for key, value in rule_def.items():
                        if key in rules[entity_type]:
                            rules[entity_type][key].extend(value)
                        else:
                            rules[entity_type][key] = value
                else:
                    # Add new rule
                    rules[entity_type] = rule_def
        
        return rules
    
    def classify(self, entity, text, embeddings=None):
        """
        Classify entity based on rules
        
        Args:
            entity: Entity to classify
            text: Full text for context
            embeddings: Optional embeddings for semantic analysis
            
        Returns:
            Tuple of (entity_type, confidence)
        """
        entity_text = entity['text']
        entity_context = self._get_entity_context(entity, text)
        
        scores = {}
        
        # Check if already classified by NER
        if 'ner_type' in entity:
            ner_type = entity['ner_type']
            for entity_type, rule in self.rules.items():
                if 'ner_types' in rule and ner_type in rule['ner_types']:
                    scores[entity_type] = 0.8  # High confidence from NER
        
        # Apply rules for each entity type
        for entity_type, rule in self.rules.items():
            score = 0.0
            
            # Check keywords in entity text
            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    if keyword.lower() in entity_text.lower() or keyword.lower() in entity_context.lower():
                        score += 0.3
                        break
            
            # Check patterns
            if 'patterns' in rule:
                for pattern in rule['patterns']:
                    import re
                    if re.search(pattern, entity_text, re.IGNORECASE):
                        score += 0.7
                        break
            
            # Store score if positive
            if score > 0:
                scores[entity_type] = min(1.0, score)
        
        # Find highest scoring type
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return best_type
        
        # Default classification
        if 'PERSON' in entity.get('ner_type', ''):
            return 'PERSON', 0.7
        elif 'ORG' in entity.get('ner_type', ''):
            return 'ORGANIZATION', 0.7
        elif 'LOC' in entity.get('ner_type', '') or 'GPE' in entity.get('ner_type', ''):
            return 'LOCATION', 0.7
        else:
            return 'UNKNOWN', 0.5
    
    def _get_entity_context(self, entity, text, window_size=50):
        """Get the context around an entity"""
        start = max(0, entity['start'] - window_size)
        end = min(len(text), entity['end'] + window_size)
        return text[start:end]
