import logging
import re
import json
import stanza
import spacy
from src.ner.linkers import WikidataEntityLinker, CustomKnowledgeBaseLinker  # Import WikidataEntityLinker and CustomKnowledgeBaseLinker

from src.ner.classifiers import RuleBasedEntityClassifier, MLEntityClassifier

class EntityExtractor:
    """
    Enhanced entity extraction with proper NER, classification and regulatory linking
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.ner_model = self._initialize_ner_model()
        self.entity_classifier = self._initialize_entity_classifier()
        self.entity_linker = self._initialize_entity_linker()
        self.pii_patterns = self._compile_pii_patterns()
        self.phi_patterns = self._compile_phi_patterns()
        
    def _initialize_ner_model(self):
        """Initialize NER model based on configuration"""
        ner_type = self.config.get("ner_type", "spacy")
        
        if ner_type == "spacy":
            try:
                # Load appropriate spaCy model based on language
                lang = self.config.get("language", "en")
                if lang == "en":
                    return spacy.load("en_core_web_trf")  # Use transformer model for better accuracy
                else:
                    return spacy.load(f"{lang}_core_news_lg")
            except ImportError:
                logging.warning("spaCy not installed, falling back to regex patterns")
                return None
        elif ner_type == "stanza":
            try:
                lang = self.config.get("language", "en")
                return stanza.Pipeline(lang=lang, processors='tokenize,ner')
            except ImportError:
                logging.warning("Stanza not installed, falling back to regex patterns")
                return None
        else:
            return None
            
    def _initialize_entity_classifier(self):
        """Initialize entity classifier model"""
        classifier_type = self.config.get("entity_classifier", "rule_based")
        
        if classifier_type == "rule_based":
            return RuleBasedEntityClassifier(self.config)
        elif classifier_type == "ml_based":
            return MLEntityClassifier(self.config)
        else:
            return RuleBasedEntityClassifier(self.config)  # Default
    
    def _initialize_entity_linker(self):
        """Initialize entity linker to knowledge base"""
        linker_type = self.config.get("entity_linker", "none")
        
        if linker_type == "wikidata":
            return WikidataEntityLinker(self.config)
        elif linker_type == "custom_kb":
            return CustomKnowledgeBaseLinker(self.config)
        else:
            return None
    
    def _compile_pii_patterns(self):
        """Compile regex patterns for PII detection"""
        import re
        patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ssn': re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
            'phone': re.compile(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'date_of_birth': re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
        }
        return patterns
    
    def _compile_phi_patterns(self):
        """Compile regex patterns for PHI detection (healthcare)"""
        import re
        patterns = {
            'mrn': re.compile(r'\b(?:MRN|Medical Record Number)[:\s]*\d{4,10}\b', re.IGNORECASE),
            'patient_id': re.compile(r'\b(?:Patient ID|PID)[:\s]*\d{4,10}\b', re.IGNORECASE),
            'diagnosis_code': re.compile(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'),  # ICD-10 format
            'npi': re.compile(r'\bNPI[:\s]*\d{10}\b', re.IGNORECASE),  # National Provider Identifier
        }
        return patterns
    
    def extract(self, text, embeddings=None, compliance_mode='strict'):
        """
        Extract entities from text with proper NER and classification
        
        Args:
            text: Input text
            embeddings: Optional text embeddings (for semantic analysis)
            compliance_mode: 'strict' or 'soft' enforcement mode
            
        Returns:
            Dictionary with extracted entities and compliance information
        """
        # Run NER to identify base entities
        entities = self._extract_entities(text)
        
        # Run regex pattern matching for specialized entity types
        regex_entities = self._extract_regex_entities(text)
        
        # Merge NER and regex entities with deduplication
        all_entities = self._merge_entities(entities, regex_entities)
        
        # Classify entities by type (including PII/PHI detection)
        classified_entities = self._classify_entities(all_entities, text, embeddings)
        
        # Link entities to knowledge base if linker available
        if self.entity_linker:
            classified_entities = self._link_entities(classified_entities, text)
        
        # Verify entity compliance
        compliant_entities = []
        violations = []
        for entity in classified_entities:
            entity_compliance = self._verify_entity_compliance(entity, compliance_mode)
            if entity_compliance['is_compliant']:
                # Add compliance score to entity
                entity['compliance_score'] = entity_compliance['compliance_score']
                compliant_entities.append(entity)
            else:
                # Record violation
                violations.append({
                    'entity': entity,
                    'compliance_error': entity_compliance['error'],
                    'severity': entity_compliance['severity']
                })
        
        # Determine overall compliance
        is_compliant = len(violations) == 0 or (
            compliance_mode == 'soft' and
            not any(v['severity'] == 'high' for v in violations)
        )
        
        return {
            'entities': compliant_entities,
            'is_compliant': is_compliant,
            'violations': violations if not is_compliant else [],
            'metadata': {
                'total_entities': len(classified_entities),
                'compliant_entities': len(compliant_entities),
                'violation_count': len(violations),
                'entity_types': self._count_entity_types(classified_entities)
            }
        }
    
    def _extract_entities(self, text):
        """Extract entities using NER model"""
        entities = []
        
        if self.ner_model:
            # Using spaCy NER
            if hasattr(self.ner_model, 'pipe_names') and 'ner' in self.ner_model.pipe_names:
                doc = self.ner_model(text)
                for ent in doc.ents:
                    entities.append({
                        'id': f"e{len(entities)}",
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'ner_type': ent.label_,
                        'detection_method': 'spacy_ner'
                    })
            # Using Stanza NER
            elif hasattr(self.ner_model, 'processors') and 'ner' in self.ner_model.processors:
                doc = self.ner_model(text)
                for sent in doc.sentences:
                    for ent in sent.ents:
                        entities.append({
                            'id': f"e{len(entities)}",
                            'text': ent.text,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'ner_type': ent.type,
                            'detection_method': 'stanza_ner'
                        })
        
        # If no NER model or no entities found, fallback to basic extraction
        if not entities:
            import re
            # Extract potential named entities (capitalized words)
            for match in re.finditer(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text):
                entities.append({
                    'id': f"e{len(entities)}",
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'ner_type': 'UNKNOWN',
                    'detection_method': 'regex_fallback'
                })
        
        return entities
    
    def _extract_regex_entities(self, text):
        """Extract entities using regex patterns"""
        entities = []
        
        # Extract PII entities
        for pii_type, pattern in self.pii_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'id': f"pii{len(entities)}",
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'ner_type': 'PII',
                    'pii_type': pii_type,
                    'detection_method': 'regex_pattern'
                })
        
        # Extract PHI entities
        for phi_type, pattern in self.phi_patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'id': f"phi{len(entities)}",
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'ner_type': 'PHI',
                    'phi_type': phi_type,
                    'detection_method': 'regex_pattern'
                })
        
        return entities
    
    def _merge_entities(self, ner_entities, regex_entities):
        """Merge NER and regex entities with deduplication"""
        all_entities = []
        used_spans = set()
        
        # First add NER entities
        for entity in ner_entities:
            span = (entity['start'], entity['end'])
            if span not in used_spans:
                all_entities.append(entity)
                used_spans.add(span)
        
        # Then add regex entities, avoiding duplicates
        for entity in regex_entities:
            span = (entity['start'], entity['end'])
            # Check for overlapping spans
            overlapping = False
            for used_span in used_spans:
                if (entity['start'] <= used_span[1] and entity['end'] >= used_span[0]):
                    overlapping = True
                    break
                    
            if not overlapping:
                all_entities.append(entity)
                used_spans.add(span)
        
        # Re-assign IDs to ensure sequential ordering
        for i, entity in enumerate(all_entities):
            entity['id'] = f"e{i}"
            
        return all_entities
    
    def _classify_entities(self, entities, text, embeddings=None):
        """Classify entities by type with enhanced analysis"""
        classified_entities = []
        
        for entity in entities:
            # Skip if already classified by regex patterns
            if 'pii_type' in entity or 'phi_type' in entity:
                classified_entities.append(entity)
                continue
                
            # Get classification from entity classifier
            entity_type, confidence = self.entity_classifier.classify(entity, text, embeddings)
            
            # Create classified entity
            classified_entity = entity.copy()
            classified_entity['type'] = entity_type
            classified_entity['classification_confidence'] = confidence
            
            # Add additional metadata for sensitive entities
            if entity_type in ['PII', 'PHI', 'sensitive']:
                classified_entity['sensitivity'] = 'high'
                classified_entity['requires_consent'] = True
                
                # Try to determine specific PII/PHI type if not already set
                if 'pii_type' not in classified_entity and 'phi_type' not in classified_entity:
                    specific_type = self._determine_specific_sensitive_type(classified_entity, text)
                    if specific_type:
                        if entity_type == 'PII':
                            classified_entity['pii_type'] = specific_type
                        else:
                            classified_entity['phi_type'] = specific_type
            
            classified_entities.append(classified_entity)
        
        return classified_entities
    
    def _determine_specific_sensitive_type(self, entity, text):
        """Determine specific sensitive data type"""
        entity_text = entity['text'].lower()
        entity_context = self._get_entity_context(entity, text)
        
        # Check for common PII types by context
        if any(kw in entity_context.lower() for kw in ['email', '@']):
            return 'email'
        elif any(kw in entity_context.lower() for kw in ['ssn', 'social security', 'social']):
            return 'ssn'
        elif any(kw in entity_context.lower() for kw in ['phone', 'mobile', 'cell', 'tel']):
            return 'phone'
        elif any(kw in entity_context.lower() for kw in ['birth', 'dob', 'born']):
            return 'date_of_birth'
        elif any(kw in entity_context.lower() for kw in ['address', 'street', 'ave', 'avenue']):
            return 'address'
        
        # Check for PHI types by context
        if any(kw in entity_context.lower() for kw in ['patient', 'mrn', 'medical record']):
            return 'mrn'
        elif any(kw in entity_context.lower() for kw in ['diagnosis', 'condition', 'disease']):
            return 'diagnosis'
        
        return None
    
    def _get_entity_context(self, entity, text, window_size=50):
        """Get the context around an entity"""
        start = max(0, entity['start'] - window_size)
        end = min(len(text), entity['end'] + window_size)
        return text[start:end]
    
    def _link_entities(self, entities, text):
        """Link entities to knowledge base entries"""
        if not self.entity_linker:
            return entities
            
        linked_entities = []
        for entity in entities:
            linked_entity = entity.copy()
            # Try to link entity
            link_result = self.entity_linker.link(entity, text)
            if link_result:
                linked_entity.update(link_result)
            linked_entities.append(linked_entity)
            
        return linked_entities
    
    def _verify_entity_compliance(self, entity, compliance_mode):
        """Verify if an entity complies with regulatory requirements"""
        entity_type = entity.get('type')
        
        # Always flag PII/PHI in strict mode unless explicitly allowed
        if entity_type in ['PII', 'PHI'] and compliance_mode == 'strict':
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'error': f"Protected {entity_type} entity detected: {entity['text']}",
                'severity': 'high'
            }
            
        # Check for specific categories
        if 'pii_type' in entity and entity['pii_type'] == 'credit_card' and compliance_mode == 'strict':
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'error': f"Credit card information detected: {entity['text']}",
                'severity': 'high'
            }
        
        # Calculate compliance score based on entity type and context
        compliance_score = self._calculate_entity_compliance_score(entity)
        
        is_compliant = compliance_score >= 0.7 or compliance_mode == 'soft'
        
        if not is_compliant:
            return {
                'is_compliant': False,
                'compliance_score': compliance_score,
                'error': f"Entity '{entity['text']}' of type {entity_type} has low compliance score",
                'severity': 'high' if entity_type in ['PII', 'PHI'] else 'medium'
            }
        
        return {
            'is_compliant': True,
            'compliance_score': compliance_score,
            'error': None,
            'severity': 'none'
        }
    
    def _calculate_entity_compliance_score(self, entity):
        """Calculate compliance score based on entity characteristics"""
        entity_type = entity.get('type')
        
        # Base compliance score by type
        if entity_type in ['PII', 'PHI']:
            base_score = 0.3
        elif entity_type in ['quasi-identifier', 'sensitive']:
            base_score = 0.6
        else:
            base_score = 0.9
            
        # Adjust score based on additional factors
        adjustments = 0.0
        
        # Adjust for specificity - more specific entities may be more sensitive
        if 'classification_confidence' in entity:
            confidence = entity['classification_confidence']
            # Higher confidence means we're more certain about the classification
            # For sensitive classes, this should lower the compliance score
            if entity_type in ['PII', 'PHI', 'sensitive', 'quasi-identifier']:
                adjustments -= (confidence - 0.5) * 0.2  # Max -0.1 for high confidence
            else:
                adjustments += (confidence - 0.5) * 0.2  # Max +0.1 for high confidence
                
        # Adjust for knowledge base linkage
        if 'kb_link' in entity:
            # Linked entities are more specific, thus potentially more sensitive
            if entity_type in ['PII', 'PHI', 'sensitive']:
                adjustments -= 0.1
        
        # Calculate final score with bounds
        final_score = max(0.0, min(1.0, base_score + adjustments))
        return final_score
    
    def _count_entity_types(self, entities):
        """Count entities by type for metadata"""
        counts = {}
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in counts:
                counts[entity_type] = 0
            counts[entity_type] += 1
        return counts
