
import logging

import json
import os
import re
import numpy as np
import pandas as pd
import spacy
import benepar
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class RelationExtractor:
    """
    Enhanced relation extraction with dependency parsing and semantic role labeling
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.nlp_model = self._initialize_nlp_model()
        self.relation_patterns = self._initialize_relation_patterns()
        self.srl_model = self._initialize_srl_model()
        
    def _initialize_nlp_model(self):
        """Initialize NLP model for dependency parsing"""
        nlp_type = self.config.get("nlp_model", "spacy")
        
        if nlp_type == "spacy":
            try:
                # Load appropriate spaCy model with dependency parser
                lang = self.config.get("language", "en")
                if lang == "en":
                    return spacy.load("en_core_web_trf")
                else:
                    return spacy.load(f"{lang}_core_news_lg")
            except ImportError:
                logging.warning("spaCy not installed, falling back to basic patterns")
                return None
        elif nlp_type == "stanza":
            try:
                lang = self.config.get("language", "en")
                return stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,depparse')
            except ImportError:
                logging.warning("Stanza not installed, falling back to basic patterns")
                return None
        else:
            return None


    def _initialize_srl_model(self):
        """Initialize semantic role labeling using Hugging Face Transformers"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("vblagoje/bert-english-uncased-finetuned-srl")
            model = AutoModelForTokenClassification.from_pretrained("vblagoje/bert-english-uncased-finetuned-srl")
            srl_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            return srl_pipeline
        except Exception as e:
            logging.warning(f"Failed to initialize Hugging Face SRL pipeline: {str(e)}")
            return None
    
    def _initialize_relation_patterns(self):
        """Initialize relation extraction patterns"""
        # Default patterns for common relation types
        patterns = {
            'works_for': [
                {
                    'description': 'Person works for organization',
                    'entity_types': ['PERSON', 'ORGANIZATION'],
                    'dependency_paths': [
                        # employee -> works -> for -> company
                        [{'dep': 'nsubj'}, {'lemma': 'work'}, {'lemma': 'for'}, {'dep': 'pobj'}],
                        # employee -> is -> employed -> by -> company
                        [{'dep': 'nsubj'}, {'lemma': 'be'}, {'lemma': 'employ'}, {'lemma': 'by'}, {'dep': 'pobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:work|works|working|worked)\s+(?:for|at|with)\s+(\w+)",
                        r"(\w+)\s+(?:is|was|are|were)\s+(?:employed|hired)\s+by\s+(\w+)"
                    ]
                }
            ],
            'has_role': [
                {
                    'description': 'Person has role',
                    'entity_types': ['PERSON', None],
                    'dependency_paths': [
                        # person -> is -> role
                        [{'dep': 'nsubj'}, {'lemma': 'be'}, {'dep': 'attr'}],
                        # person -> serves -> as -> role
                        [{'dep': 'nsubj'}, {'lemma': 'serve'}, {'lemma': 'as'}, {'dep': 'pobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:is|as|was)\s+(?:the|a|an)?\s+(\w+)",
                        r"(\w+)\s+(?:serves|served|acting)\s+as\s+(?:the|a|an)?\s+(\w+)"
                    ]
                }
            ],
            'located_at': [
                {
                    'description': 'Entity located at location',
                    'entity_types': [None, 'LOCATION'],
                    'dependency_paths': [
                        # entity -> is -> located -> in -> location
                        [{'dep': 'nsubj'}, {'lemma': 'be'}, {'lemma': 'locate'}, {'lemma': 'in'}, {'dep': 'pobj'}],
                        # entity -> is -> in -> location
                        [{'dep': 'nsubj'}, {'lemma': 'be'}, {'lemma': 'in'}, {'dep': 'pobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:is|are|was|were)\s+(?:located|based|situated)\s+(?:in|at)\s+(\w+)",
                        r"(\w+)\s+(?:is|are|was|were)\s+in\s+(\w+)"
                    ]
                }
            ],
            'has_access_to': [
                {
                    'description': 'Person has access to data/system',
                    'entity_types': ['PERSON', None],
                    'dependency_paths': [
                        # person -> has -> access -> to -> data
                        [{'dep': 'nsubj'}, {'lemma': 'have'}, {'lemma': 'access'}, {'lemma': 'to'}, {'dep': 'pobj'}],
                        # person -> can -> access -> data
                        [{'dep': 'nsubj'}, {'lemma': 'can'}, {'lemma': 'access'}, {'dep': 'dobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:has|have|had)\s+access\s+to\s+(\w+)",
                        r"(\w+)\s+(?:can|could|may)\s+access\s+(\w+)"
                    ]
                }
            ],
            'part_of': [
                {
                    'description': 'Entity is part of larger entity',
                    'entity_types': [None, None],
                    'dependency_paths': [
                        # entity -> is -> part -> of -> whole
                        [{'dep': 'nsubj'}, {'lemma': 'be'}, {'lemma': 'part'}, {'lemma': 'of'}, {'dep': 'pobj'}],
                        # entity -> belongs -> to -> whole
                        [{'dep': 'nsubj'}, {'lemma': 'belong'}, {'lemma': 'to'}, {'dep': 'pobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:is|are|was|were)\s+(?:part|member)\s+of\s+(\w+)",
                        r"(\w+)\s+belongs\s+to\s+(\w+)"
                    ]
                }
            ],
            'discloses': [
                {
                    'description': 'Entity discloses information',
                    'entity_types': [None, None],
                    'dependency_paths': [
                        # entity -> discloses -> information
                        [{'dep': 'nsubj'}, {'lemma': 'disclose'}, {'dep': 'dobj'}],
                        # entity -> provides -> information -> to -> recipient
                        [{'dep': 'nsubj'}, {'lemma': 'provide'}, {'dep': 'dobj'}, {'lemma': 'to'}, {'dep': 'pobj'}]
                    ],
                    'regex_patterns': [
                        r"(\w+)\s+(?:discloses|disclosed|shares|shared)\s+(\w+)",
                        r"(\w+)\s+(?:provides|provided|gives|gave)\s+(\w+)\s+to\s+(\w+)"
                    ]
                }
            ]
        }
        
        # Add patterns from config if available
        custom_patterns = self.config.get("relation_patterns", {})
        for relation_type, patterns_list in custom_patterns.items():
            if relation_type in patterns:
                patterns[relation_type].extend(patterns_list)
            else:
                patterns[relation_type] = patterns_list
                
        return patterns
    
    def extract(self, text, entities, embeddings=None, compliance_mode='strict'):
        """
        Extract relations between entities with dependency parsing and SRL
        
        Args:
            text: Input text
            entities: Extracted entities
            embeddings: Optional text embeddings
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Dictionary with extracted relations and compliance information
        """
        # Extract relations using different methods
        relations = []
        
        # Use dependency parsing if available
        if self.nlp_model:
            dep_relations = self._extract_dependency_relations(text, entities)
            relations.extend(dep_relations)
        
        # Use semantic role labeling if available
        if self.srl_model:
            srl_relations = self._extract_srl_relations(text, entities)
            relations.extend(srl_relations)
        
        # Use regex patterns as fallback
        regex_relations = self._extract_regex_relations(text, entities)
        relations.extend(regex_relations)
        
        # Deduplicate relations
        relations = self._deduplicate_relations(relations)
        
        # Verify relation compliance
        compliant_relations = []
        violations = []
        for relation in relations:
            relation_compliance = self._verify_relation_compliance(relation, entities, compliance_mode)
            if relation_compliance['is_compliant']:
                # Add compliance score to relation
                relation['compliance_score'] = relation_compliance['compliance_score']
                compliant_relations.append(relation)
            else:
                # Record violation
                violations.append({
                    'relation': relation,
                    'compliance_error': relation_compliance['error'],
                    'severity': relation_compliance['severity']
                })
        
        # Determine overall compliance
        is_compliant = len(violations) == 0 or (
            compliance_mode == 'soft' and
            not any(v['severity'] == 'high' for v in violations)
        )
        
        return {
            'relations': compliant_relations,
            'is_compliant': is_compliant,
            'violations': violations if not is_compliant else [],
            'metadata': {
                'total_relations': len(relations),
                'compliant_relations': len(compliant_relations),
                'violation_count': len(violations),
                'relation_types': self._count_relation_types(relations)
            }
        }
    
    def _extract_dependency_relations(self, text, entities):
        """Extract relations using dependency parsing"""
        relations = []
        
        # Skip if no NLP model available
        if not self.nlp_model:
            return relations
            
        # Parse text with dependency parser
        doc = self.nlp_model(text)
        
        # Create entity span mapping for lookup
        entity_spans = {}
        for entity in entities:
            start = entity['start']
            end = entity['end']
            entity_spans[(start, end)] = entity
        
        # Extract relations using dependency patterns
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # Extract relations for each pattern
                for dep_path in pattern.get('dependency_paths', []):
                    # This is a simplified implementation
                    # A real implementation would follow the dependency paths
                    # and check if they connect entity pairs
                    
                    # For each sentence in the document
                    for sent in doc.sents:
                        # Find entities in this sentence
                        sent_entities = self._find_entities_in_span(
                            entities, sent.start_char, sent.end_char
                        )
                        
                        if len(sent_entities) < 2:
                            continue  # Need at least 2 entities for a relation
                            
                        # Check for compatible entity type pairs
                        for i, entity1 in enumerate(sent_entities):
                            for entity2 in sent_entities[i+1:]:
                                # Check entity type compatibility with pattern
                                if self._check_entity_type_match(
                                    [entity1, entity2], 
                                    pattern.get('entity_types', [None, None])
                                ):
                                    # Check if dependency path connects these entities
                                    if self._check_dependency_path(sent, entity1, entity2, dep_path):
                                        relation = self._create_relation(
                                            relation_type, entity1, entity2, 'dependency_parsing'
                                        )
                                        relations.append(relation)
        
        return relations
    
    def _extract_srl_relations(self, text, entities):
        """Extract relations using semantic role labeling"""
        relations = []
        
        # Skip if no SRL model available
        if not self.srl_model:
            return relations
            
        # Get SRL predictions
        predictions = self.srl_model.predict(text)
        
        # Process each predicted verb and its arguments
        for verb_info in predictions.get('verbs', []):
            # Get the verb and tags
            verb = verb_info.get('verb', '')
            tags = verb_info.get('tags', [])
            
            # Find arguments and their roles
            arguments = {}
            current_arg = None
            current_tag = None
            current_start = None
            
            for i, tag in enumerate(tags):
                if tag.startswith('B-'):
                    # Begin a new argument
                    if current_arg:
                        arguments[current_tag] = {
                            'text': current_arg,
                            'start': current_start,
                            'end': i
                        }
                    current_tag = tag[2:]  # Remove 'B-' prefix
                    current_arg = text.split()[i] if i < len(text.split()) else ''
                    current_start = i
                elif tag.startswith('I-'):
                    # Continue current argument
                    if current_arg and i < len(text.split()):
                        current_arg += ' ' + text.split()[i]
                elif tag == 'O':
                    # End current argument
                    if current_arg:
                        arguments[current_tag] = {
                            'text': current_arg,
                            'start': current_start,
                            'end': i
                        }
                        current_arg = None
                        current_tag = None
                        current_start = None
            
            # Add final argument if any
            if current_arg:
                arguments[current_tag] = {
                    'text': current_arg,
                    'start': current_start,
                    'end': len(tags)
                }
            
            # Extract relations based on semantic roles
            if 'ARG0' in arguments and 'ARG1' in arguments:
                # Find entities that overlap with ARG0 and ARG1
                arg0_entities = self._find_entities_overlapping(
                    entities, arguments['ARG0']['text']
                )
                arg1_entities = self._find_entities_overlapping(
                    entities, arguments['ARG1']['text']
                )
                
                # Create relations for each entity pair
                for entity1 in arg0_entities:
                    for entity2 in arg1_entities:
                        # Determine relation type from verb
                        relation_type = self._determine_relation_type_from_verb(verb)
                        if relation_type:
                            relation = self._create_relation(
                                relation_type, entity1, entity2, 'semantic_role_labeling'
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_regex_relations(self, text, entities):
        """Extract relations using regex patterns"""
        relations = []
        
        # Process each relation type and its patterns
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # Extract using regex patterns
                for regex_pattern in pattern.get('regex_patterns', []):
                    import re
                    matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Find entities that overlap with match groups
                        if match.lastindex >= 2:  # At least 2 groups needed for relation
                            # Get entities for first group
                            group1_start = match.start(1)
                            group1_end = match.end(1)
                            group1_entities = self._find_entities_in_span(
                                entities, group1_start, group1_end
                            )
                            
                            # Get entities for second group
                            group2_start = match.start(2)
                            group2_end = match.end(2)
                            group2_entities = self._find_entities_in_span(
                                entities, group2_start, group2_end
                            )
                            
                            # Create relations for each entity pair
                            for entity1 in group1_entities:
                                for entity2 in group2_entities:
                                    # Check entity type compatibility with pattern
                                    if self._check_entity_type_match(
                                        [entity1, entity2], 
                                        pattern.get('entity_types', [None, None])
                                    ):
                                        relation = self._create_relation(
                                            relation_type, entity1, entity2, 'regex_pattern'
                                        )
                                        relations.append(relation)
        
        return relations
    
    def _find_entities_in_span(self, entities, start, end):
        """Find entities that are within the given text span"""
        return [
            entity for entity in entities
            if entity['start'] >= start and entity['end'] <= end
        ]
    
    def _find_entities_overlapping(self, entities, text):
        """Find entities that overlap with the given text"""
        # Normalize text for comparison
        norm_text = text.lower()
        return [
            entity for entity in entities
            if entity['text'].lower() in norm_text or norm_text in entity['text'].lower()
        ]
    
    def _check_entity_type_match(self, entities, required_types):
        """Check if entity types match the required types for a relation pattern"""
        if len(entities) != len(required_types):
            return False
            
        for i, required_type in enumerate(required_types):
            if required_type is None:
                continue  # Any type is acceptable
                
            entity_type = entities[i].get('type', entities[i].get('ner_type', 'UNKNOWN'))
            if required_type != entity_type:
                return False
                
        return True
    
    def _check_dependency_path(self, sentence, entity1, entity2, dep_path):
        """Check if a dependency path connects two entities"""
        # This is a placeholder for a more sophisticated implementation
        # A real implementation would check if the dependency path connects the entities
        # by following the dependency relations in the parsed sentence
        
        # Simplified check - just look for the dependency relations in the sentence
        # without checking actual connectivity
        for token in sentence:
            for dep_item in dep_path:
                if 'dep' in dep_item and token.dep_ != dep_item['dep']:
                    continue
                if 'lemma' in dep_item and token.lemma_ != dep_item['lemma']:
                    continue
                # Found a matching token
                return True
                
        return False
    
    def _determine_relation_type_from_verb(self, verb):
        """Determine relation type based on verb"""
        verb_lower = verb.lower()
        
        # Map common verbs to relation types
        verb_mapping = {
            'work': 'works_for',
            'employ': 'works_for',
            'hire': 'works_for',
            'locate': 'located_at',
            'access': 'has_access_to',
            'use': 'has_access_to',
            'belong': 'part_of',
            'include': 'part_of',
            'disclose': 'discloses',
            'share': 'discloses',
            'provide': 'discloses',
            'send': 'discloses',
            'own': 'owns',
            'possess': 'owns'
        }
        
        # Check for exact matches
        if verb_lower in verb_mapping:
            return verb_mapping[verb_lower]
        
        # Check for partial matches
        for v, rel_type in verb_mapping.items():
            if v in verb_lower:
                return rel_type
                
        # Default to a generic relation type
        return 'related_to'
    
    def _create_relation(self, relation_type, source_entity, target_entity, extraction_method):
        """Create a relation between entities"""
        return {
            'id': f"r{source_entity['id']}_{target_entity['id']}",
            'type': relation_type,
            'source': source_entity['id'],
            'target': target_entity['id'],
            'source_text': source_entity['text'],
            'target_text': target_entity['text'],
            'extraction_method': extraction_method,
            'confidence': self._calculate_relation_confidence(
                relation_type, source_entity, target_entity, extraction_method
            )
        }
    
    def _calculate_relation_confidence(self, relation_type, source_entity, target_entity, method):
        """Calculate confidence score for extracted relation"""
        # Base confidence by extraction method
        if method == 'dependency_parsing':
            base_confidence = 0.8
        elif method == 'semantic_role_labeling':
            base_confidence = 0.85
        elif method == 'regex_pattern':
            base_confidence = 0.7
        else:
            base_confidence = 0.6
            
        # Adjust confidence based on entity types and relation
        entity_type_adjustment = 0.0
        
        # Check if entity types are compatible with relation type
        if relation_type == 'works_for':
            # Ideally, source should be PERSON and target should be ORGANIZATION
            source_type = source_entity.get('type', source_entity.get('ner_type', 'UNKNOWN'))
            target_type = target_entity.get('type', target_entity.get('ner_type', 'UNKNOWN'))
            
            if source_type == 'PERSON' and target_type == 'ORGANIZATION':
                entity_type_adjustment = 0.1
            elif source_type != 'PERSON' or target_type != 'ORGANIZATION':
                entity_type_adjustment = -0.2
                
        # Similar adjustments for other relation types could be added here
                
        # Adjust for entity confidence if available
        entity_confidence_adjustment = 0.0
        if 'classification_confidence' in source_entity and 'classification_confidence' in target_entity:
            src_conf = source_entity['classification_confidence']
            tgt_conf = target_entity['classification_confidence']
            avg_conf = (src_conf + tgt_conf) / 2
            entity_confidence_adjustment = (avg_conf - 0.5) * 0.2  # -0.1 to +0.1
            
        # Calculate final confidence with bounds
        final_confidence = max(0.0, min(1.0, 
                                      base_confidence + 
                                      entity_type_adjustment + 
                                      entity_confidence_adjustment))
        return final_confidence
    
    def _deduplicate_relations(self, relations):
        """Deduplicate relations based on source, target, and type"""
        unique_relations = {}
        
        for relation in relations:
            # Create a key that uniquely identifies the relation semantically
            key = (relation['type'], relation['source'], relation['target'])
            
            if key in unique_relations:
                # Keep the one with higher confidence
                if relation['confidence'] > unique_relations[key]['confidence']:
                    unique_relations[key] = relation
            else:
                unique_relations[key] = relation
                
        return list(unique_relations.values())
    
    def _verify_relation_compliance(self, relation, entities, compliance_mode):
        """Verify if a relation complies with regulatory requirements"""
        # Find source and target entities
        source_entity = next((e for e in entities if e['id'] == relation['source']), None)
        target_entity = next((e for e in entities if e['id'] == relation['target']), None)
        
        if not source_entity or not target_entity:
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'error': "Relation references non-existent entity",
                'severity': 'medium'
            }
        
        # Check for sensitive information relationships
        source_type = source_entity.get('type')
        target_type = target_entity.get('type')
        
        # Check specific high-risk relations
        
        # 1. PII/PHI disclosure relation
        if source_type in ['PII', 'PHI'] and relation['type'] == 'discloses':
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'error': f"Relation discloses protected {source_type} information",
                'severity': 'high'
            }
        
        # 2. Unauthorized access relation
        if target_type in ['PII', 'PHI'] and relation['type'] == 'has_access_to':
            # Check if access is authorized (would require additional context)
            is_authorized = self._check_authorized_access(source_entity, target_entity)
            if not is_authorized and compliance_mode == 'strict':
                return {
                    'is_compliant': False,
                    'compliance_score': 0.2,
                    'error': f"Potentially unauthorized access to {target_type}",
                    'severity': 'high'
                }
        
        # 3. Protected data transfer relation
        if (source_type in ['PII', 'PHI'] or target_type in ['PII', 'PHI']) and \
           relation['type'] in ['sends', 'transfers', 'provides']:
            # Would require additional context to verify if transfer is compliant
            transfer_compliant = self._check_compliant_transfer(source_entity, target_entity)
            if not transfer_compliant and compliance_mode == 'strict':
                return {
                    'is_compliant': False,
                    'compliance_score': 0.3,
                    'error': f"Potentially non-compliant transfer of protected data",
                    'severity': 'high'
                }
        
        # Calculate general compliance score for the relation
        compliance_score = self._calculate_relation_compliance_score(
            relation, source_entity, target_entity
        )
        
        is_compliant = compliance_score >= 0.7 or compliance_mode == 'soft'
        
        if not is_compliant:
            return {
                'is_compliant': False,
                'compliance_score': compliance_score,
                'error': f"Relation of type '{relation['type']}' has low compliance score",
                'severity': 'medium'
            }
        
        return {
            'is_compliant': True,
            'compliance_score': compliance_score,
            'error': None,
            'severity': 'none'
        }
    
    def _check_authorized_access(self, source_entity, target_entity):
        """Check if access is authorized between entities"""
        # This is a placeholder for a more sophisticated implementation
        # A real implementation would check authorization rules
        
        # Default to conservative approach for PII/PHI
        if target_entity.get('type') in ['PII', 'PHI']:
            return False
            
        return True
    
    def _check_compliant_transfer(self, source_entity, target_entity):
        """Check if data transfer is compliant"""
        # This is a placeholder for a more sophisticated implementation
        # A real implementation would check transfer compliance rules
        
        # Default to conservative approach for PII/PHI
        if source_entity.get('type') in ['PII', 'PHI'] or target_entity.get('type') in ['PII', 'PHI']:
            return False
            
        return True
    
    def _calculate_relation_compliance_score(self, relation, source_entity, target_entity):
        """Calculate compliance score for a relation"""
        # Base score based on relation type
        high_risk_relations = ['discloses', 'has_access_to', 'transfers', 'provides', 'sends']
        medium_risk_relations = ['works_for', 'owns', 'controls']
        low_risk_relations = ['located_at', 'part_of', 'related_to']
        
        if relation['type'] in high_risk_relations:
            base_score = 0.4
        elif relation['type'] in medium_risk_relations:
            base_score = 0.6
        elif relation['type'] in low_risk_relations:
            base_score = 0.8
        else:
            base_score = 0.7
        
        # Adjust for entity types
        entity_adjustment = 0.0
        
        # Higher risk when sensitive data is involved
        source_type = source_entity.get('type', 'UNKNOWN')
        target_type = target_entity.get('type', 'UNKNOWN')
        
        if source_type in ['PII', 'PHI', 'sensitive'] or target_type in ['PII', 'PHI', 'sensitive']:
            entity_adjustment -= 0.2
        
        # Adjust based on relation confidence
        confidence_adjustment = (relation['confidence'] - 0.5) * 0.2  # -0.1 to +0.1
        
        # Calculate final score with bounds
        final_score = max(0.0, min(1.0, base_score + entity_adjustment + confidence_adjustment))
        return final_score
    
    def _count_relation_types(self, relations):
        """Count relations by type for metadata"""
        counts = {}
        for relation in relations:
            relation_type = relation['type']
            if relation_type not in counts:
                counts[relation_type] = 0
            counts[relation_type] += 1
        return counts
    
