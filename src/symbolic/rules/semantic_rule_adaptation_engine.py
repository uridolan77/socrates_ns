import rules.rule_adaptation_engine as RuleAdaptationEngine
import datetime
import re

class SemanticRuleAdaptationEngine(RuleAdaptationEngine):
    """
    Engine for adapting semantic rules based on feedback and performance.
    """
    def __init__(self, config):
        super().__init__(config)
        
    def generate_rules_from_text(self, regulation_text, framework_id):
        """
        Generate semantic rules from regulatory text
        
        Args:
            regulation_text: Text of regulatory document
            framework_id: ID of the regulatory framework
            
        Returns:
            List of generated rules
        """
        # This is a placeholder implementation
        # A real implementation would use NLP techniques to extract rules
        
        # Extract key concepts from regulation text
        key_concepts = self._extract_key_concepts(regulation_text)
        
        rules = []
        
        # Generate concept threshold rules
        for concept, data in key_concepts.items():
            rule = {
                'id': f"gen_semantic_{framework_id}_{concept}_{len(rules)}",
                'name': f"Generated {concept} Rule",
                'description': data.get('description', f"Generated rule for concept '{concept}'"),
                'type': 'semantic',
                'subtype': 'concept_threshold',
                'action': 'block_if_exceeds' if data.get('negative', False) else 'require_concept',
                'concept': concept,
                'threshold': data.get('threshold', 0.7),
                'severity': data.get('severity', 'medium'),
                'source': 'generated',
                'framework_id': framework_id,
                'created_at': datetime.datetime.now().isoformat()
            }
            rules.append(rule)
            
        return rules

    def _extract_key_concepts(self, text):
        """
        Extract key concepts from regulation text using NLP techniques.
        
        Args:
            text: Regulatory text to analyze
            
        Returns:
            Dictionary of extracted concepts with metadata
        """
        # 1. Preprocess text
        clean_text = self._preprocess_text(text)
        sentences = self._split_into_sentences(clean_text)
        
        # 2. Initialize NLP components if available
        try:
            import spacy
            nlp = spacy.load("en_core_web_lg")
        except (ImportError, OSError):
            # Fall back to rule-based extraction if spacy not available
            return self._rule_based_concept_extraction(clean_text)
        
        # 3. Extract named entities and noun phrases
        entities = set()
        noun_phrases = set()
        
        for sentence in sentences:
            doc = nlp(sentence)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "LAW", "GPE", "NORP"]:
                    entities.add(ent.text.lower())
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases max
                    noun_phrases.add(chunk.text.lower())
        
        # 4. Identify regulatory verbs and obligations
        obligation_terms = ["must", "shall", "required", "necessary", "obligated", "obligation"]
        prohibition_terms = ["prohibited", "forbidden", "shall not", "must not", "may not"]
        permission_terms = ["may", "can", "allowed", "permitted"]
        
        obligation_sentences = [s for s in sentences if any(term in s.lower() for term in obligation_terms)]
        prohibition_sentences = [s for s in sentences if any(term in s.lower() for term in prohibition_terms)]
        permission_sentences = [s for s in sentences if any(term in s.lower() for term in permission_terms)]
        
        # 5. Extract concepts near regulatory verbs
        obligation_concepts = self._extract_concepts_from_regulatory_sentences(obligation_sentences, nlp)
        prohibition_concepts = self._extract_concepts_from_regulatory_sentences(prohibition_sentences, nlp)
        permission_concepts = self._extract_concepts_from_regulatory_sentences(permission_sentences, nlp)
        
        # 6. Calculate term frequencies for weighting
        term_frequencies = {}
        for concept in set(list(entities) + list(noun_phrases) + 
                        list(obligation_concepts) + list(prohibition_concepts) + list(permission_concepts)):
            term_frequencies[concept] = clean_text.lower().count(concept.lower())
        
        # 7. Construct concept dictionary with metadata
        concepts = {}
        
        # Add concepts with regulatory significance
        for concept in obligation_concepts:
            concepts[concept] = {
                "type": "obligation",
                "description": f"Obligation related to {concept}",
                "frequency": term_frequencies.get(concept, 1),
                "threshold": 0.7,
                "severity": "high",
                "negative": False
            }
            
        for concept in prohibition_concepts:
            concepts[concept] = {
                "type": "prohibition",
                "description": f"Prohibition related to {concept}",
                "frequency": term_frequencies.get(concept, 1),
                "threshold": 0.6,
                "severity": "high",
                "negative": True
            }
            
        for concept in permission_concepts:
            concepts[concept] = {
                "type": "permission",
                "description": f"Permission related to {concept}",
                "frequency": term_frequencies.get(concept, 1),
                "threshold": 0.5,
                "severity": "medium",
                "negative": False
            }
        
        # 8. Add named entities as concepts
        for entity in entities:
            if entity not in concepts:
                concepts[entity] = {
                    "type": "entity",
                    "description": f"Entity: {entity}",
                    "frequency": term_frequencies.get(entity, 1),
                    "threshold": 0.6,
                    "severity": "medium",
                    "negative": False
                }
        
        # 9. Filter and rank concepts by significance
        return self._filter_and_rank_concepts(concepts)

    def _extract_concepts_from_regulatory_sentences(self, sentences, nlp):
        """Extract concepts from sentences with regulatory language"""
        concepts = set()
        
        for sentence in sentences:
            doc = nlp(sentence)
            
            # Extract subject and object relations
            for token in doc:
                # Look for nouns that are subjects or objects
                if token.dep_ in ["nsubj", "dobj", "pobj"] and token.pos_ in ["NOUN", "PROPN"]:
                    # Get the noun phrase containing this token
                    for chunk in doc.noun_chunks:
                        if token in chunk:
                            concepts.add(chunk.text.lower())
                            break
                    else:
                        # If not in a noun chunk, add the token itself
                        concepts.add(token.text.lower())
        
        return concepts

    def _filter_and_rank_concepts(self, concepts):
        """Filter and rank concepts by significance"""
        # Filter out very short concepts
        filtered_concepts = {k: v for k, v in concepts.items() if len(k) >= 4}
        
        # Rank by frequency and type
        type_weights = {"obligation": 3, "prohibition": 3, "permission": 2, "entity": 1}
        
        for concept, data in filtered_concepts.items():
            weight = type_weights.get(data["type"], 1) * data["frequency"]
            data["weight"] = weight
        
        # Sort by weight and take top concepts
        sorted_concepts = dict(sorted(
            filtered_concepts.items(), 
            key=lambda x: x[1]["weight"], 
            reverse=True
        )[:30])  # Limit to top 30 concepts
        
        return sorted_concepts

    def _preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Standardize punctuation spacing
        text = re.sub(r'(\w)([.,:;!?])', r'\1 \2', text)
        
        return text

    def _split_into_sentences(self, text):
        """Split text into sentences"""
        # Simple regex-based sentence splitting
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]