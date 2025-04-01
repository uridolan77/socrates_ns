from src.compliance.models.compliance_issue import ComplianceIssue
import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from collections import defaultdict

class NLPContentDetector:
    """
    Advanced compliance detector using NLP techniques to identify policy violations
    with semantic understanding, contextual awareness, and explainability.
    """
    def __init__(self, config):
        """
        Initialize the NLP content detector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - policy_rules: Dictionary of compliance rules
                - enabled_rules: List of enabled rule IDs
                - compliance_threshold: Threshold for compliance (default 0.7)
                - nlp_models: Configuration for NLP models
                - embeddings_cache_size: Size of embeddings cache (default 1000)
                - confidence_thresholds: Thresholds for different confidence levels
                - semantic_similarity_threshold: Threshold for semantic matching (default 0.85)
                - multi_language: Enable multi-language support (default False)
                - context_weight: Weight for contextual factors (default 0.3)
                - break_on_first_violation: Whether to stop on first violation (default False)
        """
        self.config = config
        self.policy_rules = config.get("policy_rules", {})
        self.compliance_threshold = config.get("compliance_threshold", 0.7)
        self.rules_enabled = config.get("enabled_rules", [])
        
        # Enhanced NLP-specific configuration
        self.nlp_models_config = config.get("nlp_models", {})
        self.embeddings_cache_size = config.get("embeddings_cache_size", 1000)
        self.confidence_thresholds = config.get("confidence_thresholds", {
            "high": 0.9,
            "medium": 0.75,
            "low": 0.6
        })
        self.semantic_similarity_threshold = config.get("semantic_similarity_threshold", 0.85)
        self.multi_language = config.get("multi_language", False)
        self.context_weight = config.get("context_weight", 0.3)
        
        # Initialize NLP components
        self.nlp_models = self._initialize_nlp_models()
        self.embeddings_cache = {}  # Simple cache for embeddings
        self.rule_embeddings = self._precompute_rule_embeddings()
        
        # Language detection
        self.language_detector = self._initialize_language_detector() if self.multi_language else None
        
        # History tracking for contextual analysis
        self.detection_history = defaultdict(list)
        self.max_history_size = config.get("max_history_size", 100)
        
    def _initialize_nlp_models(self):
        """Initialize NLP models needed for detection."""
        models = {}
        
        # Try to load transformer models if available
        try:
            # Check if we have transformers and torch
            import torch
            from transformers import AutoModel, AutoTokenizer, pipeline
            
            # Load text embedding model (e.g., BERT, RoBERTa, etc.)
            embedding_model_name = self.nlp_models_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            
            if embedding_model_name:
                models["tokenizer"] = AutoTokenizer.from_pretrained(embedding_model_name)
                models["embedding_model"] = AutoModel.from_pretrained(embedding_model_name)
                models["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                logging.info(f"Loaded embedding model {embedding_model_name} on {models['device']}")
                
            # Load zero-shot classification model if specified
            zs_model_name = self.nlp_models_config.get("zero_shot_model")
            if zs_model_name:
                models["zero_shot"] = pipeline("zero-shot-classification", model=zs_model_name, device=0 if torch.cuda.is_available() else -1)
                
            # Load text classification model if specified
            tc_model_name = self.nlp_models_config.get("text_classification_model")
            if tc_model_name:
                models["text_classifier"] = pipeline("text-classification", model=tc_model_name, device=0 if torch.cuda.is_available() else -1)
                
            # Load NER model if specified
            ner_model_name = self.nlp_models_config.get("ner_model")
            if ner_model_name:
                models["ner"] = pipeline("ner", model=ner_model_name, device=0 if torch.cuda.is_available() else -1)
                
        except (ImportError, ModuleNotFoundError) as e:
            logging.warning(f"Advanced NLP models could not be loaded: {e}. Falling back to simpler techniques.")
            
            # If transformer models aren't available, try to use simpler NLP libraries
            try:
                import spacy
                
                # Load SpaCy model
                spacy_model = self.nlp_models_config.get("spacy_model", "en_core_web_md")
                models["spacy"] = spacy.load(spacy_model)
                logging.info(f"Loaded SpaCy model {spacy_model}")
                
            except (ImportError, ModuleNotFoundError):
                logging.warning("SpaCy could not be loaded. Some NLP features will be limited.")
        
        return models
        
    def _initialize_language_detector(self):
        """Initialize language detection model for multi-language support."""
        detector = None
        
        try:
            # Try fastText language identification
            import fasttext
            # Download model if needed
            model_path = os.path.join(os.path.dirname(__file__), "lid.176.ftz")
            if not os.path.exists(model_path):
                # In a real implementation, you would download the model here
                logging.warning("Language detection model not found. Downloading...")
                
            detector = fasttext.load_model(model_path)
            logging.info("Loaded fastText language detection model")
            
        except (ImportError, ModuleNotFoundError):
            try:
                # Try langdetect as fallback
                import langdetect
                detector = langdetect
                logging.info("Using langdetect for language detection")
                
            except (ImportError, ModuleNotFoundError):
                logging.warning("No language detection library available. Multi-language support disabled.")
                
        return detector
        
    def _precompute_rule_embeddings(self):
        """Precompute embeddings for rule keywords and patterns for faster matching."""
        rule_embeddings = {}
        
        # Only proceed if we have an embedding model
        if not self.nlp_models.get("embedding_model"):
            return rule_embeddings
            
        for rule_id, rule in self.policy_rules.items():
            if rule_id not in self.rules_enabled:
                continue
                
            rule_type = rule.get("type", "keyword")
            
            # Handle keyword rules
            if rule_type == "keyword":
                keywords = rule.get("keywords", [])
                rule_embeddings[rule_id] = {
                    "keywords": keywords,
                    "embeddings": self._get_embeddings_batch(keywords)
                }
                
            # Handle semantic rules
            elif rule_type == "semantic":
                examples = rule.get("examples", [])
                concepts = rule.get("concepts", [])
                
                rule_embeddings[rule_id] = {
                    "examples": examples,
                    "concepts": concepts,
                    "embeddings": self._get_embeddings_batch(examples + concepts)
                }
                
        return rule_embeddings
        
    def _get_embeddings_batch(self, texts):
        """Get embeddings for a batch of texts."""
        if not texts:
            return []
            
        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # If all texts were cached, return embeddings
        if not uncached_texts:
            return embeddings
            
        # Get embeddings for uncached texts
        if "embedding_model" in self.nlp_models and "tokenizer" in self.nlp_models:
            # Using transformers
            import torch
            
            # Tokenize all texts
            inputs = self.nlp_models["tokenizer"](uncached_texts, padding=True, truncation=True, return_tensors="pt")
            
            # Move to correct device
            for key in inputs:
                inputs[key] = inputs[key].to(self.nlp_models["device"])
                
            # Get embeddings
            with torch.no_grad():
                outputs = self.nlp_models["embedding_model"](**inputs)
                
            # Use mean pooling to get a fixed-size representation
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            
            # Calculate mean embeddings
            new_embeddings = []
            for i in range(len(uncached_texts)):
                tokens_mask = attention_mask[i].unsqueeze(-1).expand(token_embeddings[i].shape)
                sum_embeddings = torch.sum(token_embeddings[i] * tokens_mask, 0)
                sum_mask = torch.sum(tokens_mask, 0)
                mean_embedding = sum_embeddings / sum_mask
                new_embeddings.append(mean_embedding.cpu().numpy())
                
        elif "spacy" in self.nlp_models:
            # Using SpaCy
            spacy_model = self.nlp_models["spacy"]
            new_embeddings = [spacy_model(text).vector for text in uncached_texts]
            
        else:
            # No embedding model available
            return [np.zeros(300) for _ in texts]  # Return empty embeddings
            
        # Update cache and embeddings list
        for i, idx in enumerate(uncached_indices):
            self.embeddings_cache[uncached_texts[i]] = new_embeddings[i]
            embeddings[idx] = new_embeddings[i]
            
        # Limit cache size
        if len(self.embeddings_cache) > self.embeddings_cache_size:
            # Simple cache eviction: remove oldest entries
            for key in list(self.embeddings_cache.keys())[:-self.embeddings_cache_size]:
                del self.embeddings_cache[key]
                
        return embeddings
        
    def _calculate_semantic_similarity(self, embedding1, embedding2):
        """Calculate semantic similarity between two embeddings using cosine similarity."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
    def check_compliance(self, text, context=None):
        """
        Check if the provided text complies with all defined policy rules using NLP techniques.
        
        Args:
            text: Input text to check
            context: Optional context information
                
        Returns:
            Dict with compliance results including:
            - is_compliant: Boolean indicating compliance
            - filtered_input: Original or modified text
            - issues: List of ComplianceIssue objects
            - modified: Boolean indicating if text was modified
            - confidence: Overall confidence in the detection
            - explanations: Detailed explanations of detections
            - language: Detected language (if multi_language is enabled)
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        # Auto-detect language if multi-language support is enabled
        language = None
        if self.multi_language and self.language_detector:
            language = self._detect_language(text)
            
        # Get text embedding for semantic matching
        text_embedding = None
        if "embedding_model" in self.nlp_models or "spacy" in self.nlp_models:
            embeddings = self._get_embeddings_batch([text])
            if embeddings:
                text_embedding = embeddings[0]
                
        # Process each enabled rule
        issues = []
        explanations = []
        is_compliant = True
        
        for rule_id in self.rules_enabled:
            if rule_id not in self.policy_rules:
                continue
                
            rule = self.policy_rules[rule_id]
            
            # Check if rule is language-specific and doesn't match detected language
            if language and "languages" in rule and language not in rule.get("languages", []):
                continue
                
            # Run different checks based on rule type
            violation_details = self._check_rule_advanced(text, rule_id, rule, text_embedding, context, language)
            
            if violation_details["violation_found"]:
                is_compliant = False
                
                # Create issue
                metadata = {
                    "rule_name": rule.get("name", ""),
                    "confidence": violation_details["confidence"],
                    "matches": violation_details.get("matches", []),
                    "detection_method": violation_details.get("detection_method", "")
                }
                
                # Add location information if available
                if "locations" in violation_details:
                    metadata["locations"] = violation_details["locations"]
                    
                issue = ComplianceIssue(
                    rule_id=rule_id,
                    severity=rule.get("severity", "medium"),
                    description=rule.get("description", "Policy violation detected"),
                    metadata=metadata
                )
                
                issues.append(issue)
                explanations.append(violation_details.get("explanation", f"Violation of rule {rule_id} detected"))
                
                # Stop checking if configured to break on first violation
                if self.config.get("break_on_first_violation", False):
                    break
        
        # Update detection history for this content
        self._update_detection_history(text, issues, context)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(issues) if issues else 1.0
        
        result = {
            "is_compliant": is_compliant,
            "filtered_input": text,  # Not modifying text in this implementation
            "issues": [issue.__dict__ for issue in issues],
            "modified": False,
            "confidence": overall_confidence,
            "explanations": explanations
        }
        
        # Add language information if detected
        if language:
            result["language"] = language
            
        return result
        
    def _check_rule_advanced(self, text, rule_id, rule, text_embedding, context, language=None):
        """
        Advanced rule checking using NLP techniques.
        
        Args:
            text: Input text
            rule_id: Rule identifier
            rule: Rule definition
            text_embedding: Pre-computed text embedding
            context: Optional context information
            language: Detected language (optional)
            
        Returns:
            Dict with violation details
        """
        rule_type = rule.get("type", "keyword")
        
        # Start with default result
        result = {
            "violation_found": False,
            "confidence": 0.0,
            "detection_method": rule_type,
            "explanation": ""
        }
        
        # Keyword matching (enhanced with stemming/lemmatization when available)
        if rule_type == "keyword":
            keywords = rule.get("keywords", [])
            
            # First try exact matching for efficiency
            matches = []
            locations = []
            
            for keyword in keywords:
                # Case-insensitive exact match
                keyword_lower = keyword.lower()
                text_lower = text.lower()
                
                if keyword_lower in text_lower:
                    matches.append(keyword)
                    
                    # Find all occurrences
                    start_idx = 0
                    while start_idx < len(text_lower):
                        pos = text_lower.find(keyword_lower, start_idx)
                        if pos == -1:
                            break
                        locations.append({"keyword": keyword, "start": pos, "end": pos + len(keyword)})
                        start_idx = pos + 1
            
            # If no exact matches but we have embeddings, try semantic matching
            if not matches and text_embedding is not None and rule_id in self.rule_embeddings:
                keyword_embeddings = self.rule_embeddings[rule_id]["embeddings"]
                rule_keywords = self.rule_embeddings[rule_id]["keywords"]
                
                for i, keyword_embedding in enumerate(keyword_embeddings):
                    similarity = self._calculate_semantic_similarity(text_embedding, keyword_embedding)
                    
                    if similarity >= self.semantic_similarity_threshold:
                        matches.append(rule_keywords[i])
                        result["detection_method"] = "semantic_keyword"
            
            # Check for lemmatized matches if no matches found and SpaCy is available
            if not matches and "spacy" in self.nlp_models:
                spacy_doc = self.nlp_models["spacy"](text)
                
                # Get lemmas of text
                text_lemmas = [token.lemma_.lower() for token in spacy_doc]
                
                # Check keywords against lemmas
                for keyword in keywords:
                    # Process keyword with SpaCy to get its lemma
                    keyword_doc = self.nlp_models["spacy"](keyword)
                    keyword_lemmas = [token.lemma_.lower() for token in keyword_doc]
                    
                    # Check if any keyword lemma appears in text lemmas
                    for lemma in keyword_lemmas:
                        if lemma in text_lemmas:
                            matches.append(keyword)
                            result["detection_method"] = "lemma_matching"
                            break
            
            if matches:
                result["violation_found"] = True
                result["confidence"] = min(1.0, 0.7 + (len(matches) / len(keywords)) * 0.3)
                result["matches"] = matches
                result["locations"] = locations if locations else None
                result["explanation"] = f"Matched keywords: {', '.join(matches)}"
            
        # Regex matching (same as original but with additional metadata)
        elif rule_type == "regex":
            pattern = rule.get("pattern", "")
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    result["violation_found"] = True
                    result["confidence"] = 0.9  # High confidence for regex matches
                    result["matches"] = [match.group(0) for match in matches]
                    result["locations"] = [{"start": match.start(), "end": match.end(), "pattern": pattern} for match in matches]
                    result["explanation"] = f"Matched regex pattern: {pattern}"
            except re.error:
                logging.error(f"Invalid regex pattern: {pattern}")
                result["violation_found"] = False
                result["explanation"] = f"Error in regex pattern: {pattern}"
            
        # Semantic matching using embeddings or zero-shot classification
        elif rule_type == "semantic":
            # Get concepts and examples from rule
            concepts = rule.get("concepts", [])
            examples = rule.get("examples", [])
            threshold = rule.get("threshold", self.semantic_similarity_threshold)
            
            # If we have the zero-shot classification model, use it
            if "zero_shot" in self.nlp_models:
                zs_result = self.nlp_models["zero_shot"](text, concepts)
                
                # Get the highest scoring concept
                top_concept = zs_result["labels"][0]
                top_score = zs_result["scores"][0]
                
                if top_score >= threshold:
                    result["violation_found"] = True
                    result["confidence"] = top_score
                    result["matches"] = [top_concept]
                    result["detection_method"] = "zero_shot"
                    result["explanation"] = f"Detected concept: {top_concept} with confidence {top_score:.2f}"
                
            # Otherwise use embeddings similarity if available
            elif text_embedding is not None and rule_id in self.rule_embeddings:
                # Compare text embedding with concept/example embeddings
                matches = []
                similarities = []
                
                for i, embedding in enumerate(self.rule_embeddings[rule_id]["embeddings"]):
                    similarity = self._calculate_semantic_similarity(text_embedding, embedding)
                    
                    if similarity >= threshold:
                        if i < len(concepts):
                            matches.append(concepts[i])
                        else:
                            matches.append(examples[i - len(concepts)])
                        similarities.append(similarity)
                
                if matches:
                    # Use maximum similarity as confidence
                    max_similarity = max(similarities)
                    result["violation_found"] = True
                    result["confidence"] = max_similarity
                    result["matches"] = matches
                    result["detection_method"] = "embedding_similarity"
                    result["explanation"] = f"Semantically similar to concepts: {', '.join(matches)}"
                
        # Named Entity Recognition based rules
        elif rule_type == "entity":
            prohibited_entities = rule.get("prohibited_entities", [])
            
            if prohibited_entities and "ner" in self.nlp_models:
                # Use transformer-based NER
                entities = self.nlp_models["ner"](text)
                
                # Filter for prohibited entity types
                matches = []
                locations = []
                
                for entity in entities:
                    if entity["entity_group"] in prohibited_entities:
                        matches.append(f"{entity['entity_group']}: {entity['word']}")
                        locations.append({
                            "entity": entity["entity_group"],
                            "word": entity["word"],
                            "start": entity["start"],
                            "end": entity["end"]
                        })
                
                if matches:
                    result["violation_found"] = True
                    result["confidence"] = min(1.0, 0.7 + (len(matches) / 10) * 0.3)  # Scale with number of entities
                    result["matches"] = matches
                    result["locations"] = locations
                    result["detection_method"] = "transformer_ner"
                    result["explanation"] = f"Detected prohibited entities: {', '.join(matches)}"
                    
            elif prohibited_entities and "spacy" in self.nlp_models:
                # Use SpaCy NER as fallback
                spacy_doc = self.nlp_models["spacy"](text)
                
                # Filter for prohibited entity types
                matches = []
                locations = []
                
                for ent in spacy_doc.ents:
                    if ent.label_ in prohibited_entities:
                        matches.append(f"{ent.label_}: {ent.text}")
                        locations.append({
                            "entity": ent.label_,
                            "word": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char
                        })
                
                if matches:
                    result["violation_found"] = True
                    result["confidence"] = min(1.0, 0.7 + (len(matches) / 10) * 0.3)
                    result["matches"] = matches
                    result["locations"] = locations
                    result["detection_method"] = "spacy_ner"
                    result["explanation"] = f"Detected prohibited entities: {', '.join(matches)}"
            
        # Text classification rules
        elif rule_type == "classification":
            labels = rule.get("labels", [])
            threshold = rule.get("threshold", 0.7)
            
            if labels and "text_classifier" in self.nlp_models:
                # Use transformer-based text classification
                classification = self.nlp_models["text_classifier"](text)
                
                # Check if any predicted label is in prohibited labels
                matches = []
                scores = []
                
                for result_item in classification:
                    if result_item["label"] in labels and result_item["score"] >= threshold:
                        matches.append(f"{result_item['label']}: {result_item['score']:.2f}")
                        scores.append(result_item["score"])
                
                if matches:
                    result["violation_found"] = True
                    result["confidence"] = max(scores)
                    result["matches"] = matches
                    result["detection_method"] = "text_classification"
                    result["explanation"] = f"Classified as prohibited categories: {', '.join(matches)}"
            
        # Contextual rules that consider both content and context
        elif rule_type == "contextual":
            # Contextual rules require context
            if context:
                context_conditions = rule.get("context_conditions", [])
                content_conditions = rule.get("content_conditions", [])
                
                # Check if context matches any conditions
                context_match = False
                for condition in context_conditions:
                    if self._check_context_condition(condition, context):
                        context_match = True
                        break
                
                # Only check content if context matches
                if context_match:
                    # Check content conditions
                    for condition in content_conditions:
                        content_match = self._check_content_condition(condition, text, text_embedding)
                        
                        if content_match["match"]:
                            result["violation_found"] = True
                            result["confidence"] = content_match["confidence"]
                            result["matches"] = content_match.get("matches", [])
                            result["detection_method"] = "contextual"
                            result["explanation"] = (f"Content matched condition in sensitive context: "
                                                   f"{content_match.get('explanation', '')}")
                            break
            
        # Call custom rule implementation if available
        elif rule_type == "custom":
            custom_impl = rule.get("implementation")
            if custom_impl and callable(custom_impl):
                try:
                    custom_result = custom_impl(text, context)
                    return {**result, **custom_result}  # Merge results
                except Exception as e:
                    logging.error(f"Error in custom rule implementation: {e}")
                    result["explanation"] = f"Error in custom rule: {str(e)}"
            else:
                result["explanation"] = "Custom rule implementation not available"
                
        # Apply context-based confidence adjustment
        if result["violation_found"] and context:
            result["confidence"] = self._adjust_confidence_with_context(
                result["confidence"], rule_id, rule, context
            )
                
        return result
        
    def _check_context_condition(self, condition, context):
        """Check if context matches a condition."""
        condition_type = condition.get("type")
        
        if condition_type == "user_attribute":
            # Check if user has specific attribute
            attribute = condition.get("attribute")
            value = condition.get("value")
            
            if "user" in context and attribute in context["user"]:
                if value is None:  # Just check if attribute exists
                    return True
                else:
                    return context["user"][attribute] == value
                    
        elif condition_type == "domain":
            # Check if in specific domain
            domain = condition.get("domain")
            return context.get("domain") == domain
            
        elif condition_type == "environment":
            # Check environment (e.g., production, test)
            env = condition.get("environment")
            return context.get("environment") == env
            
        elif condition_type == "history":
            # Check if user/session has specific history
            return self._check_history_condition(condition, context)
            
        return False
        
    def _check_content_condition(self, condition, text, text_embedding=None):
        """Check if content matches a condition."""
        condition_type = condition.get("type")
        
        result = {
            "match": False,
            "confidence": 0.0,
            "matches": [],
            "explanation": ""
        }
        
        if condition_type == "keyword":
            # Simple keyword matching
            keywords = condition.get("keywords", [])
            matches = [kw for kw in keywords if kw.lower() in text.lower()]
            
            if matches:
                result["match"] = True
                result["confidence"] = min(1.0, 0.7 + (len(matches) / len(keywords)) * 0.3)
                result["matches"] = matches
                result["explanation"] = f"Matched keywords: {', '.join(matches)}"
                
        elif condition_type == "regex":
            # Regex matching
            pattern = condition.get("pattern", "")
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    result["match"] = True
                    result["confidence"] = 0.9
                    result["matches"] = [match.group(0) for match in matches]
                    result["explanation"] = f"Matched regex pattern: {pattern}"
            except re.error:
                logging.error(f"Invalid regex pattern: {pattern}")
                
        elif condition_type == "semantic":
            # Semantic matching using embeddings
            concepts = condition.get("concepts", [])
            threshold = condition.get("threshold", self.semantic_similarity_threshold)
            
            if text_embedding is not None and concepts:
                # Get embeddings for concepts
                concept_embeddings = self._get_embeddings_batch(concepts)
                
                # Find semantic matches
                matches = []
                similarities = []
                
                for i, concept_embedding in enumerate(concept_embeddings):
                    similarity = self._calculate_semantic_similarity(text_embedding, concept_embedding)
                    
                    if similarity >= threshold:
                        matches.append(concepts[i])
                        similarities.append(similarity)
                
                if matches:
                    result["match"] = True
                    result["confidence"] = max(similarities)
                    result["matches"] = matches
                    result["explanation"] = f"Semantically similar to concepts: {', '.join(matches)}"
                    
        return result
        
    def _check_history_condition(self, condition, context):
        """Check if user/session history matches a condition."""
        # Extract identifiers from context
        user_id = context.get("user", {}).get("id")
        session_id = context.get("session_id")
        
        if not user_id and not session_id:
            return False
            
        # Determine which history to check
        history_type = condition.get("history_type", "user")
        history_key = user_id if history_type == "user" else session_id
        
        if not history_key or history_key not in self.detection_history:
            return False
            
        # Check condition against history
        condition_check = condition.get("check", "")
        rule_id = condition.get("rule_id")
        
        if condition_check == "previous_violation" and rule_id:
            # Check if this rule was previously violated
            for detection in self.detection_history[history_key]:
                for issue in detection.get("issues", []):
                    if issue.get("rule_id") == rule_id:
                        return True
                        
        elif condition_check == "violation_count":
            # Check if violation count exceeds threshold
            min_count = condition.get("min_count", 1)
            specific_rule_id = condition.get("rule_id")
            
            if specific_rule_id:
                # Count violations of specific rule
                count = sum(
                    1 for detection in self.detection_history[history_key]
                    for issue in detection.get("issues", [])
                    if issue.get("rule_id") == specific_rule_id
                )
            else:
                # Count all violations
                count = sum(
                    len(detection.get("issues", []))
                    for detection in self.detection_history[history_key]
                )
                
            return count >= min_count
            
        return False
        
    def _adjust_confidence_with_context(self, base_confidence, rule_id, rule, context):
        """Adjust detection confidence based on context."""
        adjusted_confidence = base_confidence
        
        # Check for context factors that might increase confidence
        confidence_factors = rule.get("confidence_factors", [])
        
        for factor in confidence_factors:
            factor_type = factor.get("type")
            
            if factor_type == "user_attribute":
                # Check if user has specific attribute
                attribute = factor.get("attribute")
                value = factor.get("value")
                
                if "user" in context and attribute in context["user"]:
                    if value is None or context["user"][attribute] == value:
                        # Apply confidence adjustment
                        adjustment = factor.get("adjustment", 0.1)
                        adjusted_confidence = min(1.0, adjusted_confidence + adjustment)
                        
            elif factor_type == "domain":
                # Check if in specific domain
                domain = factor.get("domain")
                if context.get("domain") == domain:
                    adjustment = factor.get("adjustment", 0.1)
                    adjusted_confidence = min(1.0, adjusted_confidence + adjustment)
                    
            elif factor_type == "history":
                # Check user history
                user_id = context.get("user", {}).get("id")
                session_id = context.get("session_id")
                
                if user_id and user_id in self.detection_history:
                    # Apply history-based adjustment
                    history_adjustment = self._calculate_history_adjustment(user_id, rule_id, factor)
                    adjusted_confidence = min(1.0, adjusted_confidence + history_adjustment)
                    
                elif session_id and session_id in self.detection_history:
                    # Apply session-based adjustment
                    session_adjustment = self._calculate_history_adjustment(session_id, rule_id, factor)
                    adjusted_confidence = min(1.0, adjusted_confidence + session_adjustment)
                    
        return adjusted_confidence
        
    def _calculate_history_adjustment(self, history_key, rule_id, factor):
        """Calculate confidence adjustment based on history."""
        adjustment = 0.0
        
        # Get detection history for this key
        history = self.detection_history.get(history_key, [])
        
        if not history:
            return adjustment
            
        # Check if this rule was previously violated
        previous_violations = sum(
            1 for detection in history
            for issue in detection.get("issues", [])
            if issue.get("rule_id") == rule_id
        )
        
        # Apply adjustment based on number of previous violations
        if previous_violations > 0:
            scaling_factor = factor.get("scaling_factor", 0.05)
            max_adjustment = factor.get("max_adjustment", 0.2)
            
            adjustment = min(max_adjustment, previous_violations * scaling_factor)
            
        return adjustment
        
    def _update_detection_history(self, text, issues, context=None):
        """Update detection history with current detection results."""
        if not context:
            return
            
        # Extract identifiers from context
        user_id = context.get("user", {}).get("id")
        session_id = context.get("session_id")
        
        # Only update history if we have an identifier
        if not user_id and not session_id:
            return
            
        # Prepare detection record
        detection = {
            "timestamp": self._get_current_timestamp(),
            "issues": [issue.__dict__ for issue in issues],
            "context": context
        }
        
        # Add to user history if available
        if user_id:
            if user_id not in self.detection_history:
                self.detection_history[user_id] = []
                
            self.detection_history[user_id].append(detection)
            
            # Limit history size
            if len(self.detection_history[user_id]) > self.max_history_size:
                self.detection_history[user_id] = self.detection_history[user_id][-self.max_history_size:]
                
        # Add to session history if available
        if session_id:
            if session_id not in self.detection_history:
                self.detection_history[session_id] = []
                
            self.detection_history[session_id].append(detection)
            
            # Limit history size
            if len(self.detection_history[session_id]) > self.max_history_size:
                self.detection_history[session_id] = self.detection_history[session_id][-self.max_history_size:]
                
    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def _calculate_overall_confidence(self, issues):
        """Calculate overall confidence from multiple issues."""
        if not issues:
            return 1.0
            
        # Use maximum confidence for final result
        confidences = [issue.metadata.get("confidence", 0.5) for issue in issues if hasattr(issue, "metadata")]
        return max(confidences) if confidences else 0.5
        
    def _detect_language(self, text):
        """Detect language of text."""
        if not self.language_detector or not text:
            return "en"  # Default to English
            
        try:
            # Check type of language detector
            if hasattr(self.language_detector, "predict"):
                # FastText detector
                predictions = self.language_detector.predict(text, k=1)
                lang_code = predictions[0][0].replace("__label__", "")
                return lang_code
            else:
                # Langdetect detector
                import langdetect
                lang_code = langdetect.detect(text)
                return lang_code
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            return "en"  # Default to English