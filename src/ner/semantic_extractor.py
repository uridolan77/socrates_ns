
import logging
import re
import numpy as np
# Try to load sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from bertopic import BERTopic

class SemanticConceptExtractor:
    """
    Enhanced semantic concept extractor with topic modeling, sentiment analysis,
    and regulatory concept detection.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.topic_model = self._initialize_topic_model()
        self.concept_detectors = self._initialize_concept_detectors()
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        self.harmful_content_detector = self._initialize_harmful_content_detector()
        self.regulatory_concepts = self._load_regulatory_concepts()
        
    def _initialize_topic_model(self):
        """Initialize topic modeling capabilities"""
        topic_model_type = self.config.get("topic_model", "none")
        
        if topic_model_type == "lda":
            try:
                from sklearn.decomposition import LatentDirichletAllocation
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Initialize vectorizer and LDA model
                vectorizer = CountVectorizer(
                    max_df=0.95, min_df=2, stop_words='english', max_features=1000
                )
                lda_model = LatentDirichletAllocation(
                    n_components=10, random_state=42, learning_method='online'
                )
                
                return {
                    'type': 'lda',
                    'vectorizer': vectorizer,
                    'model': lda_model,
                    'is_trained': False  # Will be set to True once trained
                }
            except ImportError:
                logging.warning("scikit-learn not installed, LDA topic modeling not available")
                return None
        elif topic_model_type == "nmf":
            try:
                from sklearn.decomposition import NMF
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Initialize vectorizer and NMF model
                vectorizer = TfidfVectorizer(
                    max_df=0.95, min_df=2, stop_words='english', max_features=1000
                )
                nmf_model = NMF(
                    n_components=10, random_state=42, max_iter=1000
                )
                
                return {
                    'type': 'nmf',
                    'vectorizer': vectorizer,
                    'model': nmf_model,
                    'is_trained': False  # Will be set to True once trained
                }
            except ImportError:
                logging.warning("scikit-learn not installed, NMF topic modeling not available")
                return None
        elif topic_model_type == "bertopic":
            try:
                
                
                bertopic_model = BERTopic()
                
                return {
                    'type': 'bertopic',
                    'model': bertopic_model,
                    'is_trained': False
                }
            except ImportError:
                logging.warning("BERTopic not installed, BERTopic modeling not available")
                return None
                
        return None
            
    def _initialize_concept_detectors(self):
        """Initialize concept detection models"""
        detectors = {}
        
        # Initialize keyword-based concept detectors
        detectors['keyword'] = self._initialize_keyword_detector()
        
        # Initialize embedding-based concept detector if available
        use_embeddings = self.config.get("use_embedding_concepts", True)
        if use_embeddings:
            detectors['embedding'] = self._initialize_embedding_detector()
            
        # Initialize ML-based concept detector if available
        ml_detector_path = self.config.get("ml_concept_detector_path")
        if ml_detector_path:
            detectors['ml'] = self._initialize_ml_detector(ml_detector_path)
            
        return detectors
    
    def _initialize_keyword_detector(self):
        """Initialize keyword-based concept detector"""
        # Load concept keywords from config or default list
        concept_keywords = self.config.get("concept_keywords", {})
        
        # Default keywords for common concepts if not provided
        default_keywords = {
            'data_privacy': ['privacy', 'personal data', 'data protection', 'gdpr', 'ccpa', 
                           'confidential', 'sensitive information'],
            'consent': ['consent', 'permission', 'authorize', 'opt-in', 'opt-out', 'approval'],
            'data_security': ['security', 'encryption', 'protection', 'safeguard', 'breach',
                            'secure', 'vulnerability', 'threat'],
            'compliance': ['compliance', 'regulation', 'policy', 'requirement', 'standard',
                         'guideline', 'law', 'rule'],
            'data_processing': ['processing', 'collect', 'store', 'analyze', 'use', 'share',
                              'transfer', 'delete', 'retention'],
            'accountability': ['accountability', 'responsibility', 'oversight', 'governance',
                             'audit', 'monitor', 'report'],
            'transparency': ['transparency', 'disclose', 'inform', 'notice', 'explain',
                           'clear', 'open'],
            'risk': ['risk', 'impact', 'assessment', 'mitigation', 'reduce', 'prevent',
                    'likelihood', 'severity'],
            'user_rights': ['rights', 'access', 'rectification', 'erasure', 'portability',
                          'object', 'restriction', 'automated decision'],
            'health_data': ['health', 'medical', 'clinical', 'patient', 'treatment',
                          'diagnosis', 'prognosis', 'medication', 'prescription']
        }
        
        # Merge provided keywords with defaults for any missing concepts
        for concept, keywords in default_keywords.items():
            if concept not in concept_keywords:
                concept_keywords[concept] = keywords
                
        return {
            'type': 'keyword',
            'concepts': concept_keywords,
            # Compile regex patterns for efficient matching
            'patterns': {
                concept: [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                        for kw in keywords]
                for concept, keywords in concept_keywords.items()
            }
        }
    
    def _initialize_embedding_detector(self):
        """Initialize embedding-based concept detector"""
        try:

            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
            
            # Load concept reference embeddings
            concept_references = self.config.get("concept_references", {})
            
            # Default concept references if not provided
            if not concept_references:
                concept_references = {
                    'data_privacy': [
                        "Personal data must be protected and kept private.",
                        "Organizations must adhere to data privacy regulations like GDPR and CCPA.",
                        "Privacy policies should clearly state how data is collected and used."
                    ],
                    'consent': [
                        "Valid consent must be freely given, specific, informed, and unambiguous.",
                        "Users should be able to opt-in or opt-out of data collection.",
                        "Consent should be obtained before processing personal data."
                    ],
                    # Add more concepts with reference sentences
                }
                
            # Generate embeddings for concept references
            concept_embeddings = {}
            for concept, references in concept_references.items():
                if references:
                    # Generate embeddings for each reference
                    reference_embeddings = model.encode(references)
                    # Store average embedding for the concept
                    concept_embeddings[concept] = np.mean(reference_embeddings, axis=0)
            
            return {
                'type': 'embedding',
                'model': model,
                'concept_embeddings': concept_embeddings
            }
        except ImportError:
            logging.warning("sentence-transformers not installed, embedding-based concept detection not available")
            return None
    
    def _initialize_ml_detector(self, model_path):
        """Initialize ML-based concept detector"""
        try:
            # Try to load a trained model
            import pickle
            
            with open(model_path, 'rb') as f:
                ml_model = pickle.load(f)
                
            # Load vectorizer or tokenizer if available
            vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
            vectorizer = None
            try:
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
            except:
                logging.warning(f"Vectorizer not found at {vectorizer_path}")
                
            return {
                'type': 'ml',
                'model': ml_model,
                'vectorizer': vectorizer
            }
        except:
            logging.warning(f"Could not load ML concept detector from {model_path}")
            return None
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis capability"""
        sentiment_analyzer_type = self.config.get("sentiment_analyzer", "none")
        
        if sentiment_analyzer_type == "vader":
            try:
                
                return SentimentIntensityAnalyzer()
            except ImportError:
                logging.warning("NLTK VADER not installed, sentiment analysis not available")
                return None
        elif sentiment_analyzer_type == "textblob":
            try:
                
                return {'type': 'textblob'}
            except ImportError:
                logging.warning("TextBlob not installed, sentiment analysis not available")
                return None
        elif sentiment_analyzer_type == "transformers":
            try:
                from transformers import pipeline
                sentiment_pipeline = pipeline("sentiment-analysis")
                return sentiment_pipeline
            except ImportError:
                logging.warning("Transformers not installed, sentiment analysis not available")
                return None
                
        return None
    
    def _initialize_harmful_content_detector(self):
        """Initialize harmful content detection capability"""
        harmful_detector_type = self.config.get("harmful_content_detector", "none")
        
        if harmful_detector_type == "custom":
            # Load custom harmful content categories and patterns
            harmful_categories = self.config.get("harmful_categories", {})
            
            # Default harmful categories if not provided
            default_categories = {
                'harmful_misleading': [
                    'fake news', 'misinformation', 'disinformation', 'misleading',
                    'false claim', 'conspiracy', 'pseudoscience'
                ],
                'harmful_offensive': [
                    'offensive', 'inappropriate', 'explicit', 'obscene', 'vulgar',
                    'profanity', 'slur', 'racist', 'sexist', 'discriminatory'
                ],
                'harmful_dangerous': [
                    'dangerous', 'harmful', 'hazardous', 'illegal', 'unethical',
                    'exploit', 'manipulate', 'weapon', 'violence'
                ],
                'harmful_privacy': [
                    'doxing', 'personal information', 'private data', 'confidential',
                    'leaked', 'exposed', 'breach', 'unauthorized access'
                ]
            }
            
            # Merge provided categories with defaults for any missing ones
            for category, keywords in default_categories.items():
                if category not in harmful_categories:
                    harmful_categories[category] = keywords
            
            return {
                'type': 'custom',
                'categories': harmful_categories,
                # Compile regex patterns for efficient matching
                'patterns': {
                    category: [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                             for kw in keywords]
                    for category, keywords in harmful_categories.items()
                }
            }
        elif harmful_detector_type == "transformers":
            try:
                from transformers import pipeline
                toxic_pipeline = pipeline("text-classification", 
                                        model="unitary/toxic-bert",
                                        return_all_scores=True)
                return {
                    'type': 'transformers',
                    'model': toxic_pipeline
                }
            except ImportError:
                logging.warning("Transformers not installed, harmful content detection not available")
                return None
                
        return None
    
    def _load_regulatory_concepts(self):
        """Load regulatory concepts from config or predefined list"""
        # Load from config if available
        regulatory_concepts = self.config.get("regulatory_concepts", {})
        
        # Default regulatory concepts if not provided
        default_concepts = {
            'GDPR': {
                'concepts': [
                    'lawfulness', 'fairness', 'transparency', 'purpose_limitation',
                    'data_minimization', 'accuracy', 'storage_limitation',
                    'integrity', 'confidentiality', 'accountability',
                    'data_subject_rights', 'consent', 'legitimate_interest',
                    'data_protection_officer', 'data_protection_impact_assessment',
                    'data_breach_notification', 'cross_border_transfers'
                ],
                'severity': 'high'
            },
            'HIPAA': {
                'concepts': [
                    'privacy_rule', 'security_rule', 'breach_notification_rule',
                    'minimum_necessary', 'authorization', 'designated_record_set',
                    'business_associate', 'covered_entity', 'protected_health_information',
                    'electronic_protected_health_information', 'de_identification',
                    'privacy_officer', 'notice_of_privacy_practices', 
                    'administrative_safeguards', 'physical_safeguards', 'technical_safeguards'
                ],
                'severity': 'high'
            },
            'CCPA': {
                'concepts': [
                    'personal_information', 'business', 'service_provider',
                    'third_party', 'sell', 'consumer_rights', 'right_to_know',
                    'right_to_delete', 'right_to_opt_out', 'financial_incentives',
                    'notice_at_collection', 'privacy_policy', 'methods_for_submitting_requests',
                    'verification', 'household', 'do_not_sell'
                ],
                'severity': 'medium'
            },
            'FINRA': {
                'concepts': [
                    'suitability', 'know_your_customer', 'communications_with_public',
                    'outside_business_activities', 'selling_away', 'private_securities_transactions',
                    'supervisory_procedures', 'records_retention', 'compliance_policies',
                    'financial_reporting', 'advertising', 'social_media_communications',
                    'anti_money_laundering', 'cybersecurity', 'best_execution'
                ],
                'severity': 'high'
            }
        }
        
        # Merge provided concepts with defaults for any missing regulations
        for regulation, data in default_concepts.items():
            if regulation not in regulatory_concepts:
                regulatory_concepts[regulation] = data
                
        return regulatory_concepts
    
    def extract(self, text, embeddings=None, compliance_mode='strict'):
        """
        Extract semantic concepts from text with compliance verification
        
        Args:
            text: Input text
            embeddings: Optional text embeddings for faster processing
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Dictionary with extracted concepts and compliance information
        """
        # Initialize result data structures
        extracted_concepts = {}
        
        # Extract basic concepts using keyword matching
        keyword_concepts = self._extract_keyword_concepts(text)
        extracted_concepts.update(keyword_concepts)
        
        # Extract concepts using embeddings if available
        if 'embedding' in self.concept_detectors and embeddings is not None:
            embedding_concepts = self._extract_embedding_concepts(text, embeddings)
            self._merge_concepts(extracted_concepts, embedding_concepts)
        
        # Extract concepts using ML model if available
        if 'ml' in self.concept_detectors:
            ml_concepts = self._extract_ml_concepts(text)
            self._merge_concepts(extracted_concepts, ml_concepts)
        
        # Extract topics if topic model is available
        if self.topic_model:
            topics = self._extract_topics(text)
            self._merge_concepts(extracted_concepts, topics)
            
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text, embeddings)
        extracted_concepts.update(sentiment)
        
        # Detect harmful content categories
        harmful_categories = self._detect_harmful_categories(text, embeddings)
        extracted_concepts.update(harmful_categories)
        
        # Identify regulatory concepts
        regulatory_concepts = self._identify_regulatory_concepts(extracted_concepts)
        
        # Verify concept compliance
        compliant_concepts = {}
        violations = []
        
        for concept, data in extracted_concepts.items():
            concept_compliance = self._verify_concept_compliance(concept, data, compliance_mode)
            if concept_compliance['is_compliant']:
                # Add compliance score to concept data
                data['compliance_score'] = concept_compliance['compliance_score']
                compliant_concepts[concept] = data
            else:
                # Record violation
                violations.append({
                    'concept': concept,
                    'compliance_error': concept_compliance['error'],
                    'severity': concept_compliance['severity']
                })
        
        # Determine overall compliance
        is_compliant = len(violations) == 0 or (
            compliance_mode == 'soft' and
            not any(v['severity'] == 'high' for v in violations)
        )
        
        return {
            'concepts': compliant_concepts,
            'is_compliant': is_compliant,
            'violations': violations if not is_compliant else [],
            'regulatory_concepts': regulatory_concepts,
            'metadata': {
                'total_concepts': len(extracted_concepts),
                'compliant_concepts': len(compliant_concepts),
                'violation_count': len(violations),
                'concept_sources': self._count_concept_sources(extracted_concepts)
            }
        }
    
    def _extract_keyword_concepts(self, text):
        """Extract concepts using keyword matching"""
        concepts = {}
        
        if 'keyword' not in self.concept_detectors:
            return concepts
            
        detector = self.concept_detectors['keyword']
        
        # Check for each concept's keywords in the text
        for concept, patterns in detector['patterns'].items():
            # Count matches for all patterns
            matches = []
            for pattern in patterns:
                matches.extend(pattern.finditer(text))
                
            # Calculate concept activation based on matches
            if matches:
                # Calculate activation score based on number of matches and text length
                # Normalize to [0, 1] range
                activation = min(1.0, len(matches) / (len(text.split()) / 20))
                
                # Store concept with activation score and matches
                concepts[concept] = {
                    'activation': activation,
                    'match_count': len(matches),
                    'detection_method': 'keyword',
                    'confidence': 0.8  # Higher confidence for explicit matches
                }
                
        return concepts
    
    def _extract_embedding_concepts(self, text, text_embedding=None):
        """Extract concepts using embedding similarity"""
        concepts = {}
        
        if 'embedding' not in self.concept_detectors:
            return concepts
            
        detector = self.concept_detectors['embedding']
        
        # Get text embedding if not provided
        if text_embedding is None:
            text_embedding = detector['model'].encode(text)
            
        # Calculate similarity with concept reference embeddings
        for concept, concept_embedding in detector['concept_embeddings'].items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(text_embedding, concept_embedding)
            
            # Store concept if similarity is significant
            if similarity > 0.4:  # Threshold for concept activation
                concepts[concept] = {
                    'activation': similarity,
                    'detection_method': 'embedding',
                    'confidence': similarity  # Confidence equals similarity
                }
                
        return concepts
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _extract_ml_concepts(self, text):
        """Extract concepts using ML model"""
        concepts = {}
        
        if 'ml' not in self.concept_detectors:
            return concepts
            
        detector = self.concept_detectors['ml']
        
        try:
            # Prepare input for the model
            if detector['vectorizer']:
                X = detector['vectorizer'].transform([text])
            else:
                X = [text]  # Assume model can handle raw text
                
            # Get predictions
            if hasattr(detector['model'], 'predict_proba'):
                # Get probability scores for each class
                proba = detector['model'].predict_proba(X)[0]
                # Get class names
                classes = detector['model'].classes_
                
                # Store concepts with probability scores
                for i, cls in enumerate(classes):
                    if proba[i] > 0.3:  # Threshold for concept activation
                        concepts[cls] = {
                            'activation': float(proba[i]),
                            'detection_method': 'ml_model',
                            'confidence': float(proba[i])
                        }
            else:
                # Binary prediction for each concept
                predictions = detector['model'].predict(X)
                if hasattr(predictions, '__iter__'):
                    for i, pred in enumerate(predictions[0]):
                        if pred > 0.5:
                            concept = f"concept_{i}"
                            concepts[concept] = {
                                'activation': float(pred),
                                'detection_method': 'ml_model',
                                'confidence': float(pred)
                            }
        except Exception as e:
            logging.warning(f"Error using ML concept detector: {str(e)}")
            
        return concepts
    
    def _extract_topics(self, text):
        """Extract topics using topic modeling"""
        topics = {}
        
        if not self.topic_model:
            return topics
            
        try:
            model_type = self.topic_model['type']
            
            if model_type == 'lda' or model_type == 'nmf':
                vectorizer = self.topic_model['vectorizer']
                model = self.topic_model['model']
                
                # Train model if not already trained
                if not self.topic_model['is_trained']:
                    # Need a corpus to train on - for now, just use this text
                    # In a real implementation, would train on a larger corpus
                    X = vectorizer.fit_transform([text])
                    model.fit(X)
                    self.topic_model['is_trained'] = True
                    
                    # Get feature names for topic interpretation
                    feature_names = vectorizer.get_feature_names_out()
                    self.topic_model['feature_names'] = feature_names
                else:
                    # Use trained model
                    X = vectorizer.transform([text])
                    feature_names = self.topic_model['feature_names']
                
                # Get topic distribution for the text
                topic_distribution = model.transform(X)[0]
                
                # Store topics with significant activation
                for i, score in enumerate(topic_distribution):
                    if score > 0.1:  # Threshold for topic activation
                        # Get top words for this topic
                        if hasattr(model, 'components_'):
                            top_words_idx = model.components_[i].argsort()[:-10:-1]
                            top_words = [feature_names[idx] for idx in top_words_idx]
                        else:
                            top_words = []
                            
                        topic_name = f"topic_{i}"
                        topics[topic_name] = {
                            'activation': float(score),
                            'detection_method': f'topic_model_{model_type}',
                            'confidence': 0.7,  # Lower confidence for unsupervised topics
                            'top_words': top_words
                        }
            
            elif model_type == 'bertopic':
                model = self.topic_model['model']
                
                # Train model if not already trained
                if not self.topic_model['is_trained']:
                    # Need a corpus to train on - for now, just use this text
                    # In a real implementation, would train on a larger corpus
                    model.fit([text])
                    self.topic_model['is_trained'] = True
                
                # Get topics for the text
                topics_pred, probs = model.transform([text])
                
                # Store topics with significant activation
                for topic_idx, prob in zip(topics_pred[0], probs[0]):
                    if prob > 0.1 and topic_idx != -1:  # Threshold and not outlier
                        # Get topic info
                        topic_info = model.get_topic(topic_idx)
                        top_words = [word for word, _ in topic_info]
                        
                        topic_name = f"topic_{topic_idx}"
                        topics[topic_name] = {
                            'activation': float(prob),
                            'detection_method': 'topic_model_bertopic',
                            'confidence': 0.7,  # Lower confidence for unsupervised topics
                            'top_words': top_words
                        }
                        
        except Exception as e:
            logging.warning(f"Error extracting topics: {str(e)}")
            
        return topics
    
    def _analyze_sentiment(self, text, embeddings=None):
        """Analyze sentiment in text"""
        sentiment = {}
        
        if not self.sentiment_analyzer:
            return sentiment
            
        try:
            if hasattr(self.sentiment_analyzer, 'polarity_scores'):
                # VADER sentiment analyzer
                scores = self.sentiment_analyzer.polarity_scores(text)
                
                sentiment['sentiment_positive'] = float(scores['pos'])
                sentiment['sentiment_negative'] = float(scores['neg'])
                sentiment['sentiment_neutral'] = float(scores['neu'])
                sentiment['sentiment_compound'] = float(scores['compound'])
                
                # Add detection method
                for key in sentiment:
                    sentiment[key] = {
                        'activation': sentiment[key],
                        'detection_method': 'vader_sentiment',
                        'confidence': 0.8
                    }
                    
            elif isinstance(self.sentiment_analyzer, dict) and self.sentiment_analyzer.get('type') == 'textblob':
                # TextBlob sentiment
                from textblob import TextBlob
                blob = TextBlob(text)
                
                # TextBlob provides polarity (-1 to 1) and subjectivity (0 to 1)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert polarity to positive/negative scores
                if polarity >= 0:
                    sentiment['sentiment_positive'] = {
                        'activation': float(polarity),
                        'detection_method': 'textblob_sentiment',
                        'confidence': 0.7
                    }
                    sentiment['sentiment_negative'] = {
                        'activation': 0.0,
                        'detection_method': 'textblob_sentiment',
                        'confidence': 0.7
                    }
                else:
                    sentiment['sentiment_positive'] = {
                        'activation': 0.0,
                        'detection_method': 'textblob_sentiment',
                        'confidence': 0.7
                    }
                    sentiment['sentiment_negative'] = {
                        'activation': float(-polarity),
                        'detection_method': 'textblob_sentiment',
                        'confidence': 0.7
                    }
                    
                sentiment['sentiment_subjectivity'] = {
                    'activation': float(subjectivity),
                    'detection_method': 'textblob_sentiment',
                    'confidence': 0.7
                }
                
            else:
                # Assume transformers pipeline
                result = self.sentiment_analyzer(text)[0]
                
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    sentiment['sentiment_positive'] = {
                        'activation': float(score),
                        'detection_method': 'transformers_sentiment',
                        'confidence': float(score)
                    }
                    sentiment['sentiment_negative'] = {
                        'activation': 0.0,
                        'detection_method': 'transformers_sentiment',
                        'confidence': float(score)
                    }
                elif 'negative' in label:
                    sentiment['sentiment_positive'] = {
                        'activation': 0.0,
                        'detection_method': 'transformers_sentiment',
                        'confidence': float(score)
                    }
                    sentiment['sentiment_negative'] = {
                        'activation': float(score),
                        'detection_method': 'transformers_sentiment',
                        'confidence': float(score)
                    }
                else:
                    # Neutral sentiment
                    sentiment['sentiment_neutral'] = {
                        'activation': float(score),
                        'detection_method': 'transformers_sentiment',
                        'confidence': float(score)
                    }
                
        except Exception as e:
            logging.warning(f"Error analyzing sentiment: {str(e)}")
            
        return sentiment
    
    def _detect_harmful_categories(self, text, embeddings=None):
        """Detect potentially harmful content categories"""
        harmful_categories = {}
        
        if not self.harmful_content_detector:
            return harmful_categories
            
        try:
            detector_type = self.harmful_content_detector.get('type')
            
            if detector_type == 'custom':
                # Use custom keyword patterns
                for category, patterns in self.harmful_content_detector['patterns'].items():
                    # Count matches for all patterns
                    matches = []
                    for pattern in patterns:
                        matches.extend(pattern.finditer(text))
                        
                    # Calculate category activation based on matches
                    if matches:
                        # Calculate activation score
                        activation = min(1.0, len(matches) / (len(text.split()) / 20))
                        
                        # Store category with activation score
                        harmful_categories[category] = {
                            'activation': activation,
                            'match_count': len(matches),
                            'detection_method': 'keyword_pattern',
                            'confidence': 0.7  # Moderate confidence for keyword detection
                        }
            
            elif detector_type == 'transformers':
                # Use transformer-based toxicity detection
                model = self.harmful_content_detector['model']
                result = model(text)[0]
                
                # Extract categories and scores
                for category_result in result:
                    label = category_result['label'].lower()
                    score = category_result['score']
                    
                    # Map to our category naming scheme
                    if 'toxic' in label or 'toxicity' in label:
                        category = 'harmful_offensive'
                    elif 'obscene' in label:
                        category = 'harmful_offensive'
                    elif 'threat' in label:
                        category = 'harmful_dangerous'
                    elif 'insult' in label:
                        category = 'harmful_offensive'
                    elif 'identity' in label and ('hate' in label or 'attack' in label):
                        category = 'harmful_offensive'
                    else:
                        category = f"harmful_{label}"
                    
                    # Store if score is significant
                    if score > 0.3:  # Threshold for harmful content detection
                        harmful_categories[category] = {
                            'activation': float(score),
                            'detection_method': 'transformers_toxicity',
                            'confidence': float(score)
                        }
        
        except Exception as e:
            logging.warning(f"Error detecting harmful categories: {str(e)}")
            
        return harmful_categories
    
    def _identify_regulatory_concepts(self, concepts):
        """Identify which regulatory frameworks are relevant based on concepts"""
        regulatory_relevance = {}
        
        for regulation, data in self.regulatory_concepts.items():
            # Count how many regulation-specific concepts are present
            matches = []
            for reg_concept in data['concepts']:
                # Convert to standardized format for comparison
                std_concept = reg_concept.lower().replace(' ', '_')
                
                # Check for exact or partial matches
                for concept in concepts.keys():
                    if std_concept == concept.lower() or std_concept in concept.lower():
                        matches.append(concept)
                        break
            
            # Calculate relevance score based on matches
            if matches:
                relevance_score = len(matches) / len(data['concepts'])
                
                regulatory_relevance[regulation] = {
                    'relevance_score': relevance_score,
                    'matched_concepts': matches,
                    'severity': data['severity']
                }
        
        return regulatory_relevance
    
    def _merge_concepts(self, base_concepts, new_concepts):
        """Merge new concepts into base concepts with smart handling of duplicates"""
        for concept, data in new_concepts.items():
            if concept in base_concepts:
                # Concept already exists, merge data
                existing_data = base_concepts[concept]
                
                # Take maximum of activation scores
                activation = max(existing_data.get('activation', 0), data.get('activation', 0))
                
                # Take value from higher confidence source
                if data.get('confidence', 0) > existing_data.get('confidence', 0):
                    # Use new data but preserve activation
                    base_concepts[concept] = data.copy()
                    base_concepts[concept]['activation'] = activation
                else:
                    # Keep existing data but update activation
                    base_concepts[concept]['activation'] = activation
                    
                # Track multiple detection methods
                if 'detection_methods' not in base_concepts[concept]:
                    base_concepts[concept]['detection_methods'] = [existing_data.get('detection_method', 'unknown')]
                
                base_concepts[concept]['detection_methods'].append(data.get('detection_method', 'unknown'))
            else:
                # New concept, add it to base
                base_concepts[concept] = data
    
    def _verify_concept_compliance(self, concept, data, compliance_mode):
        """Verify if a concept complies with regulatory requirements"""
        # Check if concept is explicitly prohibited
        if concept.startswith('harmful_') and data['activation'] > 0.5:
            severity = 'high' if data['activation'] > 0.8 else 'medium'
            return {
                'is_compliant': False,
                'compliance_score': 0.0,
                'error': f"Prohibited content category detected: {concept}",
                'severity': severity
            }
        
        # Check regulatory concept compliance
        for regulation, reg_data in self.regulatory_concepts.items():
            # Convert to standardized format for comparison
            std_concepts = [c.lower().replace(' ', '_') for c in reg_data['concepts']]
            
            if concept.lower() in std_concepts and compliance_mode == 'strict':
                # This is a regulatory concept, verify proper handling
                # In a real implementation, would check if handling is correct
                # For now, just flag it as potentially requiring attention
                return {
                    'is_compliant': True,  # Set to compliant but with warning
                    'compliance_score': 0.6,
                    'error': f"Regulatory concept '{concept}' detected from {regulation}",
                    'severity': 'low'
                }
        
        # Calculate compliance score (based on concept type and activation)
        compliance_score = self._calculate_concept_compliance_score(concept, data)
        
        is_compliant = compliance_score >= 0.7 or compliance_mode == 'soft'
        
        if not is_compliant:
            return {
                'is_compliant': False,
                'compliance_score': compliance_score,
                'error': f"Concept '{concept}' has low compliance score",
                'severity': 'medium' if compliance_score < 0.5 else 'low'
            }
        
        return {
            'is_compliant': True,
            'compliance_score': compliance_score,
            'error': None,
            'severity': 'none'
        }
    
    def _calculate_concept_compliance_score(self, concept, data):
        """Calculate compliance score for a concept"""
        # Base score depends on concept type
        if concept.startswith('harmful_'):
            # Harmful concepts have base compliance inversely proportional to activation
            base_score = 1.0 - data['activation']
        elif concept.startswith('sentiment_'):
            # Sentiment concepts are generally compliant
            base_score = 0.9
        elif concept.startswith('topic_'):
            # Topics are generally compliant
            base_score = 0.8
        else:
            # Other concepts get moderate base score
            base_score = 0.7
            
        # Adjust based on confidence
        confidence_adjustment = 0.0
        if 'confidence' in data:
            # Higher confidence can both increase or decrease score
            if concept.startswith('harmful_'):
                # For harmful content, higher confidence lowers compliance score
                confidence_adjustment = -0.1 * (data['confidence'] - 0.5)
            else:
                # For other concepts, higher confidence increases compliance score
                confidence_adjustment = 0.1 * (data['confidence'] - 0.5)
                
        # Calculate final score with bounds
        final_score = max(0.0, min(1.0, base_score + confidence_adjustment))
        return final_score
    
    def _count_concept_sources(self, concepts):
        """Count concepts by detection method for metadata"""
        counts = {}
        for concept, data in concepts.items():
            method = data.get('detection_method', 'unknown')
            if method not in counts:
                counts[method] = 0
            counts[method] += 1
        return counts
