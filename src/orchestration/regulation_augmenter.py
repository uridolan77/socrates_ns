import numpy as np
import re
from collections import Counter, defaultdict
import logging

class EfficientRegulationAugmenter:
    """
    System for augmenting prompts with relevant regulatory context
    to improve compliance during generation.
    """
    def __init__(self, regulatory_retriever, language_model):
        self.retriever = regulatory_retriever
        self.language_model = language_model
        
        # Augmentation configuration
        self.max_regulations = 5
        self.max_tokens_per_regulation = 200
        self.min_relevance_score = 0.65
        
        # Initialize NLP components for concept extraction
        self._initialize_nlp_components()
        
        # Tracking statistics
        self.augmentation_stats = {
            "avg_regulations_used": 0,
            "avg_total_tokens_added": 0,
            "augmentation_count": 0
        }
        
        # Cached concept extractions for efficiency
        self.concept_cache = {}
        
        # Domain-specific concept mappings
        self.domain_concepts = self._initialize_domain_concepts()
        
        # Regulatory relevance boosters by domain
        self.relevance_boosters = self._initialize_relevance_boosters()
        
    def _initialize_nlp_components(self):
        """Initialize NLP components for concept extraction"""
        # In a real implementation, these would be actual models or APIs
        # For this implementation, we'll use simpler approaches
        self.nlp_components = {
            "tokenizer": None,  # Would be an actual tokenizer
            "embedding_model": None,  # Would be an actual embedding model
            "keyword_extractor": None,  # Would be a keyword extraction model
            "entity_recognizer": None  # Would be a named entity recognition model
        }
        
    def _initialize_domain_concepts(self):
        """Initialize domain-specific concept mappings"""
        return {
            "data_privacy": [
                "personal data", "data subject", "consent", "processing", "controller", 
                "processor", "legitimate interest", "data protection", "anonymization",
                "pseudonymization", "right to access", "right to be forgotten", "data breach",
                "data minimization", "purpose limitation", "storage limitation"
            ],
            "healthcare": [
                "protected health information", "phi", "covered entity", "business associate",
                "privacy rule", "security rule", "notice of privacy practices", "authorization",
                "minimum necessary", "treatment", "payment", "healthcare operations", 
                "patient rights", "medical records", "health information"
            ],
            "finance": [
                "aml", "kyc", "anti-money laundering", "know your customer", "customer due diligence",
                "suspicious activity", "transaction monitoring", "regulatory reporting", 
                "beneficial ownership", "risk assessment", "politically exposed person",
                "financial crime", "fraud prevention", "sanctions screening"
            ],
            "ai_ethics": [
                "fairness", "transparency", "accountability", "explainability", "bias",
                "discrimination", "privacy", "security", "governance", "impact assessment",
                "human oversight", "algorithmic accountability", "robustness", "safety",
                "inclusivity", "diversity", "ethical ai", "responsible ai"
            ]
        }
        
    def _initialize_relevance_boosters(self):
        """Initialize relevance boosters for different domains"""
        return {
            "data_privacy": {
                "gdpr": 1.5,  # Higher relevance for GDPR in data privacy context
                "ccpa": 1.3,  # Higher relevance for CCPA in data privacy context
                "personal_data": 1.4  # Higher relevance for personal data concepts
            },
            "healthcare": {
                "hipaa": 1.5,  # Higher relevance for HIPAA in healthcare context
                "phi": 1.4,  # Higher relevance for PHI in healthcare context
                "medical": 1.3  # Higher relevance for medical concepts
            },
            "finance": {
                "finreg": 1.5,  # Higher relevance for financial regulations
                "aml": 1.4,  # Higher relevance for AML in finance context
                "kyc": 1.4  # Higher relevance for KYC in finance context
            }
        }
        
    def augment_prompt(self, prompt, context=None, available_tokens=1000):
        """
        Augment prompt with relevant regulatory content
        
        Args:
            prompt: Original prompt text
            context: Optional context information
            available_tokens: Maximum tokens available for augmentation
            
        Returns:
            Tuple of (augmented_prompt, used_regulations)
        """
        # Skip augmentation if no tokens available
        if available_tokens <= 0:
            return prompt, []
            
        # Extract domain from context if available
        domain = self._extract_domain(context)
        
        # Analyze prompt to identify regulatory concepts
        relevant_concepts = self._extract_relevant_concepts(prompt, domain)
        logging.info(f"Extracted {len(relevant_concepts)} regulatory concepts from prompt")
        
        # Retrieve relevant regulatory content
        retrieved_regulations = self._retrieve_relevant_regulations(
            prompt, context, relevant_concepts, domain
        )
        logging.info(f"Retrieved {len(retrieved_regulations)} relevant regulations")
        
        # Filter regulations based on relevance and token budget
        selected_regulations = self._select_regulations(
            retrieved_regulations, 
            available_tokens,
            self.max_regulations
        )
        logging.info(f"Selected {len(selected_regulations)} regulations within token budget")
        
        # Format regulations for insertion
        formatted_regulations = self._format_regulations(selected_regulations, domain)
        
        # Create augmented prompt
        augmented_prompt = self._create_augmented_prompt(prompt, formatted_regulations, domain)
        
        # Update statistics
        self._update_augmentation_stats(selected_regulations)
        
        # Return augmented prompt and regulation metadata
        regulation_metadata = [
            {
                "id": reg["document_id"],
                "framework": reg["framework_id"],
                "relevance": reg["score"]
            }
            for reg in selected_regulations
        ]
        
        return augmented_prompt, regulation_metadata
        
    def _extract_relevant_concepts(self, prompt, domain=None):
        """
        Extract relevant regulatory concepts from prompt using NLP techniques.
        
        Args:
            prompt: Input prompt text
            domain: Optional domain for more targeted extraction
            
        Returns:
            List of relevant regulatory concepts
        """
        # Check cache first
        cache_key = f"{hash(prompt)}:{domain or 'general'}"
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
            
        # In a real implementation, this would use sophisticated NLP models
        # For this implementation, we'll use a combination of simpler techniques
        
        # 1. Tokenize and clean prompt
        prompt_lower = prompt.lower()
        prompt_tokens = re.findall(r'\b\w+\b', prompt_lower)
        
        # 2. Extract keywords (simple frequency-based)
        word_counts = Counter(prompt_tokens)
        keywords = [word for word, count in word_counts.most_common(20) 
                   if len(word) > 3 and word not in STOPWORDS]
        
        # 3. Extract n-grams (for multi-word concepts)
        ngrams = self._extract_ngrams(prompt_lower, n=3)
        
        # 4. Combine with domain-specific concepts if available
        domain_specific_concepts = []
        if domain and domain in self.domain_concepts:
            domain_specific_concepts = [
                concept for concept in self.domain_concepts[domain]
                if concept.lower() in prompt_lower or
                any(keyword in concept.lower() for keyword in keywords)
            ]
        
        # 5. Expand with semantically related concepts
        expanded_concepts = self._expand_concepts(keywords, domain)
        
        # 6. Combine all concepts and rank by relevance
        all_concepts = list(set(keywords + ngrams + domain_specific_concepts + expanded_concepts))
        ranked_concepts = self._rank_concepts(all_concepts, prompt_lower, domain)
        
        # Store in cache for future use
        self.concept_cache[cache_key] = ranked_concepts
        
        return ranked_concepts
        
    def _extract_ngrams(self, text, n=3):
        """Extract n-grams from text that match known regulatory concepts"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Generate n-grams up to n words
        ngrams = []
        for i in range(1, min(n+1, len(words)+1)):
            for j in range(len(words)-i+1):
                ngram = " ".join(words[j:j+i])
                # Only include if it's a known regulatory concept
                if self._is_known_regulatory_concept(ngram):
                    ngrams.append(ngram)
                    
        return ngrams
        
    def _is_known_regulatory_concept(self, ngram):
        """Check if an n-gram is a known regulatory concept"""
        # Check if ngram appears in any domain's concepts
        for domain_concepts in self.domain_concepts.values():
            if any(concept.lower() == ngram.lower() or ngram.lower() in concept.lower() 
                  for concept in domain_concepts):
                return True
                
        # List of common regulatory terms not covered by domain concepts
        common_regulatory_terms = [
            "compliance", "regulation", "policy", "standard", "requirement",
            "guideline", "framework", "law", "rule", "legislation", "directive",
            "statute", "legal", "regulatory", "compliance program", "audit",
            "assessment", "controls", "safeguards", "protection", "governance"
        ]
        
        return ngram.lower() in common_regulatory_terms
        
    def _expand_concepts(self, concepts, domain=None):
        """Expand concepts with semantically related regulatory concepts"""
        # In a real implementation, this would use embeddings or a knowledge graph
        # For this implementation, we'll use a simple mapping
        
        expanded = []
        concept_mappings = {
            "privacy": ["data protection", "confidentiality", "personal data"],
            "data": ["information", "personal information", "records"],
            "consent": ["permission", "authorization", "opt-in"],
            "security": ["safeguards", "protection", "controls"],
            "compliance": ["adherence", "conformity", "regulations"],
            "user": ["data subject", "individual", "consumer"],
            "health": ["medical", "healthcare", "patient"],
            "financial": ["monetary", "banking", "economic"],
            "report": ["disclosure", "notification", "documentation"]
        }
        
        for concept in concepts:
            if concept in concept_mappings:
                expanded.extend(concept_mappings[concept])
                
        # Add domain-specific expansions
        if domain == "data_privacy":
            privacy_mappings = {
                "user": ["data subject"],
                "delete": ["right to be forgotten", "erasure"],
                "collect": ["processing", "data collection"],
                "use": ["processing", "purpose"],
                "store": ["storage", "retention"]
            }
            for concept in concepts:
                if concept in privacy_mappings:
                    expanded.extend(privacy_mappings[concept])
        
        elif domain == "healthcare":
            health_mappings = {
                "medical": ["healthcare", "clinical", "treatment"],
                "doctor": ["provider", "covered entity"],
                "patient": ["individual", "data subject"],
                "records": ["PHI", "medical records"]
            }
            for concept in concepts:
                if concept in health_mappings:
                    expanded.extend(health_mappings[concept])
        
        elif domain == "finance":
            finance_mappings = {
                "customer": ["client", "account holder"],
                "verify": ["KYC", "due diligence"],
                "suspicious": ["AML", "fraud", "risk"],
                "transaction": ["payment", "transfer", "activity"]
            }
            for concept in concepts:
                if concept in finance_mappings:
                    expanded.extend(finance_mappings[concept])
                    
        return list(set(expanded))
        
    def _rank_concepts(self, concepts, prompt_text, domain=None):
        """Rank concepts by relevance to the prompt and domain"""
        ranked_concepts = []
        
        for concept in concepts:
            # Calculate base score from frequency in prompt
            frequency = prompt_text.count(concept.lower())
            length_factor = min(1.0, len(concept) / 20)  # Longer concepts are more specific
            base_score = frequency * length_factor
            
            # Apply domain boost if applicable
            domain_boost = 1.0
            if domain and domain in self.relevance_boosters:
                boosters = self.relevance_boosters[domain]
                for booster_term, boost_factor in boosters.items():
                    if booster_term in concept.lower():
                        domain_boost = boost_factor
                        break
                        
            # Final score
            final_score = base_score * domain_boost
            
            ranked_concepts.append((concept, final_score))
            
        # Sort by score and extract just the concept names
        ranked_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, score in ranked_concepts if score > 0]
        
    def _retrieve_relevant_regulations(self, prompt, context, concepts, domain=None):
        """
        Retrieve relevant regulatory content based on prompt and concepts.
        
        Args:
            prompt: Original prompt text
            context: Optional context information
            concepts: Extracted regulatory concepts
            domain: Optional domain for targeted retrieval
            
        Returns:
            List of relevant regulatory documents
        """
        # Perform base retrieval using prompt
        base_retrieved = self.retriever.retrieve(
            prompt, context=context, top_k=self.max_regulations * 2
        )
        
        # Also search by specific concepts
        concept_regulations = []
        for concept in concepts[:5]:  # Use top 5 concepts for efficiency
            concept_results = self.retriever.retrieve_by_concept(
                concept, top_k=3
            )
            concept_regulations.extend(concept_results)
            
        # Combine results, removing duplicates
        combined_regulations = base_retrieved.copy()
        seen_doc_ids = {reg["document_id"] for reg in base_retrieved}
        
        for reg in concept_regulations:
            if reg["document_id"] not in seen_doc_ids:
                combined_regulations.append(reg)
                seen_doc_ids.add(reg["document_id"])
                
        # Apply domain-specific boosts if available
        if domain:
            combined_regulations = self._apply_domain_boosts(combined_regulations, domain)
            
        # Filter by minimum relevance
        filtered_regulations = [
            reg for reg in combined_regulations
            if reg["score"] >= self.min_relevance_score
        ]
        
        # Sort by relevance
        filtered_regulations.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered_regulations
        
    def _apply_domain_boosts(self, regulations, domain):
        """Apply domain-specific relevance boosts to regulations"""
        boosted_regulations = []
        
        # Get domain boosters if available
        domain_boosters = self.relevance_boosters.get(domain, {})
        
        for reg in regulations:
            # Create a copy to avoid modifying original
            boosted_reg = dict(reg)
            
            # Check if any boosters apply
            framework_id = reg.get("framework_id", "").lower()
            content = reg.get("excerpt", "").lower()
            
            for term, boost in domain_boosters.items():
                if term in framework_id or term in content:
                    # Apply boost to score
                    boosted_reg["score"] = min(1.0, reg["score"] * boost)
                    boosted_reg["boosted"] = True
                    break
                    
            boosted_regulations.append(boosted_reg)
            
        return boosted_regulations
        
    def _select_regulations(self, regulations, available_tokens, max_count):
        """
        Select regulations within token budget using a sophisticated selection algorithm.
        
        Args:
            regulations: Retrieved regulations
            available_tokens: Maximum tokens available
            max_count: Maximum number of regulations to include
            
        Returns:
            Selected regulations
        """
        if not regulations:
            return []
            
        # Greedy selection approach with diversity
        selected = []
        tokens_used = 0
        framework_counts = defaultdict(int)
        
        # First pass: estimate tokens for each regulation
        for reg in regulations:
            # Use accurate token estimation
            reg["estimated_tokens"] = self._estimate_tokens_accurately(reg["excerpt"])
            
        # Sort by score for initial processing
        scored_regulations = sorted(regulations, key=lambda x: x["score"], reverse=True)
        
        # First, pick the highest scored regulation from each framework (diversity)
        seen_frameworks = set()
        
        for reg in scored_regulations:
            framework_id = reg.get("framework_id", "unknown")
            
            if framework_id not in seen_frameworks:
                # Check if we can add this regulation
                if tokens_used + reg["estimated_tokens"] <= available_tokens and len(selected) < max_count:
                    selected.append(reg)
                    tokens_used += reg["estimated_tokens"]
                    seen_frameworks.add(framework_id)
                    framework_counts[framework_id] += 1
        
        # Second pass: add more regulations, favoring diversity
        for reg in scored_regulations:
            if reg in selected:
                continue
                
            framework_id = reg.get("framework_id", "unknown")
            
            # Apply diversity penalty (lower score for frameworks already represented)
            diversity_penalty = framework_counts[framework_id] * 0.1
            adjusted_score = reg["score"] - diversity_penalty
            
            # Skip low-scoring regulations after penalty
            if adjusted_score < self.min_relevance_score:
                continue
                
            # Truncate if needed
            truncated_tokens = min(reg["estimated_tokens"], self.max_tokens_per_regulation)
            if truncated_tokens < reg["estimated_tokens"]:
                # Create a truncated copy
                truncated_reg = dict(reg)
                truncated_reg["excerpt"] = self._truncate_text(
                    reg["excerpt"], 
                    self.max_tokens_per_regulation
                )
                truncated_reg["estimated_tokens"] = truncated_tokens
                truncated_reg["truncated"] = True
                reg = truncated_reg
                
            # Check if we can add this regulation
            if tokens_used + truncated_tokens <= available_tokens and len(selected) < max_count:
                selected.append(reg)
                tokens_used += truncated_tokens
                framework_counts[framework_id] += 1
            else:
                # Stop if we've reached token limit or max count
                break
                
        # Sort final selection by relevance
        selected.sort(key=lambda x: x["score"], reverse=True)
        
        return selected
        
    def _format_regulations(self, regulations, domain=None):
        """
        Format regulations for insertion into prompt with domain-specific formatting.
        
        Args:
            regulations: Selected regulations to format
            domain: Optional domain for domain-specific formatting
            
        Returns:
            Formatted regulatory text
        """
        if not regulations:
            return ""
            
        # Base formatting
        formatted_text = "\n# RELEVANT REGULATORY GUIDELINES\n"
        
        # Apply domain-specific formatting if available
        if domain == "data_privacy":
            return self._format_privacy_regulations(regulations)
        elif domain == "healthcare":
            return self._format_healthcare_regulations(regulations)
        elif domain == "finance":
            return self._format_finance_regulations(regulations)
        elif domain == "ai_ethics":
            return self._format_ai_ethics_regulations(regulations)
            
        # Default formatting
        for i, reg in enumerate(regulations, 1):
            formatted_text += f"## {i}. {reg.get('framework_id', 'Regulatory Framework')}\n"
            formatted_text += f"**Source**: {reg.get('document_id', 'Regulatory Document')}\n\n"
            formatted_text += f"{reg['excerpt']}\n\n"
            
        return formatted_text
        
    def _format_privacy_regulations(self, regulations):
        """Format privacy-specific regulations"""
        formatted_text = "\n# DATA PRIVACY REGULATORY GUIDELINES\n"
        formatted_text += "_The following privacy regulations are relevant to your request:_\n\n"
        
        for i, reg in enumerate(regulations, 1):
            framework = reg.get('framework_id', 'Privacy Framework')
            formatted_text += f"## {i}. {framework}\n"
            
            # Add specific formatting for known frameworks
            if "GDPR" in framework:
                formatted_text += f"**GDPR Provision** - {reg.get('document_id', 'Article')}\n\n"
            elif "CCPA" in framework:
                formatted_text += f"**CCPA Provision** - {reg.get('document_id', 'Section')}\n\n"
            else:
                formatted_text += f"**Source**: {reg.get('document_id', 'Regulatory Document')}\n\n"
                
            formatted_text += f"{reg['excerpt']}\n\n"
            
            # Add clarification for privacy regulations
            if i == len(regulations):
                formatted_text += "_Note: When processing personal data, ensure proper legal basis, transparency, and respect for data subject rights._\n\n"
                
        return formatted_text
        
    def _format_healthcare_regulations(self, regulations):
        """Format healthcare-specific regulations"""
        formatted_text = "\n# HEALTHCARE REGULATORY GUIDELINES\n"
        formatted_text += "_The following healthcare regulations are relevant to your request:_\n\n"
        
        for i, reg in enumerate(regulations, 1):
            framework = reg.get('framework_id', 'Healthcare Framework')
            formatted_text += f"## {i}. {framework}\n"
            
            # Add specific formatting for known frameworks
            if "HIPAA" in framework:
                formatted_text += f"**HIPAA Provision** - {reg.get('document_id', 'Section')}\n\n"
            else:
                formatted_text += f"**Source**: {reg.get('document_id', 'Regulatory Document')}\n\n"
                
            formatted_text += f"{reg['excerpt']}\n\n"
            
            # Add clarification for healthcare regulations
            if i == len(regulations):
                formatted_text += "_Note: Healthcare information requires special protection. Always consider patient privacy and minimum necessary principle._\n\n"
                
        return formatted_text
        
    def _format_finance_regulations(self, regulations):
        """Format finance-specific regulations"""
        formatted_text = "\n# FINANCIAL REGULATORY GUIDELINES\n"
        formatted_text += "_The following financial regulations are relevant to your request:_\n\n"
        
        for i, reg in enumerate(regulations, 1):
            framework = reg.get('framework_id', 'Financial Framework')
            formatted_text += f"## {i}. {framework}\n"
            formatted_text += f"**Source**: {reg.get('document_id', 'Regulatory Document')}\n\n"
            formatted_text += f"{reg['excerpt']}\n\n"
            
            # Add clarification for financial regulations
            if i == len(regulations):
                formatted_text += "_Note: Financial regulations require careful adherence. Consider KYC, AML, and reporting requirements where applicable._\n\n"
                
        return formatted_text
        
    def _format_ai_ethics_regulations(self, regulations):
        """Format AI ethics-specific regulations"""
        formatted_text = "\n# AI ETHICS GUIDELINES\n"
        formatted_text += "_The following AI ethics guidelines are relevant to your request:_\n\n"
        
        for i, reg in enumerate(regulations, 1):
            framework = reg.get('framework_id', 'AI Ethics Framework')
            formatted_text += f"## {i}. {framework}\n"
            formatted_text += f"**Source**: {reg.get('document_id', 'Guidelines Document')}\n\n"
            formatted_text += f"{reg['excerpt']}\n\n"
            
            # Add clarification for AI ethics
            if i == len(regulations):
                formatted_text += "_Note: AI systems should be designed and deployed with fairness, transparency, accountability, and human oversight in mind._\n\n"
                
        return formatted_text
        
    def _create_augmented_prompt(self, prompt, formatted_regulations, domain=None):
        """
        Create augmented prompt with regulations, using domain-specific placement.
        
        Args:
            prompt: Original prompt text
            formatted_regulations: Formatted regulatory text
            domain: Optional domain for domain-specific placement
            
        Returns:
            Augmented prompt with regulations
        """
        if not formatted_regulations:
            return prompt
            
        # Domain-specific placement
        if domain == "data_privacy":
            # For privacy, regulations typically should come before specific instructions
            return self._insert_before_instructions(prompt, formatted_regulations)
        elif domain in ["healthcare", "finance"]:
            # For healthcare and finance, regulations are crucial context
            return formatted_regulations + "\n" + prompt
        elif domain == "ai_ethics":
            # For AI ethics, regulations should frame the response
            return formatted_regulations + "\n" + prompt
        else:
            # Default placement: add regulations before prompt
            return formatted_regulations + "\n" + prompt
            
    def _insert_before_instructions(self, prompt, formatted_regulations):
        """Insert regulations before specific instructions in the prompt"""
        # Look for common instruction patterns
        instruction_patterns = [
            r"\bplease\s+(write|create|generate|provide)",
            r"\bI\s+need\s+(you\s+)?to\s+(write|create|generate|provide)",
            r"\bYour\s+task\s+is\s+to",
            r"\bWrite\s+a",
            r"\bGenerate\s+a"
        ]
        
        for pattern in instruction_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                # Insert before the instruction
                start_idx = match.start()
                return prompt[:start_idx] + formatted_regulations + "\n" + prompt[start_idx:]
                
        # Default: add at beginning
        return formatted_regulations + "\n" + prompt
        
    def _estimate_tokens_accurately(self, text):
        """
        Estimate token count more accurately based on the language model's tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # If we have access to the language model's tokenizer, use it directly
        if hasattr(self.language_model, 'tokenize') and callable(self.language_model.tokenize):
            try:
                tokens = self.language_model.tokenize(text)
                return len(tokens)
            except:
                pass  # Fall back to heuristic methods if tokenization fails
                
        # Advanced heuristic based on token patterns
        # Most tokenizers split on spaces and punctuation, with subtleties for
        # capitalization, special characters, etc.
        
        # 1. Count words (split by whitespace)
        words = text.split()
        word_count = len(words)
        
        # 2. Count punctuation characters and symbols
        punctuation_count = sum(1 for c in text if c in ".,;:!?()[]{}\"'`-_+=<>@#$%^&*|\\~/")
        
        # 3. Count numbers (many tokenizers treat digits specially)
        number_count = len(re.findall(r'\b\d+\b', text))
        
        # 4. Count special tokens (uppercase words often tokenized differently)
        uppercase_count = sum(1 for word in words if word.isupper())
        
        # 5. Adjust for typical tokenizer behavior
        # - Most tokenizers use ~1.3 tokens per word for English
        # - Punctuation usually counts as separate tokens
        # - Numbers and special formats may have specific treatment
        base_estimate = word_count * 1.3 + punctuation_count * 0.8 + number_count * 0.2 + uppercase_count * 0.5
        
        # 6. Round to integer (tokens are discrete)
        return int(base_estimate)
        
    def _truncate_text(self, text, max_tokens):
        """
        Truncate text to fit within token limit while preserving meaning.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        # If text is already under the limit, return as is
        current_tokens = self._estimate_tokens_accurately(text)
        if current_tokens <= max_tokens:
            return text
            
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Progressive truncation strategy:
        # 1. Try keeping full sentences up to the token limit
        truncated_text = ""
        tokens_used = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens_accurately(sentence)
            if tokens_used + sentence_tokens <= max_tokens:
                truncated_text += sentence + " "
                tokens_used += sentence_tokens
            else:
                # No more full sentences can fit
                break
                
        # 2. If we still have room, add a partial sentence with ellipsis
        if tokens_used < max_tokens and len(sentences) > len(truncated_text.split('.')) and truncated_text:
            remaining_tokens = max_tokens - tokens_used
            if remaining_tokens >= 10:  # Only add partial if we have room for meaningful content
                next_sentence = sentences[len(truncated_text.split('.')) - 1]
                words = next_sentence.split()
                
                partial = ""
                for word in words:
                    word_tokens = self._estimate_tokens_accurately(word + " ")
                    if tokens_used + word_tokens + 3 <= max_tokens:  # +3 for ellipsis
                        partial += word + " "
                        tokens_used += word_tokens
                    else:
                        break
                        
                if partial:
                    truncated_text += partial + "..."
                    
        # 3. If we couldn't fit any full sentences, truncate the first sentence
        if not truncated_text and sentences:
            first_sentence = sentences[0]
            words = first_sentence.split()
            
            for i in range(len(words), 0, -1):
                partial_sentence = " ".join(words[:i]) + "..."
                if self._estimate_tokens_accurately(partial_sentence) <= max_tokens:
                    truncated_text = partial_sentence
                    break
                    
        # Fallback: If all else fails, just truncate characters
        if not truncated_text:
            char_per_token = len(text) / current_tokens
            safe_chars = int(max_tokens * char_per_token * 0.9)  # 10% safety margin
            truncated_text = text[:safe_chars] + "..."
            
        return truncated_text
        
    def _update_augmentation_stats(self, selected_regulations):
        """Update augmentation statistics"""
        # Calculate tokens added
        total_tokens = sum(reg.get("estimated_tokens", self._estimate_tokens_accurately(reg["excerpt"])) 
                         for reg in selected_regulations)
        
        # Update moving averages
        count = self.augmentation_stats["augmentation_count"]
        if count > 0:
            # Update with exponential moving average
            self.augmentation_stats["avg_regulations_used"] = (
                0.9 * self.augmentation_stats["avg_regulations_used"] +
                0.1 * len(selected_regulations)
            )
            self.augmentation_stats["avg_total_tokens_added"] = (
                0.9 * self.augmentation_stats["avg_total_tokens_added"] +
                0.1 * total_tokens
            )
        else:
            # First update
            self.augmentation_stats["avg_regulations_used"] = len(selected_regulations)
            self.augmentation_stats["avg_total_tokens_added"] = total_tokens
            
        # Increment count
        self.augmentation_stats["augmentation_count"] += 1
        
    def _extract_domain(self, context):
        """Extract domain from context if available"""
        if not context:
            return None
            
        if isinstance(context, dict) and "domain" in context:
            return context["domain"]
            
        if isinstance(context, dict) and "metadata" in context:
            metadata = context["metadata"]
            if isinstance(metadata, dict) and "domain" in metadata:
                return metadata["domain"]
                
        return None

# Common English stopwords for keyword filtering
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", 
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", 
    "you", "your", "yours", "yourself", "yourselves"
])