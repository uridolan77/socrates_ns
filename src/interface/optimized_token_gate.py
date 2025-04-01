from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.cache.lru_cache import LRUCache
import numpy as np
import time
import logging
from collections import deque
import hashlib

class OptimizedTokenLevelComplianceGate:
    """
    Optimized token-level compliance enforcement during generation
    with parallel checking and batched processing.
    """
    def __init__(self, language_model, regulatory_constraints, max_workers=8):
        self.language_model = language_model
        self.regulatory_constraints = regulatory_constraints
        self.max_workers = max_workers
        
        # Token filtering components
        self.constraint_compiler = self._initialize_constraint_compiler()
        self.token_validator = self._initialize_token_validator()
        
        # Optimization components
        self.batch_size = 128
        self.cached_decisions = LRUCache(maxsize=10000)
        self.recent_contexts = deque(maxlen=5)
        
        # Statistics tracking
        self.stats = {
            "tokens_processed": 0,
            "tokens_filtered": 0,
            "batch_count": 0,
            "cache_hits": 0
        }
        
    def generate_compliant_text(self, model, prompt, semantic_state, constraints, max_tokens, compliance_mode):
        """
        Generate text with compliance constraints enforced at token level
        
        Args:
            model: Language model for generation
            prompt: Input prompt
            semantic_state: Initial semantic state
            constraints: Applicable compliance constraints
            max_tokens: Maximum tokens to generate
            compliance_mode: Compliance strictness mode
            
        Returns:
            Dict with generated text and compliance info
        """
        # Tokenize prompt
        input_ids = model.tokenize(prompt)
        
        # Compile constraints for efficient checking
        compiled_constraints = self.constraint_compiler.compile(constraints, compliance_mode)
        
        # Initialize generation state
        generated_text = ""
        current_tokens = input_ids.copy()
        current_semantic_state = semantic_state
        filtered_tokens = 0
        
        # Generate tokens up to max_tokens
        for i in range(max_tokens):
            # Get next token logits from model
            logits = model.get_next_token_logits(current_tokens)
            
            # Filter logits based on compliance constraints
            filtered_logits, filtering_info = self.filter_token_logits(
                logits, 
                generated_text, 
                current_semantic_state, 
                compiled_constraints, 
                compliance_mode
            )
            
            # Update stats
            filtered_tokens += filtering_info.get("filtered_count", 0)
            
            # Sample next token from filtered distribution
            next_token = model.sample_token(filtered_logits)
            
            # Check for end of sequence
            if model.is_eos_token(next_token):
                break
                
            # Decode token to text
            token_text = model.decode_token(next_token)
            
            # Update generated text and token sequence
            generated_text += token_text
            current_tokens.append(next_token)
            
            # Update semantic state
            current_semantic_state = self._update_semantic_state(
                current_semantic_state,
                token_text,
                generated_text,
                compiled_constraints
            )
            
        # Check final compliance
        compliance_check = self._check_final_compliance(
            generated_text, compiled_constraints, compliance_mode
        )
        
        return {
            "text": generated_text,
            "is_compliant": compliance_check["is_compliant"],
            "compliance_score": compliance_check["compliance_score"],
            "tokens_filtered": filtered_tokens,
            "tokens_generated": len(generated_text.split()),
            "compliance_mode": compliance_mode
        }
def filter_token_logits(self, logits, generated_text, semantic_state, compiled_constraints, compliance_mode):
    """
    Filter token logits based on compliance constraints
    
    Args:
        logits: Token probability logits
        generated_text: Text generated so far
        semantic_state: Current semantic state
        compiled_constraints: Compiled compliance constraints
        compliance_mode: Compliance strictness mode
        
    Returns:
        Filtered logits and filtering information
    """
    # Clone logits to avoid modifying original
    filtered_logits = logits.copy()
    
    # Get context fingerprint for caching
    context_key = self._get_context_fingerprint(generated_text, semantic_state)
    
    # Get top tokens (to limit processing)
    top_tokens = self._get_top_tokens(filtered_logits)
    
    # Initialize filtering statistics
    filtering_info = {
        "total_tokens": len(top_tokens),
        "filtered_count": 0,
        "cache_hits": 0,
        "batch_processing": True
    }
    
    # Check cache for this context
    cached_decisions = {}
    if context_key in self.cached_decisions:
        cached_decisions = self.cached_decisions[context_key]
        filtering_info["cache_hits"] = len(set(top_tokens) & set(cached_decisions.keys()))
    
    # Identify tokens that need checking
    tokens_to_check = [t for t in top_tokens if t not in cached_decisions]
    
    if tokens_to_check:
        # Process tokens in batches for efficiency
        filter_results = {}
        
        for i in range(0, len(tokens_to_check), self.batch_size):
            batch = tokens_to_check[i:i+self.batch_size]
            
            # Process batch in parallel
            batch_results = self._check_token_batch(
                batch,
                generated_text,
                semantic_state,
                compiled_constraints,
                compliance_mode
            )
            
            filter_results.update(batch_results)
            filtering_info["batch_count"] = filtering_info.get("batch_count", 0) + 1
        
        # Update cache with new decisions
        cached_decisions.update(filter_results)
        self.cached_decisions[context_key] = cached_decisions
    
    # Apply filtering based on cached decisions
    for token_id in top_tokens:
        if token_id in cached_decisions and not cached_decisions[token_id]:
            filtered_logits[token_id] = float('-inf')  # Mask prohibited tokens
            filtering_info["filtered_count"] += 1
    
    # Ensure we haven't blocked all tokens
    if self._all_tokens_blocked(filtered_logits):
        # Apply fallback strategy
        filtered_logits = self._apply_fallback_strategy(logits, compliance_mode)
        filtering_info["applied_fallback"] = True
    
    # Update statistics
    self.stats["tokens_processed"] += len(top_tokens)
    self.stats["tokens_filtered"] += filtering_info["filtered_count"]
    
    return filtered_logits, filtering_info

def _initialize_constraint_compiler(self):
    """Initialize constraint compiler"""
    class ConstraintCompiler:
        def compile(self, constraints, mode):
            """Compile constraints for efficient checking"""
            # Placeholder implementation
            return {
                "constraints": constraints,
                "mode": mode,
                "compiled_at": time.time()
            }
    
    return ConstraintCompiler()

def _initialize_token_validator(self):
    """Initialize token validator"""
    class TokenValidator:
        def validate(self, token, context, semantic_state, constraints):
            """Validate if token is compliant"""
            # Placeholder implementation
            return True
    
    return TokenValidator()

def _check_token_batch(self, token_batch, generated_text, semantic_state, compiled_constraints, compliance_mode):
    """Check compliance for a batch of tokens in parallel"""
    results = {}
    
    # Process tokens in parallel
    with ThreadPoolExecutor(max_workers=min(len(token_batch), self.max_workers)) as executor:
        future_to_token = {
            executor.submit(
                self._check_token_compliance,
                token_id,
                generated_text,
                semantic_state,
                compiled_constraints,
                compliance_mode
            ): token_id for token_id in token_batch
        }
        
        # Collect results
        for future in as_completed(future_to_token):
            token_id = future_to_token[future]
            try:
                is_compliant = future.result()
                results[token_id] = is_compliant
            except Exception as e:
                # Log error but continue (assume token is compliant)
                logging.error(f"Error checking token {token_id}: {str(e)}")
                results[token_id] = True
    
    return results

    def _check_token_compliance(self, token_id, generated_text, semantic_state, compiled_constraints, compliance_mode):
        """Check if a token complies with constraints"""
        # Decode token to text
        token_text = self.language_model.decode_token(token_id)
        
        # Hypothetical continuation
        potential_text = generated_text + token_text
        
        # Check against compiled constraints
        return self.token_validator.validate(
            token_id,
            potential_text,
            semantic_state,
            compiled_constraints
        )

    def _update_semantic_state(self, state, token_text, generated_text, compiled_constraints):
        """Update semantic state with new token"""
        # Placeholder implementation
        # In a real system, this would integrate with semantic monitoring
        return state

    def _check_final_compliance(self, generated_text, compiled_constraints, compliance_mode):
        """Check compliance of complete generated text"""
        # Placeholder implementation
        return {
            "is_compliant": True,
            "compliance_score": 0.9
        }

    def _get_top_tokens(self, logits, top_k=100, threshold=0.001):
        """Get top tokens by probability"""
        # Convert logits to probabilities
        probs = self._logits_to_probs(logits)
        
        # Get indices of tokens above threshold
        top_indices = np.where(probs >= threshold)[0]
        
        # If too many tokens above threshold, limit to top_k
        if len(top_indices) > top_k:
            top_indices = np.argsort(probs)[-top_k:]
        
        return top_indices.tolist()

    def _logits_to_probs(self, logits):
        """Convert logits to probabilities"""
        # Softmax conversion
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def _all_tokens_blocked(self, logits):
        """Check if all tokens have been blocked"""
        return np.all(np.isneginf(logits))

    def _apply_fallback_strategy(self, original_logits, compliance_mode):
        """Apply fallback strategy when all tokens are blocked"""
        # Revert to original logits but only allow safe tokens
        filtered_logits = original_logits.copy()
        
        if compliance_mode == "strict":
            # In strict mode, only allow very limited safe tokens
            for i in range(len(filtered_logits)):
                if i not in self._get_safe_token_ids():
                    filtered_logits[i] = float('-inf')
        else:
            # In other modes, allow top-5 from original distribution
            top_5 = np.argsort(original_logits)[-5:]
            for i in range(len(filtered_logits)):
                if i not in top_5:
                    filtered_logits[i] = float('-inf')
        
        return filtered_logits

    def _get_safe_token_ids(self):
        """Get set of inherently safe token IDs"""
        # This would be populated with known safe tokens
        # Placeholder implementation
        return {0, 1, 2, 3, 4}  # Example of safe token IDs

    def _get_context_fingerprint(self, text, state):
        """Generate context fingerprint for caching"""
        # Use last N chars of text as context key for caching
        context_text = text[-100:] if len(text) > 100 else text
        
        # Combine with state fingerprint
        if hasattr(state, "get_fingerprint"):
            state_fp = state.get_fingerprint()
        else:
            # Create simple fingerprint from state
            state_keys = sorted(state.keys()) if isinstance(state, dict) else []
            state_fp = "-".join(state_keys)
        
        # Create combined fingerprint
        combined = f"{context_text}||{state_fp}"
        return hashlib.md5(combined.encode()).hexdigest()