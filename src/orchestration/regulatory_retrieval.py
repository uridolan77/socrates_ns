import hashlib
import random
from src.utils.cache import LRUCache
import numpy as np

class OptimizedRegulatoryRetrieval:
    """
    Optimized retrieval system for regulatory content with efficient
    indexing and context-aware search capabilities.
    """
    def __init__(self, regulatory_document_store, embedding_model):
        self.document_store = regulatory_document_store
        self.embedding_model = embedding_model
        
        # Initialize index structures
        self.embedding_index = self._initialize_embedding_index()
        self.keyword_index = self._initialize_keyword_index()
        self.framework_index = self._initialize_framework_index()
        
        # Retrieval configuration
        self.max_results = 20
        self.retrieval_settings = {
            "hybrid_alpha": 0.7,  # Weight for semantic vs keyword search
            "min_similarity": 0.6,  # Minimum similarity threshold
            "max_context_window": 2000  # Maximum context window size
        }
        
        # Cache for frequent queries
        self.query_cache = LRUCache(maxsize=200)
        
    def retrieve(self, query, context=None, frameworks=None, top_k=5):
        """
        Retrieve relevant regulatory content with context awareness
        
        Args:
            query: Query text or embedding
            context: Optional context information
            frameworks: Optional list of frameworks to search within
            top_k: Number of results to return
            
        Returns:
            List of relevant regulatory content snippets
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, context, frameworks, top_k)
        cached_result = self.query_cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Convert query to embedding if needed
        query_embedding = self._get_query_embedding(query)
        
        # Perform hybrid retrieval
        semantic_results = self._semantic_search(query_embedding, frameworks, top_k * 2)
        keyword_results = self._keyword_search(query, frameworks, top_k * 2)
        
        # Merge results with hybrid ranking
        merged_results = self._merge_results(
            semantic_results, 
            keyword_results,
            self.retrieval_settings["hybrid_alpha"]
        )
        
        # Rerank with context if available
        if context:
            merged_results = self._rerank_with_context(merged_results, context)
        
        # Truncate to top_k
        final_results = merged_results[:top_k]
        
        # Process results to include context windows
        processed_results = self._process_results(final_results, query)
        
        # Cache results
        self.query_cache[cache_key] = processed_results
        
        return processed_results
        
    def retrieve_by_concept(self, concept, frameworks=None, top_k=5):
        """
        Retrieve content related to a specific regulatory concept
        
        Args:
            concept: Regulatory concept to search for
            frameworks: Optional list of frameworks to search within
            top_k: Number of results to return
            
        Returns:
            List of relevant regulatory content snippets
        """
        # Search specifically for concept
        concept_results = self._concept_search(concept, frameworks, top_k)
        
        # Process results
        processed_results = self._process_results(concept_results, concept)
        
        return processed_results
        
    def _initialize_embedding_index(self):
        """Initialize embedding index for semantic search"""
        # In a real system, this would initialize a vector store
        # Placeholder implementation
        return {}
        
    def _initialize_keyword_index(self):
        """Initialize keyword index for text search"""
        # In a real system, this would initialize an inverted index
        # Placeholder implementation
        return {}
        
    def _initialize_framework_index(self):
        """Initialize index for framework-specific lookups"""
        # Group documents by framework
        framework_index = {}
        
        for document in self.document_store.get_all_documents():
            framework_id = document.get("framework_id")
            if framework_id:
                if framework_id not in framework_index:
                    framework_index[framework_id] = []
                framework_index[framework_id].append(document)
                
        return framework_index
        
    def _get_query_embedding(self, query):
        """Get embedding for query text or use provided embedding"""
        if isinstance(query, np.ndarray):
            return query
            
        # Generate embedding for text query
        return self.embedding_model.get_embeddings(query)
        
    def _semantic_search(self, query_embedding, frameworks=None, top_k=10):
        """Perform semantic search using embeddings"""
        # In a real system, this would search in a vector database
        # Placeholder implementation returning random results
        results = []
        
        # Use only documents from specified frameworks if provided
        candidate_docs = []
        if frameworks:
            for framework in frameworks:
                if framework.id in self.framework_index:
                    candidate_docs.extend(self.framework_index[framework.id])
        else:
            candidate_docs = self.document_store.get_all_documents()
        
        # Simulate semantic search
        for doc in candidate_docs[:top_k]:
            # Calculate similarity (placeholder - would use real similarity)
            similarity = random.uniform(0.6, 0.95)
            
            results.append({
                "document_id": doc.get("id"),
                "content": doc.get("content", ""),
                "score": similarity,
                "framework_id": doc.get("framework_id"),
                "source": "semantic"
            })
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
        
    def _keyword_search(self, query, frameworks=None, top_k=10):
        """Perform keyword-based search"""
        # In a real system, this would use an inverted index or full-text search
        # Placeholder implementation
        results = []
        
        # Use only documents from specified frameworks if provided
        candidate_docs = []
        if frameworks:
            for framework in frameworks:
                if framework.id in self.framework_index:
                    candidate_docs.extend(self.framework_index[framework.id])
        else:
            candidate_docs = self.document_store.get_all_documents()
        
        # Simulate keyword search
        for doc in candidate_docs[:top_k]:
            # Calculate relevance (placeholder)
            relevance = random.uniform(0.5, 0.9)
            
            results.append({
                "document_id": doc.get("id"),
                "content": doc.get("content", ""),
                "score": relevance,
                "framework_id": doc.get("framework_id"),
                "source": "keyword"
            })
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
        
    def _concept_search(self, concept, frameworks=None, top_k=10):
        """Search for content related to specific concept"""
        # This would use concept-specific indices in a real system
        # Placeholder implementation
        results = []
        
        # Use only documents from specified frameworks if provided
        candidate_docs = []
        if frameworks:
            for framework in frameworks:
                if framework.id in self.framework_index:
                    candidate_docs.extend(self.framework_index[framework.id])
        else:
            candidate_docs = self.document_store.get_all_documents()
        
        # Simulate concept-specific search
        for doc in candidate_docs[:top_k]:
            # Calculate concept relevance (placeholder)
            relevance = random.uniform(0.6, 0.95)
            
            results.append({
                "document_id": doc.get("id"),
                "content": doc.get("content", ""),
                "score": relevance,
                "framework_id": doc.get("framework_id"),
                "source": "concept",
                "concept": concept
            })
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
        
    def _merge_results(self, semantic_results, keyword_results, alpha=0.7):
        """Merge semantic and keyword results with hybrid ranking"""
        # Create map of document IDs to results
        result_map = {}
        
        # Add semantic results with weight alpha
        for result in semantic_results:
            doc_id = result["document_id"]
            result_map[doc_id] = {
                "document_id": doc_id,
                "content": result["content"],
                "score": alpha * result["score"],
                "framework_id": result["framework_id"],
                "sources": ["semantic"],
                "original_scores": {"semantic": result["score"]}
            }
            
        # Add or update with keyword results with weight (1-alpha)
        for result in keyword_results:
            doc_id = result["document_id"]
            if doc_id in result_map:
                # Update existing entry
                result_map[doc_id]["score"] += (1 - alpha) * result["score"]
                result_map[doc_id]["sources"].append("keyword")
                result_map[doc_id]["original_scores"]["keyword"] = result["score"]
            else:
                # Add new entry
                result_map[doc_id] = {
                    "document_id": doc_id,
                    "content": result["content"],
                    "score": (1 - alpha) * result["score"],
                    "framework_id": result["framework_id"],
                    "sources": ["keyword"],
                    "original_scores": {"keyword": result["score"]}
                }
                
        # Convert map to list and sort by combined score
        merged_results = list(result_map.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        return merged_results
        
    def _rerank_with_context(self, results, context):
        """Rerank results considering the context"""
        # This would use context-aware reranking in a real system
        # Simple implementation: boost results that match context keywords
        
        # Extract keywords from context (placeholder)
        context_keywords = context.split() if isinstance(context, str) else []
        
        # Rerank based on keyword match
        for result in results:
            context_match_score = 0.0
            
            # Check for keyword matches
            for keyword in context_keywords:
                if keyword in result["content"]:
                    context_match_score += 0.05  # Small boost per match
                    
            # Apply context boost
            result["score"] = result["score"] * (1.0 + context_match_score)
            
        # Re-sort by adjusted score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
        
    def _process_results(self, results, query):
        """Process results to include context windows around matches"""
        processed_results = []
        
        for result in results:
            # Find relevant excerpt
            excerpt = self._extract_relevant_excerpt(result["content"], query)
            
            # Add to processed results
            processed_results.append({
                "document_id": result["document_id"],
                "framework_id": result["framework_id"],
                "score": result["score"],
                "excerpt": excerpt,
                "sources": result.get("sources", [])
            })
            
        return processed_results
        
    def _extract_relevant_excerpt(self, content, query):
        """Extract relevant excerpt from document content"""
        # In a real system, this would find the most relevant passage
        # Simplified implementation: return beginning of content
        max_len = self.retrieval_settings["max_context_window"]
        
        if len(content) <= max_len:
            return content
            
        # Simple excerpt from beginning (would be more sophisticated in real system)
        return content[:max_len] + "..."
        
    def _generate_cache_key(self, query, context, frameworks, top_k):
        """Generate cache key for query"""
        # Hash query
        if isinstance(query, str):
            query_hash = hashlib.md5(query.encode()).hexdigest()
        else:
            # For embeddings, use hash of first few values
            query_hash = hashlib.md5(str(query[:5]).encode()).hexdigest()
            
        # Hash context if available
        context_hash = hashlib.md5(str(context).encode()).hexdigest() if context else "no_context"
        
        # Framework string
        framework_str = "-".join(sorted([f.id for f in frameworks])) if frameworks else "all"
        
        return f"q:{query_hash[:8]}_c:{context_hash[:8]}_f:{framework_str}_k:{top_k}"