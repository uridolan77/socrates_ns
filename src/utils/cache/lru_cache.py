    
class LRUCache:
    """Enhanced LRU Cache with size tracking and statistics"""
    
    def __init__(self, maxsize=128, stats_window=100):
        self.cache = {}
        self.maxsize = maxsize
        self.order = []
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0,
            "size_bytes": 0
        }
        self.access_history = []  # Track last N accesses
        self.stats_window = stats_window
    
    def get(self, key):
        """Get an item from cache with stats tracking"""
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            
            # Update stats
            self.stats["hits"] += 1
            self._record_access(key, True)
            
            return self.cache[key]
        else:
            # Update stats
            self.stats["misses"] += 1
            self._record_access(key, False)
            
            return None
    
    def __setitem__(self, key, value):
        """Add/update an item in the cache"""
        if key in self.cache:
            # Update existing entry
            old_size = self._get_item_size(self.cache[key])
            new_size = self._get_item_size(value)
            self.stats["size_bytes"] += (new_size - old_size)
            
            # Update
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new entry
            self.cache[key] = value
            self.order.append(key)
            self.stats["size_bytes"] += self._get_item_size(value)
            
            # Check size limit
            while len(self.cache) > self.maxsize:
                # Remove least recently used
                lru_key = self.order.pop(0)
                self.stats["size_bytes"] -= self._get_item_size(self.cache[lru_key])
                del self.cache[lru_key]
                self.stats["evictions"] += 1
    
    def _record_access(self, key, hit):
        """Record access for hit rate calculation"""
        self.access_history.append(hit)
        if len(self.access_history) > self.stats_window:
            self.access_history.pop(0)
        
        # Update hit rate
        if self.access_history:
            self.stats["hit_rate"] = sum(self.access_history) / len(self.access_history)
    
    def _get_item_size(self, item):
        """Estimate memory size of cached item"""
        import sys
        
        # For strings
        if isinstance(item, str):
            return len(item) * 2  # Approximate for Python strings
        
        # For tensors
        if hasattr(item, 'element_size') and hasattr(item, 'nelement'):
            return item.element_size() * item.nelement()
        
        # For dictionaries
        if isinstance(item, dict):
            return sum(self._get_item_size(k) + self._get_item_size(v) 
                      for k, v in item.items())
        
        # For lists
        if isinstance(item, list):
            return sum(self._get_item_size(x) for x in item)
        
        # Default approximation
        return sys.getsizeof(item)
