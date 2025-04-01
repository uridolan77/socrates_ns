import logging
import time

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold=5, reset_timeout=60, half_open_timeout=30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
        
    def can_execute(self):
        """Check if request can be executed based on circuit state"""
        current_time = time.time()
        
        if self.state == "OPEN":
            # Check if reset timeout has elapsed
            if current_time - self.last_failure_time > self.reset_timeout:
                self.logger.info("Circuit transitioning from OPEN to HALF_OPEN")
                self.state = "HALF_OPEN"
                return True
            return False
            
        elif self.state == "HALF_OPEN":
            # In half-open state, only allow one request to test the service
            return True
            
        # In closed state, allow all requests
        return True
        
    def record_success(self):
        """Record successful execution"""
        if self.state == "HALF_OPEN":
            self.logger.info("Circuit transitioning from HALF_OPEN to CLOSED")
            self.state = "CLOSED"
            self.failure_count = 0
            
    def record_failure(self):
        """Record failed execution"""
        current_time = time.time()
        self.last_failure_time = current_time
        
        if self.state == "HALF_OPEN":
            self.logger.warning("Failure in HALF_OPEN state, circuit transitioning to OPEN")
            self.state = "OPEN"
            return
            
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.logger.warning(f"Failure threshold reached ({self.failure_count}), circuit transitioning to OPEN")
            self.state = "OPEN"
