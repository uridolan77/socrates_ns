import time
import uuid
import json
import redis
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("distributed.rules")

@dataclass
class RuleTask:
    """Task for distributed rule processing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    content: str = ""
    content_type: str = "text"
    framework_id: str = ""
    created_at: float = field(default_factory=time.time)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    completed: bool = False
    error: Optional[str] = None


class DistributedRuleProcessor:
    """
    System for distributing rule processing across multiple workers
    to improve scalability for compliance verification.
    """
    
    def __init__(self, 
                redis_url: str,
                rule_registry: Any,
                worker_id: str = None,
                worker_count: int = 4,
                batch_size: int = 10,
                poll_interval: float = 0.5):
        """
        Initialize the distributed rule processor
        
        Args:
            redis_url: URL for Redis connection
            rule_registry: Registry of rules to process
            worker_id: Unique ID for this worker (auto-generated if None)
            worker_count: Number of worker threads
            batch_size: Number of tasks to process in one batch
            poll_interval: Seconds between polling for new tasks
        """
        self.redis = redis.from_url(redis_url)
        self.rule_registry = rule_registry
        self.worker_id = worker_id or f"worker-{str(uuid.uuid4())[:8]}"
        self.worker_count = worker_count
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        
        # Keys for Redis data structures
        self.task_queue_key = "rule_tasks:queue"
        self.processing_key = "rule_tasks:processing"
        self.results_key = "rule_tasks:results"
        self.task_key_prefix = "rule_task:"
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        
        # Control flags
        self.running = False
        self.worker_threads = []
        
        logger.info(f"Distributed rule processor initialized: {self.worker_id}")
        
    def start(self):
        """Start the processing workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
            
        logger.info(f"Started {self.worker_count} worker threads")
        
    def stop(self):
        """Stop all processing workers"""
        self.running = False
        
        # Wait for threads to terminate
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
            
        self.worker_threads = []
        logger.info("All worker threads stopped")
        
    def submit_task(self, task: RuleTask) -> str:
        """
        Submit a rule processing task
        
        Args:
            task: The rule processing task
            
        Returns:
            Task ID for tracking
        """
        # Store task details
        task_key = f"{self.task_key_prefix}{task.id}"
        self.redis.set(task_key, json.dumps(asdict(task)))
        
        # Add to processing queue with priority
        self.redis.zadd(self.task_queue_key, {task.id: task.priority})
        
        logger.debug(f"Submitted task {task.id} for rule {task.rule_id}")
        return task.id
        
    def batch_submit(self, tasks: List[RuleTask]) -> List[str]:
        """
        Submit multiple tasks in a batch
        
        Args:
            tasks: List of rule processing tasks
            
        Returns:
            List of task IDs
        """
        if not tasks:
            return []
            
        # Use pipeline for efficiency
        pipe = self.redis.pipeline()
        
        task_ids = []
        for task in tasks:
            task_key = f"{self.task_key_prefix}{task.id}"
            pipe.set(task_key, json.dumps(asdict(task)))
            pipe.zadd(self.task_queue_key, {task.id: task.priority})
            task_ids.append(task.id)
            
        pipe.execute()
        
        logger.debug(f"Batch submitted {len(tasks)} tasks")
        return task_ids
        
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of a task, optionally waiting for completion
        
        Args:
            task_id: Task ID to check
            timeout: Seconds to wait for result (None to return immediately)
            
        Returns:
            Task result or None if not completed
        """
        task_key = f"{self.task_key_prefix}{task_id}"
        
        # If no timeout, just check current state
        if timeout is None:
            task_data = self.redis.get(task_key)
            if not task_data:
                return None
                
            task = RuleTask(**json.loads(task_data))
            return task.result if task.completed else None
            
        # Wait for completion with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            task_data = self.redis.get(task_key)
            if task_data:
                task = RuleTask(**json.loads(task_data))
                if task.completed:
                    return task.result
                    
            # Sleep briefly before checking again
            time.sleep(0.1)
            
        # Timeout reached, return current state
        task_data = self.redis.get(task_key)
        if not task_data:
            return None
            
        task = RuleTask(**json.loads(task_data))
        return task.result if task.completed else None
        
    def get_batch_results(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """
        Get results for multiple tasks
        
        Args:
            task_ids: List of task IDs to check
            timeout: Seconds to wait for all results (None to return immediately)
            
        Returns:
            Dictionary of task ID to result (None for incomplete tasks)
        """
        if not task_ids:
            return {}
            
        # If no timeout, get current state of all tasks
        if timeout is None:
            results = {}
            pipe = self.redis.pipeline()
            
            for task_id in task_ids:
                task_key = f"{self.task_key_prefix}{task_id}"
                pipe.get(task_key)
                
            task_data_list = pipe.execute()
            
            for i, task_id in enumerate(task_ids):
                task_data = task_data_list[i]
                if task_data:
                    task = RuleTask(**json.loads(task_data))
                    results[task_id] = task.result if task.completed else None
                else:
                    results[task_id] = None
                    
            return results
            
        # Wait for all to complete with timeout
        pending_ids = set(task_ids)
        results = {task_id: None for task_id in task_ids}
        
        start_time = time.time()
        while pending_ids and time.time() - start_time < timeout:
            pipe = self.redis.pipeline()
            
            for task_id in list(pending_ids):
                task_key = f"{self.task_key_prefix}{task_id}"
                pipe.get(task_key)
                
            task_data_list = pipe.execute()
            
            for i, task_id in enumerate(list(pending_ids)):
                task_data = task_data_list[i]
                if task_data:
                    task = RuleTask(**json.loads(task_data))
                    if task.completed:
                        results[task_id] = task.result
                        pending_ids.remove(task_id)
                        
            # If all complete, we're done
            if not pending_ids:
                break
                
            # Sleep briefly before checking again
            time.sleep(0.1)
            
        return results
    
    def verify_content(self, 
                     content: str, 
                     framework_ids: List[str],
                     content_type: str = "text",
                     wait_for_result: bool = True,
                     timeout: float = 30.0) -> Dict[str, Any]:
        """
        High-level method to verify content against multiple frameworks
        
        Args:
            content: Content to verify
            framework_ids: Frameworks to verify against
            content_type: Type of content
            wait_for_result: Whether to wait for result
            timeout: Seconds to wait for result
            
        Returns:
            Verification result
        """
        # Get all applicable rules
        all_rule_ids = []
        for framework_id in framework_ids:
            framework_rules = self.rule_registry.get_rules_for_framework(framework_id)
            all_rule_ids.extend([(rule_id, framework_id) for rule_id in framework_rules])
            
        # Create tasks for each rule
        tasks = []
        for rule_id, framework_id in all_rule_ids:
            task = RuleTask(
                rule_id=rule_id,
                content=content,
                content_type=content_type,
                framework_id=framework_id,
                priority=self._get_rule_priority(rule_id, framework_id)
            )
            tasks.append(task)
            
        # Submit tasks
        task_ids = self.batch_submit(tasks)
        
        # Wait for results if requested
        if wait_for_result:
            results = self.get_batch_results(task_ids, timeout)
            
            # Process and combine results
            verification_result = self._process_rule_results(results, framework_ids)
            return verification_result
        else:
            # Return task IDs for later retrieval
            return {
                "task_ids": task_ids,
                "framework_ids": framework_ids,
                "pending": True
            }
    
    def _worker_loop(self, worker_index: int):
        """
        Main loop for worker thread
        
        Args:
            worker_index: Index of this worker thread
        """
        logger.info(f"Worker thread {worker_index} started")
        
        while self.running:
            try:
                # Try to get a batch of tasks
                tasks = self._claim_tasks()
                
                if tasks:
                    # Process each task
                    for task in tasks:
                        self._process_task(task)
                else:
                    # No tasks, sleep before polling again
                    time.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in worker thread {worker_index}: {str(e)}")
                time.sleep(1)  # Sleep on error to avoid tight loop
                
        logger.info(f"Worker thread {worker_index} stopped")
    
    def _claim_tasks(self) -> List[RuleTask]:
        """
        Claim a batch of tasks for processing
        
        Returns:
            List of tasks to process
        """
        # Use Lua script to atomically claim tasks
        claim_script = """
        local queue_key = KEYS[1]
        local processing_key = KEYS[2]
        local task_prefix = KEYS[3]
        local batch_size = tonumber(ARGV[1])
        local worker_id = ARGV[2]
        
        -- Get tasks from queue
        local task_ids = redis.call('ZRANGE', queue_key, 0, batch_size - 1)
        if #task_ids == 0 then
            return {}
        end
        
        -- Claim tasks
        local tasks = {}
        for i, task_id in ipairs(task_ids) do
            -- Remove from queue
            redis.call('ZREM', queue_key, task_id)
            
            -- Mark as processing by this worker
            redis.call('HSET', processing_key, task_id, worker_id)
            
            -- Get task data
            local task_data = redis.call('GET', task_prefix .. task_id)
            if task_data then
                table.insert(tasks, task_data)
            end
        end
        
        return tasks
        """
        
        result = self.redis.eval(
            claim_script,
            3,  # number of keys
            self.task_queue_key,
            self.processing_key,
            self.task_key_prefix,
            self.batch_size,
            self.worker_id
        )
        
        # Parse task data
        tasks = []
        for task_data in result:
            try:
                task_dict = json.loads(task_data)
                task = RuleTask(**task_dict)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error parsing task data: {str(e)}")
                
        return tasks
    
    def _process_task(self, task: RuleTask):
        """
        Process a single rule task
        
        Args:
            task: Task to process
        """
        logger.debug(f"Processing task {task.id} for rule {task.rule_id}")
        
        try:
            # Get rule implementation
            rule = self.rule_registry.get_rule(task.rule_id, task.framework_id)
            if not rule:
                raise ValueError(f"Rule not found: {task.rule_id}")
                
            # Apply rule to content
            result = rule.apply(task.content, task.content_type)
            
            # Update task with result
            task.result = result
            task.completed = True
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {str(e)}")
            task.error = str(e)
            task.completed = True
            
        # Store updated task
        self._save_task_result(task)
    
    def _save_task_result(self, task: RuleTask):
        """
        Save task result to Redis
        
        Args:
            task: Completed task
        """
        # Update task in Redis
        task_key = f"{self.task_key_prefix}{task.id}"
        self.redis.set(task_key, json.dumps(asdict(task)))
        
        # Remove from processing set
        self.redis.hdel(self.processing_key, task.id)
        
        # Add to results sorted set (scored by completion time)
        self.redis.zadd(self.results_key, {task.id: time.time()})
        
        logger.debug(f"Saved result for task {task.id}")
    
    def _get_rule_priority(self, rule_id: str, framework_id: str) -> int:
        """
        Get priority for rule processing
        
        Args:
            rule_id: Rule ID
            framework_id: Framework ID
            
        Returns:
            Priority value (lower is higher priority)
        """
        # Get rule metadata
        rule = self.rule_registry.get_rule(rule_id, framework_id)
        if not rule:
            return 100  # Default low priority
            
        # Assign priority based on rule severity
        severity = getattr(rule, 'severity', None)
        if severity == 'critical':
            return 0
        elif severity == 'high':
            return 10
        elif severity == 'medium':
            return 50
        else:
            return 100
    
    def _process_rule_results(self, 
                            results: Dict[str, Any], 
                            framework_ids: List[str]) -> Dict[str, Any]:
        """
        Process and combine results from multiple rules
        
        Args:
            results: Dictionary of task ID to rule result
            framework_ids: List of frameworks applied
            
        Returns:
            Combined verification result
        """
        # Initialize result
        verification_result = {
            "is_compliant": True,
            "compliance_score": 1.0,
            "frameworks": framework_ids,
            "violations": []
        }
        
        # Process each rule result
        rule_violations = []
        rule_scores = []
        
        for task_id, result in results.items():
            if result is None:
                # Task didn't complete
                continue
                
            # Check if rule passed
            is_compliant = result.get("is_compliant", True)
            rule_score = result.get("compliance_score", 1.0 if is_compliant else 0.0)
            
            rule_scores.append(rule_score)
            
            if not is_compliant:
                # Rule violation
                verification_result["is_compliant"] = False
                
                # Add violation details
                violation = result.get("violation", {})
                if violation:
                    rule_violations.append(violation)
                    
        # Calculate overall compliance score
        if rule_scores:
            verification_result["compliance_score"] = sum(rule_scores) / len(rule_scores)
            
        # Add violations
        verification_result["violations"] = rule_violations
        
        return verification_result


class AsyncDistributedRuleProcessor:
    """
    Async wrapper for the distributed rule processor.
    """
    
    def __init__(self, processor: DistributedRuleProcessor):
        """
        Initialize the async wrapper
        
        Args:
            processor: The underlying distributed rule processor
        """
        self.processor = processor
        self.loop = asyncio.get_event_loop()
        
    async def submit_task(self, task: RuleTask) -> str:
        """
        Submit a rule processing task asynchronously
        
        Args:
            task: The rule processing task
            
        Returns:
            Task ID for tracking
        """
        return await self.loop.run_in_executor(
            None, self.processor.submit_task, task
        )
        
    async def batch_submit(self, tasks: List[RuleTask]) -> List[str]:
        """
        Submit multiple tasks in a batch asynchronously
        
        Args:
            tasks: List of rule processing tasks
            
        Returns:
            List of task IDs
        """
        return await self.loop.run_in_executor(
            None, self.processor.batch_submit, tasks
        )
        
    async def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of a task asynchronously
        
        Args:
            task_id: Task ID to check
            timeout: Seconds to wait for result
            
        Returns:
            Task result or None if not completed
        """
        return await self.loop.run_in_executor(
            None, functools.partial(self.processor.get_task_result, task_id, timeout)
        )
        
    async def get_batch_results(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """
        Get results for multiple tasks asynchronously
        
        Args:
            task_ids: List of task IDs to check
            timeout: Seconds to wait for all results
            
        Returns:
            Dictionary of task ID to result
        """
        return await self.loop.run_in_executor(
            None, functools.partial(self.processor.get_batch_results, task_ids, timeout)
        )
        
    async def verify_content(self, 
                          content: str, 
                          framework_ids: List[str],
                          content_type: str = "text",
                          wait_for_result: bool = True,
                          timeout: float = 30.0) -> Dict[str, Any]:
        """
        Verify content against multiple frameworks asynchronously
        
        Args:
            content: Content to verify
            framework_ids: Frameworks to verify against
            content_type: Type of content
            wait_for_result: Whether to wait for result
            timeout: Seconds to wait for result
            
        Returns:
            Verification result
        """
        return await self.loop.run_in_executor(
            None, 
            functools.partial(
                self.processor.verify_content,
                content,
                framework_ids,
                content_type,
                wait_for_result,
                timeout
            )
        )


# Example usage of the distributed rule processor:
"""
# Initialize Redis connection and rule registry
redis_url = "redis://localhost:6379/0"
rule_registry = YourRuleRegistry()  # Replace with your rule registry

# Create processor
processor = DistributedRuleProcessor(
    redis_url=redis_url,
    rule_registry=rule_registry,
    worker_count=4
)

# Start processing workers
processor.start()

# Submit tasks for processing
task = RuleTask(
    rule_id="GDPR.Art.5.1.a",
    content="This is a test document with PII data: SSN 123-45-6789",
    content_type="text",
    framework_id="GDPR"
)
task_id = processor.submit_task(task)

# Wait for and get result
result = processor.get_task_result(task_id, timeout=10.0)
print(f"Task result: {result}")

# High-level verification
verification_result = processor.verify_content(
    content="This is a test document with PII data: SSN 123-45-6789",
    framework_ids=["GDPR", "HIPAA"],
    wait_for_result=True
)
print(f"Verification result: {verification_result}")

# Stop processor when done
processor.stop()
"""