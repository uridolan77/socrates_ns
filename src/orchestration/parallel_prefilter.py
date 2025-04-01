from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import prompt
from src.utils.cache import CacheManager
from src.compliance.filtering.content_detector import ContentComplianceDetector
from src.compliance.filtering.sensitive_data_detector import SensitiveDataDetector
from src.compliance.filtering.sensitive_token_detector import SensitiveTokenDetector
from src.compliance.filtering.topic_analyzer import ComplianceTopicAnalyzer
from src.compliance.filtering.content_detector import RegexPatternMatcher

class ParallelCompliancePrefilter:
    """
    Parallel input filtering system that efficiently screens content 
    against compliance requirements before generation.
    """
    def __init__(self, filter_config):
        self.config = filter_config
        self.max_workers = filter_config.get("max_workers", 4)
        
        # Initialize filter components
        self.content_detector = ContentComplianceDetector(filter_config)
        self.pattern_matcher = RegexPatternMatcher(filter_config.get("patterns", []))
        self.topic_analyzer = ComplianceTopicAnalyzer(filter_config)
        self.sensitive_data_detector = SensitiveDataDetector(filter_config)
        
        # Initialize allowlists and denylists
        self.allowlists = filter_config.get("allowlists", {})
        self.denylists = filter_config.get("denylists", {})
        
        # Create filter chain
        self.filter_chain = self._initialize_filter_chain()
        
    def filter_input_parallel(self, prompt, context=None):
        """
        Apply compliance filters to input in parallel
        
        Args:
            prompt: Input prompt to filter
            context: Optional context information
            
        Returns:
            Dict with filtering results
        """
        # Skip filtering for empty prompts
        if not prompt or prompt.strip() == "":
            return {
                "is_compliant": True,
                "filtered_input": prompt,
                "issues": []
            }
            
        # Apply filters in parallel
        filter_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit filter tasks
            future_to_filter = {
                executor.submit(filter_fn, prompt, context): filter_name
                for filter_name, filter_fn in self.filter_chain.items()
            }
            
            # Collect results
            for future in as_completed(future_to_filter):
                filter_name = future_to_filter[future]
                try:
                    result = future.result()
                    filter_results.append((filter_name, result))
                except Exception as e:
                    # Log filter failure but continue with other filters
                    logging.error(f"Filter {filter_name} failed: {str(e)}")
                    traceback.print_exc()
        
        # Aggregate filter results
        aggregated_result = self._aggregate_filter_results(filter_results)
        
        # Apply post-filtering if needed
        if aggregated_result["is_compliant"] and self.config.get("apply_post_filtering", True):
            # Apply any post-filtering logic (e.g., minor modifications)
            aggregated_result["filtered_input"] = self._apply_post_filtering(
                aggregated_result["filtered_input"], 
                context
            )
            
        return aggregated_result
        
    def _initialize_filter_chain(self):
        """Initialize filter chain functions"""
        return {
            "content_compliance": self.content_detector.check_compliance,
            "pattern_matching": self.pattern_matcher.check_patterns,
            "topic_analysis": self.topic_analyzer.analyze_topics,
            "sensitive_data": self.sensitive_data_detector.detect
        }
        
    def _aggregate_filter_results(self, filter_results):
        """Aggregate results from parallel filters"""
        is_compliant = True
        issues = []
        filtered_input = None
        modified = False
        
        # Sort results by filter priority if specified
        filter_priority = self.config.get("filter_priority", {})
        filter_results.sort(
            key=lambda x: filter_priority.get(x[0], 999)
        )
        
        # Process results in priority order
        for filter_name, result in filter_results:
            # Track compliance
            if not result.get("is_compliant", True):
                is_compliant = False
                
            # Collect issues
            filter_issues = result.get("issues", [])
            for issue in filter_issues:
                if not any(existing["rule_id"] == issue.get("rule_id") for existing in issues):
                    issues.append(issue)
                    
            # Use filtered input from this filter if provided
            if "filtered_input" in result:
                if filtered_input is None:
                    filtered_input = result["filtered_input"]
                elif result.get("modified", False):
                    # This filter modified content, use its version
                    filtered_input = result["filtered_input"]
                    modified = True
                    
        # If no filter provided filtered input, use original
        if filtered_input is None:
            filtered_input = prompt
            
        return {
            "is_compliant": is_compliant,
            "filtered_input": filtered_input,
            "issues": issues,
            "modified": modified
        }
        
    def _apply_post_filtering(self, input_text, context):
        """Apply post-filtering modifications"""
        # Placeholder implementation
        # In a real system, this would apply minor modifications
        # for compliance without blocking the input
        return input_text