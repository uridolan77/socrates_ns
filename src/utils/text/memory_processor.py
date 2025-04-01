import os
import re
import mmap
from typing import Callable, Union, Optional, Any, Pattern
import tempfile
from typing import Iterator, Pattern, Match, List, Dict, Any, Generator, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("efficient.text")

@dataclass
class TextChunk:
    """A chunk of text with position information"""
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """A pattern match with position information"""
    pattern_id: str
    text: str
    start_pos: int
    end_pos: int
    rule_id: Optional[str] = None
    framework_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryEfficientTextProcessor:
    """
    Process large text documents efficiently with minimal memory usage
    by using memory mapping and streaming techniques.
    """
    
    def __init__(self, chunk_size: int = 1024 * 1024, overlap: int = 1000):
        """
        Initialize the text processor
        
        Args:
            chunk_size: Size of text chunks to process (bytes)
            overlap: Overlap between chunks to avoid missing patterns at boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.temp_files = []
        
    def __del__(self):
        """Clean up temporary files"""
        self._cleanup_temp_files()
        
    def process_file(self, file_path: str, processor_func, *args, **kwargs) -> Any:
        """
        Process a file with a given function
        
        Args:
            file_path: Path to file
            processor_func: Function to process text chunks
            *args, **kwargs: Additional arguments for processor_func
            
        Returns:
            Result from processor_func
        """
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Process with memory mapping
        with open(file_path, 'r') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return self._process_mmap(mm, processor_func, *args, **kwargs)
    
    def process_text(self, text: str, processor_func, *args, **kwargs) -> Any:
        """
        Process a text string with a given function
        
        Args:
            text: Text to process
            processor_func: Function to process text chunks
            *args, **kwargs: Additional arguments for processor_func
            
        Returns:
            Result from processor_func
        """
        # For very large texts, write to temp file and process with mmap
        if len(text) > self.chunk_size * 2:
            return self._process_large_text(text, processor_func, *args, **kwargs)
            
        # For smaller texts, process directly
        return processor_func(text, 0, len(text), *args, **kwargs)
    
    def detect_patterns(self, 
                      text_or_file: Union[str, os.PathLike],
                      patterns: Dict[str, Pattern],
                      is_file: bool = False) -> List[PatternMatch]:
        """
        Detect patterns in text efficiently
        
        Args:
            text_or_file: Text content or file path
            patterns: Dictionary of pattern ID to compiled regex
            is_file: Whether text_or_file is a file path
            
        Returns:
            List of pattern matches
        """
        if is_file:
            return self.process_file(text_or_file, self._detect_patterns_in_chunk, patterns)
        else:
            return self.process_text(text_or_file, self._detect_patterns_in_chunk, patterns)
    
    def redact_sensitive_info(self,
                            text_or_file: Union[str, os.PathLike],
                            patterns: Dict[str, Pattern],
                            is_file: bool = False,
                            replacement: str = "[REDACTED]") -> str:
        """
        Redact sensitive information from text
        
        Args:
            text_or_file: Text content or file path
            patterns: Dictionary of pattern ID to compiled regex
            is_file: Whether text_or_file is a file path
            replacement: Replacement string for matches
            
        Returns:
            Redacted text
        """
        if is_file:
            return self.process_file(
                text_or_file, self._redact_sensitive_info_in_chunk, 
                patterns, replacement
            )
        else:
            return self.process_text(
                text_or_file, self._redact_sensitive_info_in_chunk, 
                patterns, replacement
            )
    
    def calculate_compliance_metrics(self,
                                  text_or_file: Union[str, os.PathLike],
                                  patterns: Dict[str, Pattern],
                                  is_file: bool = False) -> Dict[str, Any]:
        """
        Calculate compliance metrics for text
        
        Args:
            text_or_file: Text content or file path
            patterns: Dictionary of pattern ID to compiled regex
            is_file: Whether text_or_file is a file path
            
        Returns:
            Dictionary of compliance metrics
        """
        if is_file:
            return self.process_file(
                text_or_file, self._calculate_compliance_metrics_in_chunk, 
                patterns
            )
        else:
            return self.process_text(
                text_or_file, self._calculate_compliance_metrics_in_chunk, 
                patterns
            )
    
    def _process_mmap(self, mm, processor_func, *args, **kwargs) -> Any:
        """Process a memory-mapped file with chunking"""
        file_size = mm.size()
        
        if file_size <= self.chunk_size:
            # Small enough to process in one go
            text = mm[:].decode('utf-8', errors='replace')
            return processor_func(text, 0, file_size, *args, **kwargs)
            
        # Process in chunks
        results = []
        pos = 0
        
        while pos < file_size:
            # Determine chunk end
            chunk_end = min(pos + self.chunk_size, file_size)
            
            # Get chunk with overlap
            if chunk_end < file_size:
                end_with_overlap = min(chunk_end + self.overlap, file_size)
            else:
                end_with_overlap = chunk_end
                
            # Extract chunk
            chunk_bytes = mm[pos:end_with_overlap]
            chunk_text = chunk_bytes.decode('utf-8', errors='replace')
            
            # Process chunk
            result = processor_func(chunk_text, pos, end_with_overlap, *args, **kwargs)
            results.append(result)
            
            # Move to next chunk
            pos = chunk_end
            
        # Combine results
        return self._combine_results(results, processor_func.__name__)
    
    def _process_large_text(self, text: str, processor_func, *args, **kwargs) -> Any:
        """Process large text by writing to temp file"""
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt')
        self.temp_files.append(temp_path)
        
        try:
            # Write text to file
            with os.fdopen(temp_fd, 'w') as f:
                f.write(text)
                
            # Process file
            return self.process_file(temp_path, processor_func, *args, **kwargs)
            
        finally:
            # Clean up temp file
            if temp_path in self.temp_files:
                try:
                    os.unlink(temp_path)
                    self.temp_files.remove(temp_path)
                except:
                    pass
    
    def _cleanup_temp_files(self):
        """Clean up any temporary files"""
        for path in self.temp_files[:]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                self.temp_files.remove(path)
            except:
                pass
    
    def _combine_results(self, results: List[Any], processor_name: str) -> Any:
        """
        Combine results from multiple chunks
        
        Args:
            results: List of results from each chunk
            processor_name: Name of processor function
            
        Returns:
            Combined result
        """
        if not results:
            return None
            
        # Handle different result types based on processor
        if processor_name == '_detect_patterns_in_chunk':
            # Combine lists of pattern matches
            combined = []
            for result in results:
                combined.extend(result)
            return self._deduplicate_matches(combined)
            
        elif processor_name == '_redact_sensitive_info_in_chunk':
            # Combine strings
            return ''.join(results)
            
        elif processor_name == '_calculate_compliance_metrics_in_chunk':
            # Combine metrics dictionaries
            combined = {
                'patterns_found': 0,
                'pattern_counts': {},
                'content_length': 0,
                'sensitive_data_ratio': 0
            }
            
            for result in results:
                combined['patterns_found'] += result.get('patterns_found', 0)
                combined['content_length'] += result.get('content_length', 0)
                
                for pattern_id, count in result.get('pattern_counts', {}).items():
                    if pattern_id not in combined['pattern_counts']:
                        combined['pattern_counts'][pattern_id] = 0
                    combined['pattern_counts'][pattern_id] += count
                    
            # Recalculate ratio
            if combined['content_length'] > 0:
                combined['sensitive_data_ratio'] = combined['patterns_found'] / combined['content_length']
                
            return combined
            
        else:
            # Default behavior - return list of results
            return results
    
    def _detect_patterns_in_chunk(self, 
                               text: str,
                               start_pos: int,
                               end_pos: int,
                               patterns: Dict[str, Pattern]) -> List[PatternMatch]:
        """
        Detect patterns in a text chunk
        
        Args:
            text: Text chunk
            start_pos: Start position of chunk in original text
            end_pos: End position of chunk in original text
            patterns: Dictionary of pattern ID to compiled regex
            
        Returns:
            List of pattern matches
        """
        matches = []
        
        for pattern_id, regex in patterns.items():
            for match in regex.finditer(text):
                # Calculate absolute positions
                abs_start = start_pos + match.start()
                abs_end = start_pos + match.end()
                
                pattern_match = PatternMatch(
                    pattern_id=pattern_id,
                    text=match.group(0),
                    start_pos=abs_start,
                    end_pos=abs_end
                )
                matches.append(pattern_match)
                
        return matches
    
    def _redact_sensitive_info_in_chunk(self,
                                      text: str,
                                      start_pos: int,
                                      end_pos: int,
                                      patterns: Dict[str, Pattern],
                                      replacement: str) -> str:
        """
        Redact sensitive information in a text chunk
        
        Args:
            text: Text chunk
            start_pos: Start position of chunk in original text
            end_pos: End position of chunk in original text
            patterns: Dictionary of pattern ID to compiled regex
            replacement: Replacement string for matches
            
        Returns:
            Redacted text
        """
        # Get all matches
        matches = self._detect_patterns_in_chunk(text, start_pos, end_pos, patterns)
        
        # Sort matches by position (in reverse order to avoid index shifting)
        matches.sort(key=lambda m: m.start_pos, reverse=True)
        
        # Apply redactions
        redacted_text = text
        for match in matches:
            # Convert to chunk-relative positions
            rel_start = match.start_pos - start_pos
            rel_end = match.end_pos - start_pos
            
            # Only redact if the match is fully within this chunk
            if 0 <= rel_start and rel_end <= len(redacted_text):
                redacted_text = redacted_text[:rel_start] + replacement + redacted_text[rel_end:]
                
        return redacted_text
    
    def _calculate_compliance_metrics_in_chunk(self,
                                            text: str,
                                            start_pos: int,
                                            end_pos: int,
                                            patterns: Dict[str, Pattern]) -> Dict[str, Any]:
        """
        Calculate compliance metrics for a text chunk
        
        Args:
            text: Text chunk
            start_pos: Start position of chunk in original text
            end_pos: End position of chunk in original text
            patterns: Dictionary of pattern ID to compiled regex
            
        Returns:
            Dictionary of compliance metrics
        """
        # Get all matches
        matches = self._detect_patterns_in_chunk(text, start_pos, end_pos, patterns)
        
        # Calculate metrics
        pattern_counts = {}
        for match in matches:
            if match.pattern_id not in pattern_counts:
                pattern_counts[match.pattern_id] = 0
            pattern_counts[match.pattern_id] += 1
            
        metrics = {
            'patterns_found': len(matches),
            'pattern_counts': pattern_counts,
            'content_length': len(text),
            'sensitive_data_ratio': len(matches) / len(text) if len(text) > 0 else 0
        }
        
        return metrics
    
    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Deduplicate pattern matches from overlapping chunks
        
        Args:
            matches: List of pattern matches
            
        Returns:
            Deduplicated list of pattern matches
        """
        # Sort matches by position
        sorted_matches = sorted(matches, key=lambda m: (m.start_pos, m.end_pos))
        
        # Deduplicate
        deduplicated = []
        prev_end = -1
        prev_hash = None
        
        for match in sorted_matches:
            # Create a hash of the match to identify duplicates
            match_hash = hashlib.md5(
                f"{match.pattern_id}:{match.start_pos}:{match.end_pos}:{match.text}".encode()
            ).hexdigest()
            
            # Check if this is a duplicate
            if match.start_pos >= prev_end and match_hash != prev_hash:
                deduplicated.append(match)
                prev_end = match.end_pos
                prev_hash = match_hash
                
        return deduplicated


class StreamingTextAnalyzer:
    """
    Analyze text in a streaming fashion to minimize memory usage.
    """
    
    def __init__(self, text_processor: MemoryEfficientTextProcessor = None):
        """
        Initialize the streaming analyzer
        
        Args:
            text_processor: Text processor to use
        """
        self.text_processor = text_processor or MemoryEfficientTextProcessor()
        
    def stream_analyze(self, 
                     text_or_file: Union[str, os.PathLike],
                     analyzer_func: Callable,
                     is_file: bool = False,
                     *args, **kwargs) -> Generator[Any, None, None]:
        """
        Analyze text in a streaming fashion
        
        Args:
            text_or_file: Text content or file path
            analyzer_func: Function to analyze each chunk
            is_file: Whether text_or_file is a file path
            *args, **kwargs: Additional arguments for analyzer_func
            
        Yields:
            Analysis results for each chunk
        """
        # Define chunk processor that yields results
        def yielding_processor(text, start_pos, end_pos, *pargs, **pkwargs):
            result = analyzer_func(text, start_pos, end_pos, *pargs, **pkwargs)
            return result
            
        # Process text and yield results
        if is_file:
            with open(text_or_file, 'r') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    file_size = mm.size()
                    pos = 0
                    
                    while pos < file_size:
                        # Determine chunk end
                        chunk_end = min(pos + self.text_processor.chunk_size, file_size)
                        
                        # Get chunk with overlap
                        if chunk_end < file_size:
                            end_with_overlap = min(chunk_end + self.text_processor.overlap, file_size)
                        else:
                            end_with_overlap = chunk_end
                            
                        # Extract chunk
                        chunk_bytes = mm[pos:end_with_overlap]
                        chunk_text = chunk_bytes.decode('utf-8', errors='replace')
                        
                        # Process chunk
                        result = yielding_processor(chunk_text, pos, end_with_overlap, *args, **kwargs)
                        yield result
                        
                        # Move to next chunk
                        pos = chunk_end
        else:
            # For text content
            text = text_or_file
            text_len = len(text)
            pos = 0
            
            while pos < text_len:
                # Determine chunk end
                chunk_end = min(pos + self.text_processor.chunk_size, text_len)
                
                # Get chunk with overlap
                if chunk_end < text_len:
                    end_with_overlap = min(chunk_end + self.text_processor.overlap, text_len)
                else:
                    end_with_overlap = chunk_end
                    
                # Extract chunk
                chunk_text = text[pos:end_with_overlap]
                
                # Process chunk
                result = yielding_processor(chunk_text, pos, end_with_overlap, *args, **kwargs)
                yield result
                
                # Move to next chunk
                pos = chunk_end
    
    @contextmanager
    def file_analyzer(self, file_path: str, StreamingTextAnalyzer = None) -> Generator[str, None, None]:   
        """
        Context manager for analyzing a file
        
        Args:
            file_path: Path to file
            
        Yields:
            Self for method chaining
        """
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Open file and provide analyzer
        try:
            self.current_file_path = file_path
            yield self
        finally:
            if hasattr(self, 'current_file_path'):
                del self.current_file_path
    
    def detect_entities(self, 
                      entity_patterns: Dict[str, Pattern],
                      max_entities: int = None) -> List[Dict[str, Any]]:
        """
        Detect entities in text or file
        
        Args:
            entity_patterns: Dictionary of entity type to regex pattern
            max_entities: Maximum number of entities to detect
            
        Returns:
            List of detected entities
        """
        if not hasattr(self, 'current_file_path'):
            raise ValueError("Must be used within file_analyzer context")
            
        # Compile patterns if needed
        compiled_patterns = {}
        for entity_type, pattern in entity_patterns.items():
            if isinstance(pattern, str):
                compiled_patterns[entity_type] = re.compile(pattern)
            else:
                compiled_patterns[entity_type] = pattern
                
        # Analyze file
        entities = []
        
        for matches in self.stream_analyze(
            self.current_file_path, 
            self._detect_entities_in_chunk,
            is_file=True,
            patterns=compiled_patterns
        ):
            entities.extend(matches)
            
            # Check max entities
            if max_entities and len(entities) >= max_entities:
                entities = entities[:max_entities]
                break
                
        return entities
    
    def _detect_entities_in_chunk(self,
                                text: str,
                                start_pos: int,
                                end_pos: int,
                                patterns: Dict[str, Pattern]) -> List[Dict[str, Any]]:
        """
        Detect entities in a text chunk
        
        Args:
            text: Text chunk
            start_pos: Start position of chunk in original text
            end_pos: End position of chunk in original text
            patterns: Dictionary of entity type to regex pattern
            
        Returns:
            List of detected entities
        """
        entities = []
        
        for entity_type, pattern in patterns.items():
            for match in pattern.finditer(text):
                # Calculate absolute positions
                abs_start = start_pos + match.start()
                abs_end = start_pos + match.end()
                
                entity = {
                    'type': entity_type,
                    'text': match.group(0),
                    'start': abs_start,
                    'end': abs_end
                }
                
                # Add any named groups as metadata
                if match.groupdict():
                    entity['metadata'] = match.groupdict()
                    
                entities.append(entity)
                
        return entities

def get_pii_patterns() -> Dict[str, Pattern]:
    """Get common PII regex patterns"""
    pii_patterns = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'ssn': re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
        'phone': re.compile(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        'credit_card': re.compile(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'date_of_birth': re.compile(r'\b(?:0[1-9]|1[0-2])[/.-](?:0[1-9]|[12][0-9]|3[01])[/.-](?:19|20)\d{2}\b')
    }
    return pii_patterns

# Example usage of the memory-efficient text processor:
"""
# Create processor
processor = MemoryEfficientTextProcessor()

# Get common PII patterns
pii_patterns = get_pii_patterns()

# Process a large file
matches = processor.detect_patterns(
    "large_document.txt",
    pii_patterns,
    is_file=True
)
print(f"Found {len(matches)} PII matches")

# Redact sensitive information
redacted_text = processor.redact_sensitive_info(
    "large_document.txt",
    pii_patterns,
    is_file=True
)
with open("redacted_document.txt", "w") as f:
    f.write(redacted_text)

# Calculate compliance metrics
metrics = processor.calculate_compliance_metrics(
    "large_document.txt",
    pii_patterns,
    is_file=True
)
print(f"Compliance metrics: {metrics}")

# Use streaming analyzer
analyzer = StreamingTextAnalyzer()
with analyzer.file_analyzer("large_document.txt") as file_analyzer:
    entities = file_analyzer.detect_entities(pii_patterns)
    print(f"Found {len(entities)} entities")
"""