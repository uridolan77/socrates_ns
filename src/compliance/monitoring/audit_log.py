import datetime
import json
import os
import re
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict

@dataclass
class ComplianceAuditEntry:
    """Data class representing a single compliance audit log entry."""
    entry_id: str
    timestamp: str
    event_type: str
    framework_id: Optional[str] = None
    content_hash: Optional[str] = None
    compliance_result: Optional[bool] = None
    compliance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

class ComplianceAuditLog:
    """
    Manages a comprehensive audit log for regulatory compliance actions.
    
    This class provides mechanisms to record, store, and retrieve compliance
    audit events for accountability and regulatory reporting requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the compliance audit log system.
        
        Args:
            config: Configuration dictionary with audit log settings
        """
        self.config = config
        self.storage_path = config.get("audit_log_path", "./compliance_audit_logs")
        self.retention_days = config.get("retention_days", 365)
        self.log_level = config.get("log_level", "INFO")
        self.enabled = config.get("audit_log_enabled", True)
        self.in_memory_buffer = []
        self.buffer_size = config.get("buffer_size", 100)
        self.sanitize_pii = config.get("sanitize_pii", True)
        
        # Ensure storage directory exists
        if self.enabled and not self.config.get("in_memory_only", False):
            os.makedirs(self.storage_path, exist_ok=True)
            
        # Configure logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the compliance audit logger."""
        self.logger = logging.getLogger("compliance_audit")
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # Add file handler if not in memory only
        if not self.config.get("in_memory_only", False):
            log_file = os.path.join(self.storage_path, f"compliance_audit.log")
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_verification_event(self, 
                              framework_id: str,
                              content_hash: str,
                              compliance_result: bool,
                              compliance_score: float,
                              violations: List[Dict[str, Any]] = None,
                              metadata: Dict[str, Any] = None,
                              user_id: Optional[str] = None,
                              session_id: Optional[str] = None,
                              request_id: Optional[str] = None) -> str:
        """
        Log a compliance verification event.
        
        Args:
            framework_id: ID of the regulatory framework
            content_hash: Hash of the content being verified
            compliance_result: Whether content is compliant or not
            compliance_score: Numerical compliance score (0-1)
            violations: List of compliance violations found
            metadata: Additional context metadata
            user_id: Optional user identifier
            session_id: Optional session identifier
            request_id: Optional request identifier
            
        Returns:
            The entry ID of the logged event
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        # Create audit entry
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        audit_entry = ComplianceAuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            event_type="verification",
            framework_id=framework_id,
            content_hash=content_hash,
            compliance_result=compliance_result,
            compliance_score=compliance_score,
            violations=violations or [],
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )
        
        # Sanitize PII if configured
        if self.sanitize_pii:
            audit_entry = self._sanitize_entry(audit_entry)
        
        # Store the entry
        self._store_entry(audit_entry)
        
        # Log summary
        self.logger.info(
            f"Compliance verification: framework={framework_id}, "
            f"compliant={compliance_result}, score={compliance_score:.2f}, "
            f"violations={len(audit_entry.violations)}"
        )
        
        return entry_id
    
    def log_filtering_event(self,
                           content_hash: str,
                           filtered_content_hash: str,
                           violations: List[Dict[str, Any]] = None,
                           metadata: Dict[str, Any] = None,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           request_id: Optional[str] = None) -> str:
        """
        Log a content filtering event.
        
        Args:
            content_hash: Hash of original content
            filtered_content_hash: Hash of filtered content
            violations: List of violations addressed by filtering
            metadata: Additional context metadata
            user_id: Optional user identifier
            session_id: Optional session identifier
            request_id: Optional request identifier
            
        Returns:
            The entry ID of the logged event
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        # Create audit entry
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Add filtering metadata
        filter_metadata = metadata or {}
        filter_metadata.update({
            "original_content_hash": content_hash,
            "filtered_content_hash": filtered_content_hash,
            "was_modified": content_hash != filtered_content_hash
        })
        
        audit_entry = ComplianceAuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            event_type="filtering",
            content_hash=content_hash,
            violations=violations or [],
            metadata=filter_metadata,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )
        
        # Sanitize PII if configured
        if self.sanitize_pii:
            audit_entry = self._sanitize_entry(audit_entry)
        
        # Store the entry
        self._store_entry(audit_entry)
        
        # Log summary
        was_modified = content_hash != filtered_content_hash
        self.logger.info(
            f"Content filtering: modified={was_modified}, "
            f"violations={len(audit_entry.violations)}"
        )
        
        return entry_id
    
    def log_exception_event(self,
                           error_message: str,
                           error_type: str,
                           stack_trace: Optional[str] = None,
                           metadata: Dict[str, Any] = None,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           request_id: Optional[str] = None) -> str:
        """
        Log a compliance-related exception event.
        
        Args:
            error_message: Error message
            error_type: Type of error 
            stack_trace: Optional stack trace
            metadata: Additional context metadata
            user_id: Optional user identifier
            session_id: Optional session identifier
            request_id: Optional request identifier
            
        Returns:
            The entry ID of the logged event
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        # Create audit entry
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Add exception metadata
        exception_metadata = metadata or {}
        exception_metadata.update({
            "error_message": error_message,
            "error_type": error_type,
            "stack_trace": stack_trace
        })
        
        audit_entry = ComplianceAuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            event_type="exception",
            metadata=exception_metadata,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )
        
        # Sanitize PII if configured
        if self.sanitize_pii:
            audit_entry = self._sanitize_entry(audit_entry)
        
        # Store the entry
        self._store_entry(audit_entry)
        
        # Log summary (use ERROR level for exceptions)
        self.logger.error(
            f"Compliance exception: type={error_type}, "
            f"message={error_message[:100]}"
        )
        
        return entry_id
    
    def log_system_event(self,
                         event_name: str,
                         description: str,
                         metadata: Dict[str, Any] = None,
                         user_id: Optional[str] = None) -> str:
        """
        Log a compliance system event.
        
        Args:
            event_name: Name of the system event
            description: Description of the event
            metadata: Additional context metadata
            user_id: Optional user identifier
            
        Returns:
            The entry ID of the logged event
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        # Create audit entry
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Add system event metadata
        system_metadata = metadata or {}
        system_metadata.update({
            "event_name": event_name,
            "description": description
        })
        
        audit_entry = ComplianceAuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            event_type="system",
            metadata=system_metadata,
            user_id=user_id
        )
        
        # Sanitize PII if configured
        if self.sanitize_pii:
            audit_entry = self._sanitize_entry(audit_entry)
        
        # Store the entry
        self._store_entry(audit_entry)
        
        # Log summary
        self.logger.info(
            f"System event: {event_name} - {description[:100]}"
        )
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific audit log entry by ID.
        
        Args:
            entry_id: The ID of the audit entry to retrieve
            
        Returns:
            The audit entry or None if not found
        """
        # Check in-memory buffer first
        for entry in self.in_memory_buffer:
            if entry.entry_id == entry_id:
                return asdict(entry)
        
        # Check file storage if enabled
        if not self.config.get("in_memory_only", False):
            entry_path = os.path.join(self.storage_path, f"{entry_id}.json")
            if os.path.exists(entry_path):
                try:
                    with open(entry_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.error(f"Error reading audit entry {entry_id}: {str(e)}")
        
        return None
    
    def query_logs(self, 
                  start_time: Optional[datetime.datetime] = None,
                  end_time: Optional[datetime.datetime] = None,
                  event_types: Optional[List[str]] = None,
                  framework_ids: Optional[List[str]] = None,
                  compliance_result: Optional[bool] = None,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query audit logs with filtering criteria.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            event_types: Optional list of event types to include
            framework_ids: Optional list of framework IDs to include
            compliance_result: Optional compliance result filter
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching audit entries
        """
        # Convert datetime objects to ISO strings for comparison
        start_iso = start_time.isoformat() if start_time else None
        end_iso = end_time.isoformat() if end_time else None
        
        # Start with in-memory buffer
        results = []
        
        # Filter in-memory entries
        for entry in self.in_memory_buffer:
            if self._matches_criteria(entry, start_iso, end_iso, event_types,
                                    framework_ids, compliance_result, user_id, session_id):
                results.append(asdict(entry))
        
        # If we need more results and file storage is enabled, scan files
        if len(results) < limit and not self.config.get("in_memory_only", False):
            # List all json files in storage directory
            try:
                json_files = [f for f in os.listdir(self.storage_path) 
                             if f.endswith('.json') and f != 'audit_metadata.json']
                
                # Process files to find matching entries
                for filename in json_files:
                    if len(results) >= limit:
                        break
                        
                    file_path = os.path.join(self.storage_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            entry_dict = json.load(f)
                            # Convert dict to ComplianceAuditEntry for filtering
                            entry = ComplianceAuditEntry(**entry_dict)
                            
                            if self._matches_criteria(entry, start_iso, end_iso, event_types,
                                                    framework_ids, compliance_result, user_id, session_id):
                                results.append(entry_dict)
                    except Exception as e:
                        self.logger.error(f"Error reading audit file {filename}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error listing audit files: {str(e)}")
        
        # Sort results by timestamp (most recent first)
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def generate_compliance_report(self, 
                                 start_time: datetime.datetime,
                                 end_time: datetime.datetime,
                                 framework_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a compliance report for a specific time period and framework.
        
        Args:
            start_time: Start time for the report period
            end_time: End time for the report period
            framework_id: Optional framework ID to filter by
            
        Returns:
            Compliance report with statistics and summaries
        """
        # Query logs for the specified period and framework
        framework_ids = [framework_id] if framework_id else None
        logs = self.query_logs(
            start_time=start_time,
            end_time=end_time,
            event_types=["verification"],
            framework_ids=framework_ids,
            limit=10000  # Large limit to get comprehensive data
        )
        
        # Analyze verification results
        total_verifications = len(logs)
        compliant_count = sum(1 for log in logs if log.get('compliance_result', False))
        non_compliant_count = total_verifications - compliant_count
        
        # Calculate compliance rate
        compliance_rate = (compliant_count / total_verifications) if total_verifications > 0 else 1.0
        
        # Count violations by type
        violation_types = {}
        for log in logs:
            for violation in log.get('violations', []):
                violation_type = violation.get('type', 'unknown')
                if violation_type not in violation_types:
                    violation_types[violation_type] = 0
                violation_types[violation_type] += 1
        
        # Get top 5 violation types
        top_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.datetime.now().isoformat(),
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "framework_id": framework_id,
            "summary": {
                "total_verifications": total_verifications,
                "compliant_count": compliant_count,
                "non_compliant_count": non_compliant_count,
                "compliance_rate": compliance_rate
            },
            "violations": {
                "total_violations": sum(violation_types.values()),
                "unique_violation_types": len(violation_types),
                "top_violations": top_violations
            }
        }
        
        # Add framework-specific metrics if applicable
        if framework_id:
            report["framework_specific"] = {
                "name": framework_id,
                "average_compliance_score": self._calculate_avg_compliance_score(logs)
            }
        
        return report
    
    def export_logs(self, 
                   start_time: datetime.datetime,
                   end_time: datetime.datetime,
                   format: str = "json",
                   output_path: Optional[str] = None) -> str:
        """
        Export audit logs to a file.
        
        Args:
            start_time: Start time for logs to export
            end_time: End time for logs to export
            format: Export format ('json' or 'csv')
            output_path: Optional output file path
            
        Returns:
            Path to the exported file
        """
        # Query logs for the specified period
        logs = self.query_logs(
            start_time=start_time,
            end_time=end_time,
            limit=100000  # Large limit to get comprehensive data
        )
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output path
        if not output_path:
            filename = f"compliance_audit_export_{timestamp}.{format}"
            output_path = os.path.join(self.storage_path, filename)
        
        # Export based on format
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
        elif format.lower() == "csv":
            import csv
            
            # Flatten nested structures for CSV
            flattened_logs = []
            for log in logs:
                flat_log = {
                    "entry_id": log.get("entry_id", ""),
                    "timestamp": log.get("timestamp", ""),
                    "event_type": log.get("event_type", ""),
                    "framework_id": log.get("framework_id", ""),
                    "content_hash": log.get("content_hash", ""),
                    "compliance_result": str(log.get("compliance_result", "")),
                    "compliance_score": str(log.get("compliance_score", "")),
                    "user_id": log.get("user_id", ""),
                    "session_id": log.get("session_id", ""),
                    "request_id": log.get("request_id", ""),
                    "violation_count": str(len(log.get("violations", []))),
                    "metadata": json.dumps(log.get("metadata", {}))
                }
                flattened_logs.append(flat_log)
            
            with open(output_path, 'w', newline='') as f:
                if flattened_logs:
                    writer = csv.DictWriter(f, fieldnames=flattened_logs[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_logs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path
    
    def purge_old_logs(self, days: Optional[int] = None) -> int:
        """
        Purge audit logs older than specified days.
        
        Args:
            days: Number of days to keep (default: use retention_days config)
            
        Returns:
            Number of logs purged
        """
        if not self.enabled or self.config.get("in_memory_only", False):
            return 0
            
        retention_days = days if days is not None else self.retention_days
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()
        
        purged_count = 0
        
        # List all json files in storage directory
        try:
            json_files = [f for f in os.listdir(self.storage_path) 
                         if f.endswith('.json') and f != 'audit_metadata.json']
            
            # Check each file and remove if older than cutoff
            for filename in json_files:
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        entry = json.load(f)
                        timestamp = entry.get('timestamp', '')
                        
                        if timestamp < cutoff_iso:
                            os.remove(file_path)
                            purged_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing file {filename} for purge: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error listing audit files for purge: {str(e)}")
        
        # Log purge results
        self.logger.info(f"Purged {purged_count} audit logs older than {retention_days} days")
        
        return purged_count
    
    def _store_entry(self, entry: ComplianceAuditEntry):
        """Store an audit entry."""
        # Add to in-memory buffer
        self.in_memory_buffer.append(entry)
        
        # Trim buffer if it exceeds max size
        if len(self.in_memory_buffer) > self.buffer_size:
            self.in_memory_buffer = self.in_memory_buffer[-self.buffer_size:]
        
        # Store to file if not in-memory only
        if not self.config.get("in_memory_only", False):
            try:
                entry_dict = asdict(entry)
                entry_path = os.path.join(self.storage_path, f"{entry.entry_id}.json")
                
                with open(entry_path, 'w') as f:
                    json.dump(entry_dict, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error storing audit entry: {str(e)}")
    
    def _sanitize_entry(self, entry: ComplianceAuditEntry) -> ComplianceAuditEntry:
        """Sanitize PII from audit entry."""
        # Create a copy to avoid modifying the original
        sanitized = ComplianceAuditEntry(
            entry_id=entry.entry_id,
            timestamp=entry.timestamp,
            event_type=entry.event_type,
            framework_id=entry.framework_id,
            content_hash=entry.content_hash,
            compliance_result=entry.compliance_result,
            compliance_score=entry.compliance_score,
            metadata=entry.metadata.copy(),
            violations=[v.copy() for v in entry.violations],
            user_id=entry.user_id,
            session_id=entry.session_id,
            request_id=entry.request_id
        )
        
        # Remove sensitive fields from violations
        for violation in sanitized.violations:
            if "matched_content" in violation:
                violation["matched_content"] = self._redact_text(violation["matched_content"])
            
            if "context" in violation:
                violation["context"] = self._redact_text(violation["context"])
        
        # Sanitize metadata
        if "user_details" in sanitized.metadata:
            if isinstance(sanitized.metadata["user_details"], dict):
                user_details = sanitized.metadata["user_details"]
                sanitized_details = {}
                
                # Keep non-PII fields
                safe_fields = ["role", "access_level", "authenticated"]
                for field in safe_fields:
                    if field in user_details:
                        sanitized_details[field] = user_details[field]
                
                sanitized.metadata["user_details"] = sanitized_details
            else:
                # If not a dict, remove it entirely
                del sanitized.metadata["user_details"]
        
        return sanitized
    
    def _redact_text(self, text: str) -> str:
        """Redact potentially sensitive information from text."""
        if not text:
            return text
            
        # Redact common PII patterns
        pii_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN]'),
            (r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
            (r'\b(?:\d{4}[- ]?){3}\d{4}\b', '[CREDIT_CARD]')
        ]
        
        redacted = text
        for pattern, replacement in pii_patterns:
            redacted = re.sub(pattern, replacement, redacted)
            
        return redacted
    
    def _matches_criteria(self,
                         entry: ComplianceAuditEntry,
                         start_iso: Optional[str],
                         end_iso: Optional[str],
                         event_types: Optional[List[str]],
                         framework_ids: Optional[List[str]],
                         compliance_result: Optional[bool],
                         user_id: Optional[str],
                         session_id: Optional[str]) -> bool:
        """Check if an entry matches the specified criteria."""
        # Check timestamp range
        if start_iso and entry.timestamp < start_iso:
            return False
            
        if end_iso and entry.timestamp > end_iso:
            return False
            
        # Check event type
        if event_types and entry.event_type not in event_types:
            return False
            
        # Check framework ID
        if framework_ids and entry.framework_id not in framework_ids:
            return False
            
        # Check compliance result
        if compliance_result is not None and entry.compliance_result != compliance_result:
            return False
            
        # Check user ID
        if user_id and entry.user_id != user_id:
            return False
            
        # Check session ID
        if session_id and entry.session_id != session_id:
            return False
            
        # All criteria matched (or weren't specified)
        return True
    
    def _calculate_avg_compliance_score(self, logs: List[Dict[str, Any]]) -> float:
        """Calculate average compliance score from logs."""
        scores = [log.get('compliance_score', 0.0) for log in logs 
                 if log.get('compliance_score') is not None]
        
        return sum(scores) / len(scores) if scores else 0.0