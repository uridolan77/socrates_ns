import re
import datetime
import collections
from typing import List, Dict, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass, field
import time

class PatternAnalyzer:
    """
    Analyzes compliance violations to identify contextual patterns.
    
    This class detects patterns in compliance violations across different dimensions
    such as content type, platform, time, location, and multi-context environments.
    """
    
    def __init__(self, config=None):
        """Initialize the pattern analyzer with configuration."""
        self.config = config or {}
        self.min_pattern_confidence = self.config.get("min_pattern_confidence", 0.6)
        self.min_violations_for_pattern = self.config.get("min_violations_for_pattern", 3)
        
        # Pattern matching thresholds
        self.thresholds = {
            "content_type": self.config.get("content_type_threshold", 0.7),
            "platform": self.config.get("platform_threshold", 0.6),
            "time": self.config.get("time_threshold", 0.5),
            "location": self.config.get("location_threshold", 0.6),
            "multi_context": self.config.get("multi_context_threshold", 0.75),
            "sequential": self.config.get("sequential_threshold", 0.65)
        }
        
        # Pattern recognition state
        self.content_type_statistics = collections.defaultdict(int)
        self.platform_statistics = collections.defaultdict(int)
        self.time_statistics = collections.defaultdict(int)
        self.location_statistics = collections.defaultdict(int)
        self.sequential_patterns = []
        
        # Historical violation data
        self.historical_violations = []
        self.max_history_size = self.config.get("max_history_size", 1000)
    
    def analyze_content_type_patterns(self, violations, context=None):
        """
        Analyze patterns in content types that frequently have compliance issues.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of content type patterns with confidence scores
        """
        if not violations:
            return []
        
        # Track content types for current violations
        current_content_types = self._extract_content_types(violations, context)
        
        # Update statistics with current content types
        for content_type, count in current_content_types.items():
            self.content_type_statistics[content_type] += count
        
        # Identify patterns based on frequency and recency
        patterns = []
        total_violations = sum(self.content_type_statistics.values())
        
        if total_violations >= self.min_violations_for_pattern:
            for content_type, count in self.content_type_statistics.items():
                # Calculate frequency ratio
                frequency = count / total_violations
                
                # Check if this content type appears in current violations
                is_current = content_type in current_content_types
                
                # Calculate confidence score
                confidence = self._calculate_pattern_confidence(
                    frequency, 
                    is_current,
                    threshold=self.thresholds["content_type"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "content_type",
                        "content_type": content_type,
                        "frequency": frequency,
                        "violation_count": count,
                        "confidence": confidence,
                        "examples": self._get_content_type_examples(content_type)
                    }
                    patterns.append(pattern)
        
        # Sort patterns by confidence (highest first)
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return patterns
    
    def _extract_content_types(self, violations, context=None):
        """Extract content types from violations and context."""
        content_types = collections.defaultdict(int)
        
        # Extract from violations
        for violation in violations:
            metadata = violation.get("metadata", {})
            
            # Try to find content type in various fields
            content_type = (
                metadata.get("content_type") or 
                metadata.get("document_type") or 
                metadata.get("media_type")
            )
            
            if content_type:
                content_types[content_type] += 1
                continue
            
            # Try to infer from rule ID or description
            rule_id = violation.get("rule_id", "")
            description = violation.get("description", "")
            
            inferred_type = self._infer_content_type(rule_id, description)
            if inferred_type:
                content_types[inferred_type] += 1
                continue
            
            # Fallback to "unknown"
            content_types["unknown"] += 1
        
        # Extract from context if provided
        if context:
            ctx_content_type = context.get("content_type")
            if ctx_content_type:
                content_types[ctx_content_type] += 1
        
        return dict(content_types)
    
    def _infer_content_type(self, rule_id, description):
        """Infer content type from rule ID and description."""
        # Common content types to check for
        content_type_keywords = {
            "text": ["text", "document", "article", "content", "message", "post"],
            "image": ["image", "picture", "photo", "graphic", "visual"],
            "video": ["video", "stream", "clip", "footage", "recording"],
            "audio": ["audio", "sound", "voice", "recording", "podcast"],
            "structured_data": ["data", "record", "dataset", "database", "structured"],
            "form": ["form", "submission", "application", "entry", "input"],
            "email": ["email", "message", "communication"],
            "social_media": ["social", "post", "tweet", "status", "comment"],
            "marketing": ["marketing", "advertisement", "promotion", "campaign"],
            "report": ["report", "summary", "analysis", "assessment"],
            "user_generated": ["user", "generated", "submitted", "uploaded"],
            "financial": ["financial", "transaction", "payment", "invoice"],
            "health": ["health", "medical", "clinical", "patient"],
            "legal": ["legal", "contract", "agreement", "terms"]
        }
        
        # Convert to lowercase for case-insensitive matching
        rule_id_lower = rule_id.lower()
        description_lower = description.lower()
        combined_text = rule_id_lower + " " + description_lower
        
        # Score each content type based on keyword matches
        scores = {}
        for content_type, keywords in content_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[content_type] = score
        
        # Return content type with highest score, if any
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _get_content_type_examples(self, content_type, max_examples=3):
        """Get examples of violations for this content type."""
        examples = []
        count = 0
        
        for violation in reversed(self.historical_violations):
            metadata = violation.get("metadata", {})
            violation_content_type = (
                metadata.get("content_type") or 
                metadata.get("document_type") or 
                metadata.get("media_type")
            )
            
            # Skip if not matching the target content type
            if violation_content_type != content_type:
                # Try inference if direct match fails
                if not self._infer_content_type(
                    violation.get("rule_id", ""), 
                    violation.get("description", "")
                ) == content_type:
                    continue
            
            # Add example
            examples.append({
                "rule_id": violation.get("rule_id", "unknown"),
                "description": violation.get("description", ""),
                "severity": violation.get("severity", "medium")
            })
            
            count += 1
            if count >= max_examples:
                break
        
        return examples
    
    def analyze_platform_patterns(self, violations, context=None):
        """
        Analyze patterns in platforms that frequently have compliance issues.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of platform patterns with confidence scores
        """
        if not violations:
            return []
        
        # Track platforms for current violations
        current_platforms = self._extract_platforms(violations, context)
        
        # Update statistics with current platforms
        for platform, count in current_platforms.items():
            self.platform_statistics[platform] += count
        
        # Identify patterns based on frequency and recency
        patterns = []
        total_violations = sum(self.platform_statistics.values())
        
        if total_violations >= self.min_violations_for_pattern:
            for platform, count in self.platform_statistics.items():
                # Calculate frequency ratio
                frequency = count / total_violations
                
                # Check if this platform appears in current violations
                is_current = platform in current_platforms
                
                # Calculate confidence score
                confidence = self._calculate_pattern_confidence(
                    frequency, 
                    is_current,
                    threshold=self.thresholds["platform"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    # Get platform-specific characteristics
                    platform_characteristics = self._get_platform_characteristics(platform)
                    
                    pattern = {
                        "pattern_type": "platform",
                        "platform": platform,
                        "frequency": frequency,
                        "violation_count": count,
                        "confidence": confidence,
                        "platform_characteristics": platform_characteristics,
                        "examples": self._get_platform_examples(platform)
                    }
                    patterns.append(pattern)
        
        # Sort patterns by confidence (highest first)
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return patterns
    
    def _extract_platforms(self, violations, context=None):
        """Extract platforms from violations and context."""
        platforms = collections.defaultdict(int)
        
        # Extract from violations
        for violation in violations:
            metadata = violation.get("metadata", {})
            
            # Try to find platform in various fields
            platform = (
                metadata.get("platform") or 
                metadata.get("source") or 
                metadata.get("channel") or
                metadata.get("system")
            )
            
            if platform:
                platforms[platform] += 1
                continue
            
            # Try to infer from rule ID or description
            rule_id = violation.get("rule_id", "")
            description = violation.get("description", "")
            
            inferred_platform = self._infer_platform(rule_id, description)
            if inferred_platform:
                platforms[inferred_platform] += 1
                continue
            
            # Fallback to "unknown"
            platforms["unknown"] += 1
        
        # Extract from context if provided
        if context:
            ctx_platform = (
                context.get("platform") or
                context.get("source") or
                context.get("channel")
            )
            if ctx_platform:
                platforms[ctx_platform] += 1
        
        return dict(platforms)
    
    def _infer_platform(self, rule_id, description):
        """Infer platform from rule ID and description."""
        # Common platforms to check for
        platform_keywords = {
            "web": ["web", "website", "browser", "online", "internet", "site"],
            "mobile": ["mobile", "app", "ios", "android", "phone", "tablet"],
            "email": ["email", "mail", "smtp", "message"],
            "social_media": ["social", "facebook", "twitter", "instagram", "linkedin", "youtube"],
            "internal_systems": ["internal", "system", "intranet", "network", "enterprise"],
            "cloud": ["cloud", "aws", "azure", "google cloud", "s3", "storage"],
            "desktop": ["desktop", "windows", "macos", "linux", "application"],
            "api": ["api", "service", "endpoint", "interface", "integration"],
            "cms": ["cms", "content management", "wordpress", "drupal"],
            "ecommerce": ["ecommerce", "shop", "store", "cart", "checkout", "payment"],
            "iot": ["iot", "device", "sensor", "connected", "smart"]
        }
        
        # Convert to lowercase for case-insensitive matching
        rule_id_lower = rule_id.lower()
        description_lower = description.lower()
        combined_text = rule_id_lower + " " + description_lower
        
        # Score each platform based on keyword matches
        scores = {}
        for platform, keywords in platform_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[platform] = score
        
        # Return platform with highest score, if any
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _get_platform_characteristics(self, platform):
        """Get characteristics of the platform."""
        # Define known platform characteristics
        characteristics = {
            "web": {
                "user_controls": "varied",
                "data_persistence": "cookies and local storage",
                "compliance_challenges": ["cross-border data transfers", "user tracking", "accessibility"]
            },
            "mobile": {
                "user_controls": "limited",
                "data_persistence": "app storage and device",
                "compliance_challenges": ["location data", "device identifiers", "notification permissions"]
            },
            "email": {
                "user_controls": "minimal",
                "data_persistence": "high",
                "compliance_challenges": ["marketing consent", "record retention", "sensitive data transmission"]
            },
            "social_media": {
                "user_controls": "platform-dependent",
                "data_persistence": "high",
                "compliance_challenges": ["public exposure", "third-party data sharing", "content moderation"]
            },
            "internal_systems": {
                "user_controls": "organization-defined",
                "data_persistence": "centralized",
                "compliance_challenges": ["access controls", "data classification", "audit trails"]
            },
            "cloud": {
                "user_controls": "varied",
                "data_persistence": "distributed",
                "compliance_challenges": ["data sovereignty", "shared responsibility", "deletion verification"]
            },
            "desktop": {
                "user_controls": "high",
                "data_persistence": "local with potential backup",
                "compliance_challenges": ["local storage security", "update management", "data leakage"]
            },
            "api": {
                "user_controls": "minimal",
                "data_persistence": "transaction-based",
                "compliance_challenges": ["authentication", "rate limiting", "data validation"]
            },
            "cms": {
                "user_controls": "role-based",
                "data_persistence": "structured and searchable",
                "compliance_challenges": ["version control", "content approval", "metadata management"]
            },
            "ecommerce": {
                "user_controls": "account-based",
                "data_persistence": "transaction and profile",
                "compliance_challenges": ["payment data", "tax compliance", "product information accuracy"]
            },
            "iot": {
                "user_controls": "minimal",
                "data_persistence": "sensor data streams",
                "compliance_challenges": ["continuous monitoring", "firmware updates", "physical security"]
            },
            "unknown": {
                "user_controls": "undefined",
                "data_persistence": "undefined",
                "compliance_challenges": ["comprehensive assessment needed"]
            }
        }
        
        return characteristics.get(platform, characteristics["unknown"])
    
    def _get_platform_examples(self, platform, max_examples=3):
        """Get examples of violations for this platform."""
        examples = []
        count = 0
        
        for violation in reversed(self.historical_violations):
            metadata = violation.get("metadata", {})
            violation_platform = (
                metadata.get("platform") or 
                metadata.get("source") or 
                metadata.get("channel") or
                metadata.get("system")
            )
            
            # Skip if not matching the target platform
            if violation_platform != platform:
                # Try inference if direct match fails
                if not self._infer_platform(
                    violation.get("rule_id", ""), 
                    violation.get("description", "")
                ) == platform:
                    continue
            
            # Add example
            examples.append({
                "rule_id": violation.get("rule_id", "unknown"),
                "description": violation.get("description", ""),
                "severity": violation.get("severity", "medium")
            })
            
            count += 1
            if count >= max_examples:
                break
        
        return examples
    
    def analyze_time_patterns(self, violations, context=None):
        """
        Analyze temporal patterns in compliance violations.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of time-based patterns with confidence scores
        """
        if not violations:
            return []
        
        # Get current time for reference
        current_time = datetime.datetime.now()
        
        # Add current violations to historical record with timestamps
        for violation in violations:
            violation_copy = violation.copy()
            
            # Add timestamp if not present
            if "timestamp" not in violation_copy:
                violation_copy["timestamp"] = current_time.isoformat()
            
            self.historical_violations.append(violation_copy)
        
        # Limit historical violations to configured max size
        if len(self.historical_violations) > self.max_history_size:
            self.historical_violations = self.historical_violations[-self.max_history_size:]
        
        # Extract time features from violations
        time_features = self._extract_time_features(self.historical_violations)
        
        # Analyze for patterns
        patterns = []
        
        # Check for hourly patterns
        hourly_patterns = self._analyze_hourly_patterns(time_features)
        patterns.extend(hourly_patterns)
        
        # Check for daily patterns
        daily_patterns = self._analyze_daily_patterns(time_features)
        patterns.extend(daily_patterns)
        
        # Check for weekly patterns
        weekly_patterns = self._analyze_weekly_patterns(time_features)
        patterns.extend(weekly_patterns)
        
        # Check for monthly patterns
        monthly_patterns = self._analyze_monthly_patterns(time_features)
        patterns.extend(monthly_patterns)
        
        # Check for frequency change patterns
        frequency_patterns = self._analyze_frequency_changes(time_features)
        patterns.extend(frequency_patterns)
        
        # Sort patterns by confidence (highest first)
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return patterns
    
    def _extract_time_features(self, violations):
        """Extract time features from violations."""
        features = {
            "hours": collections.defaultdict(int),
            "days": collections.defaultdict(int),
            "weekdays": collections.defaultdict(int),
            "months": collections.defaultdict(int),
            "timestamps": []
        }
        
        for violation in violations:
            # Get timestamp
            timestamp_str = violation.get("timestamp")
            if not timestamp_str:
                continue
            
            try:
                # Parse timestamp
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                
                # Extract features
                features["hours"][timestamp.hour] += 1
                features["days"][timestamp.day] += 1
                features["weekdays"][timestamp.weekday()] += 1
                features["months"][timestamp.month] += 1
                features["timestamps"].append(timestamp)
                
            except (ValueError, TypeError):
                # Skip if timestamp parsing fails
                continue
        
        return features
    
    def _analyze_hourly_patterns(self, time_features):
        """Analyze patterns in violation distribution by hour of day."""
        patterns = []
        
        # Get hour counts
        hours = time_features["hours"]
        total_violations = sum(hours.values())
        
        if total_violations < self.min_violations_for_pattern:
            return []
        
        # Find peak hours (hours with unusually high violation counts)
        avg_hourly = total_violations / 24  # Expected even distribution
        threshold = max(avg_hourly * 1.5, 3)  # At least 50% above average or minimum count
        
        peak_hours = []
        for hour, count in hours.items():
            if count >= threshold:
                peak_hours.append((hour, count))
        
        # Group consecutive peak hours into time ranges
        if peak_hours:
            peak_hours.sort()
            
            hour_ranges = []
            current_range = [peak_hours[0]]
            
            for i in range(1, len(peak_hours)):
                current_hour = peak_hours[i]
                prev_hour = peak_hours[i-1]
                
                # Check if consecutive
                if current_hour[0] == (prev_hour[0] + 1) % 24:
                    current_range.append(current_hour)
                else:
                    hour_ranges.append(current_range)
                    current_range = [current_hour]
            
            hour_ranges.append(current_range)
            
            # Create pattern for each time range
            for hour_range in hour_ranges:
                start_hour = hour_range[0][0]
                end_hour = hour_range[-1][0]
                range_count = sum(h[1] for h in hour_range)
                
                # Format time range
                if start_hour == end_hour:
                    time_range = f"{start_hour:02d}:00-{(start_hour+1) % 24:02d}:00"
                else:
                    time_range = f"{start_hour:02d}:00-{(end_hour+1) % 24:02d}:00"
                
                # Calculate range proportion
                range_proportion = range_count / total_violations
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    range_proportion,
                    True,  # Time patterns are always considered current
                    threshold=self.thresholds["time"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "time_hourly",
                        "time_range": time_range,
                        "violation_count": range_count,
                        "proportion": range_proportion,
                        "confidence": confidence,
                        "description": f"Higher violation frequency during {time_range}"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_daily_patterns(self, time_features):
        """Analyze patterns in violation distribution by day of month."""
        patterns = []
        
        # Get day counts
        days = time_features["days"]
        total_violations = sum(days.values())
        
        if total_violations < self.min_violations_for_pattern:
            return []
        
        # Calculate typical month length for average
        avg_days_in_month = 30.44  # Average days in a month (365.25/12)
        avg_daily = total_violations / avg_days_in_month
        threshold = max(avg_daily * 1.5, 3)  # At least 50% above average or minimum count
        
        peak_days = []
        for day, count in days.items():
            if count >= threshold:
                peak_days.append((day, count))
        
        # Create patterns for beginning, middle, and end of month
        day_groups = {
            "beginning": [d for d in range(1, 11)],
            "middle": [d for d in range(11, 21)],
            "end": [d for d in range(21, 32)]
        }
        
        for group_name, group_days in day_groups.items():
            group_count = sum(days.get(day, 0) for day in group_days)
            
            if group_count >= threshold:
                group_proportion = group_count / total_violations
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    group_proportion,
                    True,  # Time patterns are always considered current
                    threshold=self.thresholds["time"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "time_daily",
                        "time_period": f"{group_name} of month",
                        "violation_count": group_count,
                        "proportion": group_proportion,
                        "confidence": confidence,
                        "description": f"Higher violation frequency during the {group_name} of the month"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_weekly_patterns(self, time_features):
        """Analyze patterns in violation distribution by day of week."""
        patterns = []
        
        # Get weekday counts
        weekdays = time_features["weekdays"]
        total_violations = sum(weekdays.values())
        
        if total_violations < self.min_violations_for_pattern:
            return []
        
        # Calculate average violations per day
        avg_daily = total_violations / 7  # 7 days per week
        threshold = max(avg_daily * 1.5, 3)  # At least 50% above average or minimum count
        
        # Day name mapping
        day_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        
        # Check for weekday vs weekend patterns
        weekday_count = sum(weekdays.get(d, 0) for d in range(0, 5))  # Monday-Friday
        weekend_count = sum(weekdays.get(d, 0) for d in range(5, 7))  # Saturday-Sunday
        
        weekday_avg = weekday_count / 5
        weekend_avg = weekend_count / 2
        
        # Check if weekdays have significantly more violations than weekends
        if weekday_avg > weekend_avg * 1.5 and weekday_count >= threshold:
            weekday_proportion = weekday_count / total_violations
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                weekday_proportion,
                True,  # Time patterns are always considered current
                threshold=self.thresholds["time"]
            )
            
            if confidence >= self.min_pattern_confidence:
                pattern = {
                    "pattern_type": "time_weekly",
                    "time_period": "weekdays",
                    "violation_count": weekday_count,
                    "proportion": weekday_proportion,
                    "confidence": confidence,
                    "description": "Higher violation frequency on weekdays (Monday-Friday)"
                }
                patterns.append(pattern)
        
        # Check if weekends have significantly more violations than weekdays
        elif weekend_avg > weekday_avg * 1.5 and weekend_count >= threshold:
            weekend_proportion = weekend_count / total_violations
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                weekend_proportion,
                True,  # Time patterns are always considered current
                threshold=self.thresholds["time"]
            )
            
            if confidence >= self.min_pattern_confidence:
                pattern = {
                    "pattern_type": "time_weekly",
                    "time_period": "weekends",
                    "violation_count": weekend_count,
                    "proportion": weekend_proportion,
                    "confidence": confidence,
                    "description": "Higher violation frequency on weekends (Saturday-Sunday)"
                }
                patterns.append(pattern)
        
        # Check for specific days with high violation counts
        for day, count in weekdays.items():
            if count >= threshold:
                day_proportion = count / total_violations
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    day_proportion,
                    True,  # Time patterns are always considered current
                    threshold=self.thresholds["time"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "time_weekly",
                        "time_period": day_names[day],
                        "violation_count": count,
                        "proportion": day_proportion,
                        "confidence": confidence,
                        "description": f"Higher violation frequency on {day_names[day]}"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_monthly_patterns(self, time_features):
        """Analyze patterns in violation distribution by month."""
        patterns = []
        
        # Get month counts
        months = time_features["months"]
        total_violations = sum(months.values())
        
        if total_violations < self.min_violations_for_pattern:
            return []
        
        # Calculate average violations per month
        avg_monthly = total_violations / 12  # 12 months per year
        threshold = max(avg_monthly * 1.5, 3)  # At least 50% above average or minimum count
        
        # Month name mapping
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        
        # Check for quarterly patterns
        quarters = {
            "Q1": [1, 2, 3],
            "Q2": [4, 5, 6],
            "Q3": [7, 8, 9],
            "Q4": [10, 11, 12]
        }
        
        for quarter, quarter_months in quarters.items():
            quarter_count = sum(months.get(m, 0) for m in quarter_months)
            
            if quarter_count >= threshold * 3:  # 3 months per quarter
                quarter_proportion = quarter_count / total_violations
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    quarter_proportion,
                    True,  # Time patterns are always considered current
                    threshold=self.thresholds["time"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "time_monthly",
                        "time_period": quarter,
                        "violation_count": quarter_count,
                        "proportion": quarter_proportion,
                        "confidence": confidence,
                        "description": f"Higher violation frequency during {quarter}"
                    }
                    patterns.append(pattern)
        
        # Check for specific months with high violation counts
        for month, count in months.items():
            if count >= threshold:
                month_proportion = count / total_violations
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    month_proportion,
                    True,  # Time patterns are always considered current
                    threshold=self.thresholds["time"]
                )
                
                if confidence >= self.min_pattern_confidence:
                    pattern = {
                        "pattern_type": "time_monthly",
                        "time_period": month_names[month-1],  # Month index is 1-based
                        "violation_count": count,
                        "proportion": month_proportion,
                        "confidence": confidence,
                        "description": f"Higher violation frequency in {month_names[month-1]}"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_frequency_changes(self, time_features):
        """Analyze changes in violation frequency over time."""
        patterns = []
        
        # Get timestamps sorted by time
        timestamps = sorted(time_features.get("timestamps", []))
        
        if len(timestamps) < self.min_violations_for_pattern:
            return []
        
        # Divide the timestamps into time bins
        now = datetime.datetime.now()
        
        # Create time bins for analysis (last day, week, month, quarter, year)
        time_bins = {
            "last_day": now - datetime.timedelta(days=1),
            "last_week": now - datetime.timedelta(days=7),
            "last_month": now - datetime.timedelta(days=30),
            "last_quarter": now - datetime.timedelta(days=90),
            "last_year": now - datetime.timedelta(days=365),
        }
        
        # Count violations in each bin
        bin_counts = {}
        for bin_name, bin_start in time_bins.items():
            bin_counts[bin_name] = sum(1 for ts in timestamps if ts >= bin_start)
        
        # Calculate rates per day for each bin
        bin_rates = {}
        bin_rates["last_day"] = bin_counts["last_day"] / 1
        bin_rates["last_week"] = bin_counts["last_week"] / 7
        bin_rates["last_month"] = bin_counts["last_month"] / 30
        bin_rates["last_quarter"] = bin_counts["last_quarter"] / 90
        bin_rates["last_year"] = bin_counts["last_year"] / 365
        
        # Check for significant frequency changes between bins
        rate_changes = {}
        
        # Compare adjacent bins
        if bin_rates["last_day"] > bin_rates["last_week"] * 1.5:
            rate_changes["day_vs_week"] = {
                "ratio": bin_rates["last_day"] / max(bin_rates["last_week"], 0.001),
                "description": "Significant increase in violations in the last day compared to the weekly average",
                "confidence": min(0.5 + (bin_rates["last_day"] / max(bin_rates["last_week"], 0.001) - 1) * 0.1, 0.95)
            }
        
        if bin_rates["last_week"] > bin_rates["last_month"] * 1.5:
            rate_changes["week_vs_month"] = {
                "ratio": bin_rates["last_week"] / max(bin_rates["last_month"], 0.001),
                "description": "Significant increase in violations in the last week compared to the monthly average",
                "confidence": min(0.5 + (bin_rates["last_week"] / max(bin_rates["last_month"], 0.001) - 1) * 0.1, 0.95)
            }
        
        if bin_rates["last_month"] > bin_rates["last_quarter"] * 1.5:
            rate_changes["month_vs_quarter"] = {
                "ratio": bin_rates["last_month"] / max(bin_rates["last_quarter"], 0.001),
                "description": "Significant increase in violations in the last month compared to the quarterly average",
                "confidence": min(0.5 + (bin_rates["last_month"] / max(bin_rates["last_quarter"], 0.001) - 1) * 0.1, 0.95)
            }
        
        if bin_rates["last_quarter"] > bin_rates["last_year"] * 1.5:
            rate_changes["quarter_vs_year"] = {
                "ratio": bin_rates["last_quarter"] / max(bin_rates["last_year"], 0.001),
                "description": "Significant increase in violations in the last quarter compared to the yearly average",
                "confidence": min(0.5 + (bin_rates["last_quarter"] / max(bin_rates["last_year"], 0.001) - 1) * 0.1, 0.95)
            }
        
        # Check for decreases as well
        if bin_rates["last_day"] < bin_rates["last_week"] * 0.5:
            rate_changes["day_vs_week_decrease"] = {
                "ratio": max(bin_rates["last_week"], 0.001) / max(bin_rates["last_day"], 0.001),
                "description": "Significant decrease in violations in the last day compared to the weekly average",
                "confidence": min(0.5 + (max(bin_rates["last_week"], 0.001) / max(bin_rates["last_day"], 0.001) - 1) * 0.1, 0.95)
            }
        
        # Create patterns from identified rate changes
        for change_key, change_data in rate_changes.items():
            if change_data["confidence"] >= self.min_pattern_confidence:
                pattern = {
                    "pattern_type": "time_frequency_change",
                    "change_type": change_key,
                    "ratio": change_data["ratio"],
                    "confidence": change_data["confidence"],
                    "description": change_data["description"]
                }
                patterns.append(pattern)
        
        return patterns