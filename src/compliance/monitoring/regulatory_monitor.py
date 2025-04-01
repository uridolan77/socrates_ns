import requests
import feedparser
import json
import os
import re
import logging
import hashlib
import time
import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import sqlite3
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urlparse, urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("regulatory.monitor")

@dataclass
class RegulatorySource:
    """Source for regulatory updates"""
    id: str
    name: str
    url: str
    type: str  # 'rss', 'web', 'api'
    country: str = ""
    region: str = ""
    frameworks: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    auth: Dict[str, str] = field(default_factory=dict)
    parser_config: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class RegulatoryUpdate:
    """A regulatory update from a source"""
    id: str
    source_id: str
    title: str
    summary: str
    url: str
    published_date: str
    frameworks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    content: str = ""
    impact_level: str = "unknown"  # 'high', 'medium', 'low', 'unknown'
    processed: bool = False
    hash: str = ""


class RegulatoryMonitor:
    """
    System for monitoring regulatory updates from various sources.
    Tracks changes to compliance frameworks and alerts on important updates.
    """
    
    def __init__(self, 
                db_path: str,
                config_path: str = None,
                user_agent: str = "RegulatoryMonitor/1.0"):
        """
        Initialize the regulatory monitor
        
        Args:
            db_path: Path to SQLite database
            config_path: Path to configuration file
            user_agent: User-Agent header for requests
        """
        self.db_path = db_path
        self.user_agent = user_agent
        self.sources = []
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            
        # Initialize database
        self._init_database()
        
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load sources
            for source_config in config.get('sources', []):
                source = RegulatorySource(
                    id=source_config.get('id', f"source_{len(self.sources)}"),
                    name=source_config.get('name', ''),
                    url=source_config.get('url', ''),
                    type=source_config.get('type', 'web'),
                    country=source_config.get('country', ''),
                    region=source_config.get('region', ''),
                    frameworks=source_config.get('frameworks', []),
                    keywords=source_config.get('keywords', []),
                    headers=source_config.get('headers', {}),
                    auth=source_config.get('auth', {}),
                    parser_config=source_config.get('parser_config', {}),
                    active=source_config.get('active', True)
                )
                
                self.sources.append(source)
                
            logger.info(f"Loaded {len(self.sources)} sources from configuration")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sources table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    url TEXT,
                    type TEXT,
                    country TEXT,
                    region TEXT,
                    frameworks TEXT,
                    keywords TEXT,
                    headers TEXT,
                    auth TEXT,
                    parser_config TEXT,
                    active INTEGER,
                    last_checked TEXT
                )
                ''')
                
                # Create updates table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS updates (
                    id TEXT PRIMARY KEY,
                    source_id TEXT,
                    title TEXT,
                    summary TEXT,
                    url TEXT,
                    published_date TEXT,
                    frameworks TEXT,
                    tags TEXT,
                    content TEXT,
                    impact_level TEXT,
                    processed INTEGER,
                    hash TEXT,
                    found_date TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources (id)
                )
                ''')
                
                # Create alerts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    update_id TEXT,
                    alert_type TEXT,
                    sent_date TEXT,
                    recipients TEXT,
                    message TEXT,
                    FOREIGN KEY (update_id) REFERENCES updates (id)
                )
                ''')
                
                conn.commit()
                
            # Save sources to database
            self._save_sources()
            
            logger.info("Database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def _save_sources(self) -> None:
        """Save sources to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for source in self.sources:
                    cursor.execute('''
                    INSERT OR REPLACE INTO sources (
                        id, name, url, type, country, region, frameworks, keywords,
                        headers, auth, parser_config, active, last_checked
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        source.id,
                        source.name,
                        source.url,
                        source.type,
                        source.country,
                        source.region,
                        json.dumps(source.frameworks),
                        json.dumps(source.keywords),
                        json.dumps(source.headers),
                        json.dumps(source.auth),
                        json.dumps(source.parser_config),
                        1 if source.active else 0,
                        None
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving sources: {str(e)}")
    
    def add_source(self, source: RegulatorySource) -> bool:
        """
        Add a new source for monitoring
        
        Args:
            source: Source to add
            
        Returns:
            True if added successfully
        """
        try:
            # Check if source already exists
            if any(s.id == source.id for s in self.sources):
                logger.warning(f"Source with ID {source.id} already exists")
                return False
                
            # Add to sources list
            self.sources.append(source)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO sources (
                    id, name, url, type, country, region, frameworks, keywords,
                    headers, auth, parser_config, active, last_checked
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    source.id,
                    source.name,
                    source.url,
                    source.type,
                    source.country,
                    source.region,
                    json.dumps(source.frameworks),
                    json.dumps(source.keywords),
                    json.dumps(source.headers),
                    json.dumps(source.auth),
                    json.dumps(source.parser_config),
                    1 if source.active else 0,
                    None
                ))
                
                conn.commit()
                
            logger.info(f"Added source: {source.name} ({source.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding source: {str(e)}")
            return False
    
    def load_sources_from_database(self) -> None:
        """Load sources from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM sources')
                rows = cursor.fetchall()
                
                self.sources = []
                for row in rows:
                    source = RegulatorySource(
                        id=row['id'],
                        name=row['name'],
                        url=row['url'],
                        type=row['type'],
                        country=row['country'],
                        region=row['region'],
                        frameworks=json.loads(row['frameworks']),
                        keywords=json.loads(row['keywords']),
                        headers=json.loads(row['headers']),
                        auth=json.loads(row['auth']),
                        parser_config=json.loads(row['parser_config']),
                        active=bool(row['active'])
                    )
                    
                    self.sources.append(source)
                    
            logger.info(f"Loaded {len(self.sources)} sources from database")
            
        except Exception as e:
            logger.error(f"Error loading sources from database: {str(e)}")
    
    def check_for_updates(self, parallel: bool = True) -> List[RegulatoryUpdate]:
        """
        Check all active sources for updates
        
        Args:
            parallel: Whether to check sources in parallel
            
        Returns:
            List of new updates found
        """
        # Filter active sources
        active_sources = [s for s in self.sources if s.active]
        if not active_sources:
            logger.warning("No active sources to check")
            return []
            
        logger.info(f"Checking {len(active_sources)} sources for updates")
        
        # Check sources
        new_updates = []
        
        if parallel:
            # Check in parallel with thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_source = {
                    executor.submit(self._check_source, source): source
                    for source in active_sources
                }
                
                for future in concurrent.futures.as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        updates = future.result()
                        if updates:
                            new_updates.extend(updates)
                    except Exception as e:
                        logger.error(f"Error checking source {source.id}: {str(e)}")
        else:
            # Check sequentially
            for source in active_sources:
                try:
                    updates = self._check_source(source)
                    if updates:
                        new_updates.extend(updates)
                except Exception as e:
                    logger.error(f"Error checking source {source.id}: {str(e)}")
        
        # Process and store new updates
        if new_updates:
            logger.info(f"Found {len(new_updates)} new updates")
            self._process_updates(new_updates)
            
        return new_updates
    
    def _check_source(self, source: RegulatorySource) -> List[RegulatoryUpdate]:
        """
        Check a source for updates
        
        Args:
            source: Source to check
            
        Returns:
            List of new updates
        """
        logger.debug(f"Checking source: {source.name} ({source.id})")
        
        try:
            updates = []
            
            # Check based on source type
            if source.type == 'rss':
                updates = self._check_rss_source(source)
            elif source.type == 'web':
                updates = self._check_web_source(source)
            elif source.type == 'api':
                updates = self._check_api_source(source)
            else:
                logger.warning(f"Unknown source type: {source.type}")
                
            # Update last checked timestamp
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE sources SET last_checked = ? WHERE id = ?',
                    (datetime.datetime.now().isoformat(), source.id)
                )
                conn.commit()
                
            return updates
            
        except Exception as e:
            logger.error(f"Error checking source {source.id}: {str(e)}")
            return []
    
    def _check_rss_source(self, source: RegulatorySource) -> List[RegulatoryUpdate]:
        """
        Check an RSS source for updates
        
        Args:
            source: RSS source to check
            
        Returns:
            List of new updates
        """
        # Setup headers
        headers = {
            'User-Agent': self.user_agent
        }
        headers.update(source.headers)
        
        # Parse feed
        feed = feedparser.parse(source.url, request_headers=headers)
        
        # Process entries
        updates = []
        for entry in feed.entries:
            # Generate unique ID
            entry_id = entry.get('id', entry.get('link', ''))
            if not entry_id:
                continue
                
            # Check if already exists in database
            if self._update_exists(entry_id):
                continue
                
            # Create update
            published = entry.get('published', entry.get('updated', ''))
            if isinstance(published, time.struct_time):
                published = time.strftime('%Y-%m-%d %H:%M:%S', published)
                
            update = RegulatoryUpdate(
                id=entry_id,
                source_id=source.id,
                title=entry.get('title', ''),
                summary=entry.get('summary', ''),
                url=entry.get('link', ''),
                published_date=published,
                frameworks=source.frameworks,
                content=entry.get('content', ''),
                hash=self._generate_content_hash(entry.get('title', '') + entry.get('summary', ''))
            )
            
            # Set tags based on keywords
            update.tags = self._extract_tags(update, source.keywords)
            
            updates.append(update)
            
        return updates
    
    def _check_web_source(self, source: RegulatorySource) -> List[RegulatoryUpdate]:
        """
        Check a web source for updates
        
        Args:
            source: Web source to check
            
        Returns:
            List of new updates
        """
        # Setup headers
        headers = {
            'User-Agent': self.user_agent
        }
        headers.update(source.headers)
        
        # Make request
        response = requests.get(source.url, headers=headers, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {source.url}: HTTP {response.status_code}")
            return []
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract updates based on parser configuration
        updates = []
        
        # Get container element if specified
        container_selector = source.parser_config.get('container_selector', 'body')
        container = soup.select_one(container_selector) or soup
        
        # Get update elements
        update_selector = source.parser_config.get('update_selector', 'article, .news-item, .update-item')
        update_elements = container.select(update_selector)
        
        for element in update_elements:
            # Extract update details using selectors from config
            config = source.parser_config
            
            # Get title
            title_selector = config.get('title_selector')
            title_element = element.select_one(title_selector) if title_selector else None
            title = title_element.get_text(strip=True) if title_element else element.get('title', '')
            
            # Get URL
            url_selector = config.get('url_selector')
            url_element = element.select_one(url_selector) if url_selector else None
            url = url_element.get('href') if url_element else element.get('href', '')
            
            # Normalize URL
            if url and not url.startswith(('http://', 'https://')):
                url = urljoin(source.url, url)
                
            # Skip if no URL
            if not url:
                continue
                
            # Get summary
            summary_selector = config.get('summary_selector')
            summary_element = element.select_one(summary_selector) if summary_selector else None
            summary = summary_element.get_text(strip=True) if summary_element else ''
            
            # Get date
            date_selector = config.get('date_selector')
            date_element = element.select_one(date_selector) if date_selector else None
            published_date = date_element.get_text(strip=True) if date_element else ''
            
            # Try to parse date if found
            if published_date:
                date_format = config.get('date_format')
                if date_format:
                    try:
                        parsed_date = datetime.datetime.strptime(published_date, date_format)
                        published_date = parsed_date.isoformat()
                    except:
                        pass
            else:
                published_date = datetime.datetime.now().isoformat()
                
            # Generate ID from URL
            update_id = hashlib.md5(url.encode()).hexdigest()
            
            # Check if already exists
            if self._update_exists(update_id):
                continue
                
            # Create update
            update = RegulatoryUpdate(
                id=update_id,
                source_id=source.id,
                title=title,
                summary=summary,
                url=url,
                published_date=published_date,
                frameworks=source.frameworks,
                hash=self._generate_content_hash(title + summary)
            )
            
            # Set tags based on keywords
            update.tags = self._extract_tags(update, source.keywords)
            
            updates.append(update)
            
        return updates
    
    def _check_api_source(self, source: RegulatorySource) -> List[RegulatoryUpdate]:
        """
        Check an API source for updates
        
        Args:
            source: API source to check
            
        Returns:
            List of new updates
        """
        # Setup headers
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json'
        }
        headers.update(source.headers)
        
        # Setup auth if provided
        auth = None
        if 'username' in source.auth and 'password' in source.auth:
            auth = (source.auth['username'], source.auth['password'])
            
        # Make request
        response = requests.get(
            source.url, 
            headers=headers, 
            auth=auth, 
            timeout=30
        )
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {source.url}: HTTP {response.status_code}")
            return []
            
        # Parse JSON
        try:
            data = response.json()
        except:
            logger.error(f"Failed to parse JSON from {source.url}")
            return []
            
        # Extract updates based on parser configuration
        updates = []
        config = source.parser_config
        
        # Get items path
        items_path = config.get('items_path', '')
        items = data
        
        # Traverse path to items
        if items_path:
            for key in items_path.split('.'):
                if key.isdigit():
                    key = int(key)
                if isinstance(items, (list, tuple)) and isinstance(key, int) and 0 <= key < len(items):
                    items = items[key]
                elif isinstance(items, dict) and key in items:
                    items = items[key]
                else:
                    logger.warning(f"Invalid items path: {items_path}")
                    return []
                    
        # Ensure items is a list
        if not isinstance(items, list):
            items = [items]
            
        # Process items
        for item in items:
            # Get fields based on config
            title = self._get_nested_value(item, config.get('title_field', 'title'))
            url = self._get_nested_value(item, config.get('url_field', 'url'))
            summary = self._get_nested_value(item, config.get('summary_field', 'summary'))
            published_date = self._get_nested_value(item, config.get('date_field', 'published_date'))
            
            # Skip if missing required fields
            if not title or not url:
                continue
                
            # Generate ID from URL
            update_id = hashlib.md5(url.encode()).hexdigest()
            
            # Check if already exists
            if self._update_exists(update_id):
                continue
                
            # Create update
            update = RegulatoryUpdate(
                id=update_id,
                source_id=source.id,
                title=title,
                summary=summary,
                url=url,
                published_date=published_date,
                frameworks=source.frameworks,
                hash=self._generate_content_hash(title + summary)
            )
            
            # Set tags based on keywords
            update.tags = self._extract_tags(update, source.keywords)
            
            updates.append(update)
            
        return updates
    
    def _process_updates(self, updates: List[RegulatoryUpdate]) -> None:
        """
        Process and store new updates
        
        Args:
            updates: List of updates to process
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for update in updates:
                    # Assess impact level
                    update.impact_level = self._assess_impact_level(update)
                    
                    # Store update
                    cursor.execute('''
                    INSERT OR IGNORE INTO updates (
                        id, source_id, title, summary, url, published_date, frameworks,
                        tags, content, impact_level, processed, hash, found_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        update.id,
                        update.source_id,
                        update.title,
                        update.summary,
                        update.url,
                        update.published_date,
                        json.dumps(update.frameworks),
                        json.dumps(update.tags),
                        update.content,
                        update.impact_level,
                        1 if update.processed else 0,
                        update.hash,
                        datetime.datetime.now().isoformat()
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error processing updates: {str(e)}")
    
    def _update_exists(self, update_id: str) -> bool:
        """
        Check if an update already exists in the database
        
        Args:
            update_id: Update ID to check
            
        Returns:
            True if update exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM updates WHERE id = ?', (update_id,))
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Error checking if update exists: {str(e)}")
            return False
    
    def _generate_content_hash(self, content: str) -> str:
        """
        Generate a hash of content for change detection
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash
        """
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_tags(self, update: RegulatoryUpdate, keywords: List[str]) -> List[str]:
        """
        Extract tags from update based on keywords
        
        Args:
            update: Update to extract tags from
            keywords: Keywords to look for
            
        Returns:
            List of tags
        """
        tags = []
        content = (update.title + " " + update.summary).lower()
        
        for keyword in keywords:
            if keyword.lower() in content:
                tags.append(keyword)
                
        return tags
    
    def _assess_impact_level(self, update: RegulatoryUpdate) -> str:
        """
        Assess the impact level of an update
        
        Args:
            update: Update to assess
            
        Returns:
            Impact level ('high', 'medium', 'low', 'unknown')
        """
        # Look for high-impact indicators in title and summary
        content = (update.title + " " + update.summary).lower()
        
        # High impact indicators
        high_impact_terms = [
            'urgent', 'critical', 'mandatory', 'required', 'deadline',
            'compliance required', 'immediate action', 'significant change',
            'major update', 'breaking change', 'new requirement'
        ]
        
        # Medium impact indicators
        medium_impact_terms = [
            'update', 'change', 'revision', 'amendment', 'modification',
            'guidance', 'recommendation', 'advisory', 'clarification'
        ]
        
        # Check for high impact
        for term in high_impact_terms:
            if term in content:
                return 'high'
                
        # Check for medium impact
        for term in medium_impact_terms:
            if term in content:
                return 'medium'
                
        # Default to low
        return 'low'
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        Get a nested value from a dictionary
        
        Args:
            data: Dictionary to extract from
            field_path: Path to field (dot notation)
            
        Returns:
            Field value or empty string if not found
        """
        if not field_path:
            return ''
            
        parts = field_path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, (list, tuple)) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return ''
            else:
                return ''
                
        return current if current is not None else ''
    
    def get_recent_updates(self, 
                         days: int = 30, 
                         frameworks: List[str] = None,
                         impact_level: str = None) -> List[Dict[str, Any]]:
        """
        Get recent regulatory updates
        
        Args:
            days: Number of days to look back
            frameworks: Filter by frameworks
            impact_level: Filter by impact level
            
        Returns:
            List of updates
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build query
                query = '''
                SELECT u.*, s.name as source_name
                FROM updates u
                JOIN sources s ON u.source_id = s.id
                WHERE u.found_date >= date('now', ?)
                '''
                params = [f'-{days} days']
                
                # Add framework filter
                if frameworks:
                    placeholders = ','.join('?' for _ in frameworks)
                    query += f" AND EXISTS (SELECT 1 FROM json_each(u.frameworks) WHERE json_each.value IN ({placeholders}))"
                    params.extend(frameworks)
                    
                # Add impact level filter
                if impact_level:
                    query += " AND u.impact_level = ?"
                    params.append(impact_level)
                    
                # Order by date
                query += " ORDER BY u.published_date DESC"
                
                # Execute query
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                updates = []
                for row in rows:
                    update = dict(row)
                    
                    # Parse JSON fields
                    for field in ['frameworks', 'tags']:
                        if update[field]:
                            update[field] = json.loads(update[field])
                        else:
                            update[field] = []
                            
                    updates.append(update)
                    
                return updates
                
        except Exception as e:
            logger.error(f"Error getting recent updates: {str(e)}")
            return []
    
    def fetch_full_content(self, update_id: str) -> bool:
        """
        Fetch the full content of an update
        
        Args:
            update_id: ID of update to fetch content for
            
        Returns:
            True if successful
        """
        try:
            # Get update details
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM updates WHERE id = ?', (update_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Update not found: {update_id}")
                    return False
                    
                update = dict(row)
                url = update['url']
                
                # Skip if already has content
                if update['content']:
                    return True
                    
                # Make request
                headers = {'User-Agent': self.user_agent}
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch content for {update_id}: HTTP {response.status_code}")
                    return False
                    
                # Extract content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find main content
                content_element = None
                
                # Common content selectors
                content_selectors = [
                    'article', 'main', '.content', '#content', '.post-content',
                    '.entry-content', '.article-content', '.main-content'
                ]
                
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content_element = element
                        break
                        
                # Fallback to body
                if not content_element:
                    content_element = soup.body
                    
                # Extract text
                content = content_element.get_text(separator='\n', strip=True)
                
                # Update database
                cursor.execute(
                    'UPDATE updates SET content = ? WHERE id = ?',
                    (content, update_id)
                )
                
                conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Error fetching full content for {update_id}: {str(e)}")
            return False
    
    def generate_impact_analysis(self, 
                               update_id: str, 
                               framework_rules: Dict[str, List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate analysis of impact on compliance framework
        
        Args:
            update_id: ID of update to analyze
            framework_rules: Dictionary of framework rules for analysis
            
        Returns:
            Impact analysis
        """
        try:
            # Get update details
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM updates WHERE id = ?', (update_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Update not found: {update_id}")
                    return {}
                    
                update = dict(row)
                
                # Parse JSON fields
                for field in ['frameworks', 'tags']:
                    if update[field]:
                        update[field] = json.loads(update[field])
                    else:
                        update[field] = []
                        
                # Ensure we have content
                if not update['content']:
                    if not self.fetch_full_content(update_id):
                        # Use summary if content fetch fails
                        update['content'] = update['summary']
                        
                # Simple keyword-based impact analysis
                analysis = {
                    'update_id': update_id,
                    'frameworks': update['frameworks'],
                    'impact_level': update['impact_level'],
                    'affected_rules': [],
                    'requires_action': update['impact_level'] in ['high', 'medium'],
                    'summary': f"This update may affect compliance with {', '.join(update['frameworks'])}."
                }
                
                # Analyze against framework rules if provided
                if framework_rules:
                    affected_rules = []
                    
                    for framework in update['frameworks']:
                        if framework in framework_rules:
                            rules = framework_rules[framework]
                            
                            for rule in rules:
                                rule_id = rule.get('id', '')
                                rule_desc = rule.get('description', '')
                                
                                # Look for keywords in content
                                keywords = rule.get('keywords', [])
                                for keyword in keywords:
                                    if keyword.lower() in update['content'].lower():
                                        affected_rules.append({
                                            'framework': framework,
                                            'rule_id': rule_id,
                                            'description': rule_desc,
                                            'matched_keyword': keyword
                                        })
                                        break
                                        
                    analysis['affected_rules'] = affected_rules
                    analysis['requires_action'] = len(affected_rules) > 0 or analysis['requires_action']
                    
                    if affected_rules:
                        analysis['summary'] = f"This update affects {len(affected_rules)} compliance rules across {len(set(r['framework'] for r in affected_rules))} frameworks."
                        
                return analysis
                
        except Exception as e:
            logger.error(f"Error generating impact analysis for {update_id}: {str(e)}")
            return {}
    
    def create_alert(self, 
                   update_id: str, 
                   alert_type: str = 'email',
                   recipients: List[str] = None,
                   message: str = None) -> bool:
        """
        Create an alert for a regulatory update
        
        Args:
            update_id: ID of update to alert for
            alert_type: Type of alert ('email', 'slack', 'webhook')
            recipients: List of recipients
            message: Alert message
            
        Returns:
            True if alert created successfully
        """
        try:
            # Get update details
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM updates WHERE id = ?', (update_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Update not found: {update_id}")
                    return False
                    
                update = dict(row)
                
                # Generate message if not provided
                if not message:
                    message = (
                        f"Regulatory Update Alert\n\n"
                        f"Title: {update['title']}\n"
                        f"Impact: {update['impact_level']}\n"
                        f"URL: {update['url']}\n\n"
                        f"Summary: {update['summary']}\n"
                    )
                    
                # Create alert record
                alert_id = hashlib.md5(f"{update_id}:{int(time.time())}".encode()).hexdigest()
                
                cursor.execute('''
                INSERT INTO alerts (id, update_id, alert_type, sent_date, recipients, message)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id,
                    update_id,
                    alert_type,
                    datetime.datetime.now().isoformat(),
                    json.dumps(recipients or []),
                    message
                ))
                
                conn.commit()
                
                logger.info(f"Created alert {alert_id} for update {update_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating alert for {update_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about regulatory monitoring
        
        Returns:
            Dictionary of statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {
                    'sources': {},
                    'updates': {},
                    'alerts': {}
                }
                
                # Source stats
                cursor.execute('SELECT COUNT(*) FROM sources')
                stats['sources']['total'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM sources WHERE active = 1')
                stats['sources']['active'] = cursor.fetchone()[0]
                
                # Update stats
                cursor.execute('SELECT COUNT(*) FROM updates')
                stats['updates']['total'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM updates WHERE found_date >= date("now", "-30 days")')
                stats['updates']['last_30_days'] = cursor.fetchone()[0]
                
                # Impact level breakdown
                cursor.execute('''
                SELECT impact_level, COUNT(*) 
                FROM updates 
                GROUP BY impact_level
                ''')
                stats['updates']['by_impact'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Framework breakdown
                cursor.execute('SELECT frameworks FROM updates')
                framework_counts = {}
                for row in cursor.fetchall():
                    if row[0]:
                        frameworks = json.loads(row[0])
                        for framework in frameworks:
                            if framework not in framework_counts:
                                framework_counts[framework] = 0
                            framework_counts[framework] += 1
                            
                stats['updates']['by_framework'] = framework_counts
                
                # Alert stats
                cursor.execute('SELECT COUNT(*) FROM alerts')
                stats['alerts']['total'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM alerts WHERE sent_date >= date("now", "-30 days")')
                stats['alerts']['last_30_days'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}


def create_sample_sources():
    """Create sample regulatory sources for monitoring"""
    return [
        RegulatorySource(
            id="gdpr_eu",
            name="EU GDPR News",
            url="https://edpb.europa.eu/news-events/news_en",
            type="web",
            country="EU",
            region="Europe",
            frameworks=["GDPR"],
            keywords=["data protection", "privacy", "personal data", "consent", "GDPR"],
            parser_config={
                "container_selector": ".view-content",
                "update_selector": ".views-row",
                "title_selector": "h3 a",
                "url_selector": "h3 a",
                "summary_selector": ".field-name-field-summary",
                "date_selector": ".post-date"
            }
        ),
        RegulatorySource(
            id="hipaa_hhs",
            name="HHS HIPAA Updates",
            url="https://www.hhs.gov/hipaa/newsroom/index.html",
            type="web",
            country="US",
            region="North America",
            frameworks=["HIPAA"],
            keywords=["health information", "PHI", "patient", "healthcare", "HIPAA"],
            parser_config={
                "container_selector": ".field-items",
                "update_selector": ".hipaa-feed-item",
                "title_selector": "h3 a",
                "url_selector": "h3 a",
                "summary_selector": "p",
                "date_selector": ".date-display-single"
            }
        ),
        RegulatorySource(
            id="ccpa_ca",
            name="California Attorney General CCPA Updates",
            url="https://oag.ca.gov/privacy/ccpa",
            type="web",
            country="US",
            region="North America",
            frameworks=["CCPA"],
            keywords=["consumer privacy", "personal information", "California", "CCPA"],
            parser_config={
                "container_selector": ".article-body",
                "update_selector": "h2, h3",
                "title_selector": "self",
                "url_selector": "a",
                "summary_selector": "p"
            }
        )
    ]

# Example usage:
"""
# Initialize monitor
monitor = RegulatoryMonitor(db_path="regulatory_updates.db")

# Add sources
for source in create_sample_sources():
    monitor.add_source(source)

# Check for updates
new_updates = monitor.check_for_updates()

# Get recent updates
recent_updates = monitor.get_recent_updates(days=30, frameworks=["GDPR"])

# Create alert for high-impact updates
for update in recent_updates:
    if update['impact_level'] == 'high':
        monitor.create_alert(
            update['id'],
            alert_type='email',
            recipients=['compliance@example.com'],
            message=f"Critical update: {update['title']}"
        )

# Get statistics
stats = monitor.get_stats()
print(json.dumps(stats, indent=2))
"""