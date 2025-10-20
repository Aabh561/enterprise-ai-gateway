"""
Advanced Security Service for Enterprise AI Gateway

Provides comprehensive security including PII protection, content filtering,
authentication, authorization, and threat detection.
"""

import asyncio
import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Pattern
import uuid

import structlog
from prometheus_client import Counter, Histogram

# PII Protection imports
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False

logger = structlog.get_logger(__name__)

# Metrics
SECURITY_EVENTS_COUNTER = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity', 'action']
)

PII_DETECTION_COUNTER = Counter(
    'pii_detections_total',
    'Total PII detections',
    ['entity_type', 'action']
)

CONTENT_FILTER_COUNTER = Counter(
    'content_filter_total',
    'Total content filtering events',
    ['filter_type', 'action']
)

SECURITY_LATENCY_HISTOGRAM = Histogram(
    'security_check_duration_seconds',
    'Security check duration',
    ['check_type']
)


class SecurityEventType(str, Enum):
    """Types of security events."""
    PII_DETECTED = "pii_detected"
    CONTENT_BLOCKED = "content_blocked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    INJECTION_ATTEMPT = "injection_attempt"


class SecuritySeverity(str, Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PIIEntityType(str, Enum):
    """Types of PII entities."""
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    SSN = "US_SSN"
    CREDIT_CARD = "CREDIT_CARD"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    IBAN_CODE = "IBAN_CODE"
    US_PASSPORT = "US_PASSPORT"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    severity: SecuritySeverity
    timestamp: float
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[str] = None
    blocked: bool = False


@dataclass
class PIIDetectionResult:
    """PII detection result."""
    found_pii: bool
    entities: List[Dict[str, Any]] = field(default_factory=list)
    anonymized_text: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContentFilterResult:
    """Content filtering result."""
    blocked: bool
    reasons: List[str] = field(default_factory=list)
    filtered_content: Optional[str] = None
    risk_score: float = 0.0


@dataclass
class SecurityConfig:
    """Security service configuration."""
    # PII Protection
    pii_enabled: bool = True
    pii_threshold: float = 0.8
    pii_entities: List[PIIEntityType] = field(default_factory=lambda: list(PIIEntityType))
    pii_anonymize: bool = True
    
    # Content Filtering
    content_filtering_enabled: bool = True
    max_input_length: int = 10000
    blocked_patterns: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    
    # Rate Limiting
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    
    # Threat Detection
    threat_detection_enabled: bool = True
    max_repeated_requests: int = 10
    suspicious_patterns: List[str] = field(default_factory=list)
    
    # Authentication
    require_authentication: bool = True
    token_expiry_minutes: int = 60
    
    # Logging and Monitoring
    log_security_events: bool = True
    alert_on_critical: bool = True
    
    # Custom settings
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


class PIIProtectionService:
    """PII Protection service using Microsoft Presidio."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.analyzer = None
        self.anonymizer = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize PII protection engines."""
        if not self.config.pii_enabled or self._initialized:
            return
        
        if not HAS_PRESIDIO:
            logger.warning("Presidio not available, PII protection disabled")
            return
        
        try:
            def init_sync():
                # Initialize Presidio engines
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                return True
            
            success = await asyncio.get_event_loop().run_in_executor(None, init_sync)
            
            if success:
                self._initialized = True
                logger.info("PII protection service initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize PII protection: {e}")
            raise
    
    async def detect_pii(self, text: str, language: str = "en") -> PIIDetectionResult:
        """Detect PII in text."""
        if not self._initialized or not text.strip():
            return PIIDetectionResult(found_pii=False)
        
        start_time = time.time()
        
        try:
            def analyze_sync():
                # Analyze text for PII
                results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=[e.value for e in self.config.pii_entities]
                )
                return results
            
            analysis_results = await asyncio.get_event_loop().run_in_executor(
                None, analyze_sync
            )
            
            # Filter by confidence threshold
            high_confidence_entities = [
                result for result in analysis_results
                if result.score >= self.config.pii_threshold
            ]
            
            found_pii = len(high_confidence_entities) > 0
            
            # Convert to our format
            entities = []
            confidence_scores = {}
            
            for result in high_confidence_entities:
                entity_dict = {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                }
                entities.append(entity_dict)
                confidence_scores[result.entity_type] = max(
                    confidence_scores.get(result.entity_type, 0.0),
                    result.score
                )
                
                # Record metrics
                PII_DETECTION_COUNTER.labels(
                    entity_type=result.entity_type,
                    action="detected"
                ).inc()
            
            # Anonymize if requested
            anonymized_text = None
            if found_pii and self.config.pii_anonymize:
                anonymized_text = await self._anonymize_text(text, analysis_results)
            
            SECURITY_LATENCY_HISTOGRAM.labels(
                check_type="pii_detection"
            ).observe(time.time() - start_time)
            
            return PIIDetectionResult(
                found_pii=found_pii,
                entities=entities,
                anonymized_text=anonymized_text,
                confidence_scores=confidence_scores
            )
        
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            return PIIDetectionResult(found_pii=False)
    
    async def _anonymize_text(self, text: str, analysis_results) -> str:
        """Anonymize detected PII in text."""
        try:
            def anonymize_sync():
                return self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analysis_results
                ).text
            
            return await asyncio.get_event_loop().run_in_executor(
                None, anonymize_sync
            )
        
        except Exception as e:
            logger.error(f"PII anonymization failed: {e}")
            return text
    
    async def sanitize_input(self, text: str) -> str:
        """Sanitize input text by removing/masking PII."""
        if not self.config.pii_enabled:
            return text
        
        result = await self.detect_pii(text)
        
        if result.found_pii:
            if result.anonymized_text:
                PII_DETECTION_COUNTER.labels(
                    entity_type="mixed",
                    action="anonymized"
                ).inc()
                return result.anonymized_text
            else:
                # Fallback: simple masking
                return await self._simple_pii_mask(text, result.entities)
        
        return text
    
    async def _simple_pii_mask(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Simple PII masking fallback."""
        masked_text = text
        
        # Sort entities by start position in reverse order
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        for entity in entities_sorted:
            start, end = entity["start"], entity["end"]
            entity_type = entity["entity_type"]
            
            # Create mask based on entity type
            if entity_type == PIIEntityType.EMAIL_ADDRESS.value:
                mask = "[EMAIL]"
            elif entity_type == PIIEntityType.PHONE_NUMBER.value:
                mask = "[PHONE]"
            elif entity_type == PIIEntityType.PERSON.value:
                mask = "[PERSON]"
            elif entity_type == PIIEntityType.LOCATION.value:
                mask = "[LOCATION]"
            else:
                mask = f"[{entity_type}]"
            
            # Replace the PII with mask
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text


class ContentFilter:
    """Advanced content filtering service."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._compiled_patterns: List[Pattern] = []
        self._compiled_suspicious: List[Pattern] = []
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for content filtering."""
        # Compile blocked patterns
        for pattern in self.config.blocked_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        # Compile suspicious patterns
        for pattern in self.config.suspicious_patterns:
            try:
                self._compiled_suspicious.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid suspicious pattern '{pattern}': {e}")
        
        # Default suspicious patterns
        default_suspicious = [
            r"<script[^>]*>",  # Script injection
            r"javascript:",     # JavaScript URLs
            r"eval\s*\(",      # Eval injection
            r"union\s+select",  # SQL injection
            r"drop\s+table",    # SQL injection
            r"['\"];\s*(drop|delete|insert|update)", # SQL injection
        ]
        
        for pattern in default_suspicious:
            try:
                self._compiled_suspicious.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
    
    async def filter_content(self, text: str) -> ContentFilterResult:
        """Filter content for blocked patterns and threats."""
        start_time = time.time()
        
        if not self.config.content_filtering_enabled:
            return ContentFilterResult(blocked=False)
        
        reasons = []
        risk_score = 0.0
        blocked = False
        
        # Check length
        if len(text) > self.config.max_input_length:
            reasons.append(f"Content too long ({len(text)} > {self.config.max_input_length})")
            blocked = True
            risk_score += 0.3
        
        # Check blocked patterns
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                reasons.append(f"Blocked pattern #{i} matched")
                blocked = True
                risk_score += 0.5
                
                CONTENT_FILTER_COUNTER.labels(
                    filter_type="blocked_pattern",
                    action="blocked"
                ).inc()
        
        # Check suspicious patterns
        for i, pattern in enumerate(self._compiled_suspicious):
            if pattern.search(text):
                reasons.append(f"Suspicious pattern #{i} detected")
                risk_score += 0.3
                
                CONTENT_FILTER_COUNTER.labels(
                    filter_type="suspicious_pattern",
                    action="detected"
                ).inc()
        
        # Check for potential injection attacks
        injection_score = await self._check_injection_patterns(text)
        if injection_score > 0.5:
            reasons.append("Potential injection attack detected")
            blocked = True
            risk_score += injection_score
        
        # Determine final action
        if risk_score > 0.7:
            blocked = True
        
        SECURITY_LATENCY_HISTOGRAM.labels(
            check_type="content_filter"
        ).observe(time.time() - start_time)
        
        return ContentFilterResult(
            blocked=blocked,
            reasons=reasons,
            filtered_content=text if not blocked else None,
            risk_score=min(risk_score, 1.0)
        )
    
    async def _check_injection_patterns(self, text: str) -> float:
        """Check for injection attack patterns."""
        injection_indicators = [
            "' OR '1'='1",
            "'; DROP TABLE",
            "<script>alert(",
            "javascript:void(0)",
            "onload=",
            "onerror=",
            "../../../etc/passwd",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://",  # Log4j
        ]
        
        score = 0.0
        for indicator in injection_indicators:
            if indicator.lower() in text.lower():
                score += 0.2
        
        return min(score, 1.0)


class ThreatDetector:
    """Advanced threat detection service."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history: Dict[str, List[float]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.suspicious_ips: Set[str] = set()
    
    async def analyze_request(
        self,
        user_id: Optional[str],
        source_ip: Optional[str],
        content: str,
        request_id: str
    ) -> List[SecurityEvent]:
        """Analyze request for threats."""
        if not self.config.threat_detection_enabled:
            return []
        
        events = []
        current_time = time.time()
        
        # Rate limiting check
        if user_id:
            user_requests = self.request_history.setdefault(user_id, [])
            
            # Clean old requests
            user_requests[:] = [
                req_time for req_time in user_requests
                if current_time - req_time < 3600  # Last hour
            ]
            user_requests.append(current_time)
            
            # Check rate limits
            recent_requests = [
                req_time for req_time in user_requests
                if current_time - req_time < 60  # Last minute
            ]
            
            if len(recent_requests) > self.config.requests_per_minute:
                events.append(SecurityEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    severity=SecuritySeverity.MEDIUM,
                    timestamp=current_time,
                    user_id=user_id,
                    request_id=request_id,
                    source_ip=source_ip,
                    details={"requests_per_minute": len(recent_requests)},
                    blocked=True
                ))
        
        # Suspicious activity detection
        if await self._is_suspicious_activity(content, user_id, source_ip):
            events.append(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=SecuritySeverity.HIGH,
                timestamp=current_time,
                user_id=user_id,
                request_id=request_id,
                source_ip=source_ip,
                details={"content_preview": content[:100]},
                blocked=False
            ))
        
        return events
    
    async def _is_suspicious_activity(
        self,
        content: str,
        user_id: Optional[str],
        source_ip: Optional[str]
    ) -> bool:
        """Detect suspicious activity patterns."""
        # Check for repeated identical requests
        if user_id:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            # Implementation would track content hashes per user
        
        # Check IP reputation
        if source_ip and source_ip in self.suspicious_ips:
            return True
        
        # Check content characteristics
        if len(content) < 10 and content.count(' ') == 0:
            return True  # Very short, no spaces - could be probe
        
        return False
    
    def record_failed_attempt(self, identifier: str) -> None:
        """Record a failed authentication/authorization attempt."""
        self.failed_attempts[identifier] = self.failed_attempts.get(identifier, 0) + 1
        
        if self.failed_attempts[identifier] > 5:
            self.suspicious_ips.add(identifier)
    
    def is_blocked(self, identifier: str) -> bool:
        """Check if an identifier should be blocked."""
        return (
            identifier in self.suspicious_ips or
            self.failed_attempts.get(identifier, 0) > 10
        )


class SecurityService:
    """Comprehensive security service."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pii_service = PIIProtectionService(config)
        self.content_filter = ContentFilter(config)
        self.threat_detector = ThreatDetector(config)
        self.security_events: List[SecurityEvent] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the security service."""
        if self._initialized:
            return
        
        await self.pii_service.initialize()
        self._initialized = True
        logger.info("Security service initialized")
    
    async def sanitize_input(self, text: str) -> str:
        """Sanitize input text for security."""
        # PII protection
        if self.config.pii_enabled:
            text = await self.pii_service.sanitize_input(text)
        
        # Content filtering
        filter_result = await self.content_filter.filter_content(text)
        
        if filter_result.blocked:
            # Log security event
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.CONTENT_BLOCKED,
                severity=SecuritySeverity.MEDIUM,
                timestamp=time.time(),
                details={"reasons": filter_result.reasons, "risk_score": filter_result.risk_score},
                blocked=True
            )
            await self._log_security_event(event)
            
            raise SecurityException(
                f"Content blocked: {', '.join(filter_result.reasons)}"
            )
        
        return text
    
    async def sanitize_output(self, text: str) -> str:
        """Sanitize output text for security."""
        # PII detection and anonymization for outputs
        if self.config.pii_enabled:
            result = await self.pii_service.detect_pii(text)
            if result.found_pii and result.anonymized_text:
                return result.anonymized_text
        
        return text
    
    async def check_request_security(
        self,
        content: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive security check for requests."""
        start_time = time.time()
        
        # Threat analysis
        threat_events = await self.threat_detector.analyze_request(
            user_id, source_ip, content, request_id or str(uuid.uuid4())
        )
        
        # Check if request should be blocked
        blocked_events = [event for event in threat_events if event.blocked]
        
        if blocked_events:
            # Log security events
            for event in threat_events:
                await self._log_security_event(event)
            
            return {
                "allowed": False,
                "reason": "Security threat detected",
                "events": [event.__dict__ for event in blocked_events]
            }
        
        # PII detection
        pii_result = await self.pii_service.detect_pii(content)
        
        # Content filtering
        content_result = await self.content_filter.filter_content(content)
        
        # Record non-blocking events
        for event in threat_events:
            await self._log_security_event(event)
        
        if pii_result.found_pii:
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.PII_DETECTED,
                severity=SecuritySeverity.MEDIUM,
                timestamp=time.time(),
                user_id=user_id,
                request_id=request_id,
                source_ip=source_ip,
                details={
                    "entities": pii_result.entities,
                    "confidence_scores": pii_result.confidence_scores
                }
            )
            await self._log_security_event(event)
        
        total_time = time.time() - start_time
        
        return {
            "allowed": not content_result.blocked,
            "pii_detected": pii_result.found_pii,
            "pii_entities": pii_result.entities,
            "content_risk_score": content_result.risk_score,
            "security_events": len(threat_events),
            "processing_time_ms": total_time * 1000
        }
    
    async def _log_security_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self.security_events.append(event)
        
        # Record metrics
        SECURITY_EVENTS_COUNTER.labels(
            event_type=event.event_type,
            severity=event.severity,
            action="logged"
        ).inc()
        
        # Log to structured logger
        logger.warning(
            "Security event",
            event_id=event.event_id,
            event_type=event.event_type,
            severity=event.severity,
            user_id=event.user_id,
            request_id=event.request_id,
            source_ip=event.source_ip,
            blocked=event.blocked,
            details=event.details
        )
        
        # Alert on critical events
        if (event.severity == SecuritySeverity.CRITICAL and 
            self.config.alert_on_critical):
            await self._send_alert(event)
    
    async def _send_alert(self, event: SecurityEvent) -> None:
        """Send alert for critical security events."""
        # Implementation would send alerts via email, Slack, etc.
        logger.critical(
            "CRITICAL SECURITY ALERT",
            event_id=event.event_id,
            event_type=event.event_type,
            details=event.details
        )
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        current_time = time.time()
        last_hour_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600
        ]
        
        event_counts = {}
        for event in last_hour_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        return {
            "total_events": len(self.security_events),
            "events_last_hour": len(last_hour_events),
            "event_types": event_counts,
            "suspicious_ips": len(self.threat_detector.suspicious_ips),
            "failed_attempts": len(self.threat_detector.failed_attempts),
            "pii_enabled": self.config.pii_enabled,
            "content_filtering_enabled": self.config.content_filtering_enabled,
            "threat_detection_enabled": self.config.threat_detection_enabled
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check security service health."""
        health = {
            "initialized": self._initialized,
            "pii_service": self.pii_service._initialized,
            "content_filter": True,
            "threat_detector": True
        }
        
        health["overall"] = all(health.values())
        return health


class SecurityException(Exception):
    """Security-related exception."""
    pass


# Utility functions
def hash_sensitive_data(data: str, salt: str = "enterprise-ai-gateway") -> str:
    """Hash sensitive data for storage."""
    return hashlib.sha256(f"{salt}{data}".encode()).hexdigest()


def validate_input_length(text: str, max_length: int = 10000) -> bool:
    """Validate input length."""
    return len(text) <= max_length


async def check_domain_reputation(domain: str) -> Dict[str, Any]:
    """Check domain reputation (placeholder implementation)."""
    # Implementation would check against threat intelligence feeds
    return {
        "safe": True,
        "category": "unknown",
        "risk_score": 0.0
    }