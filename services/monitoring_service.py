"""
Advanced Monitoring and Observability Service for Enterprise AI Gateway

Provides comprehensive monitoring with metrics collection, distributed tracing,
structured logging, alerting, and business intelligence dashboards.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import psutil

# Tracing imports
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Alerting imports
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.95"
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel] = field(default_factory=list)
    cooldown_seconds: int = 300
    description: str = ""
    enabled: bool = True


@dataclass
class AlertEvent:
    """Alert event data."""
    alert_id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    description: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class TraceSpan:
    """Trace span data."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"


@dataclass
class MonitoringConfig:
    """Monitoring service configuration."""
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 8080
    custom_metrics: List[MetricDefinition] = field(default_factory=list)
    
    # Tracing
    tracing_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    trace_sample_rate: float = 0.1
    
    # Logging
    structured_logging: bool = True
    log_level: str = "INFO"
    log_retention_days: int = 30
    
    # Alerting
    alerting_enabled: bool = True
    alert_rules: List[AlertRule] = field(default_factory=list)
    
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_from: str = ""
    alert_email_to: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    
    # System monitoring
    system_metrics_enabled: bool = True
    system_metrics_interval: int = 60
    
    # Business metrics
    business_metrics_enabled: bool = True
    retention_period_days: int = 90


class MetricsCollector:
    """Advanced metrics collection and management."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Built-in metrics
        self._setup_builtin_metrics()
        
        # Custom metrics
        self._setup_custom_metrics()
    
    def _setup_builtin_metrics(self) -> None:
        """Setup built-in system and application metrics."""
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_token_usage = Counter(
            'llm_tokens_total',
            'Total tokens processed',
            ['provider', 'model', 'type'],
            registry=self.registry
        )
        
        self.llm_cost_total = Counter(
            'llm_cost_usd_total',
            'Total LLM costs in USD',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_type', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache_type'],
            registry=self.registry
        )
        
        # Vector DB metrics
        self.vector_operations = Counter(
            'vector_operations_total',
            'Total vector database operations',
            ['operation', 'collection', 'status'],
            registry=self.registry
        )
        
        # Security metrics
        self.security_events = Counter(
            'security_events_total',
            'Total security events',
            ['event_type', 'severity'],
            registry=self.registry
        )
        
        # Plugin metrics
        self.plugin_executions = Counter(
            'plugin_executions_total',
            'Total plugin executions',
            ['plugin_name', 'action', 'status'],
            registry=self.registry
        )
        
        # System metrics
        if self.config.system_metrics_enabled:
            self.system_cpu_usage = Gauge(
                'system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            )
            
            self.system_memory_usage = Gauge(
                'system_memory_usage_bytes',
                'System memory usage in bytes',
                registry=self.registry
            )
            
            self.system_disk_usage = Gauge(
                'system_disk_usage_bytes',
                'System disk usage in bytes',
                ['device'],
                registry=self.registry
            )
    
    def _setup_custom_metrics(self) -> None:
        """Setup custom metrics from configuration."""
        for metric_def in self.config.custom_metrics:
            if metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                buckets = metric_def.buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
                metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    buckets=buckets,
                    registry=self.registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                quantiles = metric_def.quantiles or [0.5, 0.9, 0.95, 0.99]
                metric = Summary(
                    metric_def.name,
                    metric_def.description,
                    metric_def.labels,
                    registry=self.registry
                )
            
            self.custom_metrics[metric_def.name] = metric
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0
    ) -> None:
        """Record LLM request metrics."""
        self.llm_requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        if input_tokens > 0:
            self.llm_token_usage.labels(
                provider=provider,
                model=model,
                type="input"
            ).inc(input_tokens)
        
        if output_tokens > 0:
            self.llm_token_usage.labels(
                provider=provider,
                model=model,
                type="output"
            ).inc(output_tokens)
        
        if cost > 0:
            self.llm_cost_total.labels(
                provider=provider,
                model=model
            ).inc(cost)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        if not self.config.system_metrics_enabled:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # Disk usage
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    self.system_disk_usage.labels(device=disk.device).set(usage.used)
                except (PermissionError, OSError):
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metric_value(self, metric_name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current value of a metric."""
        try:
            metric = self.custom_metrics.get(metric_name)
            if not metric:
                # Try built-in metrics
                metric = getattr(self, metric_name.replace('-', '_'), None)
            
            if metric:
                if hasattr(metric, '_value'):
                    return metric._value.get()
                elif hasattr(metric, 'collect'):
                    samples = list(metric.collect())[0].samples
                    if samples:
                        return samples[0].value
            
            return None
        except Exception as e:
            logger.error(f"Failed to get metric value for {metric_name}: {e}")
            return None
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode()


class DistributedTracer:
    """Distributed tracing service."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        if HAS_OPENTELEMETRY and config.tracing_enabled:
            self._setup_opentelemetry()
    
    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing."""
        try:
            # Configure tracer provider
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
                collector_endpoint=self.config.jaeger_endpoint,
            )
            
            # Configure span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = tracer
            logger.info("OpenTelemetry tracing initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
            self.tracer = None
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new trace span."""
        span_id = str(uuid.uuid4())
        trace_id = parent_span_id or str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self.active_spans[span_id] = span
        
        # OpenTelemetry integration
        if hasattr(self, 'tracer') and self.tracer:
            try:
                otel_span = self.tracer.start_span(operation_name)
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, str(value))
                span.otel_span = otel_span
            except Exception as e:
                logger.error(f"Failed to start OpenTelemetry span: {e}")
        
        return span_id
    
    def finish_span(
        self,
        span_id: str,
        tags: Optional[Dict[str, Any]] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        status: str = "ok"
    ) -> None:
        """Finish a trace span."""
        with self._lock:
            span = self.active_spans.pop(span_id, None)
        
        if not span:
            return
        
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status
        
        if tags:
            span.tags.update(tags)
        
        if logs:
            span.logs.extend(logs)
        
        # Finish OpenTelemetry span
        if hasattr(span, 'otel_span'):
            try:
                if status != "ok":
                    span.otel_span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.otel_span.end()
            except Exception as e:
                logger.error(f"Failed to finish OpenTelemetry span: {e}")
        
        # Store completed span
        self.completed_traces.append(span)
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self._lock:
            active_spans_count = len(self.active_spans)
        
        completed_count = len(self.completed_traces)
        
        # Calculate average duration
        if completed_count > 0:
            avg_duration = sum(
                span.duration_ms for span in self.completed_traces 
                if span.duration_ms is not None
            ) / completed_count
        else:
            avg_duration = 0
        
        return {
            "active_spans": active_spans_count,
            "completed_traces": completed_count,
            "average_duration_ms": avg_duration,
            "tracing_enabled": self.config.tracing_enabled
        }


class AlertManager:
    """Alert management and notification service."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules = {rule.name: rule for rule in config.alert_rules}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.last_alert_times: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def check_alerts(self, metrics_collector: MetricsCollector) -> List[AlertEvent]:
        """Check all alert rules against current metrics."""
        triggered_alerts = []
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_alert_time = self.last_alert_times.get(rule_name, 0)
            if current_time - last_alert_time < rule.cooldown_seconds:
                continue
            
            # Get metric value
            metric_value = metrics_collector.get_metric_value(rule.metric_name)
            if metric_value is None:
                continue
            
            # Evaluate condition
            if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                alert_event = AlertEvent(
                    alert_id=str(uuid.uuid4()),
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    description=rule.description,
                    timestamp=current_time
                )
                
                triggered_alerts.append(alert_event)
                
                with self._lock:
                    self.active_alerts[alert_event.alert_id] = alert_event
                    self.alert_history.append(alert_event)
                
                self.last_alert_times[rule_name] = current_time
                
                # Send notifications
                asyncio.create_task(self._send_alert_notifications(alert_event, rule))
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition.startswith('>'):
                return value > threshold
            elif condition.startswith('<'):
                return value < threshold
            elif condition.startswith('>='):
                return value >= threshold
            elif condition.startswith('<='):
                return value <= threshold
            elif condition.startswith('=='):
                return value == threshold
            elif condition.startswith('!='):
                return value != threshold
            else:
                return False
        except Exception:
            return False
    
    async def _send_alert_notifications(self, alert: AlertEvent, rule: AlertRule) -> None:
        """Send alert notifications through configured channels."""
        for channel in rule.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert, rule)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert, rule)
                elif channel == AlertChannel.CONSOLE:
                    await self._send_console_alert(alert, rule)
                # Add Slack integration here
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_email_alert(self, alert: AlertEvent, rule: AlertRule) -> None:
        """Send email alert notification."""
        if not HAS_EMAIL or not self.config.alert_email_to:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.alert_email_from
            msg['To'] = ', '.join(self.config.alert_email_to)
            msg['Subject'] = f"[{alert.severity.upper()}] {rule.name}"
            
            body = f"""
            Alert: {rule.name}
            Severity: {alert.severity}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value}
            Threshold: {alert.threshold}
            Description: {alert.description}
            Time: {datetime.fromtimestamp(alert.timestamp)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
            server.starttls()
            if self.config.smtp_user:
                server.login(self.config.smtp_user, self.config.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.config.alert_email_from, self.config.alert_email_to, text)
            server.quit()
            
            logger.info(f"Email alert sent for {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: AlertEvent, rule: AlertRule) -> None:
        """Send webhook alert notification."""
        import httpx
        
        payload = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "description": alert.description,
            "timestamp": alert.timestamp
        }
        
        for webhook_url in self.config.webhook_urls:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(webhook_url, json=payload, timeout=10.0)
                    response.raise_for_status()
                    logger.info(f"Webhook alert sent to {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
    
    async def _send_console_alert(self, alert: AlertEvent, rule: AlertRule) -> None:
        """Send console alert notification."""
        logger.warning(
            f"ALERT: {rule.name}",
            severity=alert.severity,
            metric=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold,
            description=alert.description
        )
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            active_count = len(self.active_alerts)
            total_count = len(self.alert_history)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity] += 1
        
        return {
            "active_alerts": active_count,
            "total_alerts": total_count,
            "alerts_by_severity": dict(severity_counts),
            "alert_rules": len(self.alert_rules),
            "alerting_enabled": self.config.alerting_enabled
        }


class BusinessMetricsCollector:
    """Business intelligence and application metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.business_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.Lock()
    
    def record_user_activity(
        self,
        user_id: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record user activity."""
        event = {
            "user_id": user_id,
            "action": action,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.business_metrics["user_activity"].append(event)
    
    def record_revenue_event(
        self,
        amount: float,
        currency: str = "USD",
        source: str = "api",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record revenue-generating event."""
        event = {
            "amount": amount,
            "currency": currency,
            "source": source,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.business_metrics["revenue"].append(event)
    
    def record_feature_usage(
        self,
        feature: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record feature usage."""
        event = {
            "feature": feature,
            "user_id": user_id,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.business_metrics["feature_usage"].append(event)
    
    def get_business_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get business metrics for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        stats = {}
        
        with self._lock:
            for metric_name, events in self.business_metrics.items():
                recent_events = [e for e in events if e["timestamp"] > cutoff_time]
                
                if metric_name == "user_activity":
                    unique_users = len(set(e["user_id"] for e in recent_events))
                    action_counts = defaultdict(int)
                    for event in recent_events:
                        action_counts[event["action"]] += 1
                    
                    stats["user_activity"] = {
                        "total_events": len(recent_events),
                        "unique_users": unique_users,
                        "actions": dict(action_counts)
                    }
                
                elif metric_name == "revenue":
                    total_revenue = sum(e["amount"] for e in recent_events)
                    currency_totals = defaultdict(float)
                    for event in recent_events:
                        currency_totals[event["currency"]] += event["amount"]
                    
                    stats["revenue"] = {
                        "total_events": len(recent_events),
                        "total_revenue": total_revenue,
                        "by_currency": dict(currency_totals)
                    }
                
                elif metric_name == "feature_usage":
                    feature_counts = defaultdict(int)
                    for event in recent_events:
                        feature_counts[event["feature"]] += 1
                    
                    stats["feature_usage"] = {
                        "total_events": len(recent_events),
                        "by_feature": dict(feature_counts)
                    }
        
        return stats


class MonitoringService:
    """Comprehensive monitoring and observability service."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.tracer = DistributedTracer(config)
        self.alert_manager = AlertManager(config)
        self.business_metrics = BusinessMetricsCollector(config)
        
        self._monitoring_tasks: List[asyncio.Task] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        if self._initialized:
            return
        
        # Start background monitoring tasks
        if self.config.system_metrics_enabled:
            task = asyncio.create_task(self._system_metrics_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.alerting_enabled:
            task = asyncio.create_task(self._alerting_loop())
            self._monitoring_tasks.append(task)
        
        self._initialized = True
        logger.info("Monitoring service initialized")
    
    async def _system_metrics_loop(self) -> None:
        """Background task to collect system metrics."""
        while True:
            try:
                self.metrics_collector.update_system_metrics()
                await asyncio.sleep(self.config.system_metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _alerting_loop(self) -> None:
        """Background task to check alerts."""
        while True:
            try:
                self.alert_manager.check_alerts(self.metrics_collector)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(5)
    
    # Metrics interface
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Record HTTP request metrics."""
        self.metrics_collector.record_http_request(method, endpoint, status, duration)
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0
    ) -> None:
        """Record LLM request metrics."""
        self.metrics_collector.record_llm_request(
            provider, model, status, input_tokens, output_tokens, cost
        )
    
    # Tracing interface
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new trace span."""
        return self.tracer.start_span(operation_name, parent_span_id, tags)
    
    def finish_span(
        self,
        span_id: str,
        tags: Optional[Dict[str, Any]] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        status: str = "ok"
    ) -> None:
        """Finish a trace span."""
        self.tracer.finish_span(span_id, tags, logs, status)
    
    # Business metrics interface
    def record_user_activity(
        self,
        user_id: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record user activity."""
        self.business_metrics.record_user_activity(user_id, action, metadata)
    
    def record_revenue_event(
        self,
        amount: float,
        currency: str = "USD",
        source: str = "api",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record revenue event."""
        self.business_metrics.record_revenue_event(amount, currency, source, metadata)
    
    def record_feature_usage(
        self,
        feature: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record feature usage."""
        self.business_metrics.record_feature_usage(feature, user_id, metadata)
    
    # Alert management
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_manager.add_alert_rule(rule)
    
    # Status and statistics
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return {
            "metrics": {
                "system_metrics_enabled": self.config.system_metrics_enabled,
                "custom_metrics_count": len(self.metrics_collector.custom_metrics)
            },
            "tracing": self.tracer.get_trace_stats(),
            "alerts": self.alert_manager.get_alert_stats(),
            "business_metrics": self.business_metrics.get_business_stats(),
            "monitoring_tasks": len(self._monitoring_tasks),
            "initialized": self._initialized
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics_collector.export_metrics()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check monitoring service health."""
        health = {
            "initialized": self._initialized,
            "metrics_collector": True,
            "tracer": self.config.tracing_enabled,
            "alert_manager": self.config.alerting_enabled,
            "business_metrics": self.config.business_metrics_enabled
        }
        
        # Check if monitoring tasks are running
        running_tasks = sum(1 for task in self._monitoring_tasks if not task.done())
        health["monitoring_tasks"] = running_tasks > 0 if self._monitoring_tasks else True
        
        health["overall"] = all(health.values())
        return health
    
    async def close(self) -> None:
        """Close monitoring service and cleanup resources."""
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        logger.info("Monitoring service closed")


# Utility functions and decorators
def monitor_async_function(
    monitoring_service: MonitoringService,
    operation_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None
):
    """Decorator to monitor async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            span_id = monitoring_service.start_span(span_name, tags=tags)
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                monitoring_service.finish_span(span_id, status="ok")
                return result
            except Exception as e:
                monitoring_service.finish_span(
                    span_id,
                    tags={"error": str(e)},
                    status="error"
                )
                raise
            finally:
                duration = time.time() - start_time
                logger.info(f"Function {span_name} completed in {duration:.3f}s")
        
        return wrapper
    return decorator


def create_default_monitoring_config() -> MonitoringConfig:
    """Create default monitoring configuration."""
    default_alert_rules = [
        AlertRule(
            name="high_error_rate",
            metric_name="http_requests_total",
            condition="> 0.05",  # 5% error rate
            threshold=0.05,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
            description="High HTTP error rate detected"
        ),
        AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_usage_percent",
            condition="> 80",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE],
            description="High CPU usage detected"
        ),
        AlertRule(
            name="critical_cpu_usage",
            metric_name="system_cpu_usage_percent",
            condition="> 95",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
            description="Critical CPU usage detected"
        )
    ]
    
    return MonitoringConfig(
        alert_rules=default_alert_rules,
        system_metrics_enabled=True,
        business_metrics_enabled=True,
        alerting_enabled=True
    )