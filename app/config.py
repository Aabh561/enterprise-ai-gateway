"""
Configuration Management for Enterprise AI Gateway

This module provides configuration management using Pydantic Settings
with support for environment-specific YAML configurations.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorsSettings(BaseSettings):
    """CORS configuration settings."""

    origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True


class APISettings(BaseSettings):
    """API configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    access_log: bool = True
    cors: CorsSettings = Field(default_factory=CorsSettings)


class AppSettings(BaseSettings):
    """Application configuration settings."""

    name: str = "Enterprise AI Gateway"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    url: str = "sqlite:///./data/dev.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 10
    cache_ttl: int = 3600


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


class VectorDBSettings(BaseSettings):
    """Vector database configuration settings."""

    provider: str = "weaviate"
    url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    class_name: str = "Documents"
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


class LLMProviderSettings(BaseSettings):
    """Individual LLM provider settings."""

    api_key: Optional[str] = None
    model: str = ""
    timeout: int = 30
    base_url: Optional[str] = None
    max_retries: int = 3


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    default_provider: str = "ollama"
    default_model: str = "llama3"
    providers: Dict[str, Any] = Field(default_factory=dict)


class APIKeySettings(BaseSettings):
    """API key security settings."""

    enabled: bool = True
    # Backward-compatible single secret; prefer 'secrets' list for rotation
    secret: str = "your-super-secret-api-key-here"
    secrets: List[str] = Field(default_factory=list)
    require_https: bool = False


class JWTSettings(BaseSettings):
    """JWT authentication settings."""

    secret_key: str = "your-jwt-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


class PIIProtectionSettings(BaseSettings):
    """PII protection settings."""

    enabled: bool = True
    threshold: float = 0.8
    analyzer_url: str = "http://localhost:5001"
    anonymizer_url: str = "http://localhost:5002"


class ContentFilteringSettings(BaseSettings):
    """Content filtering settings."""

    enabled: bool = False
    max_input_length: int = 8192
    blocked_patterns: List[str] = Field(default_factory=list)


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    api_keys: APIKeySettings = Field(default_factory=APIKeySettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    encryption: Dict[str, Any] = Field(default_factory=dict)
    pii_protection: PIIProtectionSettings = Field(default_factory=PIIProtectionSettings)
    content_filtering: ContentFilteringSettings = Field(
        default_factory=ContentFilteringSettings
    )


class SandboxSettings(BaseSettings):
    """Plugin sandbox settings."""

    enabled: bool = False
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50


class PluginSettings(BaseSettings):
    """Plugin configuration settings."""

    directory: str = "./plugins"
    timeout: int = 30
    max_retries: int = 3
    enabled: List[str] = Field(default_factory=list)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)


class MetricsSettings(BaseSettings):
    """Metrics configuration settings."""

    enabled: bool = True
    prometheus_port: int = 9090
    include_sensitive_labels: bool = True


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    audit_enabled: bool = True
    audit_log_path: str = "./logs/audit.log"
    structured: bool = True
    retention_days: int = 30


class TracingSettings(BaseSettings):
    """Tracing configuration settings."""

    enabled: bool = True
    service_name: str = "enterprise-ai-gateway"
    sample_rate: float = 1.0


class AlertSettings(BaseSettings):
    """Alert configuration settings."""

    enabled: bool = False
    webhook_url: Optional[str] = None


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""

    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)


class RateLimitingSettings(BaseSettings):
    """Rate limiting configuration settings."""

    per_minute: int = 60
    burst: int = 100
    enabled: bool = True
    redis_prefix: str = "rate_limit:"


class FileProcessingSettings(BaseSettings):
    """File processing configuration settings."""

    max_file_size_mb: int = 50
    supported_types: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "html", "md"]
    )
    batch_size: int = 5


class BackupSettings(BaseSettings):
    """Backup configuration settings."""

    enabled: bool = False
    schedule: str = "0 2 * * *"
    retention_days: int = 30
    storage: Dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    """
    Main application settings combining all configuration sections.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    app: AppSettings = Field(default_factory=AppSettings)
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    rate_limiting: RateLimitingSettings = Field(default_factory=RateLimitingSettings)
    file_processing: FileProcessingSettings = Field(
        default_factory=FileProcessingSettings
    )
    backup: BackupSettings = Field(default_factory=BackupSettings)

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "Settings":
        """
        Load settings from a YAML configuration file.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            Settings instance with loaded configuration
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def save_to_yaml(self, file_path: str) -> None:
        """
        Save current settings to a YAML file.

        Args:
            file_path: Path where to save the YAML configuration
        """
        config_data = self.dict()

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.

    Load order:
    1. CONFIG_PATH environment variable (absolute or relative path)
    2. ENVIRONMENT-specific YAML (configs/<environment>.yaml)
    3. Environment variables/.env defaults
    """
    # 1) Explicit config path
    explicit_path = os.getenv("CONFIG_PATH")
    if explicit_path and os.path.exists(explicit_path):
        try:
            return Settings.load_from_yaml(explicit_path)
        except Exception as e:
            print(f"Warning: Failed to load CONFIG_PATH={explicit_path}: {e}")

    # 2) Environment-based YAML
    environment = os.getenv("ENVIRONMENT", "development")
    yaml_path = f"configs/{environment}.yaml"
    if os.path.exists(yaml_path):
        try:
            return Settings.load_from_yaml(yaml_path)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_path}: {e}")

    # 3) Fall back to env vars and defaults
    return Settings()


def get_config_for_environment(environment: str) -> Settings:
    """
    Get configuration for a specific environment.

    Args:
        environment: Environment name (dev, prod, test, etc.)

    Returns:
        Settings: Configuration for the specified environment
    """
    yaml_path = f"configs/{environment}.yaml"

    if os.path.exists(yaml_path):
        return Settings.load_from_yaml(yaml_path)
    else:
        raise ValueError(f"Configuration file not found for environment: {environment}")


# Export commonly used settings
__all__ = ["Settings", "get_settings", "get_config_for_environment"]
