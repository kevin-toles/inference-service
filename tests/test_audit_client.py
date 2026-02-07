"""Tests for Neo4j Audit Client - LLM Operations Mesh Phase 5.

Tests cover:
- AuditClientConfig
- AuditClient initialization
- Log methods (model_loaded, model_unloaded, config_changed)
- Query methods (get_changes_since, get_model_history)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.audit_client import (
    AuditClient,
    AuditClientConfig,
    get_audit_client,
    set_audit_client,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAuditClientConfig:
    """Tests for AuditClientConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AuditClientConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password == ""
        assert config.database == "neo4j"
        assert config.enabled is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AuditClientConfig(
            uri="bolt://neo4j.example.com:7687",
            user="admin",
            password="secret",
            database="audit",
            enabled=False,
        )
        assert config.uri == "bolt://neo4j.example.com:7687"
        assert config.user == "admin"
        assert config.password == "secret"
        assert config.database == "audit"
        assert config.enabled is False
    
    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict("os.environ", {
            "NEO4J_URI": "bolt://test:7687",
            "NEO4J_USER": "test_user",
            "NEO4J_PASSWORD": "test_pass",
            "NEO4J_DATABASE": "test_db",
            "NEO4J_AUDIT_ENABLED": "false",
        }):
            config = AuditClientConfig.from_env()
            assert config.uri == "bolt://test:7687"
            assert config.user == "test_user"
            assert config.password == "test_pass"
            assert config.database == "test_db"
            assert config.enabled is False
    
    def test_from_env_defaults(self):
        """Test configuration uses defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear specific vars that might be set
            env = {k: v for k, v in __import__('os').environ.items() 
                   if not k.startswith('NEO4J_')}
            with patch.dict("os.environ", env, clear=True):
                config = AuditClientConfig.from_env()
                assert config.uri == "bolt://localhost:7687"
                assert config.enabled is True


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestAuditClientInit:
    """Tests for AuditClient initialization."""
    
    def test_init_with_config(self):
        """Test client initialization with config."""
        config = AuditClientConfig(
            uri="bolt://test:7687",
            user="test",
            password="test",
        )
        client = AuditClient(config)
        assert client._config == config
        assert client._driver is None
        assert client.is_connected is False
    
    def test_init_default_config(self):
        """Test client initialization with default config."""
        with patch.object(AuditClientConfig, 'from_env') as mock_from_env:
            mock_from_env.return_value = AuditClientConfig()
            client = AuditClient()
            mock_from_env.assert_called_once()
    
    def test_is_enabled(self):
        """Test is_enabled property."""
        config_enabled = AuditClientConfig(enabled=True)
        config_disabled = AuditClientConfig(enabled=False)
        
        client_enabled = AuditClient(config_enabled)
        client_disabled = AuditClient(config_disabled)
        
        assert client_enabled.is_enabled is True
        assert client_disabled.is_enabled is False


# =============================================================================
# Connection Tests
# =============================================================================


class TestAuditClientConnection:
    """Tests for AuditClient connection handling."""
    
    @pytest.mark.asyncio
    async def test_connect_disabled(self):
        """Test connect returns False when disabled."""
        config = AuditClientConfig(enabled=False)
        client = AuditClient(config)
        
        result = await client.connect()
        
        assert result is False
        assert client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_connect_no_neo4j_package(self):
        """Test connect handles missing neo4j package."""
        config = AuditClientConfig(enabled=True)
        client = AuditClient(config)
        
        with patch.dict('sys.modules', {'neo4j': None}):
            # Force ImportError
            with patch('builtins.__import__', side_effect=ImportError):
                result = await client.connect()
        
        # Should return False gracefully
        assert client.is_connected is False
    
    @pytest.mark.asyncio
    async def test_close_not_connected(self):
        """Test close when not connected."""
        client = AuditClient(AuditClientConfig())
        
        # Should not raise
        await client.close()
        
        assert client.is_connected is False


# =============================================================================
# Logging Method Tests (Mock)
# =============================================================================


class TestAuditLogging:
    """Tests for audit logging methods."""
    
    @pytest.mark.asyncio
    async def test_log_model_loaded_not_connected(self):
        """Test log_model_loaded returns None when not connected."""
        client = AuditClient(AuditClientConfig())
        
        result = await client.log_model_loaded(
            model_id="test-model",
            context_length=2048,
            memory_mb=1000,
            roles=["primary"],
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_log_model_unloaded_not_connected(self):
        """Test log_model_unloaded returns None when not connected."""
        client = AuditClient(AuditClientConfig())
        
        result = await client.log_model_unloaded("test-model")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_log_config_changed_not_connected(self):
        """Test log_config_changed returns None when not connected."""
        client = AuditClient(AuditClientConfig())
        
        result = await client.log_config_changed(
            model_id="test-model",
            field="context_length",
            old_value=2048,
            new_value=4096,
        )
        
        assert result is None


# =============================================================================
# Query Method Tests (Mock)
# =============================================================================


class TestAuditQueries:
    """Tests for audit query methods."""
    
    @pytest.mark.asyncio
    async def test_get_changes_since_not_connected(self):
        """Test get_changes_since returns empty list when not connected."""
        client = AuditClient(AuditClientConfig())
        
        result = await client.get_changes_since(hours=24)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_model_history_not_connected(self):
        """Test get_model_history returns empty list when not connected."""
        client = AuditClient(AuditClientConfig())
        
        result = await client.get_model_history("test-model")
        
        assert result == []


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Tests for global client instance management."""
    
    def test_get_audit_client_not_set(self):
        """Test get_audit_client returns None initially."""
        # Reset global
        import src.services.audit_client as module
        module._audit_client = None
        
        result = get_audit_client()
        assert result is None
    
    def test_set_and_get_audit_client(self):
        """Test set_audit_client and get_audit_client."""
        client = AuditClient(AuditClientConfig())
        
        set_audit_client(client)
        result = get_audit_client()
        
        assert result is client
        
        # Clean up
        import src.services.audit_client as module
        module._audit_client = None
