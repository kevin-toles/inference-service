"""Neo4j Audit Client for configuration change tracking.

LLM Operations Mesh - Phase 5: Neo4j Audit Trail

This module logs all model configuration changes to Neo4j for:
- Queryable history of changes
- Debugging configuration issues
- Compliance and auditing

Reference: LLM_OPERATIONS_MESH_ARCHITECTURE.md - Phase 5
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AuditClientConfig:
    """Configuration for Neo4j audit client.
    
    Attributes:
        uri: Neo4j Bolt URI (e.g., "bolt://localhost:7687")
        user: Neo4j username
        password: Neo4j password
        database: Database name (default: "neo4j")
        enabled: Whether audit logging is enabled
    """
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    enabled: bool = True
    
    @classmethod
    def from_env(cls) -> AuditClientConfig:
        """Create config from environment variables.
        
        Environment Variables:
            NEO4J_URI: Bolt URI
            NEO4J_USER: Username
            NEO4J_PASSWORD: Password
            NEO4J_DATABASE: Database name
            NEO4J_AUDIT_ENABLED: Enable audit logging (default: true)
        """
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            enabled=os.environ.get("NEO4J_AUDIT_ENABLED", "true").lower() == "true",
        )


# =============================================================================
# Audit Client
# =============================================================================


class AuditClient:
    """Neo4j client for configuration change auditing.
    
    Logs model load/unload events to Neo4j for queryable history.
    
    WBS: LLM-MESH-P5 - Neo4j Audit Trail
    
    Example:
        client = AuditClient(AuditClientConfig.from_env())
        await client.connect()
        await client.log_model_loaded("qwen3-8b", context_length=2048)
        await client.close()
    """
    
    def __init__(self, config: AuditClientConfig | None = None) -> None:
        """Initialize audit client.
        
        Args:
            config: Configuration (uses env vars if not provided)
        """
        self._config = config or AuditClientConfig.from_env()
        self._driver: Any = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected and self._driver is not None
    
    @property
    def is_enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self._config.enabled
    
    async def connect(self) -> bool:
        """Establish connection to Neo4j.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._config.enabled:
            logger.info("Neo4j audit logging is disabled")
            return False
        
        try:
            from neo4j import GraphDatabase
            
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
            )
            
            # Verify connectivity
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._driver.verify_connectivity,
            )
            
            self._connected = True
            logger.info(f"AuditClient connected to Neo4j at {self._config.uri}")
            
            # Ensure schema exists
            await self._ensure_schema()
            
            return True
            
        except ImportError:
            logger.warning("neo4j package not installed - audit logging disabled")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j for audit: {e}")
            return False
    
    async def close(self) -> None:
        """Close connection and release resources."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("AuditClient disconnected from Neo4j")
    
    async def _ensure_schema(self) -> None:
        """Ensure ConfigChange node schema exists."""
        if not self._driver:
            return
        
        def create_constraints():
            with self._driver.session(database=self._config.database) as session:
                # Create index for timestamp queries
                try:
                    session.run("""
                        CREATE INDEX config_change_timestamp IF NOT EXISTS
                        FOR (c:ConfigChange) ON (c.timestamp)
                    """)
                except Exception:
                    pass  # Index may already exist
                
                # Create index for model_id queries
                try:
                    session.run("""
                        CREATE INDEX config_change_model IF NOT EXISTS
                        FOR (c:ConfigChange) ON (c.model_id)
                    """)
                except Exception:
                    pass
        
        await asyncio.get_event_loop().run_in_executor(None, create_constraints)
        logger.debug("Neo4j audit schema verified")
    
    async def log_model_loaded(
        self,
        model_id: str,
        context_length: int,
        memory_mb: int,
        roles: list[str],
    ) -> str | None:
        """Log a model load event.
        
        Creates a ConfigChange node with event_type=MODEL_LOADED.
        
        Args:
            model_id: Model identifier
            context_length: Model's context window size
            memory_mb: Memory usage in megabytes
            roles: Model roles (e.g., ["primary", "fast"])
            
        Returns:
            Node ID if created, None if failed
        """
        if not self._connected:
            return None
        
        return await self._log_event(
            model_id=model_id,
            event_type="MODEL_LOADED",
            new_value={
                "context_length": context_length,
                "memory_mb": memory_mb,
                "roles": roles,
                "status": "loaded",
            },
            old_value=None,
        )
    
    async def log_model_unloaded(self, model_id: str) -> str | None:
        """Log a model unload event.
        
        Creates a ConfigChange node with event_type=MODEL_UNLOADED.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Node ID if created, None if failed
        """
        if not self._connected:
            return None
        
        return await self._log_event(
            model_id=model_id,
            event_type="MODEL_UNLOADED",
            new_value={"status": "unloaded"},
            old_value=None,
        )
    
    async def log_config_changed(
        self,
        model_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
    ) -> str | None:
        """Log a configuration change event.
        
        Creates a ConfigChange node with event_type=CONFIG_CHANGED.
        
        Args:
            model_id: Model identifier
            field: Field that changed (e.g., "context_length")
            old_value: Previous value
            new_value: New value
            
        Returns:
            Node ID if created, None if failed
        """
        if not self._connected:
            return None
        
        return await self._log_event(
            model_id=model_id,
            event_type="CONFIG_CHANGED",
            new_value={field: new_value},
            old_value={field: old_value} if old_value is not None else None,
        )
    
    async def _log_event(
        self,
        model_id: str,
        event_type: str,
        new_value: dict[str, Any] | None,
        old_value: dict[str, Any] | None,
    ) -> str | None:
        """Internal method to log an event to Neo4j.
        
        Args:
            model_id: Model identifier
            event_type: Type of event
            new_value: New configuration values
            old_value: Previous configuration values
            
        Returns:
            Node ID if created, None if failed
        """
        import json
        
        def create_node():
            with self._driver.session(database=self._config.database) as session:
                result = session.run("""
                    CREATE (c:ConfigChange {
                        model_id: $model_id,
                        event_type: $event_type,
                        old_value: $old_value,
                        new_value: $new_value,
                        timestamp: datetime(),
                        source: 'inference-service'
                    })
                    RETURN elementId(c) as node_id
                """, {
                    "model_id": model_id,
                    "event_type": event_type,
                    "old_value": json.dumps(old_value) if old_value else None,
                    "new_value": json.dumps(new_value) if new_value else None,
                })
                record = result.single()
                return record["node_id"] if record else None
        
        try:
            node_id = await asyncio.get_event_loop().run_in_executor(
                None, create_node
            )
            logger.info(
                f"Audit logged: {event_type} for {model_id}",
                extra={"node_id": node_id}
            )
            return node_id
            
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
            return None
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    async def get_changes_since(
        self,
        hours: int = 24,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get configuration changes in the last N hours.
        
        Args:
            hours: Number of hours to look back
            model_id: Optional filter by model
            
        Returns:
            List of change records
        """
        if not self._connected:
            return []
        
        def query_changes():
            with self._driver.session(database=self._config.database) as session:
                cypher = """
                    MATCH (c:ConfigChange)
                    WHERE c.timestamp > datetime() - duration({hours: $hours})
                """
                if model_id:
                    cypher += " AND c.model_id = $model_id"
                cypher += """
                    RETURN c.model_id as model_id,
                           c.event_type as event_type,
                           c.old_value as old_value,
                           c.new_value as new_value,
                           c.timestamp as timestamp,
                           c.source as source
                    ORDER BY c.timestamp DESC
                """
                result = session.run(cypher, {"hours": hours, "model_id": model_id})
                return [dict(record) for record in result]
        
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, query_changes
            )
        except Exception as e:
            logger.warning(f"Failed to query changes: {e}")
            return []
    
    async def get_model_history(self, model_id: str) -> list[dict[str, Any]]:
        """Get full configuration history for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of all change records for this model
        """
        if not self._connected:
            return []
        
        def query_history():
            with self._driver.session(database=self._config.database) as session:
                result = session.run("""
                    MATCH (c:ConfigChange {model_id: $model_id})
                    RETURN c.model_id as model_id,
                           c.event_type as event_type,
                           c.old_value as old_value,
                           c.new_value as new_value,
                           c.timestamp as timestamp,
                           c.source as source
                    ORDER BY c.timestamp DESC
                """, {"model_id": model_id})
                return [dict(record) for record in result]
        
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, query_history
            )
        except Exception as e:
            logger.warning(f"Failed to query model history: {e}")
            return []


# =============================================================================
# Global Instance
# =============================================================================


_audit_client: AuditClient | None = None


def get_audit_client() -> AuditClient | None:
    """Get the global audit client instance.
    
    Returns:
        AuditClient instance or None if not initialized
    """
    return _audit_client


def set_audit_client(client: AuditClient) -> None:
    """Set the global audit client instance.
    
    Args:
        client: AuditClient instance
    """
    global _audit_client
    _audit_client = client


async def init_audit_client() -> AuditClient | None:
    """Initialize and connect the global audit client.
    
    Returns:
        Connected AuditClient or None if connection failed
    """
    global _audit_client
    
    config = AuditClientConfig.from_env()
    client = AuditClient(config)
    
    if await client.connect():
        _audit_client = client
        return client
    
    return None
