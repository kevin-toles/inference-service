# Conversation Store Design

> **Version:** 1.0.0  
> **Last Updated:** 2025-12-27  
> **Status:** Design Phase  
> **Owner:** audit-service (or llm-gateway)

## Table of Contents

1. [Overview](#overview)
2. [Architecture Position](#architecture-position)
3. [Data Model](#data-model)
4. [Capture Flow](#capture-flow)
5. [Storage Options](#storage-options)
6. [API Contract](#api-contract)
7. [Reporting & Analytics](#reporting--analytics)
8. [Retention & Privacy](#retention--privacy)

---

## Overview

The **Conversation Store** is a persistent, append-only storage system for capturing full conversation history from VS Code and other clients. It is separate from the operational caching in inference-service.

### Purpose

| Capability | Description |
|------------|-------------|
| **History** | Full conversation replay and reference |
| **Reporting** | Token usage, latency, model performance |
| **Debugging** | Trace failures, reproduce issues |
| **Analytics** | Usage patterns, task types, error rates |
| **Compliance** | Audit trail for enterprise requirements |

### Key Characteristics

| Aspect | Value |
|--------|-------|
| **Storage Type** | Persistent, append-only |
| **Data Flow** | Unidirectional (capture only) |
| **Latency** | Async (non-blocking to inference) |
| **Retention** | Configurable (default: 90 days) |

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Request Flow                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   VS Code ─────▶ llm-gateway ─────▶ inference-service               │
│                     │ :8080              :8085                       │
│                     │                       │                        │
│                     │  (1) Capture          │ (2) Return metadata   │
│                     ▼   request             ▼                        │
│              ┌─────────────────┐     ┌─────────────┐                │
│              │ Conversation    │◀────│ inference   │                │
│              │ Store           │     │ metadata    │                │
│              │ (audit-service) │     └─────────────┘                │
│              │     :8084       │                                    │
│              └────────┬────────┘                                    │
│                       │                                              │
│                       ▼                                              │
│              ┌─────────────────┐                                    │
│              │ SQLite / Postgres│                                   │
│              │ (Persistent)     │                                   │
│              └─────────────────┘                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

| Service | Responsibility | Cache Type |
|---------|---------------|------------|
| **inference-service** | Operational caching (handoffs, compression) | Ephemeral |
| **audit-service** | Conversation storage, reporting, analytics | Persistent |
| **llm-gateway** | Request capture, forwarding to audit | Pass-through |

---

## Data Model

### Conversation

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

class Conversation(BaseModel):
    """A conversation session with one or more messages."""
    
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    # Context
    client: Literal["vscode", "web", "api", "cli"]
    workspace: Optional[str]          # VS Code workspace path
    project: Optional[str]            # Project identifier
    user_id: Optional[str]            # User identifier (if auth enabled)
    session_id: Optional[str]         # Client session ID
    
    # Aggregates (denormalized for queries)
    message_count: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: int = 0
    models_used: list[str] = []
    configs_used: list[str] = []
    
    # Status
    status: Literal["active", "completed", "error"] = "active"
    last_error: Optional[str] = None
```

### ConversationMessage

```python
class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    
    id: UUID
    conversation_id: UUID
    sequence: int                     # Order in conversation
    timestamp: datetime
    
    # Source
    role: Literal["system", "user", "assistant", "tool"]
    source: Literal["user", "model", "system", "tool"]
    
    # Content
    content: str
    content_type: Literal["text", "code", "markdown", "json"] = "text"
    
    # For tool calls
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    
    # Inference metadata (from inference-service response)
    model_used: Optional[str] = None
    config_used: Optional[str] = None       # D3, T1, etc.
    orchestration_mode: Optional[str] = None
    models_in_chain: Optional[list[str]] = None
    
    # Metrics
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None
    context_utilization: Optional[float] = None
    
    # Quality signals
    task_type: Optional[str] = None         # code, explain, debug, etc.
    handoff_steps: Optional[int] = None
    compression_applied: Optional[bool] = None
    
    # Error tracking
    error: Optional[str] = None
    error_type: Optional[str] = None
```

### InferenceMetadata

```python
class InferenceMetadata(BaseModel):
    """Metadata returned by inference-service for capture."""
    
    config: str                        # Configuration preset used
    orchestration_mode: str            # single, critique, debate, etc.
    models_used: list[str]             # Models in the chain
    handoff_steps: int                 # Number of model-to-model handoffs
    compression_applied: bool          # Whether context was compressed
    context_utilization: float         # 0.0-1.0, how much context was used
    
    # Timing breakdown
    total_latency_ms: int
    model_latencies: dict[str, int]    # Per-model timing
    
    # Token accounting
    tokens_by_model: dict[str, dict[str, int]]  # {model: {in, out}}
```

---

## Capture Flow

### 1. Request Capture (llm-gateway)

```python
async def capture_request(
    request: ChatCompletionRequest,
    conversation_id: UUID,
    sequence: int
) -> None:
    """Capture incoming request (async, non-blocking)."""
    
    message = ConversationMessage(
        id=uuid4(),
        conversation_id=conversation_id,
        sequence=sequence,
        timestamp=datetime.utcnow(),
        role="user",
        source="user",
        content=request.messages[-1].content,
        task_type=request.task_type,
    )
    
    await audit_client.store_message(message)
```

### 2. Response Capture (llm-gateway)

```python
async def capture_response(
    response: ChatCompletionResponse,
    conversation_id: UUID,
    sequence: int,
    latency_ms: int
) -> None:
    """Capture response with inference metadata."""
    
    metadata = response.inference_metadata
    
    message = ConversationMessage(
        id=uuid4(),
        conversation_id=conversation_id,
        sequence=sequence,
        timestamp=datetime.utcnow(),
        role="assistant",
        source="model",
        content=response.choices[0].message.content,
        
        # Inference metadata
        model_used=response.model,
        config_used=metadata.config,
        orchestration_mode=metadata.orchestration_mode,
        models_in_chain=metadata.models_used,
        
        # Metrics
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        latency_ms=latency_ms,
        context_utilization=metadata.context_utilization,
        handoff_steps=metadata.handoff_steps,
        compression_applied=metadata.compression_applied,
    )
    
    await audit_client.store_message(message)
```

### 3. Error Capture

```python
async def capture_error(
    conversation_id: UUID,
    sequence: int,
    error: Exception,
    request: ChatCompletionRequest
) -> None:
    """Capture error for debugging."""
    
    message = ConversationMessage(
        id=uuid4(),
        conversation_id=conversation_id,
        sequence=sequence,
        timestamp=datetime.utcnow(),
        role="system",
        source="system",
        content=f"Error: {str(error)}",
        error=str(error),
        error_type=type(error).__name__,
    )
    
    await audit_client.store_message(message)
```

---

## Storage Options

### Option A: SQLite (Recommended for POC)

```sql
-- conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    client TEXT NOT NULL,
    workspace TEXT,
    project TEXT,
    user_id TEXT,
    session_id TEXT,
    message_count INTEGER DEFAULT 0,
    total_tokens_in INTEGER DEFAULT 0,
    total_tokens_out INTEGER DEFAULT 0,
    total_latency_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    last_error TEXT
);

-- messages table
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    role TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',
    model_used TEXT,
    config_used TEXT,
    orchestration_mode TEXT,
    models_in_chain TEXT,  -- JSON array
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    context_utilization REAL,
    task_type TEXT,
    handoff_steps INTEGER,
    compression_applied BOOLEAN,
    error TEXT,
    error_type TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Indexes for common queries
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_conversations_workspace ON conversations(workspace);
CREATE INDEX idx_conversations_created ON conversations(created_at);
```

### Option B: PostgreSQL (Production)

Same schema, with:
- JSONB for `models_in_chain`
- UUID native type
- Partitioning by date for large volumes

### Option C: JSONL Files (Simplest)

```
conversations/
├── 2025-12-27/
│   ├── conv_abc123.jsonl
│   ├── conv_def456.jsonl
│   └── index.json
└── 2025-12-28/
    └── ...
```

Each `.jsonl` file contains one message per line, easy to stream and parse.

---

## API Contract

### Endpoints (audit-service)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/conversations` | Create new conversation |
| POST | `/conversations/{id}/messages` | Add message to conversation |
| GET | `/conversations` | List conversations (paginated) |
| GET | `/conversations/{id}` | Get conversation with messages |
| GET | `/conversations/{id}/messages` | Get messages only |
| GET | `/analytics/usage` | Token usage stats |
| GET | `/analytics/models` | Model performance stats |
| GET | `/analytics/errors` | Error analysis |

### Create Conversation

```json
POST /conversations
{
  "client": "vscode",
  "workspace": "/Users/kevintoles/POC/ai-platform-data",
  "session_id": "sess_abc123"
}

Response:
{
  "id": "conv_xyz789",
  "created_at": "2025-12-27T10:30:00Z",
  "status": "active"
}
```

### Add Message

```json
POST /conversations/conv_xyz789/messages
{
  "role": "assistant",
  "source": "model",
  "content": "Here's how to implement...",
  "model_used": "phi-4",
  "config_used": "D3",
  "orchestration_mode": "debate",
  "models_in_chain": ["phi-4", "deepseek-r1-7b"],
  "tokens_in": 150,
  "tokens_out": 500,
  "latency_ms": 2340,
  "context_utilization": 0.72
}
```

### Query Conversations

```json
GET /conversations?workspace=/Users/kevintoles/POC&limit=20&offset=0

Response:
{
  "conversations": [...],
  "total": 156,
  "limit": 20,
  "offset": 0
}
```

---

## Reporting & Analytics

### Available Reports

| Report | Query | Use Case |
|--------|-------|----------|
| Token usage by model | `SUM(tokens) GROUP BY model_used` | Cost tracking |
| Token usage by config | `SUM(tokens) GROUP BY config_used` | Config optimization |
| Latency by orchestration mode | `AVG(latency_ms) GROUP BY orchestration_mode` | Performance tuning |
| Error rate by model | `COUNT(error) / COUNT(*) GROUP BY model_used` | Reliability |
| Task type distribution | `COUNT(*) GROUP BY task_type` | Usage patterns |
| Context utilization | `AVG(context_utilization) GROUP BY model_used` | Capacity planning |
| Compression frequency | `SUM(compression_applied) / COUNT(*)` | Context pressure |
| Daily active conversations | `COUNT(DISTINCT conversation_id) GROUP BY DATE(timestamp)` | Engagement |

### Example Analytics Query

```sql
-- Model performance comparison
SELECT 
    model_used,
    config_used,
    COUNT(*) as requests,
    AVG(latency_ms) as avg_latency,
    SUM(tokens_in) as total_tokens_in,
    SUM(tokens_out) as total_tokens_out,
    AVG(context_utilization) as avg_context_util,
    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
FROM messages
WHERE timestamp > datetime('now', '-7 days')
  AND role = 'assistant'
GROUP BY model_used, config_used
ORDER BY requests DESC;
```

### Dashboard Metrics

```python
class DashboardMetrics(BaseModel):
    """Real-time dashboard metrics."""
    
    # Today
    conversations_today: int
    messages_today: int
    tokens_today: int
    
    # Performance
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    
    # Usage breakdown
    by_model: dict[str, int]          # requests per model
    by_config: dict[str, int]         # requests per config
    by_task_type: dict[str, int]      # requests per task type
    
    # Context health
    avg_context_utilization: float
    compression_rate: float
```

---

## Retention & Privacy

### Retention Policy

| Data Type | Default Retention | Configurable |
|-----------|-------------------|--------------|
| Full messages | 90 days | Yes |
| Aggregated stats | 1 year | Yes |
| Error logs | 30 days | Yes |

### Privacy Controls

```python
class PrivacySettings(BaseModel):
    """Privacy configuration for conversation store."""
    
    # Content handling
    store_content: bool = True              # Store message content
    redact_pii: bool = False                # Auto-redact PII patterns
    hash_user_id: bool = True               # Hash user identifiers
    
    # Retention
    retention_days: int = 90
    auto_delete: bool = True
    
    # Access
    require_auth: bool = False
    allowed_roles: list[str] = ["admin", "developer"]
```

### Data Cleanup

```python
async def cleanup_expired_conversations(retention_days: int = 90):
    """Delete conversations older than retention period."""
    
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    
    await db.execute("""
        DELETE FROM messages 
        WHERE conversation_id IN (
            SELECT id FROM conversations 
            WHERE created_at < ?
        )
    """, [cutoff])
    
    await db.execute("""
        DELETE FROM conversations 
        WHERE created_at < ?
    """, [cutoff])
```

---

## Implementation Notes

### For VS Code Extension

The VS Code extension (or llm-gateway) should:

1. Generate `conversation_id` on first message
2. Include `session_id` for grouping related conversations
3. Pass `workspace` path for context
4. Send capture requests async (don't block inference)

### For inference-service

inference-service should:

1. Return `inference_metadata` in every response
2. NOT store conversations (not its concern)
3. NOT block on audit capture

### For audit-service

audit-service should:

1. Accept messages async via queue or HTTP
2. Store immediately, aggregate lazily
3. Provide query APIs for reporting
4. Handle retention/cleanup automatically

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-27 | Initial design |
