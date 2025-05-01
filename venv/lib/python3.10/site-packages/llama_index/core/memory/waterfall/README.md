# WaterfallMemory

WaterfallMemory is an advanced memory module that extends ChatMemoryBuffer with a sophisticated token management system. It orchestrates a FIFO queue of chat messages with specialized memory blocks that store and process information when messages are removed from the queue.

## Key Features

- **Waterfall Mechanism**: When the FIFO queue reaches its token limit, the oldest messages are "waterfall" out of the queue and processed by memory blocks
- **Memory Blocks**: Customizable blocks that distill information from conversations (summaries, retrieval, etc.)
- **Priority-Based Token Management**: Memory blocks have priorities and token allocation limits
- **Flexible Insertion**: Memory block content can be inserted into system messages or the latest user message

## Token Management

WaterfallMemory manages three key token limits:

1. **FIFO Queue Size**: The maximum token count for the chat history queue
2. **Pressure Size**: The batch size of messages removed when the queue exceeds its limit
3. **Overall Context Balance**: How tokens are allocated between chat history and memory blocks

## Usage Example

```python
from llama_index.core.memory.waterfall import WaterfallMemory, RetrievalMemoryBlock, SummaryMemoryBlock

# Create memory blocks with priorities and token allocations
retrieval_block = RetrievalMemoryBlock(
    priority=3,  # Higher priority means this block gets precedence
    max_token_percentage=0.25,  # This block can use up to 25% of total tokens
)

summary_block = SummaryMemoryBlock(
    priority=2,
    max_token_percentage=0.15,
)

# Initialize WaterfallMemory
memory = WaterfallMemory(
    token_limit=4000,  # Max tokens for chat history
    pressure_token_limit=800,  # Batch size for removal
    memory_blocks=[retrieval_block, summary_block],
    memory_blocks_max_token_percentage=0.4,  # Memory blocks can use up to 40% of token limit
    chat_history_min_token_percentage=0.5,  # At least 50% reserved for chat history
)

# Use in conversation
await memory.aput(user_message)
await memory.aput(assistant_message)

# Get messages with memory blocks included
result = await memory.aget(input="query")
```

## Creating Custom Memory Blocks

You can create custom memory blocks by extending the `BaseMemoryBlock` class:

```python
from llama_index.core.memory.waterfall import BaseMemoryBlock

class CustomMemoryBlock(BaseMemoryBlock):
    name: str = "custom"
    priority: int = 2
    max_token_percentage: float = 0.2
    
    async def aget(self, input: Optional[str] = None, **kwargs) -> str:
        # Retrieve relevant information based on input
        return "Custom memory content"
    
    async def aput(self, messages: List[ChatMessage]) -> None:
        # Process and store information from messages
        pass
        
    async def atruncate(self, max_tokens: int) -> str:
        # Custom truncation logic (optional)
        content = await self.aget()
        # Implement smart truncation to fit within max_tokens
        return truncated_content
```

## Token Management Strategy

WaterfallMemory provides a sophisticated approach to token management:

1. **Priority System**: Higher priority blocks are given precedence when token limits are reached
2. **Block-Specific Limits**: Each block has its own maximum percentage of the total token limit
3. **Guaranteed Chat History**: A minimum percentage of tokens is always reserved for chat history
4. **Smart Truncation**: Memory blocks can implement custom truncation logic to fit within limits

This system ensures efficient use of the token context window while preserving the most important information. 