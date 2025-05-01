import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.base.llms.types import ChatMessage, ContentBlock
from llama_index.core.bridge.pydantic import BaseModel, Field, model_validator
from llama_index.core.memory.chat_memory_buffer import (
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_CHAT_STORE_KEY
)
from llama_index.core.memory.types import BaseChatStoreMemory
from llama_index.core.prompts import BasePromptTemplate, RichPromptTemplate
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer

DEFAULT_PRESSURE_SIZE = 5
DEFAULT_MEMORY_BLOCKS_TEMPLATE = RichPromptTemplate(
"""
<memory>
{% for (block_name, block_content) in memory_blocks %}
<{{ block_name }}>
{block_content}
</{{ block_name }}>
{% endfor %}
</memory>
"""
)

class InsertMethod(Enum):
    SYSTEM = "system"
    USER = "user"


class BaseMemoryBlock(BaseModel):

    name: str = Field(description="The name/identifier of the memory block.")
    description: Optional[str] = Field(default=None, description="A description of the memory block.")
    priority: int = Field(default=0, description="Priority of this memory block (0 = never truncate, 1 = highest priority, etc.).")

    @abstractmethod
    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> str:
        """Pull the memory block (async)."""

    @abstractmethod   
    async def aput(self, messages: List[ChatMessage]) -> None:
        """Push to the memory block (async)."""
        
    async def atruncate(self, content: str, max_tokens: int) -> str:
        """Truncate the memory block content to the given token limit.
        
        By default, truncation will remove the entire block content.
        """
        return ""


class WaterfallMemory(BaseChatStoreMemory):
    """A waterfall memory module.
    
    Works by orchestrating around
    - a FIFO queue of messages
    - a list of memory blocks
    - various parameters (pressure size, token limit, etc.)

    When the FIFO queue reaches the token limit, the oldest messages within the pressure size are ejected from the FIFO queue.
    The messages are then processed by each memory block.

    When pulling messages from this memory, the memory blocks are processed in order, and the messages are injected into the system message or the latest user message.
    """

    token_limit: int = Field(default=DEFAULT_TOKEN_LIMIT)
    pressure_token_limit: int = Field(default=DEFAULT_PRESSURE_SIZE, description="The token limit of the pressure size. When the token limit is reached, the oldest messages within the pressure size are ejected from the FIFO queue.")
    memory_blocks: List[BaseMemoryBlock] = Field(default_factory=list)
    memory_blocks_template: RichPromptTemplate = Field(default=DEFAULT_MEMORY_BLOCKS_TEMPLATE)
    insert_method: InsertMethod = Field(
        default=InsertMethod.SYSTEM, 
        description="Whether to inject memory blocks into the system message or into the latest user message."
    )
    tokenizer_fn: Callable[[str], List] = Field(
        default_factory=get_tokenizer,
        exclude=True,
    )
    chat_history_min_token_ratio: float = Field(
        default=0.7,
        description="Minimum percentage ratio of total token limit reserved for chat history."
    )

    @classmethod
    def class_name(cls) -> str:
        return "WaterfallMemory"

    @model_validator(mode="before")
    @classmethod
    def validate_memory(cls, values: dict) -> dict:
        # Validate token limit like ChatMemoryBuffer
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")
        
        tokenizer_fn = values.get("tokenizer_fn", None)
        if tokenizer_fn is None:
            values["tokenizer_fn"] = get_tokenizer()

        return values

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[List[ChatMessage]] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: int = DEFAULT_TOKEN_LIMIT,
        memory_blocks: Optional[List[BaseMemoryBlock]] = None,
        tokenizer_fn: Optional[Callable[[str], List]] = None,
        chat_history_min_token_ratio: float = 0.7,
    ) -> "WaterfallMemory":
        """Initialize WaterfallMemory."""

        chat_store = chat_store or SimpleChatStore()
        if chat_history is not None:
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            chat_store=chat_store,
            chat_store_key=chat_store_key,
            memory_blocks=memory_blocks or [],
            chat_history_min_token_ratio=chat_history_min_token_ratio,
        )
    
    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        # Get chat history
        chat_history = self.get_all()
        
        # Calculate token limits
        max_memory_blocks_tokens = int(self.token_limit * self.memory_blocks_max_token_percentage)
        min_chat_history_tokens = int(self.token_limit * self.chat_history_min_token_percentage)
        
        # Get memory blocks content and sort by priority
        memory_blocks_with_content = []
        for memory_block in sorted(self.memory_blocks, key=lambda x: -x.priority):  # Sort by priority (highest first)
            content = await memory_block.aget(input, **kwargs)
            if content:
                memory_blocks_with_content.append((memory_block, content))
        
        # Calculate tokens for each memory block
        memory_blocks_data = []
        total_memory_blocks_tokens = 0
        
        # Add blocks by priority until we reach the limit
        for block, content in memory_blocks_with_content:
            # Calculate content tokens
            content_tokens = len(self.tokenizer_fn(content))
            
            # Check if adding this block would exceed the block's percentage limit
            block_max_tokens = int(self.token_limit * block.max_token_percentage)
            if content_tokens > block_max_tokens:
                # Truncate content to fit within the block's limit
                content = await block.atruncate(block_max_tokens)
                content_tokens = len(self.tokenizer_fn(content))
            
            # Check if adding this block would exceed total memory blocks limit
            if total_memory_blocks_tokens + content_tokens > max_memory_blocks_tokens:
                # If this is a high priority block, try to make room by removing lower priority blocks
                if len(memory_blocks_data) > 0 and block.priority > memory_blocks_data[-1][0].priority:
                    # Remove lower priority blocks until we have room
                    while (memory_blocks_data and 
                           total_memory_blocks_tokens + content_tokens > max_memory_blocks_tokens and
                           memory_blocks_data[-1][0].priority < block.priority):
                        removed_block, removed_content = memory_blocks_data.pop()
                        removed_tokens = len(self.tokenizer_fn(removed_content))
                        total_memory_blocks_tokens -= removed_tokens
                    
                    # If we still don't have room, truncate this block
                    if total_memory_blocks_tokens + content_tokens > max_memory_blocks_tokens:
                        remaining_tokens = max_memory_blocks_tokens - total_memory_blocks_tokens
                        if remaining_tokens > 0:
                            content = await block.atruncate(remaining_tokens)
                            content_tokens = len(self.tokenizer_fn(content))
                        else:
                            # No room at all, skip this block
                            continue
                else:
                    # Lower or equal priority, so skip if no room
                    continue
            
            # Add block to the list
            memory_blocks_data.append((block, content))
            total_memory_blocks_tokens += content_tokens
        
        # Format memory blocks using template
        memory_blocks_template_data = [(block.name, content) for block, content in memory_blocks_data]
        memory_blocks_str = self.memory_blocks_template.format(memory_blocks=memory_blocks_template_data)
        
        # Recalculate actual memory blocks tokens after formatting
        memory_blocks_tokens = len(self.tokenizer_fn(memory_blocks_str))
        
        # Get messages that fit within the remaining token limit
        remaining_token_limit = max(min_chat_history_tokens, self.token_limit - memory_blocks_tokens)
        messages = self._get_messages_within_token_limit(chat_history, remaining_token_limit)
        
        # Insert memory blocks according to the specified method
        if self.insert_method == InsertMethod.SYSTEM:
            # Find system message or create a new one
            system_idx = next((i for i, msg in enumerate(messages) if msg.role == "system"), None)
            
            if system_idx is not None:
                # Update existing system message
                system_msg = messages[system_idx]
                system_content = str(system_msg.content) if system_msg.content else ""
                messages[system_idx] = ChatMessage(
                    role="system",
                    content=f"{system_content}\n\n{memory_blocks_str}"
                )
            else:
                # Create new system message at the beginning
                messages.insert(0, ChatMessage(role="system", content=memory_blocks_str))
        
        elif self.insert_method == InsertMethod.USER:
            # Find the latest user message
            user_idx = next((i for i, msg in enumerate(reversed(messages)) if msg.role == "user"), None)
            
            if user_idx is not None:
                # Adjust index for reversed search
                user_idx = len(messages) - 1 - user_idx
                # Update user message
                user_msg = messages[user_idx]
                user_content = str(user_msg.content) if user_msg.content else ""
                messages[user_idx] = ChatMessage(
                    role="user",
                    content=f"{user_content}\n\n{memory_blocks_str}"
                )
        
        return messages
    
    async def aput(self, message: ChatMessage) -> None:
        # Add the message to the chat store
        await super().aput(message)
        
        # Get all messages from chat store
        all_messages = self.get_all()
        
        # Calculate total token count
        total_tokens = self._token_count_for_messages(all_messages)
        
        # Check if we need to waterfall messages
        if total_tokens > self.token_limit:
            # Determine how many messages to waterfall based on pressure size
            messages_to_remove = []
            tokens_to_remove = 0
            
            # Start from the oldest messages
            for msg in all_messages:
                msg_tokens = self._token_count_for_messages([msg])
                messages_to_remove.append(msg)
                tokens_to_remove += msg_tokens
                
                # Stop when we've removed enough tokens or reached pressure size
                if tokens_to_remove >= total_tokens - self.token_limit or len(messages_to_remove) >= self.pressure_token_limit:
                    break
            
            # Remove messages from chat store
            remaining_messages = all_messages[len(messages_to_remove):]
            self.chat_store.set_messages(self.chat_store_key, remaining_messages)
            
            # Process the removed messages in memory blocks
            await asyncio.gather(*[
                memory_block.aput(messages_to_remove)
                for memory_block in self.memory_blocks
            ])
    
    def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        if len(messages) <= 0:
            return 0

        msg_str = " ".join(str(m.content) for m in messages)
        return len(self.tokenizer_fn(msg_str))
    
    def _get_messages_within_token_limit(self, messages: List[ChatMessage], token_limit: int) -> List[ChatMessage]:
        """Get as many messages as possible from the end of the list within the token limit."""
        if not messages:
            return []
            
        result = []
        total_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self._token_count_for_messages([msg])
            
            if total_tokens + msg_tokens > token_limit:
                break
                
            result.insert(0, msg)
            total_tokens += msg_tokens
            
        return result


# Example memory block implementations
class RetrievalMemoryBlock(BaseMemoryBlock):
    """Simple memory block that performs "retrieval" over chat history."""
    
    name: str = "retrieval"
    description: str = "Retrieves relevant information based on user query"
    priority: int = 3  # Higher priority since retrieval is usually most relevant
    max_token_percentage: float = 0.2
    stored_messages: List[str] = Field(default_factory=list)
    
    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> str:
        """Simulate retrieving relevant information based on input."""
        if not input or not self.stored_messages:
            return "No relevant information found."
            
        # Simple keyword matching to simulate retrieval
        relevant_messages = [
            msg for msg in self.stored_messages 
            if input.lower() in msg.lower()
        ]
        
        if relevant_messages:
            return f"Retrieved context: {' '.join(relevant_messages)}"
        return "No relevant information found for your query."
    
    async def aput(self, messages: List[ChatMessage]) -> None:
        """Store messages for later retrieval."""
        for message in messages:
            if message.content:
                self.stored_messages.append(str(message.content))
    
    async def atruncate(self, max_tokens: int) -> str:
        """Smart truncation that keeps the most relevant content."""
        content = await self.aget()
        tokenizer_fn = get_tokenizer()
        tokens = tokenizer_fn(content)
        
        if len(tokens) <= max_tokens:
            return content
            
        # For retrieval, prioritize most recent relevant messages
        if "Retrieved context:" in content:
            prefix = "Retrieved context: "
            context_only = content.replace(prefix, "")
            messages = context_only.split()
            
            # Keep as many full messages as possible within token limit
            truncated_messages = []
            total_tokens = len(tokenizer_fn(prefix))
            
            for msg in messages:
                msg_tokens = len(tokenizer_fn(msg + " "))
                if total_tokens + msg_tokens <= max_tokens:
                    truncated_messages.append(msg)
                    total_tokens += msg_tokens
                else:
                    break
                    
            if truncated_messages:
                return f"{prefix}{' '.join(truncated_messages)}"
                
        # Fall back to basic truncation
        return super().atruncate(max_tokens)


class SummaryMemoryBlock(BaseMemoryBlock):
    """Memory block that maintains a running summary of the conversation."""
    
    name: str = "summary"
    description: str = "Provides a summary of the conversation history"
    priority: int = 2  # Medium priority
    max_token_percentage: float = 0.15
    summary: str = Field(default="No conversation history yet.")
    
    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> str:
        """Return the current summary."""
        return f"Conversation summary: {self.summary}"
    
    async def aput(self, messages: List[ChatMessage]) -> None:
        """Update summary with new messages (simplified for demonstration)."""
        if not messages:
            return
            
        # Simplified summary update - in reality would use an LLM
        new_content = " ".join([str(m.content) for m in messages if m.content])
        if new_content:
            # Simulate summarization by keeping it short
            self.summary = f"{self.summary} + new messages about: {new_content[:50]}..."
    
    async def atruncate(self, max_tokens: int) -> str:
        """Truncate summary while preserving the most important information."""
        content = await self.aget()
        tokenizer_fn = get_tokenizer()
        tokens = tokenizer_fn(content)
        
        if len(tokens) <= max_tokens:
            return content
            
        # For summary, we want to keep the prefix and most recent summary content
        prefix = "Conversation summary: "
        prefix_tokens = len(tokenizer_fn(prefix))
        summary_tokens = max_tokens - prefix_tokens
        
        if summary_tokens <= 0:
            return prefix
            
        # Try to preserve meaningful chunks of the summary
        summary_content = self.summary
        if len(tokenizer_fn(summary_content)) > summary_tokens:
            # Just keep the most recent part if we need to truncate
            parts = summary_content.split(" + ")
            truncated_parts = []
            total_tokens = 0
            
            # Start from the most recent parts
            for part in reversed(parts):
                part_tokens = len(tokenizer_fn(part + (" + " if truncated_parts else "")))
                if total_tokens + part_tokens <= summary_tokens:
                    truncated_parts.insert(0, part)
                    total_tokens += part_tokens
                else:
                    break
                    
            summary_content = " + ".join(truncated_parts)
            
        return f"{prefix}{summary_content}"


class StaticMemoryBlock(BaseMemoryBlock):
    """Memory block that always returns the same content."""
    
    name: str = "static"
    description: str = "Provides static information regardless of input"
    priority: int = 1  # Lowest priority
    max_token_percentage: float = 0.1
    static_content: str = Field(default="This is static information that's always available.")
    
    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> str:
        """Always return the static content."""
        return self.static_content
    
    async def aput(self, messages: List[ChatMessage]) -> None:
        """Static block doesn't need to store anything."""
        pass
    
    async def atruncate(self, max_tokens: int) -> str:
        """For static content, we either include it all or truncate with an ellipsis."""
        content = self.static_content
        tokenizer_fn = get_tokenizer()
        tokens = tokenizer_fn(content)
        
        if len(tokens) <= max_tokens:
            return content
            
        # If we need to truncate, try to keep complete sentences
        if max_tokens < 10:  # Too small to be meaningful
            return content[:max_tokens]
            
        # Try to truncate at sentence boundaries
        truncated = ""
        sentences = content.split(". ")
        for i, sentence in enumerate(sentences):
            next_piece = sentence + (". " if i < len(sentences) - 1 else "")
            if len(tokenizer_fn(truncated + next_piece + "...")) <= max_tokens:
                truncated += next_piece
            else:
                break
                
        return truncated + "..."


async def main():
    """Example usage of WaterfallMemory with different memory blocks."""
    # Create the memory blocks with different priorities and token percentages
    retrieval_block = RetrievalMemoryBlock(
        priority=3,
        max_token_percentage=0.25,
        description="Retrieves context from conversation history based on query"
    )
    
    summary_block = SummaryMemoryBlock(
        priority=2,
        max_token_percentage=0.15,
        description="Maintains and provides a summary of the conversation"
    )
    
    static_block = StaticMemoryBlock(
        priority=1,
        max_token_percentage=0.1,
        static_content="I am an AI assistant that helps with information retrieval and summarization."
    )

    # Custom template for memory blocks
    memory_block_template = RichPromptTemplate(
"""
<memory>
{% for (block_name, block_content) in memory_blocks %}
<{{ block_name }}>
{{ block_content }}
</{{ block_name }}>
{% endfor %}
</memory>
"""
    )
    
    # Initialize WaterfallMemory with the blocks and token management settings
    memory = WaterfallMemory(
        token_limit=30000,
        pressure_token_limit=5000,
        memory_blocks=[retrieval_block, summary_block, static_block],
        memory_blocks_template=memory_block_template,
        memory_blocks_max_token_percentage=0.4,  # Memory blocks can use up to 40% of token limit
        chat_history_min_token_percentage=0.5,   # At least 50% reserved for chat history
        insert_method=InsertMethod.SYSTEM        # Insert memories into system message
    )
    
    # Simulate adding messages to the memory
    user_message = ChatMessage(role="user", content="What is the weather like today?")
    await memory.aput(user_message)
    
    assistant_message = ChatMessage(
        role="assistant", 
        content="I don't have real-time weather information. To get accurate weather details, you should check a weather website or app."
    )
    await memory.aput(assistant_message)
    
    # Add more messages to trigger waterfall
    for i in range(10):
        await memory.aput(ChatMessage(role="user", content=f"Test message {i} to fill the queue"))
        await memory.aput(ChatMessage(role="assistant", content=f"Response to test message {i}"))
    
    # Get messages with memory blocks included
    result = await memory.aget(input="weather")
    
    # Print result structure
    print(f"Retrieved {len(result)} messages with memory blocks")
    for i, msg in enumerate(result):
        print(f"Message {i}: role={msg.role}, content_length={len(str(msg.content))}")
        if msg.role == "system" and "<memory>" in str(msg.content):
            print("  - Contains memory blocks")

if __name__ == "__main__":
    asyncio.run(main())