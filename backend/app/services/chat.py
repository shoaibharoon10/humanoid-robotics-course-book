"""OpenAI chat completion service with streaming."""

from typing import List, AsyncGenerator, Optional
from openai import AsyncOpenAI

from app.config import get_settings
from app.models.schemas import Source


SYSTEM_PROMPT = """You are an expert AI teaching assistant for the "Physical AI & Humanoid Robotics" textbook.
Your role is to help students understand concepts about:
- ROS 2 (Robot Operating System)
- Digital twins and physics simulation
- NVIDIA Isaac Sim and perception
- Vision-Language-Action (VLA) models
- LLM integration with robotics

Guidelines:
1. Answer questions based on the provided context from the textbook
2. Be concise but thorough in your explanations
3. Use code examples when helpful (Python, ROS 2 commands)
4. If the context doesn't contain relevant information, say so honestly
5. Encourage further exploration of related topics in the textbook
6. Use technical terminology appropriately for robotics students

Context from the textbook will be provided in the user message."""


class ChatService:
    """Service for chat completions using OpenAI."""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

    async def generate_response(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """Generate a non-streaming response."""
        messages = self._build_messages(user_message, context, conversation_history)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content

    async def generate_stream(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        messages = self._build_messages(user_message, context, conversation_history)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def get_usage(
        self,
        user_message: str,
        context: str,
        response: str
    ) -> int:
        """Estimate token usage for a conversation."""
        # Simple estimation based on character count
        # More accurate would be to use tiktoken
        total_chars = len(SYSTEM_PROMPT) + len(context) + len(user_message) + len(response)
        return total_chars // 4  # Rough estimate: 4 chars per token

    def _build_messages(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> List[dict]:
        """Build the messages array for the API call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current message with context
        user_content = f"""Based on the following context from the textbook, please answer my question.

CONTEXT:
{context}

QUESTION:
{user_message}"""

        messages.append({"role": "user", "content": user_content})

        return messages


# Singleton instance
_chat_service = None


def get_chat_service() -> ChatService:
    """Get the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
