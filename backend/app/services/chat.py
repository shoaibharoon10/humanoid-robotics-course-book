"""Google Gemini chat completion service with streaming."""

from typing import List, AsyncGenerator, Optional
import google.generativeai as genai

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
    """Service for chat completions using Google Gemini."""

    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        self.model_name = settings.chat_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

        # Configure generation settings
        self.generation_config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _get_model(self):
        """Get a configured Gemini model instance."""
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=SYSTEM_PROMPT
        )

    async def generate_response(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """Generate a non-streaming response."""
        model = self._get_model()

        # Build conversation content
        contents = self._build_contents(user_message, context, conversation_history)

        # Generate response
        response = model.generate_content(contents)
        return response.text

    async def generate_stream(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        model = self._get_model()

        # Build conversation content
        contents = self._build_contents(user_message, context, conversation_history)

        # Generate streaming response
        response = model.generate_content(contents, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def get_usage(
        self,
        user_message: str,
        context: str,
        response: str
    ) -> int:
        """Estimate token usage for a conversation."""
        # Simple estimation based on character count
        # Gemini doesn't provide exact token counts in free tier
        total_chars = len(SYSTEM_PROMPT) + len(context) + len(user_message) + len(response)
        return total_chars // 4  # Rough estimate: 4 chars per token

    def _build_contents(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> List[dict]:
        """Build the contents array for the API call."""
        contents = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [msg["content"]]
                })

        # Add current message with context
        user_content = f"""Based on the following context from the textbook, please answer my question.

CONTEXT:
{context}

QUESTION:
{user_message}"""

        contents.append({
            "role": "user",
            "parts": [user_content]
        })

        return contents


# Singleton instance
_chat_service = None


def get_chat_service() -> ChatService:
    """Get the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
