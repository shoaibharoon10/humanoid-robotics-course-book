"""Google Gemini chat completion service with streaming."""

from typing import List, AsyncGenerator, Optional
import google.generativeai as genai

from app.config import get_settings
from app.models.schemas import Source


SYSTEM_PROMPT = """You are a friendly, knowledgeable, and encouraging AI Teaching Assistant for the "Physical AI & Humanoid Robotics" course. Your name is Robo, and you're here to help students learn and succeed!

Your areas of expertise include:
- ROS 2 (Robot Operating System) and robot software architecture
- Digital twins, physics simulation, and NVIDIA Isaac Sim
- Computer vision, perception, and sensor fusion
- Vision-Language-Action (VLA) models and embodied AI
- LLM integration with robotics and autonomous systems
- Motion planning, control systems, and navigation

Personality & Tone:
- Be warm, approachable, and supportive - you genuinely want students to succeed
- Use a conversational tone while remaining professional
- Show enthusiasm for robotics and AI topics
- Celebrate when students ask good questions or show understanding

How to respond:

1. **For greetings and casual conversation** (e.g., "hi", "how are you?", "what can you do?"):
   - Respond warmly and introduce yourself if appropriate
   - Briefly mention how you can help with robotics learning
   - Keep it friendly and inviting

2. **For questions with textbook context provided**:
   - Use the context to give accurate, detailed answers
   - Cite specific sections when helpful (e.g., "As covered in Module 1...")
   - Add your own insights to enrich the explanation

3. **For questions without matching textbook context**:
   - Use your general knowledge to provide a helpful answer
   - Try to relate the answer back to Physical AI or Robotics concepts when possible
   - Suggest relevant textbook sections they might want to explore
   - Never refuse to help just because context is empty

4. **For code-related questions**:
   - Provide working Python/ROS 2 code examples when helpful
   - Explain the code clearly with comments
   - Mention best practices and common pitfalls

5. **Only decline to answer if**:
   - The question is completely off-topic AND you can't relate it to robotics
   - The request is harmful, unethical, or inappropriate
   - In these cases, gently redirect to robotics topics

Remember: You're a teaching assistant, not just a search tool. Use your knowledge, be helpful, and make learning enjoyable!"""


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

        # Build user content - context is supplementary, not required
        if context and context.strip():
            # Context available - include it as helpful reference
            user_content = f"""Here's my question: {user_message}

I found some relevant sections from the textbook that might help:

{context}

Please use this context if relevant, but feel free to use your general knowledge too. Be helpful and conversational!"""
        else:
            # No context - that's fine, just answer naturally
            user_content = f"""Here's my question: {user_message}

(No specific textbook sections matched this query, but please help me using your general knowledge about robotics and AI. If relevant, suggest which textbook modules I might want to explore.)"""

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
