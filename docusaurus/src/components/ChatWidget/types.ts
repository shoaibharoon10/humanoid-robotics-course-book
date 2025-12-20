/**
 * TypeScript types for ChatWidget component.
 */

export interface Source {
  title: string;
  url_path: string;
  section?: string;
  relevance_score: number;
  snippet: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatState {
  messages: Message[];
  isOpen: boolean;
  isLoading: boolean;
  conversationId: string | null;
  sessionId: string;
  error: string | null;
}

export interface StreamEvent {
  type: 'token' | 'sources' | 'done' | 'error';
  content?: string;
  data?: Source[];
  message_id?: string;
  conversation_id?: string;
  message?: string;
}

export interface ChatConfig {
  apiUrl: string;
  title?: string;
  placeholder?: string;
  welcomeMessage?: string;
}
