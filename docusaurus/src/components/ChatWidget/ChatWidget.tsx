/**
 * ChatWidget component - Floating chat interface for RAG chatbot.
 * Note: Uses CSS-only dark mode detection to avoid ColorModeProvider dependency.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import styles from './styles.module.css';
import { useStreaming } from './useStreaming';
import type { Message, Source, ChatConfig } from './types';

// Generate a unique session ID
function generateSessionId(): string {
  return 'session_' + Math.random().toString(36).substring(2, 15);
}

// Get or create session ID from localStorage
function getSessionId(): string {
  if (typeof window === 'undefined') return generateSessionId();

  let sessionId = localStorage.getItem('chat_session_id');
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem('chat_session_id', sessionId);
  }
  return sessionId;
}

// Default configuration - apiUrl should be passed from Layout.tsx via customFields
const defaultConfig: ChatConfig = {
  apiUrl: 'https://humanoid-robotics-course-book-production.up.railway.app',
  title: 'Robotics Assistant',
  placeholder: 'Ask about humanoid robotics...',
  welcomeMessage: 'Hello! I can help you understand concepts from the Physical AI & Humanoid Robotics textbook. What would you like to learn about?'
};

interface ChatWidgetProps {
  config?: Partial<ChatConfig>;
}

export default function ChatWidget({ config = {} }: ChatWidgetProps): JSX.Element {
  const mergedConfig = { ...defaultConfig, ...config };

  // State
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId] = useState(getSessionId);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { sendMessage, cancelRequest } = useStreaming(mergedConfig.apiUrl);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  // Add welcome message on first open
  useEffect(() => {
    if (isOpen && messages.length === 0 && mergedConfig.welcomeMessage) {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: mergedConfig.welcomeMessage,
        timestamp: new Date()
      }]);
    }
  }, [isOpen, messages.length, mergedConfig.welcomeMessage]);

  const handleSend = useCallback(async () => {
    const message = inputValue.trim();
    if (!message || isLoading) return;

    setInputValue('');
    setError(null);

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);

    // Create placeholder for assistant message
    const assistantId = `assistant-${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true
    };
    setMessages(prev => [...prev, assistantMessage]);
    setIsLoading(true);

    // Stream response
    await sendMessage(
      message,
      conversationId,
      sessionId,
      {
        onToken: (token) => {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: msg.content + token }
              : msg
          ));
        },
        onSources: (sources) => {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, sources }
              : msg
          ));
        },
        onComplete: (messageId, newConversationId) => {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, id: messageId, isStreaming: false }
              : msg
          ));
          setConversationId(newConversationId);
          setIsLoading(false);
        },
        onError: (errorMsg) => {
          setError(errorMsg);
          setMessages(prev => prev.filter(msg => msg.id !== assistantId));
          setIsLoading(false);
        }
      }
    );
  }, [inputValue, isLoading, conversationId, sessionId, sendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleNewChat = useCallback(() => {
    cancelRequest();
    setMessages([]);
    setConversationId(null);
    setError(null);
    setIsLoading(false);
  }, [cancelRequest]);

  const toggleChat = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  return (
    <>
      {/* Floating Action Button */}
      <button
        className={`${styles.chatFab} ${isOpen ? styles.hidden : ''}`}
        onClick={toggleChat}
        aria-label="Open chat assistant"
        title="Ask the Robotics Assistant"
      >
        <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
        </svg>
      </button>

      {/* Chat Drawer - dark mode handled via CSS [data-theme='dark'] selector */}
      <div className={`${styles.chatDrawer} ${isOpen ? styles.open : ''}`}>
        {/* Header */}
        <div className={styles.chatHeader}>
          <div className={styles.headerTitle}>
            <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
              <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5A2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5a2.5 2.5 0 0 0 2.5 2.5a2.5 2.5 0 0 0 2.5-2.5a2.5 2.5 0 0 0-2.5-2.5z"/>
            </svg>
            <span>{mergedConfig.title}</span>
          </div>
          <div className={styles.headerActions}>
            <button
              onClick={handleNewChat}
              className={styles.iconButton}
              title="New conversation"
              aria-label="New conversation"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
                <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
              </svg>
            </button>
            <button
              onClick={toggleChat}
              className={styles.iconButton}
              title="Close"
              aria-label="Close chat"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className={styles.messagesContainer}>
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`${styles.message} ${styles[msg.role]}`}
            >
              <div className={styles.messageContent}>
                {msg.content}
                {msg.isStreaming && <span className={styles.cursor}>|</span>}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className={styles.sources}>
                  <div className={styles.sourcesTitle}>Sources:</div>
                  {msg.sources.map((source, i) => (
                    <a
                      key={i}
                      href={source.url_path}
                      className={styles.sourceLink}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {source.title}
                      {source.section && ` - ${source.section}`}
                    </a>
                  ))}
                </div>
              )}
            </div>
          ))}
          {error && (
            <div className={styles.errorMessage}>
              Error: {error}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className={styles.inputContainer}>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mergedConfig.placeholder}
            disabled={isLoading}
            className={styles.input}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
            className={styles.sendButton}
            aria-label="Send message"
          >
            {isLoading ? (
              <div className={styles.spinner} />
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            )}
          </button>
        </div>
      </div>
    </>
  );
}
