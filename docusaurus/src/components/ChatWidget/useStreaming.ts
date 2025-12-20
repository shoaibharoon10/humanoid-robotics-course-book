/**
 * Custom hook for handling SSE streaming from the chat API.
 */

import { useCallback, useRef } from 'react';
import type { Source, StreamEvent } from './types';

interface StreamingOptions {
  onToken: (token: string) => void;
  onSources: (sources: Source[]) => void;
  onComplete: (messageId: string, conversationId: string) => void;
  onError: (error: string) => void;
}

export function useStreaming(apiUrl: string) {
  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (
    message: string,
    conversationId: string | null,
    sessionId: string,
    options: StreamingOptions
  ) => {
    // Cancel any existing request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${apiUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
          session_id: sessionId,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data: StreamEvent = JSON.parse(line.slice(6));

              switch (data.type) {
                case 'token':
                  if (data.content) {
                    options.onToken(data.content);
                  }
                  break;
                case 'sources':
                  if (data.data) {
                    options.onSources(data.data);
                  }
                  break;
                case 'done':
                  if (data.message_id && data.conversation_id) {
                    options.onComplete(data.message_id, data.conversation_id);
                  }
                  break;
                case 'error':
                  if (data.message) {
                    options.onError(data.message);
                  }
                  break;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return; // Request was cancelled
      }
      options.onError(error instanceof Error ? error.message : 'Unknown error');
    }
  }, [apiUrl]);

  const cancelRequest = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  return { sendMessage, cancelRequest };
}
