/**
 * Swizzled Layout component to add ChatWidget inside the ColorModeProvider context.
 *
 * This component wraps the original Layout and adds the ChatWidget
 * which requires access to useColorMode hook.
 * ChatWidget is only rendered for authenticated users.
 */

import React, { useState, useEffect } from 'react';
import Layout from '@theme-original/Layout';
import type LayoutType from '@theme/Layout';
import type { WrapperProps } from '@docusaurus/types';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Cookies from 'js-cookie';
import axios from 'axios';

type Props = WrapperProps<typeof LayoutType>;

const API_URL = 'https://humanoid-robotics-course-book-production.up.railway.app/api/v1';
const TOKEN_COOKIE = 'auth_token';

// ChatWidget loader - only rendered in browser for authenticated users
function ChatWidgetLoader(): JSX.Element | null {
  const { siteConfig } = useDocusaurusContext();
  const chatApiUrl = (siteConfig.customFields?.chatApiUrl as string) || 'https://humanoid-robotics-course-book-production.up.railway.app';
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Check authentication status
  useEffect(() => {
    const checkAuth = async () => {
      const token = Cookies.get(TOKEN_COOKIE);
      if (!token) {
        setIsLoading(false);
        return;
      }

      try {
        await axios.get(`${API_URL}/me`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setIsAuthenticated(true);
      } catch {
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Don't render while checking auth
  if (isLoading) {
    return null;
  }

  // Only render ChatWidget for authenticated users
  if (!isAuthenticated) {
    return null;
  }

  // Dynamically import ChatWidget to avoid SSR issues
  const ChatWidget = React.lazy(() => import('@site/src/components/ChatWidget'));

  return (
    <React.Suspense fallback={null}>
      <ChatWidget
        config={{
          apiUrl: chatApiUrl,
          title: 'Robotics Assistant',
          placeholder: 'Ask about humanoid robotics...',
          welcomeMessage: 'Hello! I can help you understand concepts from the Physical AI & Humanoid Robotics textbook. What would you like to learn about?'
        }}
      />
    </React.Suspense>
  );
}

export default function LayoutWrapper(props: Props): JSX.Element {
  return (
    <>
      <Layout {...props} />
      <BrowserOnly fallback={null}>
        {() => <ChatWidgetLoader />}
      </BrowserOnly>
    </>
  );
}
