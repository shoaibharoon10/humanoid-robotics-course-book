/**
 * Root theme wrapper for Docusaurus.
 *
 * This component wraps the entire application and adds the ChatWidget
 * as a global floating component. Uses BrowserOnly to prevent SSR issues.
 */

import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

interface RootProps {
  children: React.ReactNode;
}

// Lazy-loaded ChatWidget component
function ChatWidgetLoader(): JSX.Element | null {
  const { siteConfig } = useDocusaurusContext();
  const chatApiUrl = (siteConfig.customFields?.chatApiUrl as string) || 'https://humanoid-robotics-backend.up.railway.app';

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

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <>
      {children}
      <BrowserOnly fallback={null}>
        {() => <ChatWidgetLoader />}
      </BrowserOnly>
    </>
  );
}
