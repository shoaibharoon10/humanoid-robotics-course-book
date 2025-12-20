/**
 * Root theme wrapper for Docusaurus.
 *
 * This component wraps the entire application and adds the ChatWidget
 * as a global floating component.
 */

import React from 'react';
import ChatWidget from '@site/src/components/ChatWidget';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  // Only render ChatWidget on client-side
  const [isMounted, setIsMounted] = React.useState(false);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  return (
    <>
      {children}
      {isMounted && (
        <ChatWidget
          config={{
            apiUrl: process.env.REACT_APP_CHAT_API_URL || 'http://localhost:8000',
            title: 'Robotics Assistant',
            placeholder: 'Ask about humanoid robotics...',
            welcomeMessage: 'Hello! I can help you understand concepts from the Physical AI & Humanoid Robotics textbook. What would you like to learn about?'
          }}
        />
      )}
    </>
  );
}
