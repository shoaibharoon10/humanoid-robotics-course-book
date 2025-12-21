/**
 * Swizzled Layout component to add ChatWidget inside the ColorModeProvider context.
 *
 * This component wraps the original Layout and adds the ChatWidget
 * which requires access to useColorMode hook.
 */

import React from 'react';
import Layout from '@theme-original/Layout';
import type LayoutType from '@theme/Layout';
import type { WrapperProps } from '@docusaurus/types';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

type Props = WrapperProps<typeof LayoutType>;

// ChatWidget loader - only rendered in browser, inside ColorModeProvider
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
