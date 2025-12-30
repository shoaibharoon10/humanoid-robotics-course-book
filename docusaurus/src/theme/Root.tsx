/**
 * Root theme wrapper for Docusaurus.
 *
 * This is the outermost component that wraps the entire application.
 * Provides authentication context to all child components.
 * NOTE: ChatWidget has been moved to Layout component where ColorModeProvider is available.
 */

import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { AuthProvider } from '@site/src/context/AuthContext';

interface RootProps {
  children: React.ReactNode;
}

// Auth Provider wrapper - only runs in browser
function AuthWrapper({ children }: { children: React.ReactNode }): JSX.Element {
  return <AuthProvider>{children}</AuthProvider>;
}

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <BrowserOnly fallback={<>{children}</>}>
      {() => <AuthWrapper>{children}</AuthWrapper>}
    </BrowserOnly>
  );
}
