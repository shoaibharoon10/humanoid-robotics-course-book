/**
 * Root theme wrapper for Docusaurus.
 *
 * This is the outermost component that wraps the entire application.
 * NOTE: ChatWidget has been moved to Layout component where ColorModeProvider is available.
 */

import React from 'react';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  return <>{children}</>;
}
