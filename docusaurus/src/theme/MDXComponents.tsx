/**
 * Custom MDX Components - wraps all docs with Urdu Translation button
 */

import React from 'react';
import MDXComponents from '@theme-original/MDXComponents';
import UrduTranslation from '@site/src/components/UrduTranslation';

// Export all original MDX components plus our custom additions
export default {
  ...MDXComponents,
  // Make UrduTranslation available in all MDX files
  UrduTranslation,
};
