/**
 * Swizzled DocItem/Content component to inject Urdu Translation button
 * at the top of all documentation pages.
 *
 * This wrapper preserves all theme context (including ColorMode) by
 * wrapping with a div instead of a fragment.
 */

import React from 'react';
import Content from '@theme-original/DocItem/Content';
import type ContentType from '@theme/DocItem/Content';
import type { WrapperProps } from '@docusaurus/types';
import UrduTranslation from '@site/src/components/UrduTranslation';

type Props = WrapperProps<typeof ContentType>;

export default function ContentWrapper(props: Props): JSX.Element {
  return (
    <div className="doc-content-wrapper">
      <UrduTranslation />
      <Content {...props} />
    </div>
  );
}
