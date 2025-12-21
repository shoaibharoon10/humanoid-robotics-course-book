/**
 * UrduTranslation component - Button to translate page content to Urdu using Google Translate.
 * Uses a simple URL redirect approach that works reliably on all platforms.
 */

import React, { useState, useCallback } from 'react';
import styles from './styles.module.css';

export default function UrduTranslation(): JSX.Element {
  const [isLoading, setIsLoading] = useState(false);

  const handleTranslate = useCallback(() => {
    if (typeof window === 'undefined') return;

    setIsLoading(true);

    // Get the current page URL
    const currentUrl = window.location.href;

    // Google Translate URL format - opens translated page in same tab
    const translateUrl = `https://translate.google.com/translate?sl=en&tl=ur&u=${encodeURIComponent(currentUrl)}`;

    // Navigate to the translated page
    window.location.href = translateUrl;
  }, []);

  return (
    <div className={styles.container}>
      <button
        onClick={handleTranslate}
        className={styles.translateButton}
        disabled={isLoading}
        title="Translate this page to Urdu (اردو میں ترجمہ کریں)"
        aria-label="Translate to Urdu"
      >
        <span className={styles.flag} aria-hidden="true">
          {/* Pakistan Flag SVG */}
          <svg viewBox="0 0 36 24" width="24" height="16">
            <rect fill="#01411C" width="36" height="24"/>
            <rect fill="#FFFFFF" width="9" height="24"/>
            <g transform="translate(22.5, 12)">
              <circle r="6" fill="#FFFFFF"/>
              <circle r="5" fill="#01411C" cx="1.5"/>
              <polygon
                points="0,-3.5 0.8,-1 3.5,-1 1.3,0.5 2.1,3 0,1.5 -2.1,3 -1.3,0.5 -3.5,-1 -0.8,-1"
                fill="#FFFFFF"
                transform="translate(3, -2)"
              />
            </g>
          </svg>
        </span>
        <span className={styles.buttonText}>
          {isLoading ? 'Translating...' : 'Urdu Translation'}
        </span>
        <span className={styles.urduText} dir="rtl">اردو ترجمہ</span>
      </button>
    </div>
  );
}
