/**
 * UrduTranslation component - Button to translate page content to Urdu using Google Translate.
 * Uses a standard anchor tag approach for reliable translation on all platforms.
 */

import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

export default function UrduTranslation(): JSX.Element {
  const [translateUrl, setTranslateUrl] = useState<string>('#');
  const [isLocalhost, setIsLocalhost] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const hostname = window.location.hostname;
      const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
      setIsLocalhost(isLocal);

      if (isLocal) {
        console.warn('Urdu Translation: Google Translate works best on production URLs. Local development URLs may not translate properly.');
      }

      // Build Google Translate URL
      const currentUrl = window.location.href;
      setTranslateUrl(`https://translate.google.com/translate?sl=en&tl=ur&u=${encodeURIComponent(currentUrl)}`);
    }
  }, []);

  return (
    <div className={styles.container}>
      <a
        href={translateUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={styles.translateButton}
        title={isLocalhost
          ? "Translation works best on production (اردو ترجمہ)"
          : "Translate this page to Urdu (اردو میں ترجمہ کریں)"
        }
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
          Urdu Translation
        </span>
        <span className={styles.urduText} dir="rtl">اردو ترجمہ</span>
      </a>
    </div>
  );
}
