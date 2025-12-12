import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to building intelligent humanoid robots',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://shoaibharoon10.github.io',
  baseUrl: '/humanoid-robotics-course-book/',

  organizationName: 'shoaibharoon10',
  projectName: 'humanoid-robotics-course-book',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Mermaid support
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/shoaibharoon10/humanoid-robotics-course-book/tree/main/docusaurus/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Humanoid Robotics',
      logo: {
        alt: 'Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbookSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/shoaibharoon10/humanoid-robotics-course-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Module 1: ROS 2',
              to: '/docs/module-1',
            },
            {
              label: 'Module 2: Digital Twin',
              to: '/docs/module-2',
            },
          ],
        },
        {
          title: 'Advanced Topics',
          items: [
            {
              label: 'Module 3: NVIDIA Isaac',
              to: '/docs/module-3',
            },
            {
              label: 'Module 4: VLA',
              to: '/docs/module-4',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'References',
              to: '/docs/references',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/shoaibharoon10/humanoid-robotics-course-book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
