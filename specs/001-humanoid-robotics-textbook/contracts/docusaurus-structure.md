# Content Structure Contract: Docusaurus Configuration

**Feature**: 001-humanoid-robotics-textbook
**Date**: 2025-12-11
**Type**: Configuration Schema

## Purpose

This contract defines the expected structure for Docusaurus configuration files that will render the 4-module textbook.

---

## docusaurus.config.js Contract

```javascript
// Required fields for textbook site
{
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to building intelligent humanoid robots',
  url: 'https://<username>.github.io',
  baseUrl: '/humanoid-robotics-course-book/',

  // GitHub Pages deployment
  organizationName: '<github-username>',
  projectName: 'humanoid-robotics-course-book',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  // Presets
  presets: [
    ['classic', {
      docs: {
        routeBasePath: '/',  // Docs as landing
        sidebarPath: './sidebars.js',
        editUrl: '<repo-edit-url>',
      },
      blog: false,  // Disabled for textbook
      theme: {
        customCss: './src/css/custom.css',
      },
    }],
  ],

  // Mermaid diagrams
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  // Theme configuration
  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        { type: 'docSidebar', sidebarId: 'textbook', position: 'left', label: 'Textbook' },
        { href: '<github-repo>', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [/* module links */],
      copyright: `Copyright © ${new Date().getFullYear()}. Built with Docusaurus.`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['python', 'bash', 'yaml', 'xml'],
    },
  },
}
```

---

## sidebars.js Contract

```javascript
// Required sidebar structure
module.exports = {
  textbook: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: { type: 'doc', id: 'module-1/index' },
      items: [
        'module-1/ros2-overview',
        'module-1/nodes-topics-services',
        'module-1/python-pipelines',
        'module-1/humanoid-structure',
        'module-1/ai-ros-integration',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin',
      link: { type: 'doc', id: 'module-2/index' },
      items: [
        'module-2/digital-twin-concept',
        'module-2/physics-simulation',
        'module-2/gazebo-overview',
        'module-2/unity-visualization',
        'module-2/sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain',
      link: { type: 'doc', id: 'module-3/index' },
      items: [
        'module-3/isaac-sim-overview',
        'module-3/slam-navigation-mapping',
        'module-3/isaac-ros-acceleration',
        'module-3/path-planning',
        'module-3/synthetic-data',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      link: { type: 'doc', id: 'module-4/index' },
      items: [
        'module-4/vla-framework',
        'module-4/voice-intent-action',
        'module-4/llm-ros-integration',
        'module-4/environment-understanding',
        'module-4/capstone-autonomous-humanoid',
      ],
    },
    {
      type: 'category',
      label: 'References',
      items: ['references/index'],
    },
  ],
};
```

---

## _category_.json Contract (per module)

```json
// Example: docs/module-1/_category_.json
{
  "label": "Module 1: The Robotic Nervous System",
  "position": 1,
  "link": {
    "type": "generated-index",
    "description": "Learn about ROS 2 as the foundation for humanoid robot control."
  }
}
```

**Required per module**:

| Module | Label | Position |
|--------|-------|----------|
| module-1 | Module 1: The Robotic Nervous System | 1 |
| module-2 | Module 2: The Digital Twin | 2 |
| module-3 | Module 3: The AI-Robot Brain | 3 |
| module-4 | Module 4: Vision-Language-Action | 4 |

---

## Document Frontmatter Contract

Every topic document MUST include:

```yaml
---
id: <topic-id>
title: <topic-title>
sidebar_position: <1-5>
description: <seo-description>
keywords:
  - humanoid robotics
  - <topic-specific-keyword>
---
```

**Example**:
```yaml
---
id: ros2-overview
title: What is ROS 2?
sidebar_position: 1
description: Introduction to ROS 2 as the robotic nervous system
keywords:
  - ROS 2
  - robot operating system
  - humanoid robotics
---
```

---

## Validation Checklist

- [ ] docusaurus.config.js has correct title and tagline
- [ ] GitHub Pages deployment configured
- [ ] Mermaid theme enabled
- [ ] sidebars.js contains exactly 4 module categories
- [ ] Each module category contains 5 topic items
- [ ] Each module has _category_.json with correct position
- [ ] All topic documents have required frontmatter
- [ ] No broken internal links
- [ ] `npm run build` passes without errors
