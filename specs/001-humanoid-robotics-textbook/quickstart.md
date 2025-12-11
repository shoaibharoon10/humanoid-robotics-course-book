# Quickstart: Physical AI & Humanoid Robotics Textbook

**Feature**: 001-humanoid-robotics-textbook
**Date**: 2025-12-11

## Overview

This guide provides step-by-step instructions to set up the Docusaurus project and create the 4-module textbook skeleton.

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Git installed and configured
- Text editor (VS Code recommended)

## Quick Setup

### 1. Create Docusaurus Project

```bash
# From repository root
npx create-docusaurus@latest docusaurus classic --typescript

# Navigate to project
cd docusaurus
```

### 2. Install Dependencies

```bash
# Install Mermaid for diagrams
npm install @docusaurus/theme-mermaid

# Install dev dependencies
npm install --save-dev markdownlint-cli
```

### 3. Configure Docusaurus

Update `docusaurus.config.ts`:

```typescript
const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to building intelligent humanoid robots',
  favicon: 'img/favicon.ico',
  url: 'https://your-username.github.io',
  baseUrl: '/humanoid-robotics-course-book/',

  // Mermaid support
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  // ... rest of config
};
```

### 4. Create Module Structure

```bash
# Create module directories
mkdir -p docs/module-{1,2,3,4}
mkdir -p docs/references

# Create category files
for i in 1 2 3 4; do
  touch docs/module-$i/_category_.json
  touch docs/module-$i/index.md
done
```

### 5. Configure Sidebar

Update `sidebars.ts`:

```typescript
const sidebars: SidebarsConfig = {
  textbook: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: { type: 'doc', id: 'module-1/index' },
      items: [/* topic files */],
    },
    // ... modules 2-4
  ],
};
```

### 6. Build and Test

```bash
# Development server
npm run start

# Production build
npm run build

# Serve production build locally
npm run serve
```

## File Checklist

### Configuration Files

- [ ] `docusaurus.config.ts` - Site configuration
- [ ] `sidebars.ts` - Navigation structure
- [ ] `package.json` - Dependencies

### Module Files (repeat for each module)

- [ ] `docs/module-N/_category_.json` - Category metadata
- [ ] `docs/module-N/index.md` - Module overview
- [ ] `docs/module-N/<topic-1>.md` - Topic 1
- [ ] `docs/module-N/<topic-2>.md` - Topic 2
- [ ] `docs/module-N/<topic-3>.md` - Topic 3
- [ ] `docs/module-N/<topic-4>.md` - Topic 4
- [ ] `docs/module-N/<topic-5>.md` - Topic 5

### Supporting Files

- [ ] `docs/intro.md` - Book introduction
- [ ] `docs/references/index.md` - Bibliography
- [ ] `static/img/` - Images directory
- [ ] `src/css/custom.css` - Custom styles

## Module Creation Commands

### Module 1: The Robotic Nervous System (ROS 2)

```bash
cd docs/module-1
touch index.md ros2-overview.md nodes-topics-services.md \
      python-pipelines.md humanoid-structure.md ai-ros-integration.md
```

### Module 2: The Digital Twin (Gazebo & Unity)

```bash
cd docs/module-2
touch index.md digital-twin-concept.md physics-simulation.md \
      gazebo-overview.md unity-visualization.md sensor-simulation.md
```

### Module 3: The AI-Robot Brain (NVIDIA Isaac)

```bash
cd docs/module-3
touch index.md isaac-sim-overview.md slam-navigation-mapping.md \
      isaac-ros-acceleration.md path-planning.md synthetic-data.md
```

### Module 4: Vision-Language-Action (VLA)

```bash
cd docs/module-4
touch index.md vla-framework.md voice-intent-action.md \
      llm-ros-integration.md environment-understanding.md capstone-autonomous-humanoid.md
```

## Validation Commands

```bash
# Lint markdown files
npx markdownlint docs/**/*.md

# Check for broken links
npm run build  # Build will fail on broken links

# Type check (TypeScript)
npm run typecheck
```

## Deployment

### GitHub Pages (Automated)

1. Create `.github/workflows/deploy.yml`
2. Configure GitHub Pages in repository settings
3. Push to main branch triggers deployment

### Manual Deployment

```bash
# Build production site
npm run build

# Deploy to GitHub Pages
GIT_USER=<username> npm run deploy
```

## Next Steps

After skeleton setup:

1. Run `/sp.tasks` to generate implementation tasks
2. Create skeleton content for each topic
3. Validate build passes
4. Deploy to GitHub Pages
5. Begin Iteration 2 (detailed chapter content)

## Troubleshooting

### Build Fails

```bash
# Clear cache and rebuild
rm -rf .docusaurus node_modules/.cache
npm run build
```

### Mermaid Not Rendering

Ensure theme is properly configured in `docusaurus.config.ts`:
```typescript
themes: ['@docusaurus/theme-mermaid'],
```

### Sidebar Not Showing

Check that `sidebars.ts` exports match `docusaurus.config.ts` preset configuration.
