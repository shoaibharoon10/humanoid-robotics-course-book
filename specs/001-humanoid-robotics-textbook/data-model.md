# Data Model: Physical AI & Humanoid Robotics Textbook

**Feature**: 001-humanoid-robotics-textbook
**Date**: 2025-12-11

## Overview

This document defines the content structure entities for the textbook. Since this is a documentation project (not a database application), the "data model" represents the content hierarchy and metadata schema.

## Content Entities

### Entity: Textbook

The root container for all educational content.

| Attribute | Type | Description |
|-----------|------|-------------|
| title | string | "Physical AI & Humanoid Robotics" |
| version | string | Semantic version (1.0.0 for skeleton) |
| modules | Module[] | Exactly 4 modules |
| introduction | Document | Book introduction/landing page |
| references | Document | Global bibliography |

**Constraints**:
- MUST contain exactly 4 modules
- Version follows semver for tracking iterations

---

### Entity: Module

A major thematic section of the textbook.

| Attribute | Type | Description |
|-----------|------|-------------|
| id | string | Unique identifier (e.g., "module-1") |
| title | string | Module title (e.g., "The Robotic Nervous System") |
| subtitle | string | Technology focus (e.g., "ROS 2") |
| description | string | 1-2 sentence overview |
| position | integer | Order in sidebar (1-4) |
| topics | Topic[] | 4-6 high-level topics |
| references | Citation[] | Module-specific bibliography |

**Constraints**:
- MUST have 4-6 topics
- Position determines navigation order
- Description MUST be understandable by non-experts

**Module Definitions**:

| ID | Title | Subtitle | Topics |
|----|-------|----------|--------|
| module-1 | The Robotic Nervous System | ROS 2 | 5 topics |
| module-2 | The Digital Twin | Gazebo & Unity | 5 topics |
| module-3 | The AI-Robot Brain | NVIDIA Isaac Platform | 5 topics |
| module-4 | Vision-Language-Action | VLA | 5 topics |

---

### Entity: Topic

A high-level subject area within a module.

| Attribute | Type | Description |
|-----------|------|-------------|
| id | string | URL slug (e.g., "ros2-overview") |
| title | string | Topic title |
| description | string | Brief description for sidebar/preview |
| module_id | string | Parent module reference |
| position | integer | Order within module |
| status | enum | skeleton | draft | complete |

**Constraints**:
- ID MUST be URL-safe (lowercase, hyphens)
- Status tracks content completion state

**Topic Registry**:

#### Module 1: The Robotic Nervous System (ROS 2)

| Position | ID | Title |
|----------|-----|-------|
| 1 | ros2-overview | What is ROS 2? |
| 2 | nodes-topics-services | Nodes, Topics, Services, Actions |
| 3 | python-pipelines | Python-based Control Pipelines |
| 4 | humanoid-structure | Humanoid Robot Structure & URDF |
| 5 | ai-ros-integration | AI Agent and ROS 2 Integration |

#### Module 2: The Digital Twin (Gazebo & Unity)

| Position | ID | Title |
|----------|-----|-------|
| 1 | digital-twin-concept | Digital Twins: Concept & Purpose |
| 2 | physics-simulation | Physics Simulation Fundamentals |
| 3 | gazebo-overview | Gazebo for Robot Simulation |
| 4 | unity-visualization | Unity for Visualization |
| 5 | sensor-simulation | Sensor Simulation Concepts |

#### Module 3: The AI-Robot Brain (NVIDIA Isaac)

| Position | ID | Title |
|----------|-----|-------|
| 1 | isaac-sim-overview | Isaac Sim Overview |
| 2 | slam-navigation-mapping | SLAM, Navigation, and Mapping |
| 3 | isaac-ros-acceleration | Isaac ROS Acceleration |
| 4 | path-planning | Path Planning for Humanoids |
| 5 | synthetic-data | Synthetic Data Generation |

#### Module 4: Vision-Language-Action (VLA)

| Position | ID | Title |
|----------|-----|-------|
| 1 | vla-framework | VLA Framework Overview |
| 2 | voice-intent-action | Voice → Intent → Action |
| 3 | llm-ros-integration | LLM Integration with ROS |
| 4 | environment-understanding | Environment Understanding |
| 5 | capstone-autonomous-humanoid | Capstone: Autonomous Humanoid |

---

### Entity: Citation

A bibliographic reference in APA format.

| Attribute | Type | Description |
|-----------|------|-------------|
| key | string | Citation key (e.g., "quigley2009ros") |
| authors | string[] | Author names |
| year | integer | Publication year |
| title | string | Work title |
| source | string | Journal/conference/publisher |
| doi | string? | Optional DOI link |
| url | string? | Optional URL |

**Format**: APA 7th Edition

---

### Entity: Document (Docusaurus Page)

A markdown document in the Docusaurus structure.

| Attribute | Type | Description |
|-----------|------|-------------|
| id | string | Document ID (frontmatter) |
| title | string | Page title |
| sidebar_position | integer | Order in sidebar |
| sidebar_label | string? | Optional short label |
| description | string | SEO/preview description |
| keywords | string[] | SEO keywords |
| content | markdown | MDX content body |

**Frontmatter Schema**:
```yaml
---
id: document-id
title: Document Title
sidebar_position: 1
sidebar_label: Short Label
description: Brief description for SEO
keywords: [keyword1, keyword2]
---
```

---

## Relationships

```
Textbook (1)
    │
    ├── Introduction (1)
    │
    ├── Modules (4)
    │       │
    │       └── Topics (4-6 each)
    │               │
    │               └── Documents (1 per topic)
    │
    └── References (1)
            │
            └── Citations (many)
```

## State Transitions

### Topic Status

```
skeleton → draft → complete
    │         │
    │         └── revision (back to draft)
    │
    └── (direct to complete if simple)
```

- **skeleton**: Structure only, placeholder content
- **draft**: Content written, under review
- **complete**: Validated, ready for publication

## Docusaurus Mapping

| Entity | Docusaurus Artifact |
|--------|---------------------|
| Textbook | docusaurus.config.js (title, tagline) |
| Module | docs/module-N/_category_.json |
| Topic | docs/module-N/topic-id.md |
| Citation | docs/references/index.md |
| Document | Any .md/.mdx file |

## Validation Rules

1. **Module count**: Exactly 4 modules required
2. **Topic count**: 4-6 topics per module
3. **Unique IDs**: All topic IDs must be unique across textbook
4. **URL safety**: IDs must be lowercase with hyphens only
5. **Position continuity**: No gaps in position numbers
6. **Citation format**: APA 7th Edition compliance
