# Research: Physical AI & Humanoid Robotics Textbook

**Feature**: 001-humanoid-robotics-textbook
**Date**: 2025-12-11
**Status**: Complete

## Research Summary

This document consolidates research findings for all technical decisions and unknowns identified during planning. No NEEDS CLARIFICATION items remained from the spec phase; this research focuses on validating architectural decisions and establishing authoritative sources.

## Decision Log

### D-1: Publishing Platform Selection

**Decision**: Docusaurus 3.x

**Research Findings**:
- Docusaurus 3.x released stable with React 18 support
- MDX 2.0 support enables rich interactive content
- Native Mermaid plugin available (`@docusaurus/theme-mermaid`)
- GitHub Pages deployment is first-class citizen
- Versioning built-in for future textbook editions

**Authoritative Sources**:
- Docusaurus Official Documentation: https://docusaurus.io/docs
- GitHub Pages Deployment Guide: https://docusaurus.io/docs/deployment#deploying-to-github-pages

**Alternatives Evaluated**:
| Platform | Pros | Cons | Verdict |
|----------|------|------|---------|
| Docusaurus | MDX, React, versioning, search | Heavier bundle | ✅ Selected |
| MkDocs | Simple, Python | Less customization | ❌ Rejected |
| GitBook | Beautiful defaults | Proprietary, costs | ❌ Rejected |
| VitePress | Fast, Vue | Smaller ecosystem | ❌ Rejected |

---

### D-2: ROS 2 Documentation Approach

**Decision**: Reference ROS 2 Jazzy (LTS) with Python client library (rclpy)

**Research Findings**:
- ROS 2 Jazzy Jalisco is the current LTS release (May 2024 - May 2029)
- Python client (rclpy) has full feature parity with C++ (rclcpp)
- Official tutorials use Python for accessibility
- Humanoid robot packages increasingly support Python-first

**Authoritative Sources**:
- ROS 2 Documentation: https://docs.ros.org/en/jazzy/
- rclpy API: https://docs.ros.org/en/jazzy/p/rclpy/
- ROS 2 Design: https://design.ros2.org/

**Key Concepts for Module 1**:
1. Nodes: Independent executables communicating via ROS graph
2. Topics: Publish/subscribe asynchronous messaging
3. Services: Request/response synchronous calls
4. Actions: Long-running tasks with feedback

---

### D-3: Simulation Platform Coverage

**Decision**: Cover both Gazebo (physics) and Unity (visualization)

**Research Findings**:
- Gazebo Harmonic is the current stable release with improved physics
- Unity Robotics Hub provides ROS 2 integration via TCP
- Both serve different purposes: Gazebo for control validation, Unity for HRI

**Authoritative Sources**:
- Gazebo Documentation: https://gazebosim.org/docs
- Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- ROS 2 Unity Integration: https://github.com/Unity-Technologies/ROS-TCP-Connector

**Physics Engine Comparison**:
| Feature | Gazebo (DART/ODE) | Unity (PhysX) |
|---------|-------------------|---------------|
| Accuracy | High | Medium |
| Speed | Medium | High |
| Visualization | Basic | Photorealistic |
| ROS Integration | Native | Plugin-based |

---

### D-4: NVIDIA Isaac Platform Scope

**Decision**: Cover Isaac Sim + Isaac ROS conceptually

**Research Findings**:
- Isaac Sim runs on Omniverse platform (RTX required)
- Isaac ROS provides GPU-accelerated perception packages
- VSLAM (Visual SLAM) and Nav2 integration well-documented
- Synthetic data generation via Omniverse Replicator

**Authoritative Sources**:
- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Isaac ROS: https://nvidia-isaac-ros.github.io/
- Omniverse Replicator: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html

**Key Concepts for Module 3**:
1. Isaac Sim: High-fidelity simulator on Omniverse
2. SLAM: Simultaneous Localization and Mapping
3. Nav2 Integration: Navigation stack acceleration
4. Synthetic Data: Domain randomization for training

---

### D-5: Vision-Language-Action (VLA) Framework

**Decision**: Cover VLA conceptually as emerging paradigm

**Research Findings**:
- VLA represents convergence of vision models, LLMs, and robotic action
- Key models: RT-2, PaLM-E, OpenVLA
- Integration with ROS via action servers and service interfaces
- Active research area with rapid evolution

**Authoritative Sources**:
- RT-2 Paper: Brohan et al., "RT-2: Vision-Language-Action Models" (2023)
- PaLM-E Paper: Driess et al., "PaLM-E: An Embodied Multimodal Language Model" (2023)
- OpenVLA: https://openvla.github.io/

**Key Concepts for Module 4**:
1. VLA Architecture: Vision encoder → Language model → Action decoder
2. Voice-to-Action: Speech → Intent → ROS commands
3. LLM Integration: Natural language task specification
4. Embodied AI: Grounding language in physical world

---

### D-6: URDF/SDF Robot Description

**Decision**: Use URDF with SDF migration notes

**Research Findings**:
- URDF is simpler and widely supported
- SDF is more powerful (multiple robots, sensors)
- Gazebo prefers SDF but auto-converts URDF
- Most open-source humanoid robots provide URDF

**Authoritative Sources**:
- URDF Specification: http://wiki.ros.org/urdf
- SDF Specification: http://sdformat.org/spec
- URDF Tutorials: https://docs.ros.org/en/jazzy/Tutorials/Intermediate/URDF/

**Open-Source Humanoid References**:
- NAO Robot URDF: Aldebaran/SoftBank
- WALK-MAN: IIT humanoid
- Atlas (Simulation): Boston Dynamics reference

---

### D-7: Citation and Reference Standards

**Decision**: APA 7th Edition with per-module reference sections

**Research Findings**:
- APA 7th Edition is standard for academic/technical writing
- In-text citations: (Author, Year) format
- Reference lists: Alphabetical by author surname
- DOI links preferred for journal articles

**Format Examples**:
```
In-text: (Quigley et al., 2009)
Reference: Quigley, M., Conley, K., Gerkey, B., Faust, J., Foote, T., Leibs, J.,
    Wheeler, R., & Ng, A. Y. (2009). ROS: An open-source Robot Operating System.
    ICRA Workshop on Open Source Software, 3(3.2), 5.
```

---

## Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Publishing | Docusaurus | 3.x | Static site generation |
| Diagrams | Mermaid.js | 10.x | Text-based diagrams |
| Hosting | GitHub Pages | - | Free static hosting |
| Robotics | ROS 2 | Jazzy | Robot middleware |
| Simulation | Gazebo | Harmonic | Physics simulation |
| Visualization | Unity | 2022 LTS | HRI visualization |
| AI Platform | Isaac Sim | Latest | GPU-accelerated sim |
| VLA | Conceptual | - | Emerging paradigm |

## Research Gaps (Deferred to Iteration 2)

1. **Specific code examples**: Actual Python ROS 2 code deferred
2. **URDF files**: Robot model definitions deferred
3. **Isaac Sim tutorials**: Step-by-step workflows deferred
4. **VLA implementation**: Model integration code deferred

## Authoritative Source Registry

### Official Documentation
- ROS 2: https://docs.ros.org/en/jazzy/
- Gazebo: https://gazebosim.org/docs
- Unity Robotics: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/
- Docusaurus: https://docusaurus.io/docs

### Academic References (to be cited)
- Quigley et al. (2009) - ROS architecture
- Koenig & Howard (2004) - Gazebo simulator
- Macenski et al. (2023) - Nav2 navigation stack
- Brohan et al. (2023) - RT-2 VLA model

### Standards
- APA 7th Edition
- URDF XML Specification
- SDF 1.9 Specification
