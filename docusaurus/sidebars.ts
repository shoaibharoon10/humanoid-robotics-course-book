import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Physical AI & Humanoid Robotics Textbook
 * Sidebar configuration for 4-module structure
 */
const sidebars: SidebarsConfig = {
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: {
        type: 'doc',
        id: 'module-1/index',
      },
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
      link: {
        type: 'doc',
        id: 'module-2/index',
      },
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
      link: {
        type: 'doc',
        id: 'module-3/index',
      },
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
      link: {
        type: 'doc',
        id: 'module-4/index',
      },
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
      link: {
        type: 'doc',
        id: 'references/index',
      },
      items: [],
    },
  ],
};

export default sidebars;
