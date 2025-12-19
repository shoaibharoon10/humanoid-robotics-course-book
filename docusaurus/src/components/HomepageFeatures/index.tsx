import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS 2 & Motion Control',
    emoji: '🤖',
    description: (
      <>
        Master Robot Operating System 2 fundamentals, sensor integration,
        and humanoid motion control with inverse kinematics and gait planning.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac Simulation',
    emoji: '🎮',
    description: (
      <>
        Build and test robots in high-fidelity simulation with Isaac Sim,
        GPU-accelerated perception, and synthetic data generation.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action AI',
    emoji: '🧠',
    description: (
      <>
        Integrate cutting-edge AI with VLA models, voice control via Whisper,
        and LLM-powered task planning for autonomous behavior.
      </>
    ),
  },
];

function Feature({title, emoji, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <span className={styles.featureEmoji} role="img" aria-label={title}>
          {emoji}
        </span>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
