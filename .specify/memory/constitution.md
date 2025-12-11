<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0
Bump rationale: MAJOR - Initial constitution creation for new project

Modified principles: N/A (initial creation)
Added sections:
  - Core Principles (5 principles)
  - Key Standards
  - Success Criteria
  - Governance
Removed sections: N/A (initial creation)

Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section present)
  - .specify/templates/spec-template.md: ✅ Compatible (User stories, requirements, success criteria align)
  - .specify/templates/tasks-template.md: ✅ Compatible (Phase structure supports spec-first workflow)

Follow-up TODOs: None
-->

# AI/Spec-Driven Book Creation Constitution

## Core Principles

### I. Technical Accuracy

All content MUST be based on verified and official documentation. Every code example,
command, and technical claim MUST be validated against authoritative sources before
inclusion. Unverifiable claims or outdated information MUST NOT appear in the book.

**Rationale**: Readers depend on technical accuracy to successfully follow instructions.
Incorrect or unverified content erodes trust and causes reader frustration.

### II. Structured Guidance

Content MUST provide clear, structured guidance for developers and AI learners.
Each chapter MUST follow a logical progression from concepts to implementation.
Instructions MUST be explicit and leave no ambiguity about expected actions or outcomes.

**Rationale**: Structured content reduces cognitive load and enables readers of varying
skill levels to follow along successfully.

### III. Reproducibility

All instructions, commands, and workflows MUST be reproducible. Every code example
MUST be runnable in the specified environment. Every command MUST produce the
documented output when executed correctly. Steps MUST NOT rely on implicit
knowledge or undocumented prerequisites.

**Rationale**: Reproducibility ensures readers can achieve the same results as the
author, which is essential for technical education.

### IV. Spec-First Development

Every chapter MUST map to a clear specification before content creation begins.
The spec → draft → refinement cycle MUST be strictly followed. No content MUST
be written without an approved spec defining scope, requirements, and acceptance
criteria.

**Rationale**: Spec-first development prevents scope creep, ensures consistency,
and provides clear acceptance criteria for each chapter.

### V. Original Content

All content MUST be original with zero plagiarism. References to external sources
MUST be properly attributed. Direct quotations MUST be clearly marked and limited.
Content MUST add original value beyond what existing documentation provides.

**Rationale**: Original content protects intellectual property, ensures legal
compliance, and provides unique value to readers.

## Key Standards

### Code and Command Validation

- All code examples MUST be validated and runnable in the documented environment
- All commands MUST be tested and produce the documented output
- Version-specific dependencies MUST be explicitly stated
- Environment prerequisites MUST be documented before each runnable section

### Documentation Format

- Docusaurus Markdown/MDX formatting MUST be consistent across all chapters
- Heading hierarchy MUST follow a predictable pattern
- Code blocks MUST specify the language for syntax highlighting
- Internal links MUST be relative and validated

### Reference Standards

- References MUST rely on authoritative sources (official docs preferred)
- External links MUST be validated at time of writing
- Version numbers of referenced tools MUST be documented
- Deprecated sources MUST be avoided or clearly marked

### Writing Style

- Writing tone: practical, concise, technically confident
- Avoid hedging language; be direct about what works and what does not
- Use active voice
- Keep paragraphs focused on single concepts

### Tooling Constraints

- Format: Docusaurus project deployed on GitHub Pages
- Tools: Spec-Kit Plus + Claude Code for content development
- Content MUST avoid unverifiable claims or incomplete steps
- Each chapter MUST map to a clear spec in the specs/ directory

## Success Criteria

### Build and Deployment

- The book MUST build successfully using `npm run build` without errors
- The book MUST deploy to GitHub Pages via the configured workflow
- All internal links MUST resolve correctly in the built output
- Images and assets MUST load correctly in the deployed site

### Content Consistency

- All chapters MUST be consistent with this constitution and their specs
- Content MUST be accurate, reproducible, and free of plagiarism
- All code examples MUST execute successfully in documented environments
- All commands MUST produce documented outputs

### Site Quality

- Final GitHub Pages site MUST be complete and navigable
- Navigation MUST reflect the logical structure of the book
- Search functionality (if enabled) MUST index all content
- Mobile and desktop layouts MUST render correctly

## Governance

### Amendment Process

1. Proposed amendments MUST be documented with rationale
2. Amendments MUST be reviewed for impact on existing content
3. Breaking changes require migration plan for affected chapters
4. All amendments MUST update the version following semantic versioning

### Versioning Policy

- **MAJOR**: Backward incompatible principle removals or redefinitions
- **MINOR**: New principle/section added or materially expanded guidance
- **PATCH**: Clarifications, wording, typo fixes, non-semantic refinements

### Compliance Review

- All chapter specs MUST be verified against this constitution before approval
- All chapter drafts MUST be verified against their approved specs
- Pull requests MUST not introduce content that violates stated principles
- Regular audits SHOULD verify continued compliance of published content

### Authoritative Documents

- This constitution supersedes all other project governance
- Feature specs in `specs/<feature>/spec.md` govern individual chapters
- Implementation plans in `specs/<feature>/plan.md` govern technical approach
- CLAUDE.md provides runtime development guidance for AI assistants

**Version**: 1.0.0 | **Ratified**: 2025-12-11 | **Last Amended**: 2025-12-11
