# Code Structure & Organization

## Directory Structure

### Root Level
```
/
├── .kiro/              # Kiro spec-driven development system
├── .claude/            # Claude Code slash commands
├── docs/               # Comprehensive documentation and examples
├── CLAUDE.md           # Claude Code instructions and project overview
├── README.md           # Japanese user documentation with workflow diagrams
└── .gitignore          # Git ignore file
```

### .kiro Directory
```
.kiro/
├── steering/           # Project steering documents
│   ├── product.md     # Product overview, features, use cases
│   ├── tech.md        # Architecture, tech stack, commands
│   ├── structure.md   # This file - code organization
│   └── *.md           # Custom steering documents
└── specs/             # Feature specifications
    ├── examples-pdf-diagram-explanation-app/
    ├── rubiks-cube-solver-app/
    ├── sangiiin-senkyo-realtime-app/
    └── [feature-name]/
        ├── spec.json      # Metadata and approval status
        ├── requirements.md # User requirements
        ├── design.md      # Technical design
        └── tasks.md       # Implementation tasks
```

### .claude Directory
```
.claude/
└── commands/          # Slash command definitions
    ├── spec-init.md
    ├── spec-requirements.md
    ├── spec-design.md
    ├── spec-tasks.md
    ├── spec-status.md
    ├── steering-init.md
    ├── steering-update.md
    └── steering-custom.md
```

### docs Directory
```
docs/
├── claude-code/       # Claude Code specific documentation
│   ├── hooks-guide.md # Implementation guide for hooks system
│   ├── hooks.md       # Hook system reference
│   └── slash-commands.md # Slash commands reference
└── kiro/              # Kiro IDE reference and examples
    ├── llms.txt       # Comprehensive Kiro IDE documentation
    ├── specs-example/ # Example specifications for reference
    │   ├── pdf-drawing-explainer/
    │   │   ├── design.md
    │   │   ├── requirements.md
    │   │   └── tasks.md
    │   └── task-management-service/
    │       ├── design.md
    │       ├── requirements.md
    │       └── tasks.md
    └── steering-example/ # Example steering documents
        ├── product.md
        ├── structure.md
        └── tech.md
```

## Naming Conventions

### Specifications
- **Feature Names**: Kebab-case (e.g., `pdf-diagram-explanation-app`)
- **Spec Files**: Fixed names within feature directories
  - `spec.json` - Metadata file
  - `requirements.md` - Requirements document
  - `design.md` - Technical design
  - `tasks.md` - Implementation tasks

### Steering Documents
- **Core Files**: Lowercase with `.md` extension
  - `product.md`, `tech.md`, `structure.md`
- **Custom Steering**: Descriptive names
  - `api-standards.md`, `testing-approach.md`, etc.

### Commands
- **Format**: Slash prefix with kebab-case
- **Pattern**: `/[action]-[target]`
- Examples: `/steering-init`, `/spec-requirements`

### Documentation Files
- **README.md**: User-facing documentation (Japanese)
- **llms.txt**: Reference documentation for Kiro IDE
- **Example directories**: Kebab-case with descriptive names

## File Patterns

### Markdown Files
- **Headers**: Use proper hierarchy (# for main, ## for sections)
- **Lists**: Use consistent formatting (- for bullets)
- **Code Blocks**: Triple backticks with language identifier

### JSON Files (spec.json)
```json
{
  "feature_name": "feature-name",
  "project_description": "Description of the feature",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "language": "japanese|english",
  "phase": "requirements-generated|design-generated|tasks-generated|implementation",
  "approvals": {
    "requirements": {
      "generated": true,
      "approved": false
    },
    "design": {
      "generated": true,
      "approved": false
    },
    "tasks": {
      "generated": true,
      "approved": false
    }
  },
  "progress": {
    "requirements": 100,
    "design": 100,
    "tasks": 100
  },
  "ready_for_implementation": false
}
```

## Code Organization Principles

1. **Separation of Concerns**
   - Steering: Project-wide context
   - Specs: Feature-specific planning
   - Docs: Reference documentation

2. **Progressive Disclosure**
   - Start with high-level steering
   - Drill down to specific specs
   - Implementation follows approval

3. **Version Control**
   - All markdown files tracked in Git
   - Approval status in spec.json
   - Changes tracked through commits

4. **Automation Points**
   - Slash commands enforce workflow
   - Status tracking through spec.json
   - Phase progression validation

## Best Practices

1. **File Creation**
   - Always use commands to create specs
   - Manual steering updates allowed
   - Keep files focused and concise

2. **Directory Management**
   - One directory per feature spec
   - Flat structure within directories
   - No nested feature specs

3. **Content Guidelines**
   - Clear, actionable language
   - Consistent formatting
   - Regular updates to steering
   - **Language Standards**:
     - Internal thinking: English
     - User-facing content: Japanese
     - Technical documentation: Bilingual with context-appropriate language

4. **Workflow Compliance**
   - Follow 3-phase approval process
   - Update spec.json for approvals
   - **Manual task tracking**: Update tasks.md checkboxes during implementation
   - Monitor progress with `/spec-status`