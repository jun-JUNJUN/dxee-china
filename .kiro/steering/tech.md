# Technology Stack

## Architecture Overview
A Claude Code extension system that uses hooks and slash commands to implement Kiro-style spec-driven development workflows.

## Core Technologies
- **Platform**: Claude Code CLI (darwin)
- **Language**: Markdown-based specifications and documentation
- **Automation**: Claude Code hooks system
- **Version Control**: Git

## Development Environment

### System Requirements
- Claude Code CLI
- Git repository
- macOS (darwin platform)

### Project Dependencies
- Claude Code slash commands (.claude/commands/)
- File system access for steering/spec management
- No external package dependencies (pure markdown/JSON system)
- TodoWrite tool integration for task management

### Language Specifications
- **Thinking**: English (internal processing)
- **Responses**: Japanese (user-facing content)
- **Documentation**: Bilingual with Japanese emphasis

### Task Tracking Approach
- **Manual Progress**: Checkbox manipulation in tasks.md files
- **Automatic Parsing**: Progress percentage calculation from checkboxes
- **Enhanced Tracking**: Improved hook error resolution and progress monitoring
- **TodoWrite Integration**: Active task management during implementation

## Key Commands

### Steering Commands
```bash
/steering-init          # Generate initial steering documents
/steering-update        # Update steering after changes  
/steering-custom        # Create custom steering for specialized contexts
```

### Specification Commands
```bash
/spec-init [feature-name]           # Initialize spec structure only
/spec-requirements [feature-name]   # Generate requirements
/spec-design [feature-name]         # Generate technical design
/spec-tasks [feature-name]          # Generate implementation tasks
/spec-status [feature-name]         # Check current progress and phases
```

## File Structure
```
.kiro/
├── steering/           # Project steering documents
│   ├── product.md     # Product overview
│   ├── tech.md        # Technology stack
│   └── structure.md   # Code organization
├── specs/             # Feature specifications
│   └── [feature]/
│       ├── spec.json      # Spec metadata and approval status
│       ├── requirements.md # Feature requirements
│       ├── design.md      # Technical design
│       └── tasks.md       # Implementation tasks

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

docs/                  # Comprehensive documentation
├── claude-code/       # Claude Code specific guides
│   ├── hooks-guide.md # Hook system implementation
│   ├── hooks.md       # Hook reference
│   └── slash-commands.md # Command reference
└── kiro/              # Kiro IDE reference and examples
    ├── llms.txt       # Kiro IDE documentation
    ├── specs-example/ # Example specifications
    └── steering-example/ # Example steering documents

README.md             # Japanese user documentation with workflow diagrams
```

## Integration Points
- **Claude Code CLI**: Primary interface for all commands
- **Git**: Version control for specs and steering
- **File System**: Markdown file management
- **Hooks System**: Automated tracking and compliance
- **TodoWrite Tool**: Task progress tracking and management

## Development Workflow
1. Initialize project steering with `/steering-init`
2. Create feature specifications with `/spec-init`
3. Follow 3-phase approval process (Requirements → Design → Tasks)
4. Implement with manual task tracking via checkbox manipulation
5. Monitor progress with `/spec-status`
6. Update steering as needed with `/steering-update`

## Task Progress Management
- **Manual Tracking**: Update tasks.md checkboxes during implementation
- **Progress Calculation**: Automatic percentage computation from checkbox states  
- **Enhanced Monitoring**: Improved hook error resolution and progress tracking
- **Status Monitoring**: Use `/spec-status` for current progress overview
- **TodoWrite Integration**: Track active work items during development sessions

## Security & Access
- Local file system based
- No external dependencies
- Git-based version control
- Manual approval gates for phase transitions