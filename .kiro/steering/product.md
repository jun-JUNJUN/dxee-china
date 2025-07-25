# Product Overview

## Product Name
Claude Code Spec-Driven Development System

## Product Description
A Kiro-style specification-driven development system for Claude Code that provides structured workflows, automated progress tracking, and phase-based approval processes for software development projects.

## Core Features
1. **Spec-Driven Development Workflow**
   - Three-phase approval process (Requirements → Design → Tasks)
   - Manual approval gates between phases
   - Structured document generation

2. **Steering System**
   - Project context management through markdown files
   - Core steering documents (product, tech, structure)
   - Custom steering for specialized contexts

3. **Slash Commands**
   - `/steering-init` - Initialize steering documents
   - `/steering-update` - Update steering after changes
   - `/steering-custom` - Create custom steering
   - `/spec-init` - Initialize spec structure
   - `/spec-requirements` - Generate requirements
   - `/spec-design` - Generate technical design
   - `/spec-tasks` - Generate implementation tasks
   - `/spec-status` - Check progress and phases

4. **Task Progress Tracking**
   - Manual task tracking through tasks.md checkboxes
   - Progress monitoring with checkbox parsing
   - TodoWrite integration for active work items
   - Spec compliance checking
   - Context preservation during compaction
   - Minimal automation to prevent hook errors

5. **Comprehensive Documentation**
   - Technical blog explaining implementation approach
   - Structured docs/ directory with examples and guides
   - Kiro IDE reference documentation (llms.txt)
   - Example specifications and steering documents
   - Step-by-step workflow diagrams

6. **Language Support**
   - Bilingual operation: English thinking, Japanese responses
   - Japanese README with comprehensive workflow documentation
   - Localized user interface and command descriptions

## Development Status
> **Initial Version Warning**: This is an initial version under active development and improvement based on usage feedback.

## Use Cases
- **New Feature Development**: Start with steering, create specs, implement with confidence
- **Project Documentation**: Maintain living documentation through steering files
- **Team Collaboration**: Clear approval workflow ensures alignment
- **Quality Assurance**: Phase-based approvals prevent premature implementation
- **Real-world Applications**: Proven with complex specs including Rubik's cube solver app and real-time election results tracking app

## Value Proposition
- **Reduced Development Risk**: Structured approach catches issues early
- **Improved Communication**: Clear specifications and approval process
- **Better Documentation**: Living steering documents stay current with comprehensive guides
- **Faster Iteration**: Well-defined tasks and clear progress tracking
- **Context Preservation**: Manual tracking maintains project knowledge
- **Easy Integration**: Simple copy-and-paste setup for existing projects
- **Language Flexibility**: Bilingual support (English thinking, Japanese responses)
- **Comprehensive Learning**: Technical blog and documentation provide deep understanding
- **Real-world Validation**: Proven with complex example specifications including Rubik's cube solver and election tracking applications

## Target Users
- Software development teams using Claude Code
- Projects requiring structured development workflows
- Teams needing clear documentation and approval processes
- Developers seeking better spec-driven development tools