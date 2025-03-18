# Getting Started with Cursor AI Rules 🎯

Welcome to your AI-assisted development environment with Cursor! This guide will help you get started with the rules-based development workflow.

## Initial Setup ⚙️

1. Your `.cursor` directory has been set up with the necessary configuration
2. Open your project in Cursor IDE
3. Create the following structure in your project root:
   ```
   .project/
   ├── specs/           # For your specifications
   │   └── SPECS.md     # Specification index
   ├── tasks/           # For your tasks
   │   └── TASKS.md     # Task index
   └── src/            # Your source code
   ```

## Working with Cursor AI 🤖

### 1. Creating Specifications

1. In the `.project/specs` directory, create a new specification file:
   ```
   SPEC-01-feature-name.md
   ```

2. Use this template for your specification:
   ```markdown
   # SPEC-01: Feature Name

   ## Description
   Detailed description of the feature

   ## Requirements
   - [ ] Requirement 1 [HIGH]
   - [ ] Requirement 2 [MEDIUM]

   ## Testing Criteria
   - Test case 1
   - Test case 2

   ## Acceptance Criteria
   - [ ] Criteria 1
   - [ ] Criteria 2

   ## Metadata
   - Created: YYYY-MM-DD
   - Status: Draft/In Progress/Complete
   ```

3. Update `SPECS.md` with your new specification

### 2. Task Creation

1. In the `.project/tasks` directory, create task files:
   ```
   TASK-2024-03-20-01.md
   ```

2. Use this template for tasks:
   ```markdown
   # TASK-2024-03-20-01: Implement X

   ## Specification Reference
   - SPEC-01

   ## Acceptance Criteria
   - [ ] Criteria 1
   - [ ] Criteria 2

   ## Required Tests
   - Test case 1
   - Test case 2

   ## Complexity
   Medium

   ## Status
   Not Started
   ```

### 3. Working with Cursor AI

When implementing tasks:

1. Open the task file you want to work on
2. Use Cursor's AI capabilities by:
   - Pressing `Cmd/Ctrl + K` to chat with Cursor
   - Highlight code and press `Cmd/Ctrl + L` for code explanations
   - Use `Cmd/Ctrl + /` for inline suggestions

3. Always reference your specification and task files when asking Cursor for help:
   ```
   "Please help me implement TASK-2024-03-20-01 according to SPEC-01..."
   ```

### 4. Best Practices 🌟

1. **Atomic Changes**
   - Keep tasks small and focused
   - One specification feature per task
   - Regular commits with conventional commit messages

2. **Documentation**
   - Update specification status as you progress
   - Document learnings in task files
   - Keep SPECS.md and TASKS.md up to date

3. **Testing**
   - Write tests before implementation
   - Use Cursor to help generate test cases
   - Verify all acceptance criteria

## Common Commands 🛠️

- Generate tests: "Generate unit tests for this implementation according to SPEC-XX"
- Implement feature: "Help me implement the feature described in TASK-YYYY-MM-DD-XX"
- Code review: "Review this implementation against SPEC-XX requirements"

## Need Help? 🤔

- Press `Cmd/Ctrl + K` and ask about any part of this workflow
- Reference the main README.md for detailed workflow information
- Check the specification format when unsure about requirements

Remember: The key to successful AI-assisted development is clear specifications and well-defined tasks. Let Cursor help you implement while you focus on the architecture and design decisions.
