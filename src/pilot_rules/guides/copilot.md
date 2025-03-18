# Getting Started with GitHub Copilot Rules 🚀

Welcome to your AI-assisted development environment with GitHub Copilot! This guide will help you get started with the rules-based development workflow.

## Initial Setup ⚙️

1. Your `.github` directory has been set up with the necessary configuration
2. Ensure GitHub Copilot is activated in your IDE
3. Create the following structure in your project root:
   ```
   .project/
   ├── specs/           # For your specifications
   │   └── SPECS.md     # Specification index
   ├── tasks/           # For your tasks
   │   └── TASKS.md     # Task index
   └── src/            # Your source code
   ```

## Working with GitHub Copilot 🤖

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

### 3. Working with GitHub Copilot

When implementing tasks:

1. Open the task and specification files for reference
2. Use Copilot's features effectively:
   - Let Copilot suggest code completions as you type
   - Use `//@prompt` or `//` comments to guide Copilot
   - Press `Tab` to accept suggestions
   - Use `Alt + ]` or `Alt + [` to cycle through suggestions

3. Guide Copilot with clear comments:
   ```javascript
   // Implementing TASK-2024-03-20-01: Feature X
   // Requirements from SPEC-01:
   // 1. Handle user authentication
   // 2. Validate input data
   ```

### 4. Best Practices 🌟

1. **Structured Comments**
   - Write clear, descriptive comments
   - Reference task and spec IDs in comments
   - Use natural language to describe requirements

2. **Documentation**
   - Keep specifications up to date
   - Document implementation decisions
   - Update task status regularly

3. **Testing**
   - Write test descriptions in comments
   - Let Copilot suggest test implementations
   - Verify against specification requirements

## Effective Prompting 📝

Guide Copilot with effective comments:

1. **Feature Implementation**
   ```javascript
   // Implement user authentication according to SPEC-01
   // Requirements:
   // - Use JWT tokens
   // - Include refresh token mechanism
   ```

2. **Test Generation**
   ```javascript
   // Generate tests for user authentication
   // Test cases:
   // - Valid credentials
   // - Invalid password
   // - Expired token
   ```

3. **Code Review**
   ```javascript
   // Review implementation against SPEC-01 requirements
   // Check:
   // - Security best practices
   // - Error handling
   // - Input validation
   ```

## Common Patterns 🔄

1. **Starting New Features**
   ```javascript
   // Implementing TASK-YYYY-MM-DD-XX
   // From SPEC-XX: [Brief description]
   // Requirements:
   // 1. [Requirement 1]
   // 2. [Requirement 2]
   ```

2. **Adding Tests**
   ```javascript
   // Testing requirements from SPEC-XX
   // Test cases:
   // 1. [Test case 1]
   // 2. [Test case 2]
   ```

## Need Help? 🤔

- Check GitHub Copilot documentation for advanced features
- Reference the main README.md for workflow details
- Use clear, descriptive comments to guide Copilot
- Keep specifications and tasks open while working

Remember: GitHub Copilot works best with clear guidance through comments and structured code. The more context you provide through specifications and tasks, the better the suggestions will be.
