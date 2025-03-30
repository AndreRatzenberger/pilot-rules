# Getting Started with Cursor AI Rules 🎯

This guide will help you get started with the rules-based development workflow in just three simple steps.

## Step 1: Initialize Project Structure 🏗️

1. Type `@init-project` in your editor
2. Select the rule when prompted
3. This will set up your project with the required directory structure:
   ```
   .project/
   ├── specs/           # For your specifications
   │   └── SPECS.md     # Specification index
   ├── tasks/           # For your tasks
   │   └── TASKS.md     # Task index
   └── src/            # Your source code
   ```

## Step 2: Generate Specifications 📝

1. Type `@init-specs` followed by your project description
2. Example: `@init-specs generate the specs for a webshop build with django`
3. Cursor will create structured specifications in `.project/specs/`

## Step 3: Create Tasks ✅

1. Type `@init-tasks` followed by your planning request
2. Example: `@init-tasks generate tasks for the next two days`
3. Cursor will create task files in `.project/tasks/`

That's it! Your project is now set up with specifications and tasks. Start coding with Cursor AI!

Need help? Press `Cmd/Ctrl + K` to chat with Cursor AI.
