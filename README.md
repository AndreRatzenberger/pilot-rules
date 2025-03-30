# Copilot-Toolkit 🚀 A toolkit for coding agents 🚀 

Get the most out of GitHub Copilot, Cursor and Co. Powered by flock.

This repository contains a collection of tools and configuration like custom rules for AI-assisted development that significantly enhance your productivity and code quality. These tools provide structure, automation, and consistency to your development process, regardless of which AI assistant you use.

Expand the abilities of GitHub Copilot, Cursor and other LLM-based development agents.
For example with the help of custom rules enable Copilot to follow project management best practices which will improve the quality of the code and the project tenfold.

Do custom code analysis and share the result with your coding agents, or create automatic documentations.


## Quick Setup 🛠️ - No Installation

Requirements: [`uv`](https://github.com/astral-sh/uv)

You can use everything `copilot-toolkit` has to offer without installing anything else — just make sure `uv` is available on your system.

---

## Non-agentic Features

### Scaffold Rules

```bash
# For Cursor AI
uvx pilot-rules --cursor

# For GitHub Copilot
uvx pilot-rules --copilot
```

---

### Collect Code

```bash
# Collect all .py files in the current directory and subdirectories
# Writes output to code.md in the current directory

uvx pilot-rules --collect

# which is the same as
uvx pilot-rules --collect \
  --include "py:." \
  --output repository_analysis.md
```

```bash
# Collect multiple file types from multiple roots
# Excludes paths containing 'external'
# syntax
# <file extensions>:<folder>
# *:. 
# all files in all folders
# Writes output to .project/code.md
uvx pilot-rules --collect \
  --include "py,js:." \
  --exclude "*:external" \
  --include "md:docs" \
  --output .project/code.md
```

### Agentic Features

Agentic Features are abusing making use of Gemini's immense context window.
Paired with the improved abilities of Gemini 2.5 results in tools that were not possible ever before.

```bash
# create specifications based on the current project
uvx copilot-toolkit --specs .project/

uvx copilot-toolkit --prompt specs --def specs.def --code .
```



---

### Use a Config File (Recommended for Complex Setups)

Instead of passing everything via CLI, you can define your sources and output in a simple `.toml` file.

Example `pilot-rules.toml`:

```toml
[[source]]
exts = ["py", "js"]
root = "."
exclude = ["external"]

[[source]]
exts = ["md"]
root = "docs"

output = ".project/code.md"
```

Then run:

```bash
uvx pilot-rules --collect --config pilot-rules.toml
```

---

Let your coding agents work *with* your rules — not against them.

### Permanent Installation
```bash
# Install globally with pipx (recommended)
pipx install pilot-rules

# Then use from anywhere
pilot-rules --cursor
pilot-rules --copilot
```

The tool will:
- Create the necessary directory structure
- Set up initial configuration files
- Add tool-specific templates
- Display a getting started guide


## The Future of Development is Here 🌟

AI is transforming software development, and these rules represent a significant leap forward in this evolution. This repository showcases a fundamental paradigm shift in how we build software:

**From:** Developers manually writing every line of code  
**To:** Developers creating specifications that guide AI assistants to generate implementation

This approach fundamentally changes the developer's role from manual coding to:
1. Defining clear specifications for what should be built
2. Establishing technical requirements for how code should be written
3. Supervising the AI as it implements solutions
4. Reviewing and refining the output

By adopting this rules-based approach, developers can dramatically accelerate productivity while still maintaining control.

## Core Principles 🧭

The AI-assisted development rules system is built on these foundational principles:

- **Task-driven Development:** Every code implementation must be tied to a clearly defined Task, which is associated with a corresponding Specification.
- **No Untracked Implementations:** No features should be implemented without a Task and Specification.
- **Automated Consistency:** The system continuously verifies synchronization between code, tasks, and specifications.

## Basic Workflow 🔄

The rules system introduces a streamlined workflow:

### 1. Project Initialization

Start by initializing your project structure:
- Create the `.project/specs` directory for specifications
- Initialize the `SPECS.md` index file for tracking specifications
- Create the `.project/tasks` directory for task management
- Initialize the `TASKS.md` index file for tracking tasks
- Set up basic project structure according to best practices

### 2. Specification Creation

Create specifications based on your ideas or requirements that include:
- Comprehensive specifications from your requirements
- Structured specification files in `.project/specs/` 
- Proper specification IDs (SPEC-NN-descriptive-name)
- Priority levels (HIGH/MEDIUM/LOW) for each requirement
- Testing criteria for implementation validation
- Progress tracking
- Cross-checking with existing code

### 3. Task Generation

Generate tasks for your development schedule that:
- Break down specifications into actionable tasks
- Create task files in `.project/tasks/` with unique IDs (TASK-YYYY-MM-DD-NN)
- Define clear acceptance criteria for each task
- Specify required tests for task completion
- Link tasks to their corresponding specifications
- Ensure tasks are properly sized (maximum 1 work day per task)
- Update the task index for easy tracking

## File Structure 📁

The rules system organizes your project with this structure:

```
project-root/
├── .project/
│   ├── specs/           # Specification files
│   │   └── SPEC-NN-*.md # Individual specification files
│   ├── SPECS.md         # Specification index and progress tracking
│   ├── tasks/           # Task files
│   │   └── TASK-*.md    # Individual task files
│   └── TASKS.md         # Task index and tracking
└── src/                 # Your source code
```

## Specification Format 📝

Specifications follow a standardized format:

- **Unique ID and Title:** Clear identification
- **Description:** Detailed feature requirements
- **Requirements:** Prioritized and checkable items
  ```
  - [ ] Requirement 1.1 [HIGH]: Authentication must use JWT
  - [ ] Requirement 1.2 [MEDIUM]: Token expiration after 15 minutes
  ```
- **Testing Criteria:** How implementation will be validated
- **Acceptance Criteria:** Clear success metrics
- **Metadata:** Tracking information (creation date, status, etc.)

## Task Format 📋

Tasks are structured for clarity and tracking:

- **Unique ID and Description:** Clear identification
- **Relevant Specification:** Link to corresponding spec
- **Acceptance Criteria:** Clear completion requirements
- **Tests:** Required test cases for verification
- **Metadata:** Tracking data (dates, status, etc.)
- **Complexity:** Assessment of difficulty
- **Learnings:** Insights gained during implementation

## Additional Features ✨

Beyond the core workflow, the rules system offers:

### Automatic Commit Management

The system intelligently manages git commits:
- Uses conventional commit format (`type(scope): description`)
- Determines appropriate commit types based on changes
- Manages README updates based on implementation changes
- Ensures test-verified changes are properly committed

## Benefits of AI-Assisted Development Rules 🌈

The rules system provides numerous advantages:
- **Structured Development Process** - Follow a consistent path from idea to implementation
- **Accelerated Productivity** - Skip boilerplate and focus on unique aspects of your project
- **Enhanced Quality** - Generate comprehensive specs that drive high-quality implementation
- **Improved Planning** - Create realistic task timelines with proper dependencies
- **Documentation-Driven** - Maintain thorough documentation throughout the project lifecycle


## Advanced Usage 🔧

Beyond the basic workflow, you can:

- Refine specifications with additional details
- Update priority levels as requirements evolve
- Track progress through specification completion percentages
- Regenerate tasks as priorities change
- Capture learnings during implementation for future reference

## Future Enhancements 🔮

The rules system is continuously evolving with planned additions:
- Automated testing integration
- Deployment workflow automation
- Performance analytics
- Integration with various project management tools
- Support for additional AI coding assistants

## Inspiration 💡

This structured AI development flow was inspired by [Geoffrey Huntley's method](https://ghuntley.com/specs/) of effectively using AI assistants. Huntley demonstrates how combining specifications with technical rules creates a powerful workflow that can dramatically increase development productivity.

As Huntley explains:
> "When you use '/specs' method with the 'stdlib' method in conjunction with a programming language that provides compiler soundness (driven by good types) and compiler errors, the results are incredible. You can drive hands-free output of N factor (entire weeks' worth) of co-workers in hours."

## Something Missing? 🤔

The rules system is designed to be extensible! You can create new rules that address your specific workflow needs. Check the tool-specific guides for details on how to implement custom rules with your chosen AI assistant.

## License 📜

MIT License - See [LICENSE](LICENSE) for details.

---

"If you want to build a ship, don't drum up people to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea." - Antoine de Saint-Exupéry

_In the same way, effective AI systems don't just execute code, but operate within a framework of principles and specifications that guide them toward building solutions that fulfill the true vision of what we seek to create._
