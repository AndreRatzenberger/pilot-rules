# Command Prompt for Coding Agent: Project Validation and Initialization

## Objective

Ensure the current workspace is a valid project. Support projects created with UV, projects with `.venv` and `requirements.txt`, and projects with Conda.

## Steps - Follow this exact order!

Prerequisites
Get current system information like OS and current date

1. **Check for Valid Project:**
    - Verify if the project is managed with UV by checking for a `pyproject.toml` in uv format.
    - If not project: ask the user if they want to initialize the project.
        - UV: `uv init --package` followed by `uv sync`


2. **Check for Valid Specs Structure:**
    - [specs-definition.mdc](mdc:.cursor/rules/specs-definition.mdc)
    - Verify if the `.project/specs` directory exists.
    - Check if the `SPECS.md` index file exists in the `.project` directory.

3. **Initialize if Missing:**
    - If either component is missing, create the necessary directory structure and files.
    - Create the `.project/specs` directory if it doesn't exist.
    - Create the `SPECS.md` index file with proper formatting if it doesn't exist.

4. **Report Status:**
    - Inform the user about the current status of the specs structure.
    - Provide confirmation that initialization was successful, if performed.


5. **Check for Valid Tasks Structure:**
    - [tasks-definition.mdc](mdc:.cursor/rules/tasks-definition.mdc)
    - Verify if the `.project/tasks` directory exists.
    - Check if the `TASKS.md` index file exists in the `.project` directory.

6. **Initialize if Missing:**
    - If either component is missing, create the necessary directory structure and files.
    - Create the `.project/tasks` directory if it doesn't exist.
    - Create the `TASKS.md` index file with proper formatting if it doesn't exist.

7. **Report Status:**
    - Inform the user about the current status of the tasks structure.
    - Provide confirmation that initialization was successful, if performed.
