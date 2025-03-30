---
description: 
globs: 
alwaysApply: false
---
# Command Prompt for Coding Agent: Task Management

## Objective

Ensure tasks are properly tracked, understood, and maintained throughout the project lifecycle, with special attention to recent task history and current work items.

## Steps - Follow this exact order!

Prerequisites:
Get current system information like OS and current date
[tasks-definition.mdc](mdc:.cursor/rules/tasks-definition.mdc)

1. **Read and Understand Task History:**
    - Load all task files from the `.project/tasks` directory
    - Parse the TASKS.md index file to understand the overall task landscape
    - Identify and analyze recently completed tasks:
        - Extract the last 3-5 completed tasks
        - Note their completion dates and outcomes
        - Understand patterns in implementation approaches
    - Track current work in progress:
        - Identify tasks marked as "in_progress"
        - Note their dependencies and blockers
        - Check their associated specifications

2. **Task Status Assessment:**
    - For each active task:
        - Check completion status against Definition of Done
        - Verify alignment with associated specifications
        - Review dependencies and their current states
        - Identify any blockers or risks
    - Generate a status overview showing:
        - Tasks completed in the last week
        - Currently in-progress tasks
        - Blocked or at-risk tasks
        - Upcoming tasks based on dependencies

3. **Update Tasks Based on User Requests:**
    - **Creating New Tasks:**
        - Generate a unique task ID using the format TASK-YYYY-MM-DD-NN
        - Create new task file in the `.project/tasks` directory
        - Format according to the task template
        - Link to relevant specifications
        - Update TASKS.md to include the new task
    
    - **Modifying Existing Tasks:**
        - Locate the task file to be modified
        - Update status, progress, or requirements
        - Update completion percentage if applicable
        - Add implementation notes or blockers
        - Update TASKS.md if necessary
    
    - **Completing Tasks:**
        - Verify all Definition of Done criteria are met
        - Update task status to "closed"
        - Document completion date and final outcomes
        - Update any dependent tasks
        - Update TASKS.md to reflect completion

4. **Validate Task Updates:**
    - Ensure all tasks follow the required format
    - Verify that acceptance criteria are clear and testable
    - Check that metadata and dependencies are accurate
    - Confirm that the TASKS.md index is in sync
    - Validate relationships with specifications

5. **Report Status:**
    - Provide a summary of recent task changes
    - Highlight current work in progress
    - List upcoming tasks based on dependency chain
    - Suggest next steps or potential blockers to address

6. **Handle User Requests:**
    - Parse and understand the user's task-related request
    - Verify request against existing tasks and specifications
    - Execute requested changes while maintaining consistency
    - Update all related documentation
    - Provide clear feedback on actions taken