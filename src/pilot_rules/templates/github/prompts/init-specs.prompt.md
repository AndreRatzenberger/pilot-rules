# Command Prompt for Coding Agent: Specification Management

## Objective

Ensure specifications are properly maintained, understood, and kept in sync with the codebase throughout the project lifecycle, following enhanced standards for priority levels, testing criteria, and progress tracking.

## Steps - Follow this exact order!

Prerequisites
Get current system information like OS and current date
[specs-definition.mdc](mdc:.cursor/rules/specs-definition.mdc)

1. **Read and Understand Existing Specifications:**
    - Load all specification files from the `.project/specs` directory.
    - Parse the SPECS.md index file to understand the overall specification landscape.
    - For each specification:
        - Extract key metadata (ID, status, creation date, last updated)
        - Identify requirements and their completion status, noting priority levels
        - Note any relationships to other specifications
        - Review testing criteria sections

2. **Cross-Check Specifications with Code:**
    - For each specification:
        - Identify code files and components that implement the specification
        - Verify if requirements in the specification match the implemented code
        - Check if code changes have been made that are not reflected in specifications
        - Determine if there are specification requirements without corresponding implementations
    - Generate a sync status report highlighting:
        - In-sync specifications and code
        - Out-of-date specifications
        - Code without corresponding specifications

3. **Update Specifications Based on User Requests:**
    - **Adding New Specifications:**
        - Generate a unique specification ID using the format SPEC-NN-descriptive-name
        - Create a new specification file in the `.project/specs` directory
        - Format according to the enhanced specification template including:
            - Requirement sections with priority levels [HIGH/MEDIUM/LOW]
            - Testing criteria sections for unit, integration, and other relevant tests
            - Technical details for architecture and implementation considerations
        - Update SPECS.md to include the new specification
        - Update progress tracking statistics in SPECS.md
    
    - **Modifying Existing Specifications:**
        - Locate the specification file to be modified
        - Update the content as requested by the user
        - Ensure requirements include appropriate priority levels
        - Verify testing criteria are comprehensive and aligned with requirements
        - Update the "Last Updated" metadata
        - Update completion percentage if requirements have changed
        - Update SPECS.md and progress tracking statistics
    
    - **Deleting Specifications:**
        - Confirm with the user before deleting any specification
        - Remove the specification file from the `.project/specs` directory
        - Update SPECS.md to remove references to the deleted specification
        - Update progress tracking statistics
        - Identify any tasks or other specifications that reference the deleted one

4. **Maintain SPECS.md Progress Tracking:**
    - Calculate and update completion statistics for each priority level
    - Generate progress bars for visual tracking
    - Update the specification status summary table
    - Ensure all statistics are consistent with the actual specification files

5. **Validate Updated Specifications:**
    - Ensure all specifications follow the required format
    - Verify that requirements are clear, specific, testable, and have appropriate priority levels
    - Check that testing criteria are comprehensive and aligned with requirements
    - Confirm that technical details are sufficient for implementation
    - Check that metadata is complete and accurate
    - Verify that SPECS.md index and progress tracking are in sync with specification files

6. **Report Status:**
    - Provide a summary of specification changes
    - Present progress statistics by priority level
    - Highlight any specifications that need attention (out of sync with code)
    - Suggest next steps for maintaining specification-code alignment

7. **Technical Architecture Considerations:**
    - For new projects or major features, ensure a dedicated technical architecture specification exists
    - Verify that architectural decisions are documented and rationalized
    - Ensure technology stack and dependencies are clearly specified
    - Check that security and performance considerations are addressed