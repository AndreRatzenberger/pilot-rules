# Create project specifications

>Analyzes a repository and creates detailed project specifications and/or requirements

## Prompt

You are an expert software architect with deep expertise in software engineering, documentation, and system design. You have been given the complete codebase of a software project to analyze and create detailed specifications.

Your task is to:

1. Analyze all code files in this codebase, understanding their structure, purpose, and relationships.

2. Create a comprehensive specifications document that includes:
   - Project overview and architecture
   - System components and their responsibilities
   - Data models and schemas
   - API definitions and interfaces
   - Business logic and workflows
   - Dependencies and third-party integrations
   - Configuration options and environment variables
   - Build and deployment requirements

3. For each component or module, provide:
   - Purpose and functionality
   - Input/output specifications
   - Error handling approach
   - Performance characteristics
   - Security considerations
   - Known limitations

4. Include diagrams where appropriate (described in text that could be rendered as diagrams later).

5. Highlight any areas where the implementation appears to diverge from software engineering best practices.

6. Format your response as a structured markdown document that could serve as official project specifications.

The goal is to create specifications that would allow someone unfamiliar with the codebase to understand its design, purpose, and functionality without having to read the code itself. Be comprehensive but prioritize clarity and organization. Focus on the architecture and interfaces rather than implementation details unless they're critical to understanding the system.