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



# Specification Files Naming Convention

## Index File
- **Name**: `SPECS.md`
- **Location**: Root of the specifications directory

## Individual Specification Files

### Base Location
All specification files are stored in the `specs/` directory.

### Naming Format
- Use kebab-case for all specification filenames
- Format: `specs/[component-name].md`

### Component-Specific Naming Patterns
| Component Type | Filename Pattern | Example |
|----------------|------------------|---------|
| Modules/Packages | `[module-name].md` | `specs/authentication.md` |
| Classes/Components | `[component-name].md` | `specs/user-profile.md` |
| Services | `[service-name]-service.md` | `specs/email-service.md` |
| Utilities | `[utility-name]-utils.md` | `specs/string-utils.md` |
| API Endpoints | `[endpoint-name]-api.md` | `specs/user-api.md` |
| Data Models | `[model-name]-model.md` | `specs/customer-model.md` |
| Configuration | `[config-name]-config.md` | `specs/database-config.md` |

### Directory Structure (for larger projects)
```
SPECS.md
specs/
├── core/
│   ├── core-component-1.md
│   └── core-component-2.md
├── services/
│   ├── service-1-service.md
│   └── service-2-service.md
├── models/
│   ├── model-1-model.md
│   └── model-2-model.md
├── api/
│   ├── endpoint-1-api.md
│   └── endpoint-2-api.md
└── utils/
    ├── utility-1-utils.md
    └── utility-2-utils.md
```

### Cross-Cutting Concerns
- Format: `specs/cross-cutting-[concern].md`
- Examples:
  - `specs/cross-cutting-security.md`
  - `specs/cross-cutting-logging.md`
  - `specs/cross-cutting-error-handling.md`

# Project Specifications Index - SPECS.md

## Overview
[Brief project description: 2-3 sentences explaining the project's purpose]

## System Architecture
[High-level architecture diagram description or reference]

## Components Index
[Alphabetical or hierarchical list of all components with links to their detailed specs]

## Key Interfaces
[List of critical interfaces in the system with links to their detailed specs]

## Data Models
[List of primary data structures with links to their detailed specs]

## Cross-Cutting Concerns
- Authentication & Authorization
- Logging & Monitoring
- Error Handling
- Performance Considerations
- Security Model

## Dependencies
[List of external dependencies and integrations]

## Environment Configuration
[Overview of configuration options]

## Build & Deployment
[Summary of build process and deployment requirements]

## Navigation Guide
[Instructions on how to use the specs documentation effectively]