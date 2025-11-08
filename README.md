# Requirements Engineering Knowledge Catalog System - Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Architecture & Design](#architecture--design)
4. [Usage Guide](#usage-guide)
5. [Adding New Frameworks](#adding-new-frameworks)
6. [Query API](#query-api)
7. [LLM Integration](#llm-integration)
8. [Examples](#examples)

## Overview

This system provides a hierarchical, machine-readable catalog for requirements engineering frameworks and their definitions. It's designed to support LLM-based requirements classification by providing structured knowledge about various requirement types and their characteristics according to different frameworks.

### Key Features
- Hierarchical organization of requirement categories
- Multiple framework definitions per category
- Keyword-based search capabilities
- Framework comparison tools
- LLM-ready query interface
- JSON/YAML serialization support

## Installation & Setup

### Requirements
```python
# Python 3.8+
pip install pyyaml
```

### Basic Setup
```python
from requirements_catalog import RequirementsCatalog, populate_initial_catalog

# Create a new catalog
catalog = RequirementsCatalog()

# Populate with initial data
populate_initial_catalog(catalog)
```

## Architecture & Design

### Core Classes

#### 1. **Citation**
Represents bibliographic information for framework sources.

#### 2. **FrameworkDefinition**
Contains a complete framework definition including:
- Name and authors
- Definition text
- Keywords list
- Characteristics
- Citation information
- Examples
- Related frameworks

#### 3. **RequirementCategory**
Represents a hierarchical category with:
- Parent/child relationships
- Multiple framework definitions
- Aliases and descriptions

#### 4. **RequirementsCatalog**
Main catalog class managing:
- Root hierarchy
- Category navigation
- Search operations
- Serialization

#### 5. **LLMCatalogInterface**
Standardized interface for LLM interaction with:
- Query processing
- Classification support
- Prompt templates

### Data Structure

```
Requirements (Root)
├── Functional Requirements
│   └── [Framework Definitions]
├── Non-Functional Requirements
│   ├── Performance
│   │   └── [Framework Definitions]
│   ├── Security
│   │   └── [Framework Definitions]
│   ├── Usability
│   │   └── [Framework Definitions]
│   └── [Other NFR Categories...]
└── Mixed Requirements
    └── [Framework Definitions]
```

## Usage Guide

### Creating a Catalog

```python
from requirements_catalog import RequirementsCatalog

# Initialize empty catalog
catalog = RequirementsCatalog()

# The catalog automatically creates standard categories:
# - Functional Requirements
# - Non-Functional Requirements (with subcategories)
# - Mixed Requirements
```

### Adding New Frameworks

```python
from requirements_catalog import FrameworkDefinition, Citation

# Create a framework definition
framework = FrameworkDefinition(
    name="My Framework",
    authors=["Author Name"],
    year=2024,
    definition="Framework definition text here",
    keywords=["keyword1", "keyword2", "keyword3"],
    characteristics=[
        "Characteristic 1",
        "Characteristic 2"
    ],
    citation=Citation(
        authors=["Author Name"],
        year=2024,
        title="Framework Title",
        publication="Publication Name"
    ),
    examples=[
        "Example requirement 1",
        "Example requirement 2"
    ]
)

# Add to catalog
catalog.add_framework(
    ["Non-Functional Requirements", "Performance"],  # Category path
    framework
)
```

### Adding New Categories

```python
# Add a new NFR subcategory
catalog.non_functional.add_subcategory(
    "Testability",
    "The ease with which software can be tested"
)

# Add a framework to the new category
catalog.add_framework(
    ["Non-Functional Requirements", "Testability"],
    framework_definition
)
```

## Query API

### Search by Keyword

```python
# Search for frameworks containing a keyword
results = catalog.search_by_keyword("encryption")

for category_path, framework in results:
    print(f"Found in: {category_path}")
    print(f"Framework: {framework.name}")
    print(f"Definition: {framework.definition}")
```

### Get Frameworks by Category

```python
# Get all frameworks in a category
frameworks = catalog.get_frameworks_by_category("Security", recursive=True)

for framework in frameworks:
    print(f"{framework.name}: {framework.definition}")
```

### Get Specific Framework Definition

```python
# Get definitions from a specific framework
definitions = catalog.get_definition_by_framework("NFR Framework", "Performance")

for definition in definitions:
    print(definition)
```

### Compare Frameworks

```python
# Compare all frameworks in a category
comparisons = catalog.compare_definitions("Performance")

for framework_name, definition in comparisons.items():
    print(f"{framework_name}:")
    print(f"  {definition}")
```

## LLM Integration

### Using the LLM Interface

```python
from requirements_catalog import LLMCatalogInterface

# Create interface
llm_interface = LLMCatalogInterface(catalog)

# Process queries
query = {
    "action": "search_keyword",
    "parameters": {
        "keyword": "throughput"
    }
}

result = llm_interface.process_query(query)
```

### Available Actions

1. **search_keyword**: Find frameworks containing a keyword
2. **get_definition**: Get specific framework definition
3. **classify**: Classify a requirement text
4. **compare**: Compare frameworks in a category

### Prompt Templates

```python
# Get prompt template for classification task
template = llm_interface.generate_prompt_template("classify")

# Use template with requirement text
prompt = template.format(requirement_text="The system shall process 1000 requests per second")
```

### Example LLM Query Flow

```python
# 1. User inputs requirement text
requirement = "The system shall authenticate users within 2 seconds"

# 2. LLM extracts keywords and queries catalog
classification_result = llm_interface.classify_requirement(requirement)

# 3. Process result
print(f"Classification: {classification_result['classification']}")
print(f"Keywords: {classification_result['keywords_found']}")
print(f"Matched Categories: {classification_result['matched_categories']}")
```

## Examples

### Example 1: Building a Custom Catalog

```python
# Create catalog
catalog = RequirementsCatalog()

# Add custom framework for Agile requirements
agile_framework = FrameworkDefinition(
    name="Agile User Story",
    authors=["Mike Cohn"],
    year=2004,
    definition="As a [user role], I want [goal] so that [benefit]",
    keywords=["user story", "agile", "scrum", "user role", "goal", "benefit"],
    characteristics=[
        "Focus on user value",
        "Brief and conversational",
        "Testable"
    ]
)

catalog.add_framework(["Functional Requirements"], agile_framework)
```

### Example 2: Batch Processing Requirements

```python
requirements = [
    "The system shall display user profiles",
    "Response time shall be under 3 seconds",
    "Users shall be authenticated using OAuth2"
]

for req in requirements:
    result = llm_interface.classify_requirement(req)
    print(f"{req[:50]}... -> {result['classification']}")
```

### Example 3: Exporting Catalog

```python
# Save to JSON
catalog.save_to_json("my_catalog.json")

# Save to YAML
catalog.save_to_yaml("my_catalog.yaml")

# Get statistics
stats = catalog.get_statistics()
print(f"Total frameworks: {stats['total_frameworks']}")
```

## Data Format Specifications

### JSON Structure

```json
{
  "name": "Requirements",
  "description": "Root category",
  "frameworks": [],
  "subcategories": {
    "Functional Requirements": {
      "name": "Functional Requirements",
      "description": "...",
      "frameworks": [
        {
          "name": "IEEE 830 Standard",
          "authors": ["IEEE"],
          "year": 1998,
          "definition": "...",
          "keywords": ["behavior", "function"],
          "characteristics": ["..."],
          "citation": "IEEE (1998). ...",
          "examples": ["..."]
        }
      ],
      "subcategories": {}
    }
  }
}
```

### YAML Structure

```yaml
name: Requirements
description: Root category
frameworks: []
subcategories:
  Functional Requirements:
    name: Functional Requirements
    description: ...
    frameworks:
      - name: IEEE 830 Standard
        authors: [IEEE]
        year: 1998
        definition: ...
        keywords: [behavior, function]
        characteristics: [...]
```

## Best Practices

1. **Framework Naming**: Use consistent naming convention (e.g., "Author's Framework Name")
2. **Keywords**: Include both general and specific keywords for better search
3. **Citations**: Always include proper citations for traceability
4. **Examples**: Provide concrete examples for each framework
5. **Categories**: Keep hierarchy depth reasonable (max 3-4 levels)
6. **Definitions**: Keep definitions concise but complete

## Extending the System

### Adding New Requirement Types

```python
# Add a new top-level category
catalog.root.add_subcategory(
    "Business Requirements",
    "High-level business objectives and constraints"
)
```

### Custom Search Functions

```python
def search_by_author(catalog, author_name):
    """Search for all frameworks by a specific author"""
    results = []
    
    def search_category(category):
        for framework in category.frameworks:
            if author_name in framework.authors:
                results.append((category.get_path(), framework))
        
        for subcategory in category.subcategories.values():
            search_category(subcategory)
    
    search_category(catalog.root)
    return results
```

### Integration with Your Thesis Chatbot

```python
class RequirementsClassifier:
    def __init__(self, catalog, llm_model):
        self.catalog = catalog
        self.llm_interface = LLMCatalogInterface(catalog)
        self.llm = llm_model
    
    def classify_requirement(self, text):
        # 1. Extract keywords using LLM
        keywords = self.llm.extract_keywords(text)
        
        # 2. Query catalog for matching frameworks
        matches = []
        for keyword in keywords:
            results = self.catalog.search_by_keyword(keyword)
            matches.extend(results)
        
        # 3. Use LLM with catalog knowledge for final classification
        context = self.build_context(matches)
        classification = self.llm.classify_with_context(text, context)
        
        return classification
```

## Troubleshooting

### Common Issues

1. **KeyError when accessing categories**: Ensure category exists before accessing
2. **Empty search results**: Check keyword spelling and try broader terms
3. **Serialization errors**: Ensure all objects are serializable (no lambda functions)

### Performance Optimization

- Use keyword indexing for large catalogs
- Implement caching for frequently accessed frameworks
- Consider database backend for very large catalogs (>1000 frameworks)

## Contact & Support

For questions about this catalog system in the context of your thesis on requirements classification, refer to the thesis documentation or contact your supervisor.
