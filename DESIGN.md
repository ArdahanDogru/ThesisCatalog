# Requirements Catalog System - Design Document

## Task 2: Implementation Format Recommendation

### Evaluation of Implementation Options

#### Option A: Python Classes with Nested Structure ✅ **RECOMMENDED**

**Pros:**
- Full object-oriented design with methods and behaviors
- Type hints and validation built-in via dataclasses
- Easy to extend with new functionality
- Direct integration with Python-based LLM applications
- Can serialize to JSON/YAML when needed
- Memory efficient with lazy loading potential

**Cons:**
- Requires Python runtime
- More complex initial setup
- Not directly shareable with non-Python systems without serialization

**Best for:** Your thesis project, as it provides maximum flexibility and direct integration with your LLM classifier.

#### Option B: JSON/YAML Hierarchical Structure

**Pros:**
- Language agnostic
- Human readable and editable
- Simple to share and version control
- No runtime dependencies

**Cons:**
- No built-in methods or behaviors
- Requires parsing logic in application code
- No type safety or validation
- Can become unwieldy with large datasets
- Limited querying capabilities

**Best for:** Configuration files or data exchange between different systems.

#### Option C: SQLite Database with Relational Tables

**Pros:**
- Powerful SQL querying capabilities
- ACID compliance for data integrity
- Scales well for large datasets
- Built-in indexing for performance
- Can handle complex relationships

**Cons:**
- Requires SQL knowledge
- More complex setup and maintenance
- Hierarchical data requires recursive queries
- Overhead for small datasets
- Less intuitive for hierarchical structures

**Best for:** Production systems with thousands of frameworks and complex queries.

#### Option D: Graph-Based Structure (e.g., NetworkX)

**Pros:**
- Natural representation of relationships
- Powerful graph algorithms available
- Good for complex cross-references
- Visualization capabilities

**Cons:**
- Overkill for mostly hierarchical data
- Additional dependency (NetworkX)
- More complex to understand and maintain
- Memory intensive for large graphs

**Best for:** Systems with complex, non-hierarchical relationships between requirements.

### **Recommendation: Hybrid Approach (Python Classes + JSON/YAML)**

For your thesis project, I recommend **Option A (Python Classes)** as the primary implementation with JSON/YAML serialization for persistence. This provides:

1. **Development flexibility** with OOP design
2. **Easy integration** with your LLM classifier
3. **Portability** through JSON/YAML export
4. **Scalability** path to database if needed

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         User Input (Requirements)        │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│            LLM Classifier               │
│  (Mistral/Llama via Ollama)            │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│        LLM Catalog Interface            │
│   (Query Processing & Formatting)       │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│       Requirements Catalog              │
│  (Hierarchical Knowledge Structure)     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│      Persistent Storage                 │
│      (JSON/YAML Files)                  │
└─────────────────────────────────────────┘
```

### Data Flow

1. **Input**: User provides requirement text
2. **Processing**: LLM extracts keywords and patterns
3. **Query**: Interface queries catalog for relevant frameworks
4. **Retrieval**: Catalog returns matching definitions
5. **Classification**: LLM uses catalog knowledge for classification
6. **Output**: Classified requirement with justification

---

## LLM Integration Design (Task 7)

### Query Format Specification

```python
# Standard query structure
query = {
    "action": str,           # Required: search_keyword, get_definition, classify, compare
    "parameters": {          # Required: action-specific parameters
        "keyword": str,      # For search_keyword
        "framework": str,    # For get_definition
        "category": str,     # For get_definition, compare
        "requirement_text": str,  # For classify
    },
    "options": {            # Optional: additional options
        "max_results": int,
        "include_examples": bool,
        "recursive": bool
    }
}
```

### Response Format Specification

```python
# Standard response structure
response = {
    "status": "success" | "error",
    "action": str,              # Echo of requested action
    "data": {                   # Action-specific response data
        # For search_keyword
        "keyword": str,
        "matches": [
            {
                "category": str,
                "framework": str,
                "definition": str,
                "keywords": List[str],
                "score": float  # Relevance score
            }
        ],
        
        # For classify
        "requirement": str,
        "classification": str,
        "confidence": float,
        "reasoning": str,
        "matched_frameworks": List[str]
    },
    "metadata": {
        "timestamp": str,
        "catalog_version": str,
        "processing_time": float
    },
    "error": str  # Only if status == "error"
}
```

### Example Prompt Templates

#### Classification Prompt

```python
classification_prompt = """
You are a requirements engineering expert with access to a catalog of framework definitions.

CATALOG KNOWLEDGE:
{catalog_context}

TASK: Classify the following requirement based on the catalog definitions.

REQUIREMENT: "{requirement_text}"

ANALYSIS STEPS:
1. Identify key terms and patterns in the requirement
2. Match against framework definitions in the catalog
3. Determine if the requirement is:
   - Functional (describes what the system does)
   - Non-Functional (describes quality attributes)
   - Mixed (contains both aspects)

RESPONSE FORMAT:
- Classification: [Functional/Non-Functional/Mixed]
- Confidence: [0.0-1.0]
- Reasoning: [Explain classification based on catalog]
- Matched Frameworks: [List relevant frameworks]
- Keywords Found: [List matched keywords]
"""
```

#### NFR Extraction Prompt

```python
nfr_extraction_prompt = """
Based on the NFR Framework catalog, extract non-functional requirements from this text:

TEXT: "{requirement_text}"

AVAILABLE NFR CATEGORIES:
{nfr_categories}

For each NFR found, provide:
1. Category (Performance, Security, Usability, etc.)
2. Specific aspect mentioned
3. Relevant framework definition
4. Confidence score
"""
```

### Integration with Ollama/Local LLMs

```python
import requests
import json

class LLMCatalogClassifier:
    def __init__(self, catalog, llm_endpoint="http://localhost:11434/api/generate"):
        self.catalog = catalog
        self.llm_interface = LLMCatalogInterface(catalog)
        self.llm_endpoint = llm_endpoint
    
    def classify_with_llm(self, requirement_text):
        # 1. Query catalog for relevant context
        keywords = self.extract_keywords_simple(requirement_text)
        catalog_context = self.build_catalog_context(keywords)
        
        # 2. Prepare LLM prompt
        prompt = self.build_classification_prompt(requirement_text, catalog_context)
        
        # 3. Call Ollama API
        response = requests.post(self.llm_endpoint, json={
            "model": "mistral",
            "prompt": prompt,
            "temperature": 0.3,  # Lower for more consistent classification
            "format": "json"
        })
        
        # 4. Parse and return result
        llm_response = response.json()
        return self.parse_llm_classification(llm_response)
    
    def build_catalog_context(self, keywords):
        """Build relevant context from catalog"""
        context = []
        for keyword in keywords:
            results = self.catalog.search_by_keyword(keyword)
            for category, framework in results[:3]:  # Top 3 matches
                context.append({
                    "keyword": keyword,
                    "category": category,
                    "framework": framework.name,
                    "definition": framework.definition
                })
        return json.dumps(context, indent=2)
```

---

## Performance Considerations

### Optimization Strategies

1. **Keyword Indexing**
   - Build inverted index for O(1) keyword lookups
   - Update index when adding new frameworks

2. **Caching**
   - Cache frequently accessed frameworks
   - Cache LLM classification results for similar requirements

3. **Lazy Loading**
   - Load framework details only when needed
   - Keep lightweight index in memory

### Scalability Path

```python
# Future database migration path
class CatalogRepository:
    def __init__(self, storage_backend):
        self.backend = storage_backend  # JSON, SQLite, PostgreSQL
    
    def migrate_to_database(self, catalog):
        """Migrate from in-memory to database"""
        # Create tables
        self.backend.create_schema()
        
        # Migrate data
        for category in catalog.get_all_categories():
            self.backend.insert_category(category)
        
        for framework in catalog.get_all_frameworks():
            self.backend.insert_framework(framework)
```

---

## Testing Strategy

### Unit Tests

```python
import unittest

class TestRequirementsCatalog(unittest.TestCase):
    def setUp(self):
        self.catalog = RequirementsCatalog()
        populate_initial_catalog(self.catalog)
    
    def test_search_keyword(self):
        results = self.catalog.search_by_keyword("encryption")
        self.assertGreater(len(results), 0)
    
    def test_add_framework(self):
        initial_count = len(self.catalog.get_all_frameworks())
        self.catalog.add_framework(
            ["Test Category"],
            FrameworkDefinition(name="Test", authors=["Test"], year=2024, definition="Test")
        )
        self.assertEqual(len(self.catalog.get_all_frameworks()), initial_count + 1)
```

### Integration Tests

```python
def test_llm_classification():
    catalog = RequirementsCatalog()
    populate_initial_catalog(catalog)
    interface = LLMCatalogInterface(catalog)
    
    test_requirements = [
        ("The system shall authenticate users", "Functional"),
        ("Response time shall be under 2 seconds", "Non-Functional"),
        ("The system shall encrypt and store user data securely", "Mixed")
    ]
    
    for req_text, expected_class in test_requirements:
        result = interface.classify_requirement(req_text)
        assert result['classification'] == expected_class
```

---

## Deployment Considerations

### File Structure

```
thesis_project/
├── requirements_catalog/
│   ├── __init__.py
│   ├── models.py           # Data models (Citation, FrameworkDefinition, etc.)
│   ├── catalog.py          # Main catalog class
│   ├── interface.py        # LLM interface
│   └── data/
│       ├── initial_catalog.json
│       └── custom_frameworks.yaml
├── tests/
│   ├── test_catalog.py
│   └── test_llm_interface.py
├── examples/
│   └── classification_demo.py
├── README.md
└── requirements.txt
```

### Configuration

```yaml
# config.yaml
catalog:
  storage_format: "json"
  storage_path: "./data/catalog.json"
  auto_save: true
  
llm:
  model: "mistral"
  endpoint: "http://localhost:11434/api/generate"
  temperature: 0.3
  max_tokens: 500
  
classification:
  confidence_threshold: 0.7
  use_ensemble: false
  cache_results: true
```

---

## Future Enhancements

1. **Web Interface**
   - Flask/FastAPI REST API
   - Web UI for catalog management
   - Real-time classification interface

2. **Advanced Features**
   - Similarity scoring between frameworks
   - Automatic framework extraction from papers
   - Version control for framework definitions
   - Multi-language support

3. **ML Enhancements**
   - Fine-tune LLM on catalog data
   - Learn from classification corrections
   - Automatic keyword extraction

4. **Integration**
   - VS Code extension for requirement classification
   - JIRA/Azure DevOps plugins
   - Requirements management tool integration

---

## Conclusion

This design provides a robust, extensible foundation for your requirements classification system. The Python-based implementation offers the flexibility needed for research while maintaining a clear path to production deployment. The hierarchical catalog structure aligns well with existing requirements engineering frameworks and can be easily extended as you discover new patterns in your research.
