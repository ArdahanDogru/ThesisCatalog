# Requirements Catalog - Framework Sources Documentation

## Updated Catalog Structure

The catalog has been updated to focus on established software engineering frameworks from authoritative sources in the literature. The Mixed Requirements category has been removed as requested.

## Framework Sources Included

### 1. **Boehm's Software Quality Tree (1976)**
- **Source**: Boehm, B. W., Brown, J. R., & Lipow, M. (1976). "Quantitative evaluation of software quality"
- **Key Contribution**: One of the earliest hierarchical models of software quality
- **Categories Covered**: 
  - Overall quality structure (as-is utility, maintainability, portability)
  - Reliability (accuracy, completeness, consistency)
  - Efficiency (device efficiency, accessibility, resource usage)

### 2. **McCall's Software Quality Model (1977)**
- **Source**: McCall, J. A., Richards, P. K., & Walters, G. F. (1977). "Factors in Software Quality"
- **Key Contribution**: 11 quality factors organized into three perspectives
- **Categories Covered**:
  - Product Operation (Reliability, Usability, Efficiency)
  - Product Revision (Maintainability)
  - Product Transition (Portability, Reusability)

### 3. **Grady's FURPS+ Model (1992)**
- **Source**: Grady, R. B. (1992). "Practical Software Metrics for Project Management and Process Improvement"
- **Key Contribution**: Developed at Hewlett-Packard, widely used in industry
- **Categories Covered**:
  - **F**unctionality (removed focus as requested)
  - **U**sability (human factors, aesthetics, documentation)
  - **R**eliability (MTTF, recoverability, accuracy)
  - **P**erformance (speed, efficiency, throughput)
  - **S**upportability (testability, extensibility, maintainability)
  - **+** (Design, Implementation, Interface, Physical requirements)

### 4. **Chung's NFR Framework (2000)**
- **Source**: Chung, L., Nixon, B. A., Yu, E., & Mylopoulos, J. (2000). "Non-Functional Requirements in Software Engineering"
- **Key Contribution**: Comprehensive treatment of NFRs with goal-oriented modeling
- **Categories Covered**:
  - Performance (throughput, response time, space/time constraints)
  - Security (confidentiality, integrity, availability)
  - Usability (learnability, user-friendliness, interface)
  - Reliability (accuracy, robustness, fault-tolerance)

### 5. **Sommerville's Classification (2016)**
- **Source**: Sommerville, I. (2016). "Software Engineering" (10th Edition)
- **Key Contribution**: Modern, practical classification used in education
- **Categories Covered**:
  - Performance (timing and speed constraints)
  - Security (protection against attacks, access control)

### 6. **IEEE 830 Standard (1998)**
- **Source**: IEEE Std 830-1998. "IEEE Recommended Practice for Software Requirements Specifications"
- **Key Contribution**: Industry standard for requirements specification
- **Categories Covered**:
  - Functional Requirements (basic definition with simple examples)
  - Performance (numerical requirements, speed, response time)
  - Reliability (MTBF, MTTR, availability)

## Why These Sources?

These frameworks were selected because they:

1. **Historical Significance**: Boehm (1976) and McCall (1977) are foundational works that established the field of software quality modeling

2. **Industry Adoption**: Grady's FURPS+ (1992) was developed and used at HP, representing real-world industrial practice

3. **Academic Authority**: Chung et al. (2000) provides the most comprehensive academic treatment of NFRs with extensive citations

4. **Educational Standard**: Sommerville (2016) is one of the most widely-used software engineering textbooks

5. **Standardization**: IEEE 830 represents official industry standards

## Key Differences Between Frameworks

### Performance Definitions:
- **Boehm**: Focuses on "efficiency" and resource usage
- **McCall**: Emphasizes execution and storage efficiency
- **Grady**: Practical metrics (response time, throughput)
- **Chung**: Comprehensive (time, space, throughput)
- **IEEE**: Numerical requirements focus

### Reliability Definitions:
- **Boehm**: Accuracy and precision
- **McCall**: Error tolerance and consistency
- **Grady**: MTTF and recoverability
- **Chung**: Fault tolerance and robustness
- **IEEE**: MTBF, MTTR, availability

### Evolution Over Time:
- 1970s: Focus on basic quality factors
- 1990s: Practical metrics and measurements
- 2000s: Goal-oriented approaches
- 2010s: Integration with agile and modern practices

## Usage in Your Thesis

This catalog structure directly supports your thesis objectives by:

1. **Providing Multiple Perspectives**: Each framework offers different keywords and definitions that your LLM classifier can leverage

2. **Historical Context**: Shows evolution of NFR understanding over 40+ years

3. **Keyword Richness**: Each framework provides unique keywords for classification
   - Performance: throughput, response time, efficiency, speed, capacity
   - Security: confidentiality, integrity, authentication, protection
   - Reliability: MTBF, fault-tolerance, accuracy, robustness

4. **Classification Support**: Multiple definitions help identify ambiguous requirements that might be classified differently by different frameworks

## Focus on NFRs

As requested, the catalog now emphasizes Non-Functional Requirements with:
- **23 NFR framework definitions** across different categories
- **1 simple Functional Requirement definition** (IEEE 830) with basic examples
- **Removed**: Mixed Requirements category

The functional requirements section contains only simple examples:
- "The system shall calculate the total price"
- "The system shall generate monthly reports"  
- "The system shall validate user input"

This shift allows your classifier to focus on the more challenging NFR detection and classification task.
