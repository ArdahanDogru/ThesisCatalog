"""
LLM-Based Requirements Classifier - Definition 3 Compliant
Uses NFR Framework softgoal approach to classify requirements as FR, NFR, or MIXED
"""

import pandas as pd
import re
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# LLM SETUP
# ============================================================================

MODEL = "llama3:8b"
llm = OllamaLLM(model=MODEL, temperature=0, max_tokens=300, stop=["}"])

# ============================================================================
# STEP 1: DETECT NFR SOFTGOALS (Quality Attributes)
# ============================================================================

NFR_DETECTION_PROMPT = """You are an expert in software requirements analysis using the NFR Framework.

Your task: Identify NFR softgoals (high-level quality attributes) explicitly mentioned in the requirement.

NFR Softgoals (Quality Attributes):
- Performance: Speed, response time, throughput
- Security: Protection, authorization, authentication
- Usability: Ease of use, user-friendliness
- Reliability: Availability, fault tolerance
- Maintainability: Modifiability, extensibility
- Scalability: Ability to handle growth
- Look & Feel: Appearance, visual design

CRITICAL RULES:
1. Only detect NFR softgoals that are EXPLICITLY mentioned or strongly implied by keywords
2. Temporal constraints (e.g., "within 30 seconds", "every 60 seconds") indicate Performance
3. Security language (e.g., "authorized users", "unauthorized") indicates Security
4. Visual language (e.g., "color code") indicates Look & Feel
5. If no clear quality attribute is mentioned, return empty list

Examples:

"System shall respond within 2 seconds"
→ {{"nfr_softgoals": ["Performance"]}}

"Only authorized users can access"
→ {{"nfr_softgoals": ["Security"]}}

"System shall display data from database"
→ {{"nfr_softgoals": []}}

"System must be available 99.9% of time"
→ {{"nfr_softgoals": ["Reliability"]}}

Return ONLY JSON format:
{{"nfr_softgoals": ["list", "of", "attributes"]}}

Requirement: {requirement}"""

nfr_detection_template = ChatPromptTemplate.from_template(NFR_DETECTION_PROMPT)
nfr_detection_chain = nfr_detection_template | llm

# ============================================================================
# STEP 2: DETECT OPERATIONALIZING SOFTGOALS (Mechanisms)
# ============================================================================

OP_DETECTION_PROMPT = """You are an expert in software requirements analysis using the NFR Framework.

Your task: Identify operationalizing softgoals (concrete mechanisms/operations) mentioned in the requirement.

Operationalizing Softgoals (Mechanisms):
- Authentication, Authorization, Encryption
- Caching, Load Balancing, Redundancy
- Search, Generate, Display, Refresh
- Synchronization, Validation, Logging
- Backup, Restore, Monitor

CRITICAL RULES:
1. Detect specific actions, operations, or mechanisms
2. Look for verbs describing HOW the system works
3. These are concrete implementations, not abstract qualities
4. Return empty list if only abstract behavior described

Examples:

"System shall search database within 2 seconds"
→ {{"operationalizing_softgoals": ["Search"]}}

"Authenticate users before granting access"
→ {{"operationalizing_softgoals": ["Authentication"]}}

"System shall be secure"
→ {{"operationalizing_softgoals": []}}

"Display data and refresh every 60 seconds"
→ {{"operationalizing_softgoals": ["Display", "Refresh"]}}

Return ONLY JSON format:
{{"operationalizing_softgoals": ["list", "of", "mechanisms"]}}

Requirement: {requirement}"""

op_detection_template = ChatPromptTemplate.from_template(OP_DETECTION_PROMPT)
op_detection_chain = op_detection_template | llm

# ============================================================================
# CLASSIFICATION LOGIC (Definition 3)
# ============================================================================

def detect_nfr_softgoals(requirement_text):
    """Use LLM to detect NFR softgoals in requirement."""
    try:
        result = nfr_detection_chain.invoke({"requirement": requirement_text})
        text = str(result).strip()
        
        # Try to parse JSON
        try:
            parsed = json.loads(text)
            softgoals = parsed.get('nfr_softgoals', [])
            return softgoals if isinstance(softgoals, list) else []
        except:
            # Fallback: extract from text
            match = re.search(r'"nfr_softgoals":\s*\[(.*?)\]', text)
            if match:
                items = match.group(1)
                return [s.strip(' "\'') for s in items.split(',') if s.strip()]
            return []
    except Exception as e:
        print(f"Error detecting NFR softgoals: {e}")
        return []

def detect_operationalizing_softgoals(requirement_text):
    """Use LLM to detect operationalizing softgoals in requirement."""
    try:
        result = op_detection_chain.invoke({"requirement": requirement_text})
        text = str(result).strip()
        
        # Try to parse JSON
        try:
            parsed = json.loads(text)
            softgoals = parsed.get('operationalizing_softgoals', [])
            return softgoals if isinstance(softgoals, list) else []
        except:
            # Fallback: extract from text
            match = re.search(r'"operationalizing_softgoals":\s*\[(.*?)\]', text)
            if match:
                items = match.group(1)
                return [s.strip(' "\'') for s in items.split(',') if s.strip()]
            return []
    except Exception as e:
        print(f"Error detecting operationalizing softgoals: {e}")
        return []

def classify_requirement(requirement_text):
    """
    Classify requirement based on Definition 3:
    - MIXED: Has BOTH NFR softgoal AND operationalizing softgoal
    - NFR: Has only NFR softgoal(s), no operationalizations
    - FR: Has only operationalizing softgoal(s), or neither
    """
    
    # Step 1: Detect NFR softgoals
    nfr_softgoals = detect_nfr_softgoals(requirement_text)
    
    # Step 2: Detect operationalizing softgoals
    op_softgoals = detect_operationalizing_softgoals(requirement_text)
    
    # Classification logic
    has_nfr = len(nfr_softgoals) > 0
    has_op = len(op_softgoals) > 0
    
    if has_nfr and has_op:
        classification = "MIXED"
    elif has_nfr and not has_op:
        classification = "NFR"
    else:
        classification = "FR"
    
    return {
        'classification': classification,
        'nfr_softgoals': nfr_softgoals,
        'op_softgoals': op_softgoals
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_requirement(req_text, ground_truth_label=None, dataset_flags=None):
    """Analyze a single requirement and return detailed results."""
    
    result = classify_requirement(req_text)
    classification = result['classification']
    nfr_softgoals = result['nfr_softgoals']
    op_softgoals = result['op_softgoals']
    
    # Build explanation
    if classification == "MIXED":
        explanation = "MIXED per Definition 3: Contains BOTH quality attributes AND implementation mechanisms.\n"
        explanation += f"  - NFR Softgoal(s): {', '.join(nfr_softgoals)}\n"
        explanation += f"  - Operationalizing Softgoal(s): {', '.join(op_softgoals)}\n"
        explanation += "These should be decomposed into separate requirements to enable:\n"
        explanation += "  - Individual traceability of quality vs. mechanism\n"
        explanation += "  - Contribution analysis (does mechanism HELP/HURT quality?)\n"
        explanation += "  - Alternative operationalization exploration"
    elif classification == "NFR":
        explanation = f"Pure NFR: Contains only quality attribute(s): {', '.join(nfr_softgoals)}.\n"
        explanation += "No specific implementation mechanisms mentioned.\n"
        explanation += "This represents a quality goal that can be satisfied through multiple operationalizations."
    else:  # FR
        if op_softgoals:
            explanation = f"Classified as FR: Contains operationalizing mechanism(s): {', '.join(op_softgoals)}.\n"
            explanation += "No explicit quality attributes mentioned.\n"
            explanation += "Describes WHAT the system does (functional behavior)."
        else:
            explanation = "Pure FR: Describes functional behavior without explicit quality attributes or recognized operationalizations."
    
    agrees = classification == ground_truth_label if ground_truth_label else None
    
    return {
        'requirement': req_text,
        'ground_truth': ground_truth_label,
        'dataset_flags': dataset_flags,
        'nfr_softgoals': nfr_softgoals,
        'op_softgoals': op_softgoals,
        'classification': classification,
        'explanation': explanation,
        'agrees': agrees
    }

def print_analysis(result, example_num):
    """Print detailed analysis for a single requirement."""
    
    print("\n" + "="*80)
    print(f"EXAMPLE {example_num}")
    print("="*80)
    
    # Requirement text
    print(f"\nREQUIREMENT:")
    print(f"   {repr(result['requirement'])}")
    if result['ground_truth']:
        print(f"   Ground Truth Label: {result['ground_truth']}")
    if result['dataset_flags']:
        print(f"   Dataset Flags: {result['dataset_flags']}")
    
    # Detection results
    print(f"\nLLM-BASED SOFTGOAL DETECTION:")
    
    if result['nfr_softgoals']:
        print(f"   [YES] NFR Softgoals: {', '.join(result['nfr_softgoals'])}")
    else:
        print(f"   [NO] NFR Softgoals: None detected")
    
    if result['op_softgoals']:
        print(f"   [YES] Operationalizing Softgoals: {', '.join(result['op_softgoals'])}")
    else:
        print(f"   [NO] Operationalizing Softgoals: None detected")
    
    print(f"\n   Classification: {result['classification']}")
    
    # SIG visualization
    print(f"\nSoftgoal Interdependency Graph (SIG):")
    if result['nfr_softgoals']:
        for nfr in result['nfr_softgoals']:
            print(f"  {nfr}[System] <- NFR Softgoal (quality attribute)")
    if result['nfr_softgoals'] and result['op_softgoals']:
        print(f"      ^ (contributes to)")
    if result['op_softgoals']:
        for op in result['op_softgoals']:
            print(f"  {op}[System] <- Operationalizing Softgoal (mechanism)")
    
    # Explanation
    print(f"\nEXPLANATION:")
    for line in result['explanation'].split('\n'):
        if line.strip():
            print(f"   {line}")
    
    # Agreement check
    if result['ground_truth']:
        print(f"\nCLASSIFICATION MATCH:")
        print(f"   Chatbot (Definition 3): {result['classification']}")
        print(f"   Dataset Label: {result['ground_truth']}")
        
        if result['agrees']:
            print(f"   Agreement: YES")
        else:
            print(f"   Agreement: NO")
            print(f"\n   DISAGREEMENT ANALYSIS:")
            print(f"      Dataset says {result['ground_truth']}, but Definition 3 classifies as {result['classification']}")
            
            if result['classification'] == 'FR' and not result['nfr_softgoals']:
                print(f"      Reason: LLM detected no explicit quality attributes")
            if result['classification'] == 'NFR' and not result['op_softgoals']:
                print(f"      Reason: LLM detected no operationalizations")
            
            print(f"      Note: Dataset labels may not align with Definition 3")

def analyze_batch(requirements_data):
    """Analyze a batch of requirements using LLM."""
    
    print("Analyzing {} example MIXED requirements using LLM...".format(len(requirements_data)))
    print("="*80)
    print("LLM-BASED APPROACH: Using Ollama llama3:8b for softgoal detection")
    print("="*80)
    
    results = []
    
    for idx, req_data in enumerate(requirements_data, 1):
        print(f"\nProcessing {idx}/{len(requirements_data)}...")
        
        result = analyze_requirement(
            req_data['text'],
            req_data.get('label'),
            req_data.get('flags')
        )
        results.append(result)
        print_analysis(result, idx)
    
    # Summary statistics
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    print(f"\nSUMMARY STATISTICS")
    print("="*80)
    print(f"Total analyzed: {len(results)}")
    
    if any(r['ground_truth'] for r in results):
        gt_labels = set(r['ground_truth'] for r in results if r['ground_truth'])
        if len(gt_labels) == 1:
            print(f"Dataset labeled all as: {list(gt_labels)[0]}")
    
    # Count by classification
    classifications = {}
    for r in results:
        cls = r['classification']
        classifications[cls] = classifications.get(cls, 0) + 1
    
    print(f"\nDefinition 3 Classification:")
    for cls in ['FR', 'MIXED', 'NFR']:
        count = classifications.get(cls, 0)
        pct = (count / len(results)) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    # Agreement statistics
    if any(r['ground_truth'] for r in results):
        agrees = sum(1 for r in results if r['agrees'])
        total = len(results)
        print(f"\nAgreement with Dataset:")
        print(f"   Agrees (both MIXED): {agrees} ({(agrees/total)*100:.1f}%)")
        print(f"   Disagrees: {total - agrees} ({((total-agrees)/total)*100:.1f}%)")
        print(f"\nNote: Disagreements indicate dataset 'MIXED' labels don't align with Definition 3")
        print(f"      Definition 3 requires BOTH NFR softgoal + operationalization")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test cases from your dataset
    test_requirements = [
        {
            'text': "The system shall refresh the display every 60 seconds.",
            'label': 'MIXED',
            'flags': 'Performance'
        },
        {
            'text': "The product shall ensure that it can only be accessed by authorized users. The product will be able to distinguish between authorized and unauthorized users in all access attempts",
            'label': 'MIXED',
            'flags': 'Security'
        },
        {
            'text': "The top 1/4 of the table will hold events that are to occur sequentially.",
            'label': 'MIXED',
            'flags': 'Look & Feel'
        },
        {
            'text': "The bottom 3/4 of the table will hold events that occur according to its relevance to current time.",
            'label': 'MIXED',
            'flags': 'Look & Feel'
        },
        {
            'text': "The system shall color code events according to their variance from current time.",
            'label': 'MIXED',
            'flags': 'Look & Feel'
        },
        {
            'text': "The system shall display data from the Sync Matrix 1.0 and Exercise Management Tool 1.0 applications",
            'label': 'MIXED',
            'flags': 'Look & Feel'
        },
        {
            'text': "The system shall link Events back to either the Sync Matrix 1.0 or the Exercise Managment Tool 1.0 applications for modifications.",
            'label': 'MIXED',
            'flags': 'Maintainability, Portability'
        },
        {
            'text': "The product shall produce search results in an acceptable time",
            'label': 'MIXED',
            'flags': 'Performance'
        },
        {
            'text': "The search results shall be returned no later 30 seconds after the user has entered the search criteria",
            'label': 'MIXED',
            'flags': 'Performance'
        },
        {
            'text': "The product shall generate a CMA report in an acceptable time.",
            'label': 'MIXED',
            'flags': 'Performance'
        },
    ]
    
    # Run analysis
    results = analyze_batch(test_requirements)
    
    # Save results
    df = pd.DataFrame([
        {
            'Requirement': r['requirement'],
            'Ground_Truth': r['ground_truth'],
            'Dataset_Flags': r['dataset_flags'],
            'Classification': r['classification'],
            'NFR_Softgoals': ', '.join(r['nfr_softgoals']) if r['nfr_softgoals'] else '',
            'Op_Softgoals': ', '.join(r['op_softgoals']) if r['op_softgoals'] else '',
            'Agrees': r['agrees']
        }
        for r in results
    ])
    
    output_file = 'llm_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*80)
    print("COMPLETE! LLM-based classification using Definition 3.")
    print("="*80)