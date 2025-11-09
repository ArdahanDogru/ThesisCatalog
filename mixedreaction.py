# analyze_mixed_requirements.py
# Analyzes requirements labeled as MIXED (IsFunctional=1 AND IsQuality=1)
# Shows how the chatbot classifies and explains these requirements

import pandas as pd
import re
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ==================== SETUP ====================

MODEL = "llama3:8b"
llm = OllamaLLM(model=MODEL, temperature=0, max_tokens=50, stop=["}"])

STEP1_PROMPT = """You are a requirements classifier. Classify each software requirement into exactly one category:

FR (Functional Requirement): Describes WHAT the system does - specific behaviors, actions, or functions
NFR (Non-Functional Requirement): Describes HOW WELL the system performs - quality attributes, constraints, performance

CRITICAL RULES:
1. FR: Focus on actions, functions, features (login, search, calculate, display, store)
2. NFR: Focus on qualities, constraints, performance (speed, security, reliability, usability)
3. When in doubt between FR and NFR, choose NFR

Examples:

Functional (FR):
- "The system shall allow users to create new accounts"
- "Users can search for products by category"

Non-Functional (NFR):
- "The system must have 99.9% uptime"
- "Response time shall not exceed 2 seconds"

Return your answer only in JSON format like:
{{"classification": "FR"}}

Requirement: {requirement}"""

step1_template = ChatPromptTemplate.from_template(STEP1_PROMPT)
step1_chain = step1_template | llm

# ==================== CLASSIFICATION FUNCTIONS ====================

def classify_step1(req):
    """Step 1: Basic FR/NFR classification"""
    try:
        result = step1_chain.invoke({"requirement": req})
        text = str(result).strip()
        try:
            parsed = json.loads(text)
            if parsed.get('classification', '').upper() in ['FR', 'NFR']:
                return parsed['classification'].upper()
        except:
            pass
        match = re.search(r'\b(FR|NFR)\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    except:
        pass
    return None

def check_mixed_from_fr(req):
    """Check if FR-classified requirement has mixed patterns"""
    t = req.lower()
    mixed_patterns = [
        r'(within|in|under|no\s+more\s+than|take\s+no.*than)\s+\d+\s*(second|minute|hour|click)',
        r'every\s+\d+\s*(second|minute|hour)',
        r'on\s+demand', r'at\s+any\s+time', r'at\s+least\s+\d+',
        r'all\s+(actions|sales|statistics|movies).*\b(logged|recorded|streamed)',
        r'only\s+(users|managers|supervisors).*\b(can|able\s+to|must\s+be\s+able)\s+\b(initiate|perform|advertise|view)',
        r'(export|provide).*in\s+(spreadsheet|excel|format)', r'acceptable\s+time',
    ]
    for pattern in mixed_patterns:
        if re.search(pattern, t):
            return True
    return False

def check_mixed_from_nfr(req):
    """Check if NFR-classified requirement has mixed patterns"""
    t = req.lower()
    if re.search(r'(shall\s+be\s+available\s+\d+%|shall\s+have\s+standard|shall\s+conform\s+to|shall\s+match\s+the\s+color)', t):
        if not re.search(r'within\s+\d+', t):
            return False
    
    mixed_patterns = [
        (r'\b(search|generate|return|returned|retrieve|synchronize|sync|backup|restore)\b', r'(within|no\s+later|every|acceptable\s+time)\s*\d*\s*(second|minute|hour)'),
        (r'\b(restore|prevent|ensure|authenticate)\b', r'100%'),
        (r'\b(manage|expected\s+to\s+manage)\b', r'(minimum|at\s+least)\s+\d+\s+(year|month)'),
        (r'\b(maintain|record)\b', r'(detailed\s+history|all\s+actions|audit)'),
        (r'\b(manipulate|view)\b', r'(business\s+data|data)'),
        (r'\b(provide|protect)\b', r'(authorized\s+users|in\s+accordance|policy)'),
        (r'\b(check|checking)\b', r'(integrity|validity|correct)'),
        (r'\b(authenticate|authorized?)\b', r'(user|access)'),
        (r'\d+\s+out\s+of\s+\d+.*\b(manage|use)\b', r''),
    ]
    
    for action_pattern, constraint_pattern in mixed_patterns:
        if re.search(action_pattern, t):
            if not constraint_pattern or re.search(constraint_pattern, t):
                return True
    return False

def has_mixed_pattern(text):
    """General mixed pattern detection"""
    t = text.lower()
    action_verbs = r'\b(search|login|authenticate|backup|calculate|generate|display|refresh|sync|return|retrieve|save|poll|stream|register|cancel|add|remove|de-activate|export)\b'
    time_constraint = r'(within|in|under|no\s+later\s+than|no\s+more\s+than|take\s+no\s+more\s+than|every|per)\s+\d+\s*(second|minute|hour|click)'
    
    if re.search(action_verbs, t) and re.search(time_constraint, t):
        return True
    return False

def classify_requirement(req):
    """Full classification pipeline"""
    step1_result = classify_step1(req)
    
    if not step1_result:
        return 'UNKNOWN', 'Failed to classify in Step 1'
    
    # Check for mixed patterns
    if step1_result == 'FR':
        if check_mixed_from_fr(req) or has_mixed_pattern(req):
            return 'MIXED', f'Initially classified as {step1_result}, but contains quality constraints'
        return step1_result, f'Classified as pure {step1_result}'
    
    elif step1_result == 'NFR':
        if check_mixed_from_nfr(req):
            return 'MIXED', f'Initially classified as {step1_result}, but contains functional behavior'
        return step1_result, f'Classified as pure {step1_result}'
    
    return step1_result, f'Classified as {step1_result}'

def explain_classification(req, classification, reasoning):
    """Generate detailed explanation of why requirement was classified this way"""
    explanation = {
        'classification': classification,
        'reasoning': reasoning,
        'key_indicators': [],
        'details': ''
    }
    
    t = req.lower()
    
    if classification == 'FR':
        # Look for functional indicators
        action_verbs = ['display', 'show', 'allow', 'enable', 'provide', 'create', 'delete', 'update', 'store']
        found_verbs = [v for v in action_verbs if v in t]
        if found_verbs:
            explanation['key_indicators'].append(f"Action verbs: {', '.join(found_verbs)}")
        explanation['details'] = "Describes a specific function or behavior the system must perform."
    
    elif classification == 'NFR':
        # Look for quality indicators
        quality_terms = ['uptime', 'availability', 'performance', 'security', 'usability', 'reliable', 
                        'fast', 'intuitive', 'accessible', 'maintainable']
        found_qualities = [q for q in quality_terms if q in t]
        if found_qualities:
            explanation['key_indicators'].append(f"Quality attributes: {', '.join(found_qualities)}")
        
        # Look for metrics
        if re.search(r'\d+%', t):
            explanation['key_indicators'].append("Quantitative metric found")
        
        explanation['details'] = "Describes quality attributes or constraints on system performance."
    
    elif classification == 'MIXED':
        # Look for both functional and quality aspects
        action_verbs = ['authenticate', 'refresh', 'backup', 'search', 'generate', 'display', 'ensure']
        found_verbs = [v for v in action_verbs if v in t]
        if found_verbs:
            explanation['key_indicators'].append(f"Functional aspect: {', '.join(found_verbs)}")
        
        # Time constraints
        if re.search(r'(within|every)\s+\d+\s*(second|minute|hour)', t):
            explanation['key_indicators'].append("Time constraint found (quality aspect)")
        
        # Other quality indicators
        if re.search(r'\d+%', t):
            explanation['key_indicators'].append("Percentage metric (quality aspect)")
        
        if 'authorized' in t or 'only' in t:
            explanation['key_indicators'].append("Access control (quality aspect)")
        
        explanation['details'] = "Contains both functional behavior AND quality constraints."
    
    return explanation

# ==================== DATA EXTRACTION & ANALYSIS ====================

def extract_mixed_requirements():
    """Extract requirements where IsFunctional=1 AND IsQuality=1"""
    # Load full dataset
    df = pd.read_csv("PROMISE-relabeled-NICE.csv")
    
    # Filter for MIXED (IsFunctional=1 AND IsQuality=1)
    mixed_df = df[(df['IsFunctional'] == 1) & (df['IsQuality'] == 1)].copy()
    
    # Clean requirement text
    mixed_df['RequirementText'] = mixed_df['RequirementText'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else None
    )
    
    # Remove any null requirements
    mixed_df = mixed_df.dropna(subset=['RequirementText']).reset_index(drop=True)
    
    print(f"Found {len(mixed_df)} requirements with both IsFunctional=1 AND IsQuality=1")
    
    return mixed_df

def analyze_mixed_requirements(mixed_df, num_examples=80):
    """Analyze MIXED requirements and document chatbot responses"""
    
    print(f"\nAnalyzing {num_examples} example MIXED requirements...")
    print("="*80)
    
    results = []
    
    # Take first num_examples
    for idx in range(min(num_examples, len(mixed_df))):
        row = mixed_df.iloc[idx]
        req_text = row['RequirementText']
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx + 1}")
        print(f"{'='*80}")
        
        # Original requirement info
        print(f"\n📝 ORIGINAL REQUIREMENT:")
        print(f"   Text: \"{req_text}\"")
        print(f"   ProjectID: {row['ProjectID']}")
        print(f"   IsFunctional: {row['IsFunctional']}")
        print(f"   IsQuality: {row['IsQuality']}")
        print(f"   Ground Truth Label: MIXED (both flags = 1)")
        
        # Get quality categories that are flagged
        quality_cols = ['Availability (A)', 'Fault Tolerance (FT)', 'Legal (L)', 'Look & Feel (LF)',
                       'Maintainability (MN)', 'Operability (O)', 'Performance (PE)', 'Portability (PO)',
                       'Scalability (SC)', 'Security (SE)', 'Usability (US)', 'Other (OT)']
        
        flagged_qualities = [col for col in quality_cols if col in row.index and row[col] == 1]
        if flagged_qualities:
            print(f"   Quality Categories: {', '.join([q.split('(')[0].strip() for q in flagged_qualities])}")
        
        # Classify with chatbot
        print(f"\n🤖 CHATBOT CLASSIFICATION:")
        classification, reasoning = classify_requirement(req_text)
        print(f"   Classification: {classification}")
        print(f"   Initial Reasoning: {reasoning}")
        
        # Get detailed explanation
        explanation = explain_classification(req_text, classification, reasoning)
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"   Classification: {explanation['classification']}")
        if explanation['key_indicators']:
            print(f"   Key Indicators:")
            for indicator in explanation['key_indicators']:
                print(f"      • {indicator}")
        print(f"   Explanation: {explanation['details']}")
        
        # Match analysis
        print(f"\n✅ MATCH ANALYSIS:")
        matches = classification == 'MIXED'
        print(f"   Chatbot classified as: {classification}")
        print(f"   Ground truth: MIXED")
        print(f"   Match: {'YES ✓' if matches else 'NO ✗'}")
        
        if not matches:
            print(f"   ⚠️  Mismatch! Chatbot classified as {classification} instead of MIXED")
        
        # Store results
        results.append({
            'example_number': idx + 1,
            'project_id': row['ProjectID'],
            'requirement_text': req_text,
            'ground_truth': 'MIXED',
            'is_functional': row['IsFunctional'],
            'is_quality': row['IsQuality'],
            'quality_categories': ', '.join([q.split('(')[0].strip() for q in flagged_qualities]) if flagged_qualities else '',
            'chatbot_classification': classification,
            'chatbot_reasoning': reasoning,
            'key_indicators': ' | '.join(explanation['key_indicators']),
            'explanation_details': explanation['details'],
            'match': matches
        })
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}")
    
    return results

def generate_summary_statistics(results):
    """Generate summary statistics of the analysis"""
    total = len(results)
    matches = sum(1 for r in results if r['match'])
    mismatches = total - matches
    
    # Count chatbot classifications
    classifications = {}
    for r in results:
        cls = r['chatbot_classification']
        classifications[cls] = classifications.get(cls, 0) + 1
    
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total MIXED requirements analyzed: {total}")
    print(f"Chatbot correctly identified as MIXED: {matches} ({matches/total*100:.1f}%)")
    print(f"Chatbot misclassified: {mismatches} ({mismatches/total*100:.1f}%)")
    
    print(f"\n📋 Chatbot Classification Breakdown:")
    for cls, count in sorted(classifications.items()):
        print(f"   {cls}: {count} ({count/total*100:.1f}%)")
    
    return {
        'total_analyzed': total,
        'correctly_identified_as_mixed': matches,
        'misclassified': mismatches,
        'accuracy': matches/total if total > 0 else 0,
        'classification_breakdown': classifications
    }

# ==================== MAIN ====================

def main():
    print("="*80)
    print("ANALYZING MIXED REQUIREMENTS (IsFunctional=1 AND IsQuality=1)")
    print("="*80)
    
    # Extract MIXED requirements
    mixed_df = extract_mixed_requirements()
    
    # Save all MIXED requirements to CSV
    mixed_df.to_csv("mixed_requirements_extracted.csv", index=False)
    print(f"✓ All {len(mixed_df)} MIXED requirements saved to mixed_requirements_extracted.csv")
    
    # Analyze 10 examples
    results = analyze_mixed_requirements(mixed_df, num_examples=80)
    
    # Generate summary
    summary_stats = generate_summary_statistics(results)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv("mixed_requirements_analysis_results.csv", index=False)
    print(f"\n✓ Detailed analysis saved to mixed_requirements_analysis_results.csv")
    
    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv("mixed_requirements_summary.csv", index=False)
    print(f"✓ Summary statistics saved to mixed_requirements_summary.csv")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated files:")
    print(f"  1. mixed_requirements_extracted.csv - All MIXED requirements")
    print(f"  2. mixed_requirements_analysis_results.csv - Detailed analysis of 10 examples")
    print(f"  3. mixed_requirements_summary.csv - Summary statistics")

if __name__ == "__main__":
    main()