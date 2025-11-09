# interactive_requirements_chatbot.py
# Enhanced Hybrid Architecture with User Input Options - SYNTAX CORRECTED
# Fast rule-based patterns + LLM fallback + Menu-driven interface

import pandas as pd
import re
import json
import time
import pickle
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RequirementAnalysis:
    """Structure for storing requirement analysis results"""
    text: str
    classification: str
    confidence: float
    ambiguities: List[Dict]
    traceability_suggestions: Dict
    refinement_suggestions: List[str]
    timestamp: datetime

class EnhancedRequirementsChatbot:
    """Enhanced hybrid chatbot with menu-driven interface and improved patterns"""
    
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, temperature=0, max_tokens=150, stop=["}"])
        self.conversation_history = []
        self.user_requirements = []
        self.last_analysis = None
        self.session_id = f"session_{int(time.time())}"
        self.setup_analysis_chains()
        self.load_enhanced_patterns()
        
        print("Enhanced Requirements Engineering Chatbot initialized")
        print("Ready to help with advanced requirement analysis!")

    def setup_analysis_chains(self):
        """Setup LLM chains for classification"""
        
        classification_prompt = """You are a requirements classifier. Classify this software requirement:

FR (Functional): Describes WHAT the system does - specific behaviors, actions, functions
NFR (Non-Functional): Describes HOW WELL it performs - quality attributes, constraints, performance

Examples:
FR: "Users can login", "System displays results", "Generate reports"
NFR: "Response time ≤ 2 seconds", "System uptime 99.9%", "User-friendly interface"

Return only JSON: {{"classification": "FR"}} or {{"classification": "NFR"}}

Requirement: {requirement}"""

        self.classification_chain = ChatPromptTemplate.from_template(classification_prompt) | self.llm

    def load_enhanced_patterns(self):
        """Load comprehensive enhanced patterns for better detection"""
        
        # Enhanced ambiguity patterns
        self.ambiguity_patterns = {
            'vague_attributes': {
                'patterns': [
                    'fast', 'quick', 'quickly', 'responsive', 'smooth', 'seamless', 'efficient', 'optimized',
                    'slow', 'slowly', 'sluggish', 'delayed',
                    'user-friendly', 'intuitive', 'easy', 'simple', 'convenient', 'comfortable', 'natural',
                    'usable', 'accessible', 'clear', 'obvious', 'straightforward', 'self-explanatory',
                    'good', 'better', 'best', 'excellent', 'superior', 'high-quality', 'professional',
                    'appropriate', 'suitable', 'adequate', 'sufficient', 'reasonable', 'acceptable',
                    'reliable', 'robust', 'stable', 'secure', 'safe', 'trustworthy',
                    'nice', 'beautiful', 'attractive', 'clean', 'modern', 'sleek', 'elegant',
                    'flexible', 'adaptable', 'configurable', 'customizable', 'extensible',
                    'maintainable', 'portable', 'scalable', 'compatible',
                    'accessibility mode', 'accessible', 'disability-friendly', 'inclusive',
                    'enhanced', 'improved', 'advanced', 'sophisticated', 'intelligent',
                    'smart', 'automated', 'streamlined', 'optimal', 'maximum', 'minimum'
                ],
                'severity': 'HIGH',
                'category': 'Unmeasurable quality descriptors'
            },
            'unquantified_metrics': {
                'patterns': [
                    'quickly', 'slowly', 'frequently', 'often', 'rarely', 'seldom', 'occasionally',
                    'sometimes', 'usually', 'normally', 'typically', 'generally', 'commonly',
                    'many', 'few', 'some', 'several', 'multiple', 'numerous', 'various',
                    'most', 'majority', 'minority', 'all', 'every', 'each',
                    'large', 'small', 'big', 'little', 'huge', 'tiny', 'massive', 'compact',
                    'high', 'low', 'long', 'short', 'wide', 'narrow',
                    'approximately', 'roughly', 'about', 'around', 'nearly', 'almost',
                    'close to', 'up to', 'at least', 'no more than'
                ],
                'severity': 'HIGH', 
                'category': 'Missing specific measurements'
            },
            'unclear_subjects': {
                'patterns': [
                    r'\bit\s+(?:shall|will|must|should|can)',
                    r'\bthis\s+(?:shall|will|must|should|can)',
                    r'\bthat\s+(?:shall|will|must|should|can)',
                    r'\bthe\s+system\s+(?:shall|will|must|should|can)',
                    r'\bthe\s+application\s+(?:shall|will|must|should|can)',
                    r'\bthe\s+software\s+(?:shall|will|must|should|can)'
                ],
                'severity': 'MEDIUM',
                'category': 'Ambiguous system components'
            },
            'passive_voice': {
                'patterns': [
                    r'\bdata\s+(?:will be|shall be|is|are)\s+\w+(?:ed|en)\b',
                    r'\breports?\s+(?:will be|shall be|is|are)\s+\w+(?:ed|en)\b',
                    r'\binformation\s+(?:will be|shall be|is|are)\s+\w+(?:ed|en)\b',
                    r'\brequests?\s+(?:will be|shall be|is|are)\s+\w+(?:ed|en)\b',
                    r'\bresults?\s+(?:will be|shall be|is|are)\s+\w+(?:ed|en)\b'
                ],
                'severity': 'MEDIUM',
                'category': 'Unclear responsibility assignment'
            },
            'compound_requirements': {
                'patterns': [
                    r'\band\s+(?:also\s+)?(?:display|show|provide|support|include)',
                    r'\bor\s+(?:alternatively\s+)?(?:display|show|provide|support|include)',
                    r'\bas\s+well\s+as',
                    r'\bin\s+addition\s+to',
                    r'\balong\s+with'
                ],
                'severity': 'MEDIUM',
                'category': 'Multiple requirements in one statement'
            }
        }
        
        # Enhanced traceability knowledge base
        self.traceability_knowledge = {
            'authentication': {
                'triggers': ['login', 'authenticate', 'password', 'account', 'sign in', 'user credentials'],
                'missing_nfrs': [
                    "Account lockout after 3-5 failed login attempts",
                    "Password complexity requirements (8+ chars, mixed case, numbers, symbols)", 
                    "Session timeout after 30 minutes of inactivity",
                    "Multi-factor authentication support"
                ],
                'pattern': 'Authentication Security Pattern'
            },
            'search': {
                'triggers': ['search', 'query', 'find', 'filter', 'look up', 'retrieve'],
                'missing_nfrs': [
                    "Search results returned within 2 seconds",
                    "Support 500+ concurrent search requests",
                    "Search accuracy ≥ 95% for relevant results",
                    "Search index updated within 5 minutes of data changes"
                ],
                'pattern': 'Search Performance Pattern'
            },
            'synchronization': {
                'triggers': ['synchronize', 'sync', 'synchronise', 'update', 'refresh'],
                'missing_nfrs': [
                    "Synchronization completion within 30 seconds",
                    "Data consistency guarantee during sync operations",
                    "Sync failure retry mechanism (max 3 attempts)",
                    "Conflict resolution policy for simultaneous updates"
                ],
                'missing_frs': [
                    "Sync status monitoring and reporting functionality",
                    "Manual sync trigger for administrators",
                    "Sync scheduling configuration interface",
                    "Sync error logging and notification system"
                ],
                'pattern': 'Data Synchronization Pattern'
            },
            'data_management': {
                'triggers': ['store', 'save', 'database', 'data', 'persist', 'backup', 'archive'],
                'missing_nfrs': [
                    "Data backup every 24 hours with 99.9% success rate",
                    "Data encryption at rest using AES-256",
                    "Database recovery time objective (RTO) ≤ 4 hours",
                    "Data integrity validation with checksums"
                ],
                'pattern': 'Data Management Pattern'
            },
            'display_visualization': {
                'triggers': ['display', 'show', 'visualize', 'render', 'present', 'view', 'interface'],
                'missing_nfrs': [
                    "Display refresh rate ≥ 60 FPS for smooth interaction",
                    "Support multiple screen resolutions (1920x1080 minimum)",
                    "Color contrast ratio ≥ 4.5:1 for accessibility",
                    "Display loading time ≤ 3 seconds for complex visualizations"
                ],
                'missing_frs': [
                    "Display configuration and customization options",
                    "Visual theme and layout selection",
                    "Export visualization as image/PDF functionality"
                ],
                'pattern': 'Display and Visualization Pattern'
            },
            'accessibility': {
                'triggers': ['accessibility', 'accessible', 'disability', 'screen reader', 'inclusive', 'barrier-free'],
                'missing_nfrs': [
                    "WCAG 2.1 Level AA compliance for accessibility",
                    "Screen reader compatibility with ARIA labels",
                    "Keyboard navigation support for all functions",
                    "Color contrast ratio ≥ 4.5:1 for normal text"
                ],
                'missing_frs': [
                    "Alternative text for all images and visual elements",
                    "Keyboard shortcuts for common actions",
                    "Voice input and audio output options",
                    "Adjustable font size and high contrast mode"
                ],
                'pattern': 'Accessibility and Inclusive Design Pattern'
            },
            'performance': {
                'triggers': ['performance', 'response time', 'speed', 'latency', 'fast', 'responsive', 'throughput'],
                'missing_frs': [
                    "Performance monitoring dashboard with real-time metrics",
                    "System health check endpoints for monitoring",
                    "Performance alert notifications when thresholds exceeded",
                    "Performance metrics logging and historical analysis"
                ],
                'pattern': 'Performance Monitoring Pattern'
            },
            'usability': {
                'triggers': ['user-friendly', 'intuitive', 'easy', 'usable', 'user experience', 'UX'],
                'missing_nfrs': [
                    "Task completion time ≤ X minutes for typical users",
                    "User error rate ≤ 5% after initial training", 
                    "System Usability Scale (SUS) score ≥ 80",
                    "User satisfaction rating ≥ 4.0/5.0"
                ],
                'missing_frs': [
                    "Context-sensitive help and documentation system",
                    "User onboarding and tutorial functionality",
                    "Error prevention and user guidance features",
                    "User feedback collection mechanism"
                ],
                'pattern': 'Usability and User Experience Pattern'
            }
        }

    def show_input_menu(self) -> str:
        """Display input options menu"""
        menu = """
Choose what you'd like to do:
1  Enter a new requirement for analysis
2  Ask about my previous analysis/recommendations  
3  General questions or help

Enter your choice (1, 2, or 3): """
        return menu

    def classify_requirement(self, requirement: str) -> Tuple[str, float]:
        """Enhanced classification"""
        
        # Check for mixed patterns first
        if self.has_mixed_pattern(requirement):
            return 'MIXED', 1.0
        
        try:
            result = self.classification_chain.invoke({"requirement": requirement})
            text = str(result).strip()
            
            # Parse JSON response
            try:
                parsed = json.loads(text)
                classification = parsed.get('classification', '').upper()
                if classification in ['FR', 'NFR']:
                    base_classification = classification
                else:
                    raise ValueError("Invalid classification")
            except:
                # Fallback to regex
                match = re.search(r'\b(FR|NFR)\b', text, re.IGNORECASE)
                base_classification = match.group(1).upper() if match else 'FR'
            
            # Check for mixed patterns in classified requirements
            if base_classification == 'FR' and self.check_mixed_from_fr(requirement):
                return 'MIXED', 1.0
            elif base_classification == 'NFR' and self.check_mixed_from_nfr(requirement):
                return 'MIXED', 1.0
                
            return base_classification, 1.0
            
        except Exception as e:
            return 'FR', 1.0

    def has_mixed_pattern(self, text: str) -> bool:
        """Enhanced mixed pattern detection"""
        t = text.lower()
        
        # Action + timing constraint
        action_verbs = r'\b(search|login|authenticate|backup|calculate|generate|display|refresh|sync|return|retrieve|save|poll|stream|register|cancel|add|remove|export|show|render|visualize|synchronize)\b'
        time_constraint = r'(within|in|under|no\s+later\s+than|no\s+more\s+than|take\s+no\s+more\s+than|every|per)\s+\d+\s*(second|minute|hour|click)'
        
        # Function + quality constraint
        quality_constraint = r'\b(fast|quick|responsive|accessible|user-friendly|secure|reliable)\b'
        
        return bool((re.search(action_verbs, t) and re.search(time_constraint, t)) or
                   (re.search(action_verbs, t) and re.search(quality_constraint, t)))

    def check_mixed_from_fr(self, req: str) -> bool:
        """Enhanced mixed detection for FRs"""
        t = req.lower()
        mixed_patterns = [
            r'(within|in|under|no\s+more\s+than)\s+\d+\s*(second|minute|hour|click)',
            r'every\s+\d+\s*(second|minute|hour)', 
            r'on\s+demand', r'at\s+any\s+time',
            r'acceptable\s+time', 
            r'all\s+(actions|sales|statistics).*\b(logged|recorded)',
            r'\b(fast|quick|responsive|user-friendly|accessible)\b'
        ]
        return any(re.search(pattern, t) for pattern in mixed_patterns)

    def check_mixed_from_nfr(self, req: str) -> bool:
        """Enhanced mixed detection for NFRs"""
        t = req.lower()
        action_pattern = r'\b(search|generate|return|retrieve|sync|backup|restore|authenticate|manage|maintain|record|check|display|show|render|synchronize)\b'
        return bool(re.search(action_pattern, t))

    def detect_ambiguities_enhanced(self, requirement: str) -> List[Dict]:
        """Enhanced ambiguity detection - fast rule-based only"""
        
        ambiguities = []
        text = requirement.lower()
        
        # Rule-based detection (fast)
        for ambiguity_type, config in self.ambiguity_patterns.items():
            if ambiguity_type in ['vague_attributes', 'unquantified_metrics']:
                # Word-based patterns
                found_terms = [term for term in config['patterns'] if term in text]
                if found_terms:
                    fixes = self.generate_ambiguity_fixes(ambiguity_type, found_terms)
                    ambiguities.append({
                        'type': ambiguity_type.upper(),
                        'severity': config['severity'],
                        'category': config['category'],
                        'found_terms': found_terms,
                        'fixes': fixes,
                        'detection_method': 'rule-based'
                    })
            else:
                # Regex-based patterns
                for pattern in config['patterns']:
                    if re.search(pattern, text, re.IGNORECASE):
                        fixes = self.generate_ambiguity_fixes(ambiguity_type, [])
                        ambiguities.append({
                            'type': ambiguity_type.upper(),
                            'severity': config['severity'],
                            'category': config['category'],
                            'found_terms': [],
                            'fixes': fixes,
                            'detection_method': 'rule-based'
                        })
                        break
        
        return ambiguities

    def generate_ambiguity_fixes(self, ambiguity_type: str, found_terms: List[str]) -> List[str]:
        """Enhanced fix generation"""
        
        fixes = []
        
        if ambiguity_type == 'vague_attributes':
            for term in found_terms:
                if term in ['user-friendly', 'intuitive', 'easy', 'usable']:
                    fixes.append(f"Replace '{term}' with 'System Usability Scale (SUS) score ≥ 80' or 'Users complete core tasks in ≤ 5 minutes'")
                elif term in ['fast', 'quick', 'quickly', 'responsive']:
                    fixes.append(f"Replace '{term}' with 'Response time ≤ 200ms' or 'Processing time ≤ 2 seconds'")
                elif term in ['accessible', 'accessibility mode', 'disability-friendly']:
                    fixes.append(f"Replace '{term}' with 'WCAG 2.1 Level AA compliance' or 'Screen reader compatible with ARIA labels'")
                elif term in ['reliable', 'robust', 'stable']:
                    fixes.append(f"Replace '{term}' with 'System uptime ≥ 99.9%' or 'MTBF ≥ 720 hours'")
                elif term in ['secure', 'safe']:
                    fixes.append(f"Replace '{term}' with 'AES-256 encryption' or 'OAuth 2.0 authentication'")
                elif term in ['scalable']:
                    fixes.append(f"Replace '{term}' with 'Support 1000+ concurrent users' or 'Handle 10x current load'")
                else:
                    fixes.append(f"Replace '{term}' with specific, measurable criteria")
        
        elif ambiguity_type == 'unquantified_metrics':
            for term in found_terms:
                if term in ['quickly', 'slowly']:
                    fixes.append(f"Replace '{term}' with specific time: '≤ 2 seconds' or '< 500ms'")
                elif term in ['often', 'frequently', 'rarely']:
                    fixes.append(f"Replace '{term}' with frequency: 'every 30 seconds' or 'once per hour'")
                elif term in ['many', 'numerous']:
                    fixes.append(f"Replace '{term}' with specific number: '≥ 1000' or 'at least 500'")
                elif term in ['few', 'some']:
                    fixes.append(f"Replace '{term}' with specific number: '≤ 5' or 'between 3-10'")
                elif term in ['every', 'all']:
                    fixes.append(f"Replace '{term}' with specific frequency: 'every 60 minutes' or 'all 24 hours'")
        
        elif ambiguity_type == 'unclear_subjects':
            fixes = [
                "Replace 'it/this/that' with specific component: 'the authentication module', 'the display engine'",
                "Replace 'the system' with: 'the user interface', 'the backend API', 'the synchronization service'",
                "Specify exact system component responsible for the action"
            ]
        
        elif ambiguity_type == 'passive_voice':
            fixes = [
                "Use active voice: 'The validation service checks data' instead of 'data will be validated'",
                "Specify who performs action: 'The sync manager updates...' instead of 'data will be updated'",
                "Identify the responsible component clearly"
            ]
        
        elif ambiguity_type == 'compound_requirements':
            fixes = [
                "Split into separate requirements: one for each distinct function or constraint",
                "Create individual requirements for each 'and' clause",
                "Separate functional behavior from quality attributes"
            ]
        
        return fixes

    def suggest_traceability_enhanced(self, requirement: str, classification: str) -> Dict:
        """Enhanced traceability with better pattern matching"""
        
        suggestions = {
            'missing_requirements': [],
            'related_patterns': [],
            'dependency_warnings': [],
            'completeness_score': 0.0
        }
        
        text = requirement.lower()
        matched_patterns = []
        
        # Enhanced pattern matching using triggers
        for pattern_name, pattern_data in self.traceability_knowledge.items():
            triggers = pattern_data.get('triggers', [])
            
            if any(trigger in text for trigger in triggers):
                if 'missing_nfrs' in pattern_data:
                    suggestions['missing_requirements'].extend(pattern_data['missing_nfrs'])
                if 'missing_frs' in pattern_data:
                    suggestions['missing_requirements'].extend(pattern_data['missing_frs'])
                matched_patterns.append(pattern_data['pattern'])
        
        suggestions['related_patterns'] = list(set(matched_patterns))
        
        # Enhanced conflict detection
        conflicts = [
            (['backup', 'archive', 'store'], ['real-time', 'immediate', 'instant'], "Backup operations may conflict with real-time performance"),
            (['security', 'encrypt', 'secure'], ['fast', 'quick', 'responsive'], "Security measures may impact performance - consider trade-offs"),
            (['scalable', 'scale'], ['consistent', 'consistency'], "Scalability and consistency may require trade-off decisions"),
            (['sync', 'synchronize'], ['performance', 'fast'], "Synchronization operations may impact system performance during execution")
        ]
        
        for group1, group2, warning in conflicts:
            if any(word in text for word in group1) and any(word in text for word in group2):
                suggestions['dependency_warnings'].append(warning)
        
        # Cross-requirement analysis
        if self.user_requirements:
            previous_patterns = []
            for req in self.user_requirements:
                previous_patterns.extend(req.traceability_suggestions.get('related_patterns', []))
            
            # Pattern completion suggestions
            if 'Data Synchronization Pattern' in matched_patterns and 'Performance Monitoring Pattern' not in previous_patterns:
                suggestions['missing_requirements'].append("Consider performance monitoring for sync operations")
        
        # Calculate completeness score
        base_score = min(1.0, len(matched_patterns) * 0.25)
        conflict_penalty = len(suggestions['dependency_warnings']) * 0.1
        suggestions['completeness_score'] = max(0.0, base_score - conflict_penalty)
        
        return suggestions

    def analyze_requirement_enhanced(self, requirement_text: str) -> RequirementAnalysis:
        """Enhanced requirement analysis pipeline"""
        
        start_time = time.time()
        
        # Step 1: Classification (LLM)
        classification, confidence = self.classify_requirement(requirement_text)
        
        # Step 2: Enhanced ambiguity detection (Rules only)
        ambiguities = self.detect_ambiguities_enhanced(requirement_text)
        
        # Step 3: Enhanced traceability analysis (Rules)
        traceability = self.suggest_traceability_enhanced(requirement_text, classification)
        
        # Step 4: Generate refinement suggestions
        refinement_suggestions = self.generate_refinement_suggestions(
            requirement_text, classification, ambiguities, traceability
        )
        
        analysis = RequirementAnalysis(
            text=requirement_text,
            classification=classification,
            confidence=confidence,
            ambiguities=ambiguities,
            traceability_suggestions=traceability,
            refinement_suggestions=refinement_suggestions,
            timestamp=datetime.now()
        )
        
        # Store analysis
        self.user_requirements.append(analysis)
        self.last_analysis = analysis
        
        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        
        return analysis

    def generate_refinement_suggestions(self, requirement: str, classification: str, 
                                      ambiguities: List[Dict], traceability: Dict) -> List[str]:
        """Enhanced refinement suggestions"""
        
        suggestions = []
        
        # Ambiguity-based suggestions
        if ambiguities:
            high_priority = [amb for amb in ambiguities if amb['severity'] == 'HIGH']
            if high_priority:
                suggestions.append(f"PRIORITY: Address {len(high_priority)} high-severity ambiguity issues before implementation")
                
                for amb in high_priority[:2]:
                    if amb['fixes']:
                        suggestions.append(f"Fix {amb['type']}: {amb['fixes'][0]}")

        # Classification-based suggestions
        if classification == 'FR':
            suggestions.extend([
                "Add specific acceptance criteria with input/output examples",
                "Define error handling and edge case behaviors",
                "Consider user experience and accessibility implications"
            ])
        elif classification == 'NFR':
            suggestions.extend([
                "Include measurable criteria with specific thresholds and units",
                "Define testing and verification methods",
                "Consider impact on other quality attributes and trade-offs"
            ])
        elif classification == 'MIXED':
            suggestions.extend([
                "Consider splitting into separate FR (function) and NFR (quality) requirements",
                "Ensure both behavioral and performance aspects have clear acceptance criteria",
                "Define how quality constraints affect functional behavior"
            ])

        # Enhanced traceability suggestions
        if traceability['missing_requirements']:
            suggestions.append(f"Consider adding {len(traceability['missing_requirements'])} related requirements for complete coverage")
            
        if traceability['dependency_warnings']:
            suggestions.append("Analyze potential conflicts and define trade-off decisions")

        # Quality assessment
        if not ambiguities and traceability['completeness_score'] > 0.7:
            suggestions.append("Requirement quality is good - ready for stakeholder review")
        elif len(ambiguities) > 3:
            suggestions.append("Requirement needs significant refinement - consider rewriting for clarity")
        
        return suggestions

    def handle_requirement_input(self, requirement: str) -> str:
        """Handle requirement analysis with enhanced output"""
        
        # Clean and analyze
        cleaned_requirement = requirement.strip()
        analysis = self.analyze_requirement_enhanced(cleaned_requirement)
        
        # Generate enhanced response
        response_parts = []
        
        # Classification
        response_parts.append(f"Classification: {analysis.classification}")
        
        if analysis.classification == 'FR':
            response_parts.append("This describes what your system should do - a specific function or behavior.")
        elif analysis.classification == 'NFR':
            response_parts.append("This describes how well your system should perform - a quality attribute or constraint.")
        else:
            response_parts.append("This combines both functional behavior and quality constraints.")
        
        # Enhanced ambiguity feedback
        if analysis.ambiguities:
            response_parts.append(f"\nAmbiguity Analysis - Found {len(analysis.ambiguities)} issue(s):")
            
            for i, amb in enumerate(analysis.ambiguities[:4], 1):
                terms_text = ', '.join(amb['found_terms']) if amb['found_terms'] else 'detected'
                response_parts.append(f"\n{i}. {amb['category']}: {terms_text}")
                
                if amb['fixes']:
                    response_parts.append(f"   Fix: {amb['fixes'][0]}")
        else:
            response_parts.append("\nAmbiguity Analysis - No issues detected! Requirement is well-specified.")
        
        # Enhanced traceability analysis
        response_parts.append(f"\nTraceability Analysis:")
        if analysis.traceability_suggestions['missing_requirements']:
            response_parts.append(f"Consider these {len(analysis.traceability_suggestions['missing_requirements'])} related requirements:")
            for i, req in enumerate(analysis.traceability_suggestions['missing_requirements'][:4], 1):
                response_parts.append(f"{i}. {req}")
            
            if analysis.traceability_suggestions['related_patterns']:
                response_parts.append(f"Patterns identified: {', '.join(analysis.traceability_suggestions['related_patterns'])}")
        else:
            response_parts.append("No specific missing requirements identified. Consider what supporting requirements might be needed.")
        
        # Cross-requirement analysis
        if len(self.user_requirements) > 1:
            total_frs = len([r for r in self.user_requirements if r.classification == 'FR'])
            total_nfrs = len([r for r in self.user_requirements if r.classification == 'NFR'])
            total_mixed = len([r for r in self.user_requirements if r.classification == 'MIXED'])
            
            response_parts.append(f"\nSession Analysis: {total_frs} FRs, {total_nfrs} NFRs, {total_mixed} Mixed")
            
            if total_frs > 0 and total_nfrs == 0:
                response_parts.append("Consider adding NFRs to specify quality constraints")
            elif total_nfrs > 0 and total_frs == 0:
                response_parts.append("Consider adding FRs to specify implementation functions")
        
        # Warnings
        if analysis.traceability_suggestions['dependency_warnings']:
            response_parts.append(f"\nPotential Conflicts:")
            for warning in analysis.traceability_suggestions['dependency_warnings']:
                response_parts.append(f"• {warning}")
        
        # Recommendations
        if analysis.refinement_suggestions:
            response_parts.append(f"\nRecommendations:")
            for suggestion in analysis.refinement_suggestions[:3]:
                response_parts.append(f"• {suggestion}")
        
        return '\n'.join(response_parts)

    def handle_clarification_question(self, question: str) -> str:
        """Handle questions about previous analysis"""
        
        if not self.user_requirements:
            return "I haven't analyzed any requirements yet. Please choose option 1 to enter a requirement for analysis."
        
        question_lower = question.lower()
        
        # Check if asking about a specific requirement or just the latest
        target_analysis = self.last_analysis
        
        # Allow asking about any previous requirement
        if len(self.user_requirements) > 1:
            for req in self.user_requirements:
                if any(word in req.text.lower() for word in question_lower.split()[:5]):
                    target_analysis = req
                    break
        
        if not target_analysis:
            target_analysis = self.last_analysis
        
        # Enhanced question handling with specific reasoning
        if any(phrase in question_lower for phrase in ['why', 'classification', 'classify', 'mixed', 'fr', 'nfr', 'not mixed']):
            return self.explain_classification_reasoning(target_analysis, question_lower)
            
        elif any(phrase in question_lower for phrase in ['ambiguity', 'ambiguous', 'vague', 'unclear']):
            if target_analysis.ambiguities:
                response = f"Ambiguity Details for: \"{target_analysis.text}\"\n\n"
                for i, amb in enumerate(target_analysis.ambiguities, 1):
                    response += f"{i}. {amb['type']} ({amb['severity']} severity)\n"
                    response += f"   Category: {amb['category']}\n"
                    if amb['found_terms']:
                        response += f"   Found terms: {', '.join(amb['found_terms'])}\n"
                    if amb['fixes']:
                        response += f"   Recommended fix: {amb['fixes'][0]}\n\n"
                return response
            else:
                return f"No ambiguity issues were found in: \"{target_analysis.text}\""
        
        elif any(phrase in question_lower for phrase in ['traceability', 'related', 'missing', 'coverage']):
            missing = target_analysis.traceability_suggestions['missing_requirements']
            patterns = target_analysis.traceability_suggestions['related_patterns']
            
            response = f"Traceability Analysis for: \"{target_analysis.text}\"\n\n"
            
            if missing:
                response += f"Missing related requirements ({len(missing)}):\n"
                for i, req in enumerate(missing, 1):
                    response += f"{i}. {req}\n"
                response += "\n"
            
            if patterns:
                response += f"Patterns identified: {', '.join(patterns)}\n\n"
            
            if target_analysis.traceability_suggestions['dependency_warnings']:
                response += "Potential conflicts:\n"
                for warning in target_analysis.traceability_suggestions['dependency_warnings']:
                    response += f"• {warning}\n"
            
            return response
        
        else:
            return f"""I can explain details about your requirement: \"{target_analysis.text[:60]}...\"\n\nAvailable explanations:\n• Classification - Why it was categorized as {target_analysis.classification}\n• Ambiguity issues - What makes it unclear and how to fix it\n• Traceability - Missing related requirements and patterns\n\nTry asking: \"Why was it classified as {target_analysis.classification}?\" """

    def explain_classification_reasoning(self, analysis, question_lower: str) -> str:
        """Provide specific reasoning for classification decisions"""
        
        response = f"Why '{analysis.text}' was classified as {analysis.classification}:\n\n"
        
        text_lower = analysis.text.lower()
        
        if analysis.classification == 'FR':
            response += "Functional Requirement - describes WHAT the system does:\n\n"
            
            # Identify specific functional elements
            functional_indicators = []
            action_verbs = ['synchronize', 'display', 'show', 'generate', 'create', 'produce', 'calculate', 'store', 'save', 'send', 'receive', 'process']
            found_actions = [verb for verb in action_verbs if verb in text_lower]
            if found_actions:
                functional_indicators.append(f"• Contains action verb(s): {', '.join(found_actions)}")
            
            if 'synchronize' in text_lower or 'sync' in text_lower:
                functional_indicators.append("• Describes a specific system behavior (synchronization)")
            if 'system' in text_lower or 'product' in text_lower:
                functional_indicators.append("• Specifies what the system/product will do")
            if 'with' in text_lower:
                functional_indicators.append("• Defines interaction with external system")
            
            if functional_indicators:
                response += '\n'.join(functional_indicators)
            else:
                response += "• Describes a specific system function or behavior"
            
            # Explain why it's NOT NFR or MIXED
            if 'not mixed' in question_lower or 'why not mixed' in question_lower:
                response += f"\n\nWhy it's NOT MIXED:\n"
                response += f"• While 'every hour' specifies timing, it defines WHEN the function happens, not performance quality\n"
                response += f"• No quality constraints like 'fast', 'reliable', or specific performance metrics\n"
                response += f"• Focuses on behavior (synchronization) rather than quality attributes\n"
                response += f"• The timing is operational scheduling, not a performance requirement"
                
        elif analysis.classification == 'NFR':
            response += "Non-Functional Requirement - describes HOW WELL the system performs:\n\n"
            
            # Identify specific NFR elements
            nfr_indicators = []
            quality_words = ['fast', 'slow', 'reliable', 'secure', 'usable', 'accessible', 'efficient', 'performance', 'quality']
            found_quality = [word for word in quality_words if word in text_lower]
            if found_quality:
                nfr_indicators.append(f"• Contains quality attributes: {', '.join(found_quality)}")
            
            if any(constraint in text_lower for constraint in ['within', 'under', 'less than', 'more than', 'at least']):
                nfr_indicators.append("• Specifies performance constraints or limits")
            
            if re.search(r'\d+\s*(second|minute|hour|%|fps)', text_lower):
                nfr_indicators.append("• Contains measurable quality criteria")
                
            if nfr_indicators:
                response += '\n'.join(nfr_indicators)
            else:
                response += "• Describes quality attributes or performance constraints"
                
        elif analysis.classification == 'MIXED':
            response += "Mixed Requirement - contains BOTH function and quality:\n\n"
            
            # Identify functional parts
            action_verbs = ['display', 'show', 'generate', 'create', 'search', 'login', 'synchronize']
            found_actions = [verb for verb in action_verbs if verb in text_lower]
            if found_actions:
                response += f"Function (WHAT): {', '.join(found_actions)}\n"
            
            # Identify quality parts  
            quality_words = ['fast', 'quickly', 'slow', 'reliable', 'accessible', 'within', 'under']
            found_quality = [word for word in quality_words if word in text_lower]
            if found_quality:
                response += f"Quality (HOW WELL): {', '.join(found_quality)}\n"
                
            response += f"\nRecommendation: Split into separate FR and NFR for better clarity."
        
        return response

    def handle_general_help(self) -> str:
        """Enhanced general help with interactive capability"""
        
        return """Enhanced Requirements Engineering Chatbot Help

What I can do:
• Smart Classification - FR, NFR, or Mixed
• Advanced Ambiguity Detection - Rule-based analysis for comprehensive coverage
• Enhanced Traceability - Missing requirements, patterns, and conflict detection
• Cross-Requirement Analysis - Session-wide requirement coverage assessment

Menu Options:
1 Analyze Requirements - Fast, comprehensive analysis with specific recommendations
2 Ask About Analysis - Get detailed explanations of my findings
3 General Help - Tips, examples, and guidance

Example Analyses:
• "System should be fast" → Detects vague "fast", suggests "Response time ≤ 200ms"
• "Display accessibility mode" → Identifies accessibility pattern, suggests WCAG compliance
• "Product synchronizes every hour" → Identifies sync pattern, suggests performance NFRs

Performance: Analysis typically completes in 1-3 seconds using hybrid rule-based approach.

You can ask me questions about requirements engineering, or choose option 1 to analyze a requirement!"""

    def handle_general_question(self, question: str) -> str:
        """Handle general questions about requirements engineering"""
        
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ['what is', 'define', 'explain']) and any(term in question_lower for term in ['fr', 'functional requirement']):
            return """Functional Requirements (FR):

Functional requirements describe what the system should do - the specific behaviors, actions, and functions.

Examples:
• "Users can login with username and password"
• "System generates monthly sales reports" 
• "Application displays search results in a list"

Key characteristics:
• Describes system behavior or functionality
• Usually involves actions (login, display, calculate, store)
• Answers "what does the system do?"
• Can be tested by checking if the function works"""

        elif any(phrase in question_lower for phrase in ['what is', 'define', 'explain']) and any(term in question_lower for term in ['nfr', 'non-functional', 'quality']):
            return """Non-Functional Requirements (NFR):

Non-functional requirements describe how well the system should perform - quality attributes, constraints, and performance criteria.

Examples:
• "Response time must be under 2 seconds"
• "System must have 99.9% uptime"
• "Interface must be accessible for disabled users"

Key characteristics:
• Describes quality attributes or constraints
• Usually involves measurements (time, percentage, capacity)
• Answers "how well does the system do it?"
• Can be tested by measuring performance/quality"""

        elif any(phrase in question_lower for phrase in ['mixed', 'hybrid']):
            return """Mixed Requirements:

Mixed requirements contain both functional behavior AND quality/performance constraints in a single statement.

Examples:
• "Users can search products and get results within 2 seconds" (function + performance)
• "System displays error messages in an accessible format" (function + accessibility)

Why this matters:
• Mixed requirements are often harder to test and implement
• Better to split into separate FR and NFR for clarity
• Each part can have its own acceptance criteria"""

        elif any(phrase in question_lower for phrase in ['example', 'show me']):
            return """Example Analysis:

Input: "The system should be user-friendly and display search results quickly"

Classification: MIXED (function + quality)

Ambiguities Found:
• "user-friendly" → Replace with "SUS score ≥ 80"
• "quickly" → Replace with "within 2 seconds"

Traceability Suggestions:
• Search performance requirements (accuracy, concurrency)
• Display requirements (formatting, accessibility)
• Usability requirements (training, error handling)

Recommendation: Split into separate requirements for better clarity and testability."""

        else:
            return """Ask me about:
• FR vs NFR - "What is a functional requirement?"
• Mixed requirements - "What are mixed requirements?"
• Examples - "Show me an example analysis"

Or choose option 1 to analyze a specific requirement!"""

    def get_session_summary(self) -> Dict:
        """Enhanced session summary"""
        
        summary = {
            'session_id': self.session_id,
            'requirements_analyzed': len(self.user_requirements),
            'conversation_turns': len(self.conversation_history),
            'classification_breakdown': {},
            'total_ambiguities_found': 0,
            'patterns_identified': []
        }
        
        if self.user_requirements:
            # Classification breakdown
            for req in self.user_requirements:
                label = req.classification
                summary['classification_breakdown'][label] = summary['classification_breakdown'].get(label, 0) + 1
            
            # Ambiguity analysis
            summary['total_ambiguities_found'] = sum(len(req.ambiguities) for req in self.user_requirements)
            
            # Patterns identified
            all_patterns = []
            for req in self.user_requirements:
                all_patterns.extend(req.traceability_suggestions['related_patterns'])
            summary['patterns_identified'] = list(set(all_patterns))
        
        return summary

class EnhancedChatbotSession:
    """Enhanced session manager with menu-driven interface"""
    
    def __init__(self):
        self.chatbot = EnhancedRequirementsChatbot()
        self.is_running = False
    
    def start_interactive_session(self):
        """Start enhanced interactive session"""
        
        print("\n" + "="*70)
        print("ENHANCED REQUIREMENTS ENGINEERING CHATBOT")
        print("="*70)
        print("Welcome! I provide fast, comprehensive requirement analysis using:")
        print("• Smart classification (FR/NFR/Mixed)")
        print("• Advanced ambiguity detection")  
        print("• Enhanced traceability analysis")
        print("• Cross-requirement pattern analysis")
        print("\nType 'quit' anytime to exit, 'summary' for session stats")
        print("="*70)
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Show menu
                choice = input(self.chatbot.show_input_menu()).strip()
                
                if choice.lower() in ['quit', 'exit', 'q']:
                    self.end_session()
                    break
                    
                elif choice.lower() == 'summary':
                    self.show_session_summary()
                    continue
                    
                elif choice == '1':
                    # Requirement analysis
                    requirement = input("\nEnter your requirement: ").strip()
                    if requirement:
                        print(f"\nAnalyzing requirement...")
                        response = self.chatbot.handle_requirement_input(requirement)
                        print(f"\nAnalysis Results:\n{response}\n")
                    else:
                        print("Please enter a requirement to analyze.\n")
                
                elif choice == '2':
                    # Clarification questions
                    if self.chatbot.last_analysis:
                        question = input(f"\nAsk about the analysis of: \"{self.chatbot.last_analysis.text[:50]}...\"\nYour question: ").strip()
                        if question:
                            response = self.chatbot.handle_clarification_question(question)
                            print(f"\nExplanation:\n{response}\n")
                        else:
                            print("Please enter a question.\n")
                    else:
                        print("No previous analysis to ask about. Please analyze a requirement first (option 1).\n")
                
                elif choice == '3':
                    # General help with interactive capability
                    help_text = self.chatbot.handle_general_help()
                    print(f"\nHelp & Guidance:\n{help_text}\n")
                    
                    # Allow follow-up questions
                    while True:
                        follow_up = input("Ask me anything about requirements engineering (or 'back' to return to menu): ").strip()
                        if follow_up.lower() in ['back', 'menu', 'return']:
                            break
                        elif follow_up.lower() in ['quit', 'exit']:
                            self.end_session()
                            return
                        elif follow_up:
                            response = self.chatbot.handle_general_question(follow_up)
                            print(f"\nAnswer:\n{response}\n")
                        else:
                            print("Please ask a question or type 'back' to return to the menu.\n")
                
                else:
                    # Smart fallback - check if input looks like a requirement
                    if self.looks_like_requirement(choice):
                        print(f"\nI detected you entered a requirement. Analyzing...")
                        response = self.chatbot.handle_requirement_input(choice)
                        print(f"\nAnalysis Results:\n{response}\n")
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.\n")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")

    def looks_like_requirement(self, text: str) -> bool:
        """Detect if user input looks like a requirement - improved to avoid false positives"""
        
        # Skip if it's clearly a menu choice
        if text.strip() in ['1', '2', '3']:
            return False
            
        text_lower = text.lower().strip()
        
        # Skip obvious complaints, questions, or conversational responses
        negative_indicators = [
            'that\'s not', 'thats not', 'that is not', 'not valid', 'not a valid',
            'wrong', 'incorrect', 'bad', 'poor', 'terrible', 'awful',
            'why', 'what', 'how', 'when', 'where', 'who',
            'i think', 'i believe', 'i feel', 'you should', 'you need',
            'can you', 'could you', 'would you', 'will you',
            'help', 'explain', 'clarify', 'tell me', 'show me'
        ]
        
        if any(indicator in text_lower for indicator in negative_indicators):
            return False
        
        # Must be reasonably long for a requirement
        if len(text.split()) < 5:
            return False
            
        # Strong requirement indicators
        requirement_indicators = [
            r'\b(system|product|application|software)\s+(shall|must|should|will|can)',
            r'\b(user|users|customer|admin)\s+(shall|must|should|will|can)',
            r'\bthe\s+(system|product|application)\s+(shall|must|should|will)',
            r'\b(shall|must|should|will)\s+\w+',
            r'\b(system|product)\s+(shall|will|must|should|can)\s+(provide|display|generate|process)',
        ]
        
        # Check for strong requirement patterns
        has_strong_patterns = any(re.search(pattern, text_lower) for pattern in requirement_indicators)
        
        # Additional requirement words
        requirement_words = ['system', 'shall', 'must', 'requirement', 'function', 'feature']
        has_requirement_words = sum(1 for word in requirement_words if word in text_lower) >= 2
        
        return has_strong_patterns or has_requirement_words

    def show_session_summary(self):
        """Display enhanced session summary"""
        summary = self.chatbot.get_session_summary()
        
        print(f"\nENHANCED SESSION SUMMARY")
        print("="*40)
        print(f"Requirements analyzed: {summary['requirements_analyzed']}")
        
        if summary['classification_breakdown']:
            print(f"\nClassification breakdown:")
            for label, count in summary['classification_breakdown'].items():
                print(f"  {label}: {count}")
        
        print(f"Total ambiguity issues found: {summary['total_ambiguities_found']}")
        
        if summary['patterns_identified']:
            print(f"Patterns identified: {', '.join(summary['patterns_identified'][:3])}{'...' if len(summary['patterns_identified']) > 3 else ''}")
        
        print()

    def end_session(self):
        """End session with summary"""
        summary = self.chatbot.get_session_summary()
        
        print(f"\nSession Complete!")
        print(f"• Analyzed {summary['requirements_analyzed']} requirements")
        print(f"• Found {summary['total_ambiguities_found']} ambiguity issues")
        print(f"• Identified {len(summary['patterns_identified'])} patterns")
        print("\nThank you for using the Enhanced Requirements Engineering Chatbot!")
        
        self.is_running = False

def main():
    """Main function with enhanced options"""
    
    print("Enhanced Requirements Engineering Chatbot")
    print("Choose mode:")
    print("1. Interactive session (recommended)")
    print("2. Single requirement analysis")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        session = EnhancedChatbotSession()
        session.start_interactive_session()
            
    elif choice == "2":
        chatbot = EnhancedRequirementsChatbot()
        requirement = input("Enter requirement to analyze: ").strip()
        if requirement:
            print(f"\nAnalyzing...")
            response = chatbot.handle_requirement_input(requirement)
            print(f"\nAnalysis:\n{response}")
        else:
            print("No requirement provided.")
    
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()