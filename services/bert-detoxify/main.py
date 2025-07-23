import os
import time
import logging
import threading
import requests
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bert-detoxify-service")

# Set cache directories
if os.getenv('IN_DOCKER') == 'true':
    os.environ['TORCH_HOME'] = '/tmp/cache'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache'
    os.environ['HF_HOME'] = '/tmp/cache'
    # Ensure cache directories exist
    os.makedirs('/tmp/cache', exist_ok=True)
    os.makedirs('/app/model_cache', exist_ok=True)

# Global variables for model management
model_ready = False
model_loading = False
model_load_start_time = None
startup_time = None

# FastAPI app definition
app = FastAPI(title="BERT Detoxify Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced data structures
@dataclass
class DetectionResult:
    """Structured result for pattern detection"""
    score: float
    confidence: float
    severity_level: Optional[str]
    matches: List[Dict]
    excluded_context: Optional[str]
    processing_time_ms: float

# Enums and Models
class ActualValueDtype(str, Enum):
    FLOAT = "float"

class EvaluationType(str, Enum):
    TOXICITY_EVALUATION = "toxicity_evaluation"
    HATE_SPEECH_EVALUATION = "hate_speech_evaluation"
    SEXUAL_CONTENT_EVALUATION = "sexual_content_evaluation"
    TERRORISM_EVALUATION = "terrorism_evaluation"
    VIOLENCE_EVALUATION = "violence_evaluation"
    SELF_HARM_EVALUATION = "self_harm_evaluation"

class MetricReturnModel(BaseModel):
    metric_name: EvaluationType
    actual_value: float
    actual_value_type: ActualValueDtype
    others: dict = {}

class TextRequest(BaseModel):
    text: str

# Enhanced Content Moderation Patterns
class ContentModerationPatterns:
    """Production-grade content moderation with enhanced pattern detection"""
    
    def __init__(self):
        self._init_patterns()
        self._compile_patterns()
        
    def _init_patterns(self):
        """Initialize pattern dictionaries with severity-based weights"""
        
        # Enhanced sexual content patterns with better educational exclusion
        self.sexual_patterns = {
            'explicit': {
                'weight': 0.95,
                'patterns': [
                    r'\b(hardcore|explicit|graphic)\s+(sex|porn|sexual)',
                    r'\b(penis|vagina|genitals?|naked\s+body)\s+(picture|photo|image)',
                    r'\bsend\s+(nudes?|naked\s+pics?|porn|explicit)',
                    r'\b(sexual|intimate)\s+(acts?|encounters?)\b',
                    r'\b(naked|nude)\s+(body|pictures?|photos?|images?)\b',
                    r'\b(sex|sexual)\s+(video|film|clip|content)\b',
                    r'\b(porn|pornography|adult)\s+(video|content|material)\b',
                    r'\b(genital|private\s+parts?)\s+(exposure|showing|display)\b',
                    r'\b(sexual|intimate)\s+(performance|show|act)\b',
                ]
            },
            'very_high': {
                'weight': 0.85,
                'patterns': [
                    # New tier for better score distribution
                    r'\b(explicit|graphic)\s+(sexual\s+)?(content|material|imagery)',
                    r'\b(adult\s+entertainment|pornographic\s+material)',
                    r'\b(sexual\s+)?(fantasy|desire|arousal|stimulation)',
                    r'\b(intimate|private)\s+(photos?|videos?|images?)',
                    r'\b(sexual\s+)?(content|material)\s+(warning|explicit)',
                ]
            },
            'high': {
                'weight': 0.8,
                'patterns': [
                    # Explicit terms, but exclude educational if combined with educational patterns
                    r'\b(dick|cock|pussy|tits|breast|ass)\b(?!\s*(head|istant|ume|ociation))',
                    r'\b(masturbat|orgasm|climax)\w*\b',
                    r'\b(sexual|intimate)\s+(fantasy|desire|urge)\b',
                    r'\b(erotic|sensual)\s+(content|material|imagery)\b',
                ]
            },
            'moderate': {
                'weight': 0.55,
                'patterns': [
                    # Reduced from 0.6 to allow educational context to have more impact
                    r'\b(sexy|seductive|suggestive|sensual)\b',
                    r'\b(attraction|attracted|aroused)\b',
                    r'\b(flirt|flirting|romantic)\s+(message|text|conversation)\b',
                    r'\b(beautiful|handsome|attractive)\s+(body|figure|appearance)\b',
                ]
            },
            'educational': {
                # New tier specifically for educational content - will be heavily reduced by context
                'weight': 0.4,
                'patterns': [
                    r'\b(sexual\s+education|sex\s+education|adult\s+content|mature\s+content)\b',
                    r'\b(sexual|intimate)\s+(relationships?|health)\b',
                    r'\b(sex|sexual|nude|naked|intimate|erotic)\b',  # Generic terms that get reduced
                ]
            },
            'mild': {
                'weight': 0.25,
                'patterns': [
                    # Reduced from 0.3
                    r'\b(romantic|love|affection)\b',
                    r'\b(relationship|dating|couple)\b',
                    r'\b(beautiful|pretty|handsome)\b',
                ]
            }
        }
        
        # Enhanced terrorism patterns with better score distribution and maximum coverage
        self.terrorism_patterns = {
            'imminent_threat': {
                'weight': 1.0,
                'patterns': [
                    # Immediate, specific threats with weapons/timing
                    r'\bi\s+(will|am\s+about\s+to|am\s+going\s+to)\s+(bomb|blow\s+up|attack|destroy)\s+(the|this)\s+(building|airport|school|government|mall|stadium)\s+(today|tonight|tomorrow|now)',
                    r'\bi\s+have\s+(explosive|bomb|weapon|gun|knife)\s+(ready|here|loaded|armed)\s+(to\s+)?(kill|attack|destroy)\s+(everyone|infidels|americans)',
                    r'\bthis\s+is\s+(my\s+)?(final\s+)?(warning|message)\s+before\s+(the\s+)?(attack|bombing|strike|jihad)',
                    r'\b(death\s+to\s+america|destroy\s+all\s+(infidels|christians|jews))\s+(today|tonight|now|immediately)',
                    r'\ballahu\s+akbar\s+(prepare\s+to\s+die|time\s+to\s+die|you\s+will\s+all\s+die)',
                ]
            },
            'extreme': {
                'weight': 0.98,
                'patterns': [
                    # Detailed plans and immediate threats
                    r'\bi\s+(will|am\s+going\s+to|plan\s+to)\s+(bomb|blow\s+up|attack|destroy)\s+(the|this)\s+(building|airport|school|government)',
                    r'\b(detailed\s+plan|specific\s+target|coordinated\s+attack)\s+(to\s+)?(kill|bomb|destroy|attack)',
                    r'\bi\s+have\s+(explosives?|bomb|weapons?|suicide\s+vest)\s+(ready|prepared|hidden|strapped\s+on)',
                    r'\b(tomorrow|tonight|today)\s+we\s+(attack|strike|bomb|destroy|execute\s+our\s+plan)',
                    r'\b(jihad|martyrdom)\s+(mission|operation)\s+(today|tomorrow|ready|planned|activated)',
                    # Enhanced extreme threats
                    r'\bwe\s+will\s+(massacre|slaughter|exterminate)\s+(all\s+)?(infidels|americans|christians|jews|civilians)',
                    r'\b(mass\s+casualty|maximum\s+damage)\s+(event|attack|operation|strike)',
                    r'\b(chemical|biological|nuclear)\s+(weapon|attack|device)\s+(ready|planned|armed)',
                ]
            },
            'critical': {
                'weight': 0.95,
                'patterns': [
                    r'\b(bomb|explosive|ied)\s+(threat|attack|target|plan)',
                    r'\b(kill|murder|assassinate)\s+(president|politician|leader|official)',
                    r'\b(suicide\s+bomb|car\s+bomb|explosive\s+device)',
                    r'\b(isis|al[-\s]?qaeda|terrorist\s+(group|organization|cell))',
                    r'\b(jihad|martyrdom)\s+(operation|mission|attack)',
                    # Enhanced with specific threat language
                    r'\b(mass\s+shooting|coordinated\s+attack|terror\s+cell)',
                    r'\b(radicalization|recruitment)\s+(for|to)\s+(jihad|terrorism|extremism)',
                    # Additional critical patterns
                    r'\b(hostage|kidnap|execution)\s+(video|operation|plan)',
                    r'\b(training\s+camp|weapons\s+cache|safe\s+house)\s+(location|address)',
                ]
            },
            'severe': {
                'weight': 0.85,
                'patterns': [
                    r'\b(terrorist|terrorism|extremist)\s+(attack|operation|plot|plan)',
                    r'\b(bomb|explosive|detonate|blast)\s+(plan|threat|attack)',
                    r'\b(radical|fundamentalist|militant)\s+(group|organization|movement|training)',
                    r'\b(terror\s+attack|mass\s+casualty|civilian\s+targets)',
                    r'\b(weapons\s+cache|training\s+camp|safe\s+house)',
                ]
            },
            'high': {
                'weight': 0.7,
                'patterns': [
                    r'\b(terrorist|terrorism|extremist)\s+(attack|operation)',
                    r'\b(bomb|explosive|detonate|blast)\b',
                    r'\b(radical|fundamentalist|militant)\s+(group|organization|movement)',
                    r'\b(terror\s+attack|mass\s+casualty)\b',
                ]
            },
            'moderate': {
                'weight': 0.45,
                'patterns': [
                    r'\b(terrorist|terrorism|extremist|radical|fundamentalist|militant)\b',
                    r'\b(jihad|infidel|crusade)\b',
                    r'\b(revolution|uprising|resistance)\s+(movement|group)',
                ]
            }
        }
        
        # Violence patterns with contextual awareness
        self.violence_patterns = {
            'severe': {
                'weight': 0.95,
                'patterns': [
                    r'\b(kill|murder|assassinate|execute)\s+(you|him|her|them|everyone)',
                    r'\b(shoot|stab|beat|torture)\s+(to\s+death|until\s+dead|brutally)',
                    r'\b(torture|mutilate|dismember|massacre)\b',
                    r'\b(graphic\s+violence|extreme\s+violence)\b',
                ]
            },
            'high': {
                'weight': 0.7,
                'patterns': [
                    r'\b(kill|murder|assault|attack)\s+(plan|threat|intention)',
                    r'\b(weapon|gun|knife|blade|machete)\s+(threat|attack|violence)',
                    r'\b(hurt|harm|injure|wound)\s+(badly|seriously|severely)',
                    r'\b(violence|violent)\s+(threat|attack|assault)',
                ]
            },
            'moderate': {
                'weight': 0.5,
                'patterns': [
                    r'\b(kill|murder|assault|attack|fight|violence|violent)\b',
                    r'\b(weapon|gun|knife|sword|club)\b',
                    r'\b(hurt|harm|injure|wound|bleed|blood)\b',
                ]
            },
            'mild': {
                'weight': 0.3,
                'patterns': [
                    r'\b(fight|fighting|aggressive|aggression)\b',
                    r'\b(punch|hit|slap|kick|shove)\b',
                ]
            }
        }
        
        # Enhanced self-harm patterns with maximum coverage and immediate danger detection
        self.self_harm_patterns = {
            'immediate_danger': {
                'weight': 1.0,
                'patterns': [
                    # Immediate action statements with specific methods/timing
                    r'\bi\s+am\s+(about\s+to|going\s+to)\s+(kill\s+myself|end\s+my\s+life|take\s+my\s+life)\s+(right\s+now|tonight|today|in\s+\d+\s+(minutes?|hours?))',
                    r'\bi\s+(have\s+)?(loaded\s+the\s+gun|tied\s+the\s+rope|filled\s+the\s+bathtub|taken\s+the\s+pills)',
                    r'\bthis\s+is\s+(it|the\s+end|goodbye)\s*[\.!]\s*(i\s+(can\'?t|cannot|won\'?t)\s+(go\s+on|continue|live)|nobody\s+will\s+stop\s+me)',
                    r'\bi\s+am\s+(cutting|hurting|poisoning)\s+myself\s+(right\s+now|as\s+i\s+type|while\s+writing)',
                    r'\b(if\s+you\'?re\s+reading\s+this|by\s+the\s+time\s+you\s+find\s+this)\s+(i\'?ll\s+be\s+dead|it\'?ll\s+be\s+too\s+late|i\'?m\s+already\s+gone)',
                ]
            },
            'imminent': {
                'weight': 0.98,
                'patterns': [
                    # Immediate, specific plans or actions
                    r'\bi\s+(have|got)\s+(the\s+)?(pills?|rope|knife|gun|razor|blade)\s+(ready|here|with\s+me|in\s+my\s+hand)',
                    r'\bi\s+am\s+(doing\s+this|ending\s+it|finishing\s+this)\s+(tonight|today|now|right\s+now|very\s+soon)',
                    r'\bthis\s+is\s+(my\s+)?(final\s+)?(goodbye|message|note|time)\b',
                    r'\bi\s+(can\'?t|cannot)\s+(take|handle|bear|stand)\s+(this|it|life)\s+(anymore|any\s+longer|one\s+more\s+day)',
                    r'\b(by\s+the\s+time\s+you\s+read\s+this|when\s+you\s+find\s+this|when\s+you\s+see\s+this)',
                    # Enhanced immediate planning
                    r'\bi\s+(know\s+exactly\s+how|have\s+everything\s+ready)\s+(to\s+)?(kill\s+myself|end\s+it|die)',
                    r'\b(tonight\s+is\s+the\s+night|today\s+is\s+the\s+day|this\s+is\s+my\s+last)\s+(day|night|message|post)',
                ]
            },
            'critical': {
                'weight': 0.95,
                'patterns': [
                    r'\bi\s+(want\s+to\s+|am\s+going\s+to\s+|will\s+|need\s+to\s+)?kill\s+myself\b',
                    r'\bi\s+am\s+going\s+to\s+(end|take)\s+my\s+life',
                    r'\b(tonight|today|tomorrow|soon)\s+i\s+(die|end\s+it|kill\s+myself|won\'?t\s+be\s+here)',
                    r'\b(suicide\s+(plan|method|note|attempt)|final\s+goodbye)\b',
                    r'\b(overdose|hanging|jumping|cutting|slashing)\s+(plan|method|tonight|today)\b',
                    # Enhanced immediate danger signals
                    r'\bi\s+(have\s+)?(planned|decided|chosen)\s+(how|when|where)\s+(to\s+)?(kill\s+myself|end\s+it|die)',
                    r'\b(nobody|no\s+one)\s+will\s+(miss|care\s+about|remember)\s+me\s+(when\s+)?i\'?m\s+gone',
                    # Additional critical patterns
                    r'\bi\s+(want\s+to\s+|am\s+going\s+to\s+)?(cut|hurt|harm|kill)\s+myself\s+(deeply|badly|seriously|fatally)',
                    r'\b(life\s+insurance|suicide\s+note|final\s+wishes)\s+(policy|ready|written)',
                ]
            },
            'severe': {
                'weight': 0.85,
                'patterns': [
                    r'\b(suicide|kill\s+myself|end\s+my\s+life|take\s+my\s+life)\b',
                    r'\b(want\s+to\s+die|wish\s+i\s+was\s+dead|better\s+off\s+dead|should\s+be\s+dead)\b',
                    r'\b(self\s+harm|cut\s+myself|hurt\s+myself|harm\s+myself)\b',
                    r'\b(no\s+reason\s+to\s+live|life\s+is\s+meaningless|life\s+has\s+no\s+point)\b',
                    # Enhanced severe ideation
                    r'\bi\s+(hate|despise|can\'?t\s+stand|loathe)\s+(myself|my\s+life|living|being\s+alive)',
                    r'\b(everyone\s+would\s+be\s+better\s+off|world\s+would\s+be\s+better)\s+(without\s+me|if\s+i\s+was\s+gone|if\s+i\s+died)',
                    # Additional severe patterns
                    r'\bi\s+(deserve\s+to\s+die|should\s+die|am\s+worthless|am\s+a\s+burden)',
                    r'\b(cutting|self[-\s]?injury|self[-\s]?mutilation)\s+(helps|makes\s+me\s+feel|is\s+the\s+only\s+way)',
                ]
            },
            'high': {
                'weight': 0.7,
                'patterns': [
                    r'\b(depression|depressed|hopeless|worthless|useless|suicidal)\b',
                    r'\b(no\s+one\s+cares|nobody\s+(loves|cares\s+about|understands)\s+me)\b',
                    r'\b(nothing\s+matters|give\s+up|can\'?t\s+go\s+on|can\'?t\s+continue)\b',
                    # Enhanced emotional distress
                    r'\bi\s+(feel|am)\s+(completely\s+)?(alone|isolated|abandoned|trapped|broken|destroyed)',
                    r'\b(life\s+is\s+)?(pointless|empty|unbearable|too\s+hard|not\s+worth\s+living)',
                    # Additional high-risk patterns
                    r'\b(mental\s+pain|emotional\s+pain)\s+(is\s+)?(unbearable|too\s+much|overwhelming)',
                    r'\bi\s+(can\'?t\s+|cannot\s+)(cope|handle\s+this|take\s+it)\s+(anymore|much\s+longer)',
                ]
            },
            'moderate': {
                'weight': 0.45,
                'patterns': [
                    # General distress indicators
                    r'\b(struggling|having\s+a\s+hard\s+time|going\s+through\s+a\s+tough\s+time)\b',
                    r'\b(overwhelmed|stressed|exhausted|burned\s+out)\b',
                    r'\b(don\'?t\s+know\s+what\s+to\s+do|lost|confused|desperate)\b',
                    # Additional moderate indicators
                    r'\b(mental\s+health|therapy|counseling|medication|antidepressant)\b',
                    r'\b(crisis|breakdown|falling\s+apart|losing\s+hope)\b',
                ]
            }
        }
        
        # Enhanced hate speech patterns with improved explicit detection
        self.hate_speech_patterns = {
            'critical': {
                'weight': 0.98,
                'patterns': [
                    # Direct threats and extreme hate
                    r'\b(kill\s+all|exterminate|eliminate|get\s+rid\s+of)\s+(jews|blacks|muslims|gays|lgbtq|immigrants|minorities)\b',
                    r'\b(nazi|hitler|holocaust\s+(denier|denial)|ethnic\s+cleansing|genocide)\b',
                    r'\b(lynch|hang|burn|shoot)\s+(them|all\s+of\s+them|blacks|minorities)\b',
                    r'\b(white\s+(power|supremacy)|master\s+race|racial\s+purity|final\s+solution)\b',
                ]
            },
            'severe': {
                'weight': 0.90,
                'patterns': [
                    # Explicit dehumanization and superiority claims
                    r'\b(people\s+from\s+that\s+race\s+are\s+inferior|shouldn\'?t\s+be\s+trusted)\b',
                    r'\b(destroying\s+our\s+(country|culture)|taking\s+over)\b',
                    r'\b(all\s+extremists|all\s+dangerous|all\s+criminals)\b',
                    r'\b(don\'?t\s+belong\s+in|too\s+emotional|inferior|subhuman|animals)\b',
                    r'\b(unnatural|shouldn\'?t\s+have\s+equal\s+rights|against\s+nature)\b',
                ]
            },
            'high': {
                'weight': 0.75,
                'patterns': [
                    # Strong discriminatory language
                    r'\b(racist|bigot|supremacist|fascist|xenophobe)\b',
                    r'\b(hate|hatred|disgusting|filth|dirty)\s+(jew|black|muslim|gay|immigrant)\b',
                    r'\b(go\s+back\s+to|belong\s+in|deport)\s+(africa|mexico|your\s+country)\b',
                    r'\b(inferior|superior)\s+(race|people|gender|sexuality)\b',
                    r'\b(members\s+of\s+that\s+religion|those\s+immigrants|women)\s+(are\s+all|always)\b',
                ]
            },
            'moderate': {
                'weight': 0.45,
                'patterns': [
                    # Stereotyping and generalizations - reduced weight from 0.5
                    r'\b(those\s+people\s+are\s+always|they\'?re\s+all\s+the\s+same)\b',
                    r'\b(can\'?t\s+stand\s+people\s+from\s+that\s+group)\b',
                    r'\b(always\s+complaining|causing\s+trouble)\b',
                ]
            },
            'mild': {
                'weight': 0.25,
                'patterns': [
                    # General bias indicators - new lower tier
                    r'\b(racist|bigot|prejudice|discrimination|bias)\b',
                    r'\b(stereotype|generalize|categorize)\b',
                ]
            }
        }
        
        # Enhanced exclusion patterns to reduce false positives
        self.exclusion_patterns = {
            'medical_educational': [
                r'\b(medical|clinical|educational|academic|research|scientific)\b',
                r'\b(study|analysis|therapy|treatment|diagnosis|patient)\b',
                r'\b(university|school|college|institution|journal)\b',
                r'\b(doctor|physician|therapist|counselor|psychologist)\b',
                # Enhanced educational context for sexual content
                r'\b(education|educational|learning|teaching|instruction)\b',
                r'\b(curriculum|course|class|lesson|textbook|health\s+class)\b',
                r'\b(important|crucial|necessary)\s+(for|to)\s+(understand|learn|know)\b',
                r'\b(teenagers?|students?|young\s+people)\s+(to\s+understand|should\s+know)\b',
            ],
            'media_fiction': [
                r'\b(game|gaming|video\s+game|movie|film|fiction|novel|book)\b',
                r'\b(character|player|story|plot|script|screenplay)\b',
                r'\b(tv\s+show|series|documentary|animation)\b',
                # Enhanced context for entertainment content
                r'\b(action\s+movie|thriller|drama|comedy|entertainment)\b',
                r'\b(special\s+effects|stunts|realistic|choreographed)\b',
            ],
            'historical_news': [
                r'\b(historical|history|past|ancient|medieval|war|conflict)\b',
                r'\b(museum|archive|historical\s+record|timeline)\b',
                r'\b(news|report|article|journalism|correspondent)\b',
                r'\b(world\s+war|civil\s+war|revolution|historical\s+event)\b',
                # Enhanced news context
                r'\b(reported|authorities|officials|according\s+to)\b',
                r'\b(in\s+another\s+country|overseas|international)\b',
            ],
            'legal_policy': [
                r'\b(law|legal|court|judge|attorney|lawyer)\b',
                r'\b(policy|legislation|government|official|regulation)\b',
                r'\b(debate|discussion|argument|perspective)\b',
                # Enhanced policy context
                r'\b(should\s+be\s+(properly\s+)?age[-\s]?restricted|content\s+restrictions)\b',
                r'\b(platforms?|websites?|services?)\s+(should|must|need\s+to)\b',
            ],
            'constructive_criticism': [
                # New category for constructive criticism contexts
                r'\b(poorly\s+(designed|made|written)|badly\s+(designed|executed))\b',
                r'\b(constructive\s+criticism|feedback|improvement|suggestion)\b',
                r'\b(software|application|system|program|code)\s+(is|was)\b',
                r'\b(this\s+(movie|film|book|software|app))\b',
            ],
            'motivational_context': [
                # New category for motivational/self-improvement contexts
                r'\b(work\s+harder|push\s+(myself|yourself)|strive\s+for)\b',
                r'\b(achieve\s+(my|your)\s+goals|personal\s+growth|self\s+improvement)\b',
                r'\b(deadline\s+is\s+killing|dying\s+to\s+see|killing\s+it)\b',
                r'\b(break\s+a\s+leg|knock\s+(them|it)\s+dead)\b',
            ]
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.compiled_patterns = {}
        
        for category in ['sexual_patterns', 'terrorism_patterns', 'violence_patterns', 
                        'self_harm_patterns', 'hate_speech_patterns']:
            self.compiled_patterns[category] = {}
            pattern_dict = getattr(self, category)
            
            for severity, config in pattern_dict.items():
                compiled_patterns = []
                for pattern in config['patterns']:
                    try:
                        compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e:
                        logger.warning(f"Failed to compile pattern {pattern}: {e}")
                self.compiled_patterns[category][severity] = {
                    'weight': config['weight'],
                    'patterns': compiled_patterns
                }
        
        # Compile exclusion patterns
        self.compiled_exclusions = {}
        for category, patterns in self.exclusion_patterns.items():
            compiled_patterns = []
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Failed to compile exclusion pattern {pattern}: {e}")
            self.compiled_exclusions[category] = compiled_patterns
    
    def is_excluded_context(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text appears to be in an excluded context - optimized for Cloud Run"""
        # Removed LRU cache for Cloud Run compatibility (stateless design)
        # The compiled regex patterns are already optimized for performance
        for category, patterns in self.compiled_exclusions.items():
            for pattern in patterns:
                if pattern.search(text):
                    return True, category
        return False, None
    
    def detect_weighted_patterns(self, text: str, category: str) -> DetectionResult:
        """Enhanced pattern detection with severity weights and confidence scoring"""
        start_time = time.time()
        
        if category not in self.compiled_patterns:
            # Fallback for backward compatibility
            return DetectionResult(
                score=0.0, confidence=0.0, severity_level=None, 
                matches=[], excluded_context=None, processing_time_ms=0.0
            )
        
        pattern_dict = self.compiled_patterns[category]
        
        max_score = 0.0
        matches = []
        severity_level = None
        
        # Enhanced context-aware exclusion checking
        is_excluded, excluded_context = self.is_excluded_context(text)
        
        # Apply different reduction factors based on context and category
        if is_excluded:
            if category == 'sexual_patterns' and excluded_context == 'medical_educational':
                # More aggressive reduction for sexual content in educational context
                reduction_factor = 0.15
            elif excluded_context in ['constructive_criticism', 'motivational_context']:
                # Strong reduction for criticism and motivational contexts
                reduction_factor = 0.1
            elif excluded_context in ['media_fiction', 'historical_news']:
                # Moderate reduction for media and news contexts
                reduction_factor = 0.4
            else:
                # Default reduction
                reduction_factor = 0.3
        else:
            reduction_factor = 1.0
        
        # Score each severity level
        total_matches = 0
        weighted_score = 0.0
        
        for severity, config in pattern_dict.items():
            weight = config['weight']
            patterns = config['patterns']
            severity_matches = []
            
            for pattern in patterns:
                found_matches = pattern.findall(text)
                if found_matches:
                    severity_matches.extend(found_matches)
                    total_matches += len(found_matches)
            
            if severity_matches:
                # Calculate score for this severity level with diminishing returns
                base_score = weight
                bonus_score = min((len(severity_matches) - 1) * 0.05, 0.2)
                severity_score = min(base_score + bonus_score, 1.0)
                
                if severity_score > weighted_score:
                    weighted_score = severity_score
                    severity_level = severity
                
                matches.append({
                    'severity': severity,
                    'weight': weight,
                    'matches': severity_matches,
                    'count': len(severity_matches),
                    'score': severity_score
                })
        
        # Apply context reduction
        final_score = weighted_score * reduction_factor
        
        # Calculate confidence based on multiple factors
        if total_matches > 0:
            # Base confidence from score
            confidence = final_score * 0.6
            # Boost from multiple matches
            match_boost = min(total_matches * 0.05, 0.3)
            # Boost from severity consistency
            severity_boost = 0.1 if severity_level in ['critical', 'severe', 'high'] else 0.05
            confidence = min(confidence + match_boost + severity_boost, 1.0)
        else:
            confidence = 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            score=final_score,
            confidence=confidence,
            severity_level=severity_level,
            matches=matches,
            excluded_context=excluded_context,
            processing_time_ms=processing_time
        )
    
    def detect_sexual_content(self, text: str) -> DetectionResult:
        return self.detect_weighted_patterns(text, 'sexual_patterns')
    
    def detect_terrorism(self, text: str) -> DetectionResult:
        return self.detect_weighted_patterns(text, 'terrorism_patterns')
    
    def detect_violence(self, text: str) -> DetectionResult:
        return self.detect_weighted_patterns(text, 'violence_patterns')
    
    def detect_self_harm(self, text: str) -> DetectionResult:
        return self.detect_weighted_patterns(text, 'self_harm_patterns')
    
    def detect_hate_speech(self, text: str) -> DetectionResult:
        return self.detect_weighted_patterns(text, 'hate_speech_patterns')

# Enhanced Toxic Classification with Singleton Pattern
class ToxicClassification:
    """Enhanced toxic classification with model caching and better error handling"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, device=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self, device=None):
        if not self.initialized:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.tokenizer, self.class_names = self.load_model_and_tokenizer()
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def load_model_and_tokenizer(self):
        model_cache_dir = "/app/model_cache"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cache_dir, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_cache_dir, local_files_only=True
        )
        # The class names are the model's config id2label values
        config = model.config
        class_names = [config.id2label[i] for i in range(len(config.id2label))]
        return model, tokenizer, class_names

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        """Thread-safe model prediction with error handling"""
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            out = self.model(**inputs).logits
            scores = torch.sigmoid(out).cpu().detach().numpy()
            results = {}
            for i, cla in enumerate(self.class_names):
                results[cla] = float(scores[0][i])
            return results
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {name: 0.0 for name in self.class_names}

def combine_ml_and_pattern_scores(ml_score: float, pattern_result: DetectionResult, 
                                 method: str = "weighted_max") -> Tuple[float, Dict]:
    """Enhanced score combination with context-aware balancing"""
    pattern_score = pattern_result.score
    
    if method == "weighted_max":
        # Dynamic weight assignment based on context and score agreement
        base_ml_weight = 0.6  # Give ML slightly more weight by default
        base_pattern_weight = 0.4
        
        # Adjust weights based on exclusion context
        if pattern_result.excluded_context:
            # In excluded contexts, trust ML more and reduce pattern influence
            base_ml_weight = 0.8
            base_pattern_weight = 0.2
        
        # Adjust weights based on score agreement
        score_diff = abs(ml_score - pattern_score)
        if score_diff > 0.5:
            # Large disagreement - be more conservative, trust ML more
            ml_weight = 0.75
            pattern_weight = 0.25
        else:
            # Good agreement - use base weights
            ml_weight = base_ml_weight
            pattern_weight = base_pattern_weight
        
        # Base combination with dynamic weights
        combined_score = (ml_score * ml_weight + pattern_score * pattern_weight)
        
        # Enhanced severity-based adjustments for better score distribution
        severity_adjustment = 0.0
        if pattern_result.severity_level in ['extreme', 'imminent']:
            # Maximum boost for extreme patterns - less ML dependency
            if ml_score > 0.15:
                severity_adjustment = 0.25
            else:
                severity_adjustment = 0.15  # Still boost even with lower ML scores
        elif pattern_result.severity_level in ['critical', 'explicit', 'severe']:
            # Strong boost for critical patterns
            if ml_score > 0.25:
                severity_adjustment = 0.20
            elif ml_score > 0.15:
                severity_adjustment = 0.12
        elif pattern_result.severity_level in ['high']:
            if ml_score > 0.2:
                severity_adjustment = 0.10
            elif ml_score > 0.1:
                severity_adjustment = 0.05
        elif pattern_result.severity_level in ['educational', 'mild']:
            # Apply small penalty for educational/mild patterns to prevent false positives
            severity_adjustment = -0.05
        
        # Confidence-based adjustment (more conservative)
        confidence_adjustment = pattern_result.confidence * 0.05  # Reduced from 0.1
        
        # Agreement boosting (only when both are moderately high)
        agreement_boost = 0.0
        if ml_score > 0.3 and pattern_score > 0.3:
            agreement_boost = min((ml_score + pattern_score) * 0.05, 0.1)  # Reduced
        
        # Apply all adjustments
        combined_score = combined_score + severity_adjustment + confidence_adjustment + agreement_boost
        combined_score = max(0.0, min(combined_score, 1.0))  # Ensure 0-1 range
        
    elif method == "confidence_weighted":
        # Weight by pattern confidence with full range
        confidence = pattern_result.confidence
        ml_weight = 1.0 - confidence * 0.6
        pattern_weight = confidence * 0.6
        combined_score = (ml_score * ml_weight + pattern_score * pattern_weight) / (ml_weight + pattern_weight)
        
        # Boost for high confidence patterns
        if confidence > 0.9:
            combined_score = min(combined_score * 1.15, 1.0)
            
    elif method == "adaptive_boost":
        # Adaptive boosting based on severity and confidence
        base_score = max(ml_score, pattern_score)
        
        # Boost for high severity patterns
        severity_boost = 0.0
        if pattern_result.severity_level in ['critical', 'explicit', 'severe']:
            severity_boost = 0.25
        elif pattern_result.severity_level in ['high']:
            severity_boost = 0.15
            
        # Boost for high confidence
        confidence_boost = pattern_result.confidence * 0.15
        
        combined_score = min(base_score + severity_boost + confidence_boost, 1.0)
        
    else:
        # Fallback to enhanced max with boost
        combined_score = max(ml_score, pattern_score)
        if pattern_score > 0.7:
            combined_score = min(combined_score * 1.2, 1.0)
    
    details = {
        "ml_score": ml_score,
        "pattern_score": pattern_score,
        "pattern_confidence": pattern_result.confidence,
        "pattern_severity": pattern_result.severity_level,
        "combination_method": method,
        "final_score": combined_score,
        "excluded_context": pattern_result.excluded_context,
        "pattern_matches": len(pattern_result.matches),
        "ml_weight": ml_weight if method == "weighted_max" else "N/A",
        "pattern_weight": pattern_weight if method == "weighted_max" else "N/A",
        "severity_adjustment": severity_adjustment if method == "weighted_max" else "N/A",
        "confidence_adjustment": confidence_adjustment if method == "weighted_max" else "N/A",
        "agreement_boost": agreement_boost if method == "weighted_max" else "N/A"
    }
    
    return combined_score, details

def map_predict_result(result: dict) -> dict:
    """
    Maps model output keys to expected API keys and logs the keys for debugging.
    """
    logger.info(f"Predict result keys: {list(result.keys())}")  # Debug log
    key_map = {
        "toxic": "toxicity",
        "severe_toxic": "severe_toxicity",
        "identity_hate": "identity_attack"
    }
    mapped_result = {}
    for k, v in result.items():
        mapped_key = key_map.get(k, k)
        mapped_result[mapped_key] = v
    return mapped_result

# Initialize enhanced content moderation patterns
content_patterns = ContentModerationPatterns()

# Initialize global classifier (will be singleton)
global_classifier = None

def load_model_in_background():
    global model_ready, model_loading, model_load_start_time, global_classifier
    model_loading = True
    model_load_start_time = time.time()
    startup_time = time.time()
    try:
        global_classifier = ToxicClassification(device=None)
        global_classifier.predict("test")
        model_ready = True
        logger.info("Enhanced model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
    finally:
        model_loading = False

@app.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = time.time()
    
    try:
        logger.info("BERT Detoxify service starting up...")
        
        # Start model loading in background
        thread = threading.Thread(target=load_model_in_background)
        thread.daemon = True
        thread.start()
        
        # Start background ping thread if enabled
        if os.getenv("ENABLE_PING", "false").lower() == "true":
            threading.Thread(target=background_ping, daemon=True).start()
            logger.info("Background ping service started")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# ------------------- Background Pinger -------------------
def background_ping():
    """
    Background service to ping the toxicity endpoint periodically.
    Uses configurable environment variables for flexibility.
    """
    # Get configuration from environment variables
    ping_url = os.getenv("PING_URL")
    ping_interval = int(os.getenv("PING_INTERVAL_SECONDS", "300"))  # Default: 5 minutes
    api_key = os.getenv("PING_API_KEY")  # Optional API key for gateway
    
    if not ping_url:
        logger.warning("PING_URL not configured, skipping ping service")
        return
    
    # Default payload for toxicity detection
    payload = {
        "text": os.getenv("PING_TEXT", "You are stupid and I hate you")
    }
    
    # Set up headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    logger.info(f"Starting ping service: URL={ping_url}, interval={ping_interval}s")
    
    while True:
        try:
            logger.info(f"Pinging endpoint: {ping_url}")
            response = requests.post(
                ping_url, 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Ping successful: {response.status_code}")
            else:
                logger.warning(f"Ping returned non-200 status: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            
        time.sleep(ping_interval)

@app.middleware("http")
async def check_model_readiness(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
        
    global model_ready, model_loading, model_load_start_time
    if not model_ready:
        if model_loading:
            if model_load_start_time and (time.time() - model_load_start_time) > 300:
                return Response(
                    content="Model loading timeout",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            return Response(
                content="Model is still loading, please try again later",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "10"}
            )
        else:
            return Response(
                content="Model failed to load, service is unavailable",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    return await call_next(request)

@app.get("/health")
def health_check():
    """Enhanced health check with component status"""
    status_info = {
        "status": "ok" if model_ready else "unavailable",
        "model_ready": model_ready,
        "enhanced_patterns": content_patterns is not None,
        "global_classifier": global_classifier is not None and global_classifier.initialized if global_classifier else False,
        "version": "2.0.0"
    }
    
    if model_loading:
        status_info["status"] = "initializing"
    elif not model_ready and not model_loading:
        status_info["status"] = "error"
        status_info["message"] = "Model failed to load"
    
    return status_info

@app.post("/detect/toxicity", response_model=MetricReturnModel)
async def compute_toxicity(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        result = global_classifier.predict(text)
        mapped_result = map_predict_result(result)
        toxicity_score = mapped_result["toxicity"]
        severe_toxicity_score = mapped_result["severe_toxicity"]
        obscene_score = mapped_result["obscene"]
        identity_attack_score = mapped_result["identity_attack"]
        insult_score = mapped_result["insult"]
        threat_score = mapped_result["threat"]
        overall_score = max(toxicity_score, severe_toxicity_score, obscene_score, 
                           identity_attack_score, insult_score, threat_score)
        return MetricReturnModel(
            metric_name=EvaluationType.TOXICITY_EVALUATION,
            actual_value=float(overall_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "detailed_scores": {
                    "toxicity": float(toxicity_score),
                    "severe_toxicity": float(severe_toxicity_score),
                    "obscene": float(obscene_score),
                    "identity_attack": float(identity_attack_score),
                    "insult": float(insult_score),
                    "threat": float(threat_score)
                },
                "detection_method": "ml_enhanced",
                "confidence": min(0.5 + abs(overall_score - 0.5), 1.0)
            }
        )
    except Exception as e:
        logger.error(f"Toxicity detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/hate-speech", response_model=MetricReturnModel)
async def compute_hate_speech(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        ml_result = global_classifier.predict(text)
        mapped_result = map_predict_result(ml_result)
        ml_score = max(
            mapped_result["identity_attack"],
            mapped_result["toxicity"],
            mapped_result["severe_toxicity"]
        )
        pattern_result = content_patterns.detect_hate_speech(text)
        combined_score, combination_details = combine_ml_and_pattern_scores(
            ml_score, pattern_result, "weighted_max"
        )
        return MetricReturnModel(
            metric_name=EvaluationType.HATE_SPEECH_EVALUATION,
            actual_value=float(combined_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "pattern_processing_time_ms": pattern_result.processing_time_ms,
                "detection_method": "hybrid_ml_patterns",
                "confidence": pattern_result.confidence,
                **combination_details
            }
        )
    except Exception as e:
        logger.error(f"Hate speech detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/sexual-content", response_model=MetricReturnModel)
async def compute_sexual_content(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        ml_result = global_classifier.predict(text)
        mapped_result = map_predict_result(ml_result)
        ml_score = mapped_result["obscene"]
        pattern_result = content_patterns.detect_sexual_content(text)
        combined_score, combination_details = combine_ml_and_pattern_scores(
            ml_score, pattern_result, "weighted_max"
        )
        return MetricReturnModel(
            metric_name=EvaluationType.SEXUAL_CONTENT_EVALUATION,
            actual_value=float(combined_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "pattern_processing_time_ms": pattern_result.processing_time_ms,
                "detection_method": "hybrid_ml_patterns",
                "confidence": pattern_result.confidence,
                **combination_details
            }
        )
    except Exception as e:
        logger.error(f"Sexual content detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/terrorism", response_model=MetricReturnModel)
async def compute_terrorism(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        ml_result = global_classifier.predict(text)
        mapped_result = map_predict_result(ml_result)
        ml_score = max(mapped_result["threat"], mapped_result["severe_toxicity"])
        pattern_result = content_patterns.detect_terrorism(text)
        combined_score, combination_details = combine_ml_and_pattern_scores(
            ml_score, pattern_result, "weighted_max"
        )
        return MetricReturnModel(
            metric_name=EvaluationType.TERRORISM_EVALUATION,
            actual_value=float(combined_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "pattern_processing_time_ms": pattern_result.processing_time_ms,
                "detection_method": "hybrid_ml_patterns",
                "confidence": pattern_result.confidence,
                **combination_details
            }
        )
    except Exception as e:
        logger.error(f"Terrorism detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/violence", response_model=MetricReturnModel)
async def compute_violence(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        ml_result = global_classifier.predict(text)
        mapped_result = map_predict_result(ml_result)
        ml_score = max(mapped_result["threat"], mapped_result["severe_toxicity"])
        pattern_result = content_patterns.detect_violence(text)
        combined_score, combination_details = combine_ml_and_pattern_scores(
            ml_score, pattern_result, "weighted_max"
        )
        return MetricReturnModel(
            metric_name=EvaluationType.VIOLENCE_EVALUATION,
            actual_value=float(combined_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "pattern_processing_time_ms": pattern_result.processing_time_ms,
                "detection_method": "hybrid_ml_patterns",
                "confidence": pattern_result.confidence,
                **combination_details
            }
        )
    except Exception as e:
        logger.error(f"Violence detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/detect/self-harm", response_model=MetricReturnModel)
async def compute_self_harm(req: TextRequest):
    t0 = time.time()
    try:
        text = req.text
        ml_result = global_classifier.predict(text)
        mapped_result = map_predict_result(ml_result)
        ml_score = mapped_result["threat"]
        pattern_result = content_patterns.detect_self_harm(text)
        if pattern_result.severity_level == "critical":
            combined_score = min(pattern_result.score * 1.2, 1.0)
            combination_details = {
                "ml_score": ml_score,
                "pattern_score": pattern_result.score,
                "pattern_confidence": pattern_result.confidence,
                "pattern_severity": pattern_result.severity_level,
                "combination_method": "critical_boost",
                "final_score": combined_score,
                "excluded_context": pattern_result.excluded_context,
                "pattern_matches": len(pattern_result.matches),
                "critical_alert": True
            }
        else:
            combined_score, combination_details = combine_ml_and_pattern_scores(
                ml_score, pattern_result, "weighted_max"
            )
            combination_details["critical_alert"] = False
        return MetricReturnModel(
            metric_name=EvaluationType.SELF_HARM_EVALUATION,
            actual_value=float(combined_score),
            actual_value_type=ActualValueDtype.FLOAT,
            others={
                "processing_time_ms": float(round((time.time() - t0) * 1000, 2)),
                "pattern_processing_time_ms": pattern_result.processing_time_ms,
                "detection_method": "hybrid_ml_patterns",
                "confidence": pattern_result.confidence,
                **combination_details
            }
        )
    except Exception as e:
        logger.error(f"Self-harm detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 