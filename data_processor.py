import re
import numpy as np
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def analyze_sentiment(text):
    """Analyze sentiment of the text using TextBlob."""
    if not text:
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'Neutral',
            'emotions': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        }
    
    blob = TextBlob(text)
    
    # Get polarity and subjectivity
    polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
    subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
    
    # Determine sentiment label
    if polarity > 0.1:
        sentiment_label = 'Positive'
    elif polarity < -0.1:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    # Calculate emotion scores
    positive_score = max(0, polarity)
    negative_score = max(0, -polarity)
    neutral_score = 1 - abs(polarity)
    
    # Normalize emotion scores
    total = positive_score + negative_score + neutral_score
    if total > 0:
        emotions = {
            'positive': positive_score / total,
            'negative': negative_score / total,
            'neutral': neutral_score / total
        }
    else:
        emotions = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment_label': sentiment_label,
        'emotions': emotions
    }

def analyze_bias(text):
    """Analyze political and emotional bias in the text."""
    if not text:
        return {
            'political_bias': 0.0,
            'emotional_intensity': 0.0,
            'bias_indicators': [],
            'tone': 'Neutral'
        }
    
    text_lower = text.lower()
    
    # Political bias indicators
    left_indicators = [
        'progressive', 'liberal', 'climate change', 'social justice',
        'equality', 'diversity', 'inclusion', 'regulation', 'government program'
    ]
    
    right_indicators = [
        'conservative', 'traditional values', 'free market', 'deregulation',
        'law and order', 'national security', 'fiscal responsibility', 'individual rights'
    ]
    
    # Count political indicators
    left_count = sum(1 for indicator in left_indicators if indicator in text_lower)
    right_count = sum(1 for indicator in right_indicators if indicator in text_lower)
    
    # Calculate political bias score (-1 to 1, where -1 is left, 1 is right)
    total_political = left_count + right_count
    if total_political > 0:
        political_bias = (right_count - left_count) / total_political
    else:
        political_bias = 0.0
    
    # Emotional intensity indicators
    high_emotion_words = [
        'outrageous', 'shocking', 'devastating', 'incredible', 'amazing',
        'terrible', 'wonderful', 'awful', 'fantastic', 'horrible',
        'brilliant', 'disgusting', 'magnificent', 'appalling'
    ]
    
    emotion_count = sum(1 for word in high_emotion_words if word in text_lower)
    
    # Calculate emotional intensity (0 to 1)
    word_count = len(text_lower.split())
    emotional_intensity = min(emotion_count / max(word_count, 1) * 10, 1.0)
    
    # Identify specific bias indicators found
    bias_indicators = []
    for indicator in left_indicators:
        if indicator in text_lower:
            bias_indicators.append(f"Left-leaning: {indicator}")
    
    for indicator in right_indicators:
        if indicator in text_lower:
            bias_indicators.append(f"Right-leaning: {indicator}")
    
    # Determine overall tone
    if emotional_intensity > 0.3:
        tone = 'Emotional'
    elif abs(political_bias) > 0.3:
        tone = 'Biased'
    else:
        tone = 'Neutral'
    
    return {
        'political_bias': political_bias,
        'emotional_intensity': emotional_intensity,
        'bias_indicators': bias_indicators[:5],  # Top 5 indicators
        'tone': tone
    }

def process_text(text):
    """Comprehensive text processing and analysis."""
    if not text:
        return {}
    
    # Basic text statistics
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove stopwords for analysis
    stop_words = set(stopwords.words('english'))
    content_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    # Calculate readability metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in content_words) / len(content_words) if content_words else 0
    
    # Vocabulary diversity (unique words / total words)
    vocabulary_diversity = len(set(content_words)) / len(content_words) if content_words else 0
    
    # Most frequent words
    word_freq = Counter(content_words)
    most_common_words = word_freq.most_common(10)
    
    # Text complexity indicators
    complex_words = [word for word in content_words if len(word) > 6]
    complexity_ratio = len(complex_words) / len(content_words) if content_words else 0
    
    # Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for char in text if char.isupper())
    caps_ratio = caps_count / len(text) if text else 0
    
    return {
        'basic_stats': {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'vocabulary_diversity': vocabulary_diversity
        },
        'complexity': {
            'complexity_ratio': complexity_ratio,
            'long_words_count': len(complex_words)
        },
        'style': {
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_ratio': caps_ratio
        },
        'frequent_words': most_common_words
    }

def extract_keywords(text, num_keywords=10):
    """Extract keywords from text using TF-IDF-like approach."""
    if not text:
        return []
    
    # Tokenize and clean text
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Filter words
    content_words = [
        word for word in words 
        if word.isalpha() and len(word) > 2 and word not in stop_words
    ]
    
    # Calculate word frequencies
    word_freq = Counter(content_words)
    
    # Simple keyword scoring (frequency * length)
    keyword_scores = {}
    for word, freq in word_freq.items():
        # Score based on frequency and word length
        score = freq * (len(word) ** 0.5)
        keyword_scores[word] = score
    
    # Return top keywords
    top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in top_keywords[:num_keywords]]

def detect_writing_style(text):
    """Detect writing style characteristics."""
    if not text:
        return {}
    
    text_lower = text.lower()
    
    # Formal vs informal indicators
    formal_indicators = [
        'furthermore', 'however', 'nevertheless', 'consequently',
        'therefore', 'moreover', 'additionally', 'subsequently'
    ]
    
    informal_indicators = [
        'gonna', 'wanna', 'yeah', 'ok', 'btw', 'lol',
        'omg', 'tbh', 'imo', 'fyi'
    ]
    
    formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
    informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
    
    # Academic vs journalistic style
    academic_indicators = [
        'research shows', 'studies indicate', 'according to',
        'peer-reviewed', 'methodology', 'hypothesis', 'data analysis'
    ]
    
    journalistic_indicators = [
        'breaking news', 'sources say', 'reported', 'confirmed',
        'exclusive', 'investigation reveals', 'witnesses report'
    ]
    
    academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)
    journalistic_count = sum(1 for indicator in journalistic_indicators if indicator in text_lower)
    
    # Determine dominant style
    styles = {
        'formal': formal_count,
        'informal': informal_count,
        'academic': academic_count,
        'journalistic': journalistic_count
    }
    
    dominant_style = max(styles.items(), key=lambda x: x[1])[0] if any(styles.values()) else 'neutral'
    
    return {
        'dominant_style': dominant_style,
        'style_scores': styles,
        'formality_ratio': formal_count / max(formal_count + informal_count, 1)
    }

def analyze_claims_density(text):
    """Analyze the density of factual claims in the text."""
    if not text:
        return {}
    
    sentences = sent_tokenize(text)
    
    # Claim indicators
    claim_patterns = [
        r'\d+\s*%',  # Percentages
        r'\d+\s*(million|billion|thousand)',  # Large numbers
        r'according to',  # Attribution
        r'study shows',  # Research claims
        r'data reveals',  # Data claims
        r'reports indicate',  # Report claims
    ]
    
    claim_sentences = 0
    total_claims = 0
    
    for sentence in sentences:
        sentence_claims = 0
        for pattern in claim_patterns:
            matches = len(re.findall(pattern, sentence.lower()))
            sentence_claims += matches
            total_claims += matches
        
        if sentence_claims > 0:
            claim_sentences += 1
    
    claim_density = claim_sentences / len(sentences) if sentences else 0
    claims_per_sentence = total_claims / len(sentences) if sentences else 0
    
    return {
        'claim_density': claim_density,
        'claims_per_sentence': claims_per_sentence,
        'total_claims': total_claims,
        'claim_sentences': claim_sentences
    }
