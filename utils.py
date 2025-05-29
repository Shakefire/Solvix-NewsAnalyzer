import re
import random
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_claims(text):
    """Extract factual claims from news text."""
    if not text:
        return []
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    claims = []
    claim_indicators = [
        'according to', 'reported that', 'confirmed that', 'announced that',
        'stated that', 'revealed that', 'disclosed that', 'said that',
        'claims that', 'alleges that', 'suggests that', 'indicates that'
    ]
    
    # Patterns that indicate factual claims
    fact_patterns = [
        r'\d+\s+(percent|%|people|deaths|cases|dollars)',  # Statistics
        r'(increase|decrease|rise|fall|drop).*\d+',  # Numerical changes
        r'(will|would|plans to|expected to)',  # Future claims
        r'(happened|occurred|took place).*on',  # Event claims
    ]
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        
        # Check for claim indicators
        has_indicator = any(indicator in sentence_lower for indicator in claim_indicators)
        
        # Check for fact patterns
        has_fact_pattern = any(re.search(pattern, sentence_lower) for pattern in fact_patterns)
        
        # Check for entities (proper nouns)
        words = word_tokenize(sentence)
        proper_nouns = [word for word in words if word[0].isupper() and len(word) > 1]
        has_entities = len(proper_nouns) >= 2
        
        # Determine if this sentence contains a claim
        if has_indicator or has_fact_pattern or has_entities:
            # Calculate confidence based on multiple factors
            confidence = 0.3  # Base confidence
            
            if has_indicator:
                confidence += 0.3
            if has_fact_pattern:
                confidence += 0.2
            if has_entities:
                confidence += 0.2
            
            # Categorize the claim
            category = categorize_claim(sentence)
            
            claims.append({
                'claim': sentence.strip(),
                'confidence': min(confidence, 1.0),
                'category': category,
                'position': i
            })
    
    # Sort by confidence and return top claims
    claims.sort(key=lambda x: x['confidence'], reverse=True)
    return claims[:10]  # Return top 10 claims

def categorize_claim(sentence):
    """Categorize a claim into different types."""
    sentence_lower = sentence.lower()
    
    # Define category keywords
    categories = {
        'Statistical': ['percent', '%', 'rate', 'number', 'statistics', 'data', 'study'],
        'Political': ['government', 'president', 'congress', 'election', 'vote', 'policy'],
        'Health': ['health', 'medical', 'vaccine', 'treatment', 'disease', 'hospital'],
        'Economic': ['economy', 'market', 'price', 'inflation', 'employment', 'gdp'],
        'Environmental': ['climate', 'environment', 'pollution', 'carbon', 'temperature'],
        'Social': ['people', 'community', 'society', 'culture', 'education', 'crime'],
        'Technology': ['technology', 'digital', 'internet', 'computer', 'ai', 'software'],
        'International': ['country', 'international', 'foreign', 'global', 'nation']
    }
    
    # Count matches for each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        if score > 0:
            category_scores[category] = score
    
    # Return the category with the highest score
    if category_scores:
        return max(category_scores.items(), key=lambda x: x[1])[0]
    else:
        return 'General'

def get_source_credibility(text):
    """Analyze source credibility based on text characteristics."""
    if not text:
        return {'trust_score': 0.5, 'factors': {}}
    
    # Initialize factors
    factors = {
        'Language Quality': 0.0,
        'Source Attribution': 0.0,
        'Emotional Tone': 0.0,
        'Factual Claims': 0.0,
        'Sensationalism': 0.0
    }
    
    text_lower = text.lower()
    
    # Language Quality - check for proper grammar and spelling
    blob = TextBlob(text)
    
    # Simple grammar check based on sentence structure
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
    
    if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
        factors['Language Quality'] += 0.3
    
    # Check for common misspellings or poor grammar indicators
    poor_grammar_indicators = ['ur', 'u', 'dont', 'cant', 'wont', 'shouldnt']
    if not any(indicator in text_lower for indicator in poor_grammar_indicators):
        factors['Language Quality'] += 0.4
    
    # Source Attribution - look for quotes and citations
    source_indicators = [
        'according to', 'said', 'stated', 'reported', 'confirmed',
        'spokesperson', 'official', 'expert', 'researcher', 'study'
    ]
    
    source_count = sum(1 for indicator in source_indicators if indicator in text_lower)
    factors['Source Attribution'] = min(source_count * 0.15, 0.8)
    
    # Emotional Tone - check for excessive emotional language
    emotional_words = [
        'shocking', 'unbelievable', 'amazing', 'terrible', 'awful',
        'incredible', 'outrageous', 'devastating', 'miraculous'
    ]
    
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    # Higher emotional language reduces credibility
    factors['Emotional Tone'] = max(0.8 - (emotional_count * 0.1), 0.0)
    
    # Factual Claims - look for specific facts and numbers
    factual_indicators = [
        r'\d+', r'percent', r'according to.*study', r'research shows',
        r'data indicates', r'statistics show'
    ]
    
    factual_count = sum(1 for pattern in factual_indicators if re.search(pattern, text_lower))
    factors['Factual Claims'] = min(factual_count * 0.1, 0.7)
    
    # Sensationalism - check for clickbait and sensational language
    sensational_indicators = [
        'you won\'t believe', 'shocking truth', 'they don\'t want you to know',
        'secret', 'exposed', 'revealed', 'must read', 'viral',
        'breaking:', 'urgent:', 'alert:'
    ]
    
    sensational_count = sum(1 for indicator in sensational_indicators if indicator in text_lower)
    # Higher sensationalism reduces credibility
    factors['Sensationalism'] = max(0.8 - (sensational_count * 0.15), 0.0)
    
    # Calculate overall trust score
    trust_score = sum(factors.values()) / len(factors)
    
    return {
        'trust_score': trust_score,
        'factors': factors
    }

def simplify_text(text):
    """Simplify complex text for better readability."""
    if not text:
        return ""
    
    # Split into sentences
    sentences = sent_tokenize(text)
    simplified_sentences = []
    
    for sentence in sentences:
        # Remove complex punctuation
        simplified = re.sub(r'[;:—–]', ',', sentence)
        
        # Replace complex words with simpler alternatives
        replacements = {
            'utilize': 'use',
            'implement': 'put in place',
            'demonstrate': 'show',
            'facilitate': 'help',
            'approximately': 'about',
            'subsequently': 'then',
            'consequently': 'so',
            'nevertheless': 'however',
            'furthermore': 'also',
            'therefore': 'so'
        }
        
        for complex_word, simple_word in replacements.items():
            simplified = re.sub(r'\b' + complex_word + r'\b', simple_word, simplified, flags=re.IGNORECASE)
        
        # Break down long sentences
        if len(word_tokenize(simplified)) > 20:
            # Simple sentence splitting at conjunctions
            conjunctions = [' and ', ' but ', ' however ', ' although ']
            for conj in conjunctions:
                if conj in simplified.lower():
                    parts = simplified.lower().split(conj, 1)
                    if len(parts) == 2:
                        simplified = parts[0].strip().capitalize() + '. ' + parts[1].strip().capitalize()
                        break
        
        simplified_sentences.append(simplified)
    
    return ' '.join(simplified_sentences)

def summarize_text(text):
    """Create a summary of the text."""
    if not text:
        return ""
    
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 3:
        return text
    
    # Score sentences based on various factors
    sentence_scores = {}
    
    # Get word frequencies
    words = word_tokenize(text.lower())
    word_freq = {}
    for word in words:
        if word.isalpha() and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score each sentence
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower())
        score = 0
        
        # Word frequency score
        for word in sentence_words:
            if word in word_freq:
                score += word_freq[word]
        
        # Position score (first and last sentences are important)
        if i == 0 or i == len(sentences) - 1:
            score *= 1.5
        
        # Length score (not too short, not too long)
        sentence_length = len(sentence_words)
        if 5 <= sentence_length <= 30:
            score *= 1.2
        
        sentence_scores[i] = score
    
    # Select top sentences (about 1/3 of original)
    num_sentences = max(2, len(sentences) // 3)
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Sort by original order
    selected_indices = sorted([idx for idx, score in top_sentences])
    
    summary_sentences = [sentences[i] for i in selected_indices]
    
    return ' '.join(summary_sentences)

def extract_entities(text):
    """Extract named entities from text."""
    if not text:
        return []
    
    # Simple entity extraction using patterns
    entities = {
        'PERSON': [],
        'ORGANIZATION': [],
        'LOCATION': [],
        'DATE': [],
        'MONEY': []
    }
    
    # Person names (capitalized words, often with titles)
    person_pattern = r'(?:Mr\.|Mrs\.|Dr\.|President|Senator|Rep\.|Gov\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    person_matches = re.findall(person_pattern, text)
    entities['PERSON'].extend(person_matches)
    
    # Organizations (often have Inc, Corp, Company, etc.)
    org_pattern = r'([A-Z][a-zA-Z\s]+(?:Inc|Corp|Company|Organization|Agency|Department))'
    org_matches = re.findall(org_pattern, text)
    entities['ORGANIZATION'].extend(org_matches)
    
    # Locations (capitalized words that might be places)
    location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s+(?:city|state|country|county|region))'
    location_matches = re.findall(location_pattern, text)
    entities['LOCATION'].extend(location_matches)
    
    # Dates
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}'
    date_matches = re.findall(date_pattern, text)
    entities['DATE'].extend(date_matches)
    
    # Money
    money_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+\s+(?:dollars?|billion|million|thousand)'
    money_matches = re.findall(money_pattern, text)
    entities['MONEY'].extend(money_matches)
    
    return entities
