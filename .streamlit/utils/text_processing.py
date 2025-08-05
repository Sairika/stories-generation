import re

def clean_text(text):
    """Clean and format generated text"""
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    
    # Fix contractions
    contractions = {
        " n't": "n't",
        " 's": "'s",
        " 're": "'re",
        " 've": "'ve",
        " 'll": "'ll",
        " 'm": "'m",
        " 'd": "'d"
    }
    
    for wrong, right in contractions.items():
        text = text.replace(wrong, right)
    
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?]+)', text)
    cleaned_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i % 2 == 0 and sentence.strip():  # Actual sentence content
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)
        else:
            cleaned_sentences.append(sentence)
    
    return ''.join(cleaned_sentences).strip()

def truncate_text(text, max_words=300):
    """Truncate text to maximum number of words"""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text
