import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the robot assistant bot v2.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer_v2.pkl',
    'TAG_CLASSIFIER': 'tag_classifier_v2.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer_v2.pkl',
    'ACTION_PLANS': 'action_plans.json',
}

# Domain-specific words that should NOT be removed as stopwords,
# even though NLTK considers them stopwords.
# These words carry important meaning in carpentry commands.
DOMAIN_KEEP_WORDS = {
    'down', 'up', 'off', 'out', 'over', 'through', 'between',
    'together', 'apart', 'above', 'below', 'under',
    'back', 'forward', 'here', 'there', 'where',
    'not', 'no', 'more', 'all',
}


def text_prepare(text):
    """Performs tokenization and preprocessing with domain-aware stopword filtering.

    Improvement over v1: keeps directional and spatial words that are
    important for distinguishing robot commands from chitchat.
    """
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')

    # Build a filtered stopword set that keeps domain-relevant words
    stopwords_set = set(stopwords.words('english')) - DOMAIN_KEEP_WORDS

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
