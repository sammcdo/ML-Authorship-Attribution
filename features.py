import re
import nltk
import spacy
from nltk.corpus import cmudict, stopwords
import numpy as np
from collections import Counter



nltk.download('cmudict')
cmu = cmudict.dict()
nltk.download('stopwords')
stopword_set = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm", disable=["tagger","parser","ner","lemmatizer"])
nlp.add_pipe("sentencizer")

tag = spacy.load("en_core_web_sm", disable=["parser", "ner"])
tag.add_pipe("sentencizer")

with open("./data/dale-chall.txt", 'r') as f:
    dale_chall_easy = f.readlines()

def _split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def _syllables(word):
    """Return syllable count using CMUdict, or vowels if word not found."""
    word = word.lower()
    if word in cmu:
        # Each vowel phoneme ends with a digit (stress marker)
        count = min(len([ph for ph in pron if ph[-1].isdigit()]) 
                   for pron in cmu[word])
        return max(1,count)
    
    # Basic vowel-group heuristic
    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups)

    # Adjust silent e
    if word.endswith("e"):
        count -= 1
    return max(1,count)

def std_word_length(words):
    lengths = np.array([len(w) for w in words])
    return lengths.std()

def average_syllables(words):
    total_syllables = sum(_syllables(w) for w in words)
    return total_syllables / len(words)

def std_syllables(words):
    sylls = np.array([_syllables(w) for w in words])
    return sylls.std()

def type_token_ratio(words):
    types = len(set(words))
    tokens = len(words)
    return types / tokens

def vocabulary_size(words):
    return len(set(words))

def stopword_ratio(words):
    if not words:
        return 0.0
    
    stop_count = sum(1 for w in words if w.lower() in stopword_set)
    return stop_count / len(words)

def hapax_legomena(words):
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 1) / len(words)

def hapax_dislegomena(words):
    counts = Counter(words)
    return sum(1 for w, c in counts.items() if c == 2) / len(words)

def average_syllables_of_vocabulary(words):
    words = list(set(words))
    return average_syllables(words)

def average_words_per_sentence(sentences):
    if not sentences:
        return 0.0

    total_words = sum(len([t for t in sent if t.isalpha()]) for sent in sentences)
    return total_words / len(sentences)

def std_sentence_length(sentences):
    sent = np.array([len([t for t in sent if t.isalpha()]) for sent in sentences])
    return sent.std()

def flesch_reading_ease(sentences, words):
    num_sent = len(sentences)
    num_words = len(words)

    num_syll = sum(_syllables(w) for w in words)
    
    fre = 206.835 \
          - 1.015 * (num_words / num_sent) \
          - 84.6 * (num_syll / num_words)

    return fre

def dale_chall_reading_ease(words, num_sentences):
    num_words = len(words)
    if num_words == 0 or num_sentences == 0:
        return 0.0

    difficult_words = sum(1 for w in words if w.lower() not in dale_chall_easy)

    difficult_word_percent = (difficult_words / num_words) * 100

    score = 0.1579 * difficult_word_percent + 0.0496 * (num_words / num_sentences)

    if difficult_word_percent > 5:
        score += 3.6365

    return score

def gunning_fog_index(words, num_sentences):
    num_words = len(words)
    if num_words == 0 or num_sentences == 0:
        return 0.0

    # Count complex words (3 or more syllables)
    complex_words = sum(1 for w in words if _syllables(w) >= 3)

    # Gunning Fog Index formula
    fog_index = 0.4 * ((num_words / num_sentences) + (complex_words / num_words) * 100)

    return fog_index

def average_word_len(words):
    if not words:
        return 0.0

    total_chars = sum(len(w) for w in words)
    return total_chars / len(words)

def pos_counts(words):
    # Initialize counts
    pos_categories = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "INTJ", "NUM", "PART", "PROPN", "SYM", "X"]
    counts = dict.fromkeys(pos_categories, 0)

    if not words:
        return counts

    # Process words with SpaCy
    doc = tag(" ".join(words))

    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1

    return counts

def punc_counts(text):
    cats = ["period", "exclam", "quest"]
    symb = ['.', '!', '?']

    counts = dict.fromkeys(cats, 0)

    for i in range(len(cats)):
        counts[cats[i]] = text.count(symb[i])
    
    return counts

COORD_CONJ = {"and", "but", "or", "nor", "yet", "so"}
SUBORD = {
    "because", "although", "though", "since", "unless", "if",
    "when", "while", "after", "before", "once",
    "that", "who", "whom", "whose", "which", "where", "why"
}

def sentence_type_counts(sentences):
    counts = {
        "simple": 0,
        "compound": 0,
        "complex": 0,
        "compound_complex": 0
    }

    if not sentences:
        return counts

    doc = tag(" ".join(sentences))
    sents = list(doc.sents)
    for sent in sents:
        tokens = [t.text.lower() for t in sent]
        pos = [t.pos_ for t in sent]

        # Count verbs
        verb_positions = [i for i, p in enumerate(pos) if p in ("VERB", "AUX")]

        # ----- Complex Sentence Detection -----
        has_subord = any(word in SUBORD for word in tokens)

        # ----- Compound Sentence Detection (verb ... conj ... verb) -----
        has_coord = False
        for i, t in enumerate(tokens):
            if t in COORD_CONJ:
                # check for verbs on both sides
                left_has_verb = any(v < i for v in verb_positions)
                right_has_verb = any(v > i for v in verb_positions)
                if left_has_verb and right_has_verb:
                    has_coord = True
                    break

        # ----- Final classification -----
        if has_coord and has_subord:
            counts["compound_complex"] += 1
        elif has_coord:
            counts["compound"] += 1
        elif has_subord:
            counts["complex"] += 1
        else:
            counts["simple"] += 1

    return counts

