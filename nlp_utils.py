import string
import numpy as np
import nltk
import time
import random
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

stemmer = PorterStemmer()
tfidf_transformer = TfidfTransformer()

def back_translate(text, lang="es"):
    try:
        time.sleep(1)  # Wait 1 second between requests
        translated = GoogleTranslator(source="auto", target=lang).translate(text)
        time.sleep(1)  # Wait before translating back
        return GoogleTranslator(source=lang, target="en").translate(translated)
    except:
        return text  # Return original if translation fails


# Synonym Replacement
def synonym_replace(sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_word = synonyms[0].lemmas()[0].name().replace("_", " ")
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)


# Word Shuffling
def shuffle_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)


# Augment Data for Each Intent
def augment_intents(intents_data):
    for intent in intents_data["intents"]:
        augmented_patterns = set(intent["patterns"])  # Use set to avoid duplicates
        for sentence in intent["patterns"]:
            augmented_patterns.add(back_translate(sentence))
            augmented_patterns.add(synonym_replace(sentence))
            augmented_patterns.add(shuffle_sentence(sentence))
        intent["patterns"] = list(augmented_patterns)  # Convert set back to list
    return intents_data



def remove_punctuation(sentences):
    """
    Removes punctuation from a list of sentences.
    
    Examples:
        >>> sentences = ["Hello, world!", "This is a test."]
        >>> remove_punctuation(sentences)
        -> ['Hello world', 'This is a test']
    """
    return [sentence.translate(str.maketrans("", "", string.punctuation)) for sentence in sentences]


def tokenize(sentences):
    """
    Tokenizes a list of sentences into individual words.
    
    Examples:
        >>> sentences = ["This is a sample sentence.", "Another example sentence."]
        >>> tokenize(sentences)
        -> [['this', 'is', 'a', 'sample', 'sentence', '.'], 
            ['another', 'example', 'sentence', '.']]
    """
    return [word_tokenize(sentence.lower()) for sentence in sentences]


def remove_stopwords(tokenized_sentences):
    """
    Removes stopwords from a list of tokenized sentences.
    
    Examples:
        >>> tokenized_sentences = [["this", "is", "a", "sample", "sentence"], 
                                    ["another", "example", "sentence"]]
        >>> remove_stopwords(tokenized_sentences)
        -> [['sample', 'sentence'], 
            ['another', 'example', 'sentence']]
    """
    stop_words = set(stopwords.words("english"))
    return [[word for word in tokens if word not in stop_words] for tokens in tokenized_sentences]


def stem(tokenized_sentences):
    """
    Stems a list of tokenized sentences to their root form.
    
    Examples:
        >>> words = ["organize", "organizes", "organizing"]
        >>> stem(words)
        -> ['organ', 'organ', 'organ']
    """
    return [[stemmer.stem(word.lower()) for word in tokens] for tokens in tokenized_sentences ]


def bag_of_words(tokenized_sentences, vocab):
    """
    Creates a Bag of Words (BoW) representation for a list of tokenized sentences.
    
    Examples:
        >>> tokenized_sentences = [["hello", "how", "are", "you"], 
                                    ["hi", "hello", "I", "am", "fine"]]
        >>> vocab = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        >>> bag_of_words(tokenized_sentences, vocab)
        -> [[0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0]]
    """
    
    # Initialize BoW matrix
    bow_matrix = np.zeros((len(tokenized_sentences), len(vocab)), dtype=np.float32)
    
    # Fill BoW matrix
    for i, sentence in enumerate(tokenized_sentences):
        for j, word in enumerate(vocab):
            bow_matrix[i, j] = sentence.count(word)
    
    return bow_matrix


def tf_idf(bow):
    """
    Applies TF-IDF transformation to a Bag of Words (BoW) matrix.
    Example:
        >>> bow = np.array([[0, 2, 0, 1, 0],
                            [1, 1, 1, 0, 0]])
        >>> tf_idf(bow)
        -> <2x5 sparse matrix of type '<class 'numpy.float64'>'
            with 4 stored elements in Compressed Sparse Row format>
    """
    X_tfidf = tfidf_transformer.fit_transform(bow)
    return X_tfidf


glove_model = api.load("glove-wiki-gigaword-200")  # Load 100D GloVe embeddings
def get_embedding(sentences):
    """
    Accepts a list of sentences and returns a list of their corresponding embeddings.
    
    Args:
        sentences (list): A list of strings (sentences).
    
    Returns:
        list: A list of numpy arrays, where each array is the embedding for a sentence.
    """
    embeddings = []
    for sentence in sentences:
        words = sentence.split()  # Tokenize sentence into words
        vectors = [glove_model[word] for word in words if word in glove_model]  # Get word vectors
        if vectors:  # If at least one word has an embedding
            embedding = np.mean(vectors, axis=0)  # Average word embeddings
        else:  # If no words have embeddings
            embedding = np.zeros(200)  # Return a zero vector
        embeddings.append(embedding)  # Add to the list of embeddings
    return embeddings


# def train_word2vec(tokenized_sentences, vector_size=200):
#     """Train Word2Vec model on tokenized sentences."""
#     model = Word2Vec(tokenized_sentences, vector_size=vector_size, min_count=1, workers=4)
#     return model


# def word2vec_embeddings(tokenized_sentences, model):
#     """
#     Converts a list of tokenized sentences into Word2Vec embeddings by averaging the word vectors.
#     Examples:
#         >>> from gensim.models import Word2Vec
#         >>> sentences = [["hello", "world"], ["how", "are", "you"], ["unknown", "word"]]
#         >>> model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
#         >>> embeddings = word2vec_embeddings(sentences, model)
#         >>> embeddings.shape
#         -> (3, 100)  # 3 sentences, each with a 100-dimensional embedding
#     """
#     embeddings = []
#     for sentence in tokenized_sentences:
#         vectors = [model.wv[word] for word in sentence if word in model.wv]
#         if vectors:
#             embeddings.append(np.mean(vectors, axis=0))  # Average word vectors
#         else:
#             embeddings.append(np.zeros(model.vector_size))  # Zero vector for empty sentences
#     return np.array(embeddings)


def join_tokens(token_lists):
    """Joins tokenized words back into a string."""
    return [" ".join(tokens) for tokens in token_lists]



def create_pipeline():
    pipeline = Pipeline([
        ("remove_punc", FunctionTransformer(remove_punctuation, validate=False)),
        ("tokenize", FunctionTransformer(tokenize, validate=False)),
        # ("remove_stopwords", FunctionTransformer(remove_stopwords, validate=False)),
        ("semmatize", FunctionTransformer(stem, validate=False)),
        ("join_tokens", FunctionTransformer(join_tokens, validate=False)),
        ("embedding", FunctionTransformer(get_embedding, validate=False))
    ])
    return pipeline

