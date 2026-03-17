"""
embeddings.py — Generate vector embeddings from text
=====================================================
This module uses Sentence Transformers to convert text into numerical
vectors (embeddings). These vectors capture the *meaning* of the text,
so similar text produces similar vectors.

Model used: all-MiniLM-L6-v2
    - Dimension: 384 (each text becomes a list of 384 numbers)
    - Fast and lightweight (good for beginners and prototyping)
    - Trained on 1 billion+ text pairs for semantic similarity

How it works:
    1. Load a pre-trained model (downloaded automatically on first use)
    2. Pass text to the model
    3. Get back a 384-dimensional vector
    4. This vector can be stored in Endee and compared with other vectors
"""

from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Model name — this determines the embedding quality and dimension
# all-MiniLM-L6-v2: 384 dimensions, fast, good quality
MODEL_NAME = "all-MiniLM-L6-v2"

# The dimension of the embeddings (must match what we tell Endee)
EMBEDDING_DIMENSION = 384

# Maximum text length the model can handle well
# Longer texts are truncated automatically, but we chunk smartly
MAX_TEXT_LENGTH = 8000


def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the sentence transformer model.

    The model is downloaded automatically the first time you call this.
    Subsequent calls load from cache (~/.cache/torch/sentence_transformers/).

    Returns:
        A SentenceTransformer model instance.

    Example:
        >>> model = get_embedding_model()
        >>> print(type(model))
        <class 'sentence_transformers.SentenceTransformer.SentenceTransformer'>
    """
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded! Embedding dimension: {EMBEDDING_DIMENSION}")
    return model


def generate_embedding(model: SentenceTransformer, text: str) -> list:
    """
    Convert text into a vector embedding.

    This is the core function that turns resume text into numbers
    that Endee can store and search.

    Args:
        model: The loaded SentenceTransformer model.
        text:  The text to convert (e.g., resume content).

    Returns:
        A list of floats (length 384) representing the text's meaning.

    Example:
        >>> model = get_embedding_model()
        >>> vec = generate_embedding(model, "Python developer with 5 years experience")
        >>> print(len(vec))     # 384
        >>> print(type(vec[0])) # <class 'float'>
    """
    # Truncate very long text to avoid memory issues
    # The model handles truncation internally, but we trim first
    # to keep things predictable
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    # Generate the embedding
    # model.encode() returns a numpy array, we convert to a Python list
    # because Endee expects a regular list of floats
    embedding = model.encode(text)

    # Convert numpy array to a plain Python list of floats
    embedding_list = embedding.tolist()

    return embedding_list


def generate_embeddings_batch(model: SentenceTransformer, texts: list) -> list:
    """
    Convert multiple texts into embeddings at once (faster than one-by-one).

    Args:
        model: The loaded SentenceTransformer model.
        texts: A list of text strings.

    Returns:
        A list of embedding lists.

    Example:
        >>> model = get_embedding_model()
        >>> texts = ["Python developer", "Java engineer", "Data scientist"]
        >>> vecs = generate_embeddings_batch(model, texts)
        >>> print(len(vecs))    # 3
        >>> print(len(vecs[0])) # 384
    """
    # Truncate each text
    truncated = [t[:MAX_TEXT_LENGTH] for t in texts]

    # Batch encoding is much faster than encoding one at a time
    embeddings = model.encode(truncated, show_progress_bar=True)

    # Convert each numpy array to a list
    return [emb.tolist() for emb in embeddings]


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = get_embedding_model()

    test_texts = [
        "Python developer with Django and machine learning experience",
        "Senior Java developer with Spring Boot and microservices",
        "Data scientist skilled in NLP and deep learning with PyTorch",
    ]

    for text in test_texts:
        vec = generate_embedding(model, text)
        print(f"Text: '{text[:60]}...'")
        print(f"  → Embedding dimension: {len(vec)}")
        print(f"  → First 5 values: {vec[:5]}")
        print()
