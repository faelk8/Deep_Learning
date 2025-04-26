import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# https://machinelearningmastery.com/applications-with-context-vectors/


def get_context_vectors(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the tokens (for reference)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Forward pass, get all hidden states from each layer
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # Each element in hidden states has shape (batch_size, sequence_length, hidden_size)
    # Here takes the first element in the batch from the last layer
    # Shape: (sequence_length, hidden_size)
    last_layer_vectors = hidden_states[-1][0].numpy()

    return tokens, last_layer_vectors


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_contextual_keywords(document, model, tokenizer, top_n=5):
    """extract contextual keywords from a document"""
    # Split the document into sentences (simple split by period)
    sentences = [s.strip() for s in document.split(".") if s.strip()]

    # Process each sentence to get context vectors
    all_tokens = []
    all_vectors = []
    for sentence in sentences:
        if not sentence:
            continue   # Skip empty sentences

        # Get context vectors
        tokens, vectors = get_context_vectors(sentence, model, tokenizer)

        # Store tokens and vectors (excluding special tokens [CLS] and [SEP])
        all_tokens.extend(tokens[1:-1])
        all_vectors.extend(vectors[1:-1])

    # Convert to numpy arrays, then calculate the document vector as average of all token vectors
    all_vectors = np.array(all_vectors)
    doc_vector = np.mean(all_vectors, axis=0)

    # Calculate similarity between each token vector and the document vector
    similarities = []
    for token, vec in zip(all_tokens, all_vectors):
        # Skip special tokens, punctuation, and common words
        if token in ["[CLS]", "[SEP]", ".", ",", "!", "?", "the", "a", "an", "is", "are", "was", "were"]:
            continue
        # compute similarity, then remember it with the token
        sim = cosine_similarity(vec, doc_vector)
        similarities.append((sim, token))

    # Sort the similarity and get the top N
    top_similarities = sorted(similarities, reverse=True)[:top_n]
    return top_similarities


# Example document
document = """
Artificial intelligence is transforming industries around the world.
Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions.
Natural language processing enables computers to understand and generate human language.
Computer vision systems can recognize objects and interpret visual information.
These technologies are driving innovation in healthcare, finance, transportation, and many other sectors.
"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Extract contextual keywords and print the result
top_keywords = extract_contextual_keywords(
    document, model, tokenizer, top_n=10)
print("Top contextual keywords:")
for similarity, token in top_keywords:
    print(f"{token}: {similarity:.4f}")
