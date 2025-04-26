import numpy as np
import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode


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


def disambiguate_word(word, sentences, model, tokenizer):
    """for word sense disambiguation"""

    # Get context vector of a word for each sentence
    word_vectors = []
    for sentence in sentences:
        tokens, vectors = get_context_vectors(sentence, model, tokenizer)
        for token_index, token in enumerate(tokens):
            if token == word:
                word_vectors.append({
                    'sentence': sentence,
                    'vector': vectors[token_index]
                })

    # Calculate pairwise similarities between all vectors
    n = len(word_vectors)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            value = cosine_similarity(
                word_vectors[i]['vector'], word_vectors[j]['vector'])
            similarity[i, j] = similarity[j, i] = value

    # Run simple clustering to group vectors of high similarity
    threshold = 0.60  # Similarity > threshold will be the same cluster
    clusters = []

    for i in range(n):
        # Check if this vector belongs to any existing cluster
        assigned = False
        for cluster in clusters:
            # Calculate average similarity with all vectors in the cluster
            avg_sim = np.mean([similarity[i, j] for j in cluster])
            if avg_sim > threshold:
                cluster.append(i)
                assigned = True
                break
        # If not assigned to any cluster, create a new one
        if not assigned:
            clusters.append([i])

    # Print the results
    print(f"Found {len(clusters)} different senses for '{word}':\n")
    for i, cluster in enumerate(clusters):
        print(f"Sense {i+1}:")
        for idx in cluster:
            print(f"  - {word_vectors[idx]['sentence']}")
        print()


# Example: Disambiguate the word "bank"
sentences = [
    "I'm going to the bank to deposit money.",
    "The bank approved my loan application.",
    "I'm going to sit by the river bank.",
    "The bank of the river was muddy after the rain.",
    "The central bank raised interest rates yesterday.",
    "They had to bank the fire to keep it burning through the night."
]
disambiguate_word("bank", sentences, model, tokenizer)
