import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode


def get_all_layer_vectors(sentence, model, tokenizer):
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

    # Convert from torch tensor to numpy arrays, take only the first element in the batch
    all_layers_vectors = [layer[0].numpy() for layer in hidden_states]

    return tokens, all_layers_vectors


# Get vectors from all layers for a sentence
sentence = "The quick brown fox jumps over the lazy dog."
tokens, all_layers = get_all_layer_vectors(sentence, model, tokenizer)
print(f"Number of layers (including embedding layer): {len(all_layers)}")

# Let's analyze how the representation of a word changes across layers
word = "fox"
word_idx = tokens.index(word)

# Extract the vector for this word from each layer
word_vectors_across_layers = [layer[word_idx] for layer in all_layers]

# Calculate the cosine similarity between consecutive layers


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarities = []
for i in range(len(word_vectors_across_layers) - 1):
    sim = cosine_similarity(
        word_vectors_across_layers[i], word_vectors_across_layers[i+1])
    similarities.append(sim)

# Plot the similarities
plt.figure(figsize=(10, 6))
plt.plot(similarities, marker='o')
plt.title(f"Cosine Similarity Between Consecutive Layers for '{word}'")
plt.xlabel('Layer Transition')
plt.ylabel('Cosine Similarity')
plt.xticks(range(len(similarities)), [
           f"{i}->{i+1}" for i in range(len(similarities))])
plt.grid(True)
plt.show()
