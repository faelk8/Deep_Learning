# https://machinelearningmastery.com/generating-and-visualizing-context-vectors-in-transformers/

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


# Get context vectors from example sentences with ambiguous words
sentence1 = "I'm going to the bank to deposit money."
sentence2 = "I'm going to sit by the river bank."
tokens1, vectors1 = get_context_vectors(sentence1, model, tokenizer)
tokens2, vectors2 = get_context_vectors(sentence2, model, tokenizer)

# Print the tokens for reference
print("Tokens in sentence 1:", tokens1)
print("Tokens in sentence 2:", tokens2)

# Find the index of "bank" in both sentences
bank_idx1 = tokens1.index("bank")
bank_idx2 = tokens2.index("bank")

# Get the context vectors for "bank" in both sentences
bank_vector1 = vectors1[bank_idx1]
bank_vector2 = vectors2[bank_idx2]

# Calculate cosine similarity between the two "bank" vectors
# lower similarity means meaning is different


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarity = cosine_similarity(bank_vector1, bank_vector2)
print(f"Cosine similarity between 'bank' vectors: {similarity:.4f}")
