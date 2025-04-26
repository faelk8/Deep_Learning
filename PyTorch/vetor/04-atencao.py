import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # for safety: set to evaluation mode


def get_attention_weights(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the tokens (for reference)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Forward pass, get attention weights
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        output_attentions=True)

    # One weight for each attention layer in the model
    # Each element in has shape (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions

    return tokens, attentions


def visualize_attention(tokens, attention_weights, layer, head):
    """visualize attention for a specific layer and head"""

    # Get attention weights for the specified layer and head
    # Shape: (sequence_length, sequence_length)
    attn = attention_weights[layer][0, head].numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap
    sns.heatmap(attn, xticklabels=tokens,
                yticklabels=tokens, cmap="viridis", ax=ax)
    ax.set_title(f"Attention Weights - Layer {layer+1}, Head {head+1}")
    ax.set_xlabel("Token (Key)")
    ax.set_ylabel("Token (Query)")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()


def visualize_layer_attention(tokens, attention_weights, layer):
    """visualize the average attention across all heads for a layer"""

    # Get average attention weights across all heads for the specified layer
    # Shape: (sequence_length, sequence_length)
    attn = attention_weights[layer][0].mean(dim=0).numpy()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a heatmap
    sns.heatmap(attn, xticklabels=tokens,
                yticklabels=tokens, cmap="viridis", ax=ax)
    ax.set_title(f"Average Attention Weights - Layer {layer+1}")
    ax.set_xlabel("Token (Key)")
    ax.set_ylabel("Token (Query)")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()


# Get attention weight from an example sentence
sentence = "The president of the United States visited the capital city."
tokens, attention_weights = get_attention_weights(sentence, model, tokenizer)

# Visualize attention for a specific layer and head
# BERT base has 12 layers (0-11) and 12 heads per layer (0-11)
layer_to_visualize = 5  # 6th layer (0-indexed)
head_to_visualize = 7   # 8th attention head (0-indexed)
visualize_attention(tokens, attention_weights,
                    layer_to_visualize, head_to_visualize)

# Visualize average attention for a layer
visualize_layer_attention(tokens, attention_weights, layer_to_visualize)
