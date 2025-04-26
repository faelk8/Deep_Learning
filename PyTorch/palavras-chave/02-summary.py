import numpy as np
import torch
from transformers import BertTokenizer, BertModel


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_sentence_embedding(sentence, model, tokenizer):
    """Sentence embedding extracted from the [CLS] prefix token"""
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt",
                       add_special_tokens=True, truncation=True, max_length=512)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the [CLS] token embedding at position 0 from the last layer
    cls_embedding = outputs.last_hidden_state[0, 0].numpy()
    return cls_embedding


def extractive_summarize(document, model, tokenizer, num_sentences=3):
    # Split the document into sentences
    sentences = [s.strip() for s in document.split(".") if s.strip()]
    if len(sentences) <= num_sentences:
        return document

    # Get embeddings for all sentences
    sentence_embeddings = []
    for sentence in sentences:
        embedding = get_sentence_embedding(sentence, model, tokenizer)
        sentence_embeddings.append(embedding)

    # Calculate the document embedding (average of all sentence embeddings)
    # then find the most similar sentences
    document_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = []
    for idx, embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(embedding, document_embedding)
        similarities.append((sim, idx))
    top_sentences = sorted(similarities, reverse=True)[:num_sentences]

    # Extract the sentences, preserve the original order
    top_indices = sorted([x[1] for x in top_sentences])
    summary_sentences = [sentences[i] for i in top_indices]

    # Join the sentences to form the summary
    summary = ". ".join(summary_sentences) + "."
    return summary


# Example document
document = """
Transformer models have revolutionized natural language processing by
introducing mechanisms that can effectively capture contextual relationships in
text. One of the most powerful aspects of transformers is their ability to
generate context-aware vector representations, often referred to as context
vectors. Unlike traditional word embeddings that assign a fixed vector to each
word regardless of context, transformer models generate dynamic representations
that depend on the surrounding words. This allows them to capture the nuanced
meanings of words in different contexts. For example, in the sentences "I'm
going to the bank to deposit money" and "I'm going to sit by the river bank,"
the word "bank" has different meanings. A traditional word embedding would
assign the same vector to "bank" in both sentences, but a transformer model
generates different context vectors that capture the distinct meanings based on
the surrounding words. This contextual understanding enables transformers to
excel at a wide range of NLP tasks, from question answering and sentiment
analysis to machine translation and text summarization.
"""

# Generate a summary
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
summary = extractive_summarize(document, model, tokenizer, num_sentences=3)

# Print the original document and the summary
print("Original Document:")
print(document)
print("Summary:")
print(summary)
