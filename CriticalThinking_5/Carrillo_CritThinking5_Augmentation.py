import torch
from transformers import BertTokenizer, BertForMaskedLM
import random

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def augment_text(text, top_k=7, swap_prob=0.5):
    """
    Augment the input text by swapping some words with similar words using BERT.

    Args:
    text (str): The input text to augment.
    top_k (int): The number of similar words to consider for each swap (default is 5).
    swap_prob (float): The probability of swapping each word (default is 0.3).

    Returns:
    str: The augmented text.
    """
    # Split the text into words
    words = text.split()
    augmented_text = []

    # Iterate over each word in the text
    for word in words:
        if random.random() < swap_prob:  # Decide whether to swap this word
            # Prepare the input with [MASK] token
            masked_sentence = text.replace(word, '[MASK]', 1)
            inputs = tokenizer(masked_sentence, return_tensors='pt')

            # Predict the masked word
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            # Find the index of the [MASK] token
            masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)

            # Get the top k predicted words for the masked position
            top_k_indices = torch.topk(predictions[0, masked_index], top_k).indices.tolist()
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

            # Filter out non-alphabetic tokens and the original word
            similar_words = [token for token in top_k_tokens if token.isalpha() and token != word]
            
            # Replace the word with a randomly chosen similar word if available
            if similar_words:
                new_word = random.choice(similar_words)
                augmented_text.append(new_word)
            else:
                augmented_text.append(word)
        else:
            augmented_text.append(word)  # Keep the original word

    # Join the augmented words into a single string
    return ' '.join(augmented_text)

def test_augmentation():
    # Define a static test set
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I love programming in Python.",
        "Artificial Intelligence is the future.",
        "The weather today is sunny and bright."
    ]

    # Augment each text in the test set
    for text in test_texts:
        augmented_text = augment_text(text)
        print(f"Original Text: {text}")
        print(f"Augmented Text: {augmented_text}\n")

# Run the test
test_augmentation()