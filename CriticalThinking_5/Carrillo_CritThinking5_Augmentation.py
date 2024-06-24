import torch
from transformers import BertTokenizer, BertForMaskedLM
import random

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def augment_text(text, top_k=5, swap_prob=0.3):

    words = text.split()
    augmented_text = []

    # Iterate over each word in the text
    for word in words:
        if random.random() < swap_prob:  
            masked_sentence = text.replace(word, '[MASK]', 1)
            inputs = tokenizer(masked_sentence, return_tensors='pt')

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)

            top_k_indices = torch.topk(predictions[0, masked_index], top_k).indices.tolist()
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)


            similar_words = [token for token in top_k_tokens if token.isalpha() and token != word]
            

            if similar_words:
                new_word = random.choice(similar_words)
                augmented_text.append(new_word)
            else:
                augmented_text.append(word)
        else:
            augmented_text.append(word) 


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