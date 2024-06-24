
# Text Augmentation with BERT

This script reads text data from a CSV file, performs text augmentation by swapping words with similar words predicted by a pre-trained BERT model, and saves the augmented text to a new CSV file.

## Prerequisites

Make sure you have the following libraries installed:
- `transformers`
- `torch`
- `pandas`

You can install them using pip & the requirements.txt:

```
pip install -r requirements.txt
```

## Script Explanation

The script consists of the following parts:

1. **Import Libraries**: Import necessary libraries and load the pre-trained BERT model and tokenizer.
2. **Define the Augmentation Function**: Define a function that performs text augmentation by swapping words with similar words.
3. **Read CSV, Augment Text, and Save to New CSV**: Define a function that reads text data from a CSV file, performs text augmentation, and saves the augmented text to a new CSV file.
4. **Run the Script**: Define input and output CSV files, and the text column, and process the CSV file.

## How to Use

1. **Prepare Your Data**: Create an input CSV file named `text_dataset.csv` with a column named `text` containing the text data you want to augment.

2. **Run the Script**:

3. **Check the Output**: The augmented text will be saved in a new CSV file named `augmented_dataset.csv` in the same directory.

## Example Input and Output

**Input CSV (`text_dataset.csv`)**:

| text                                         |
|----------------------------------------------|
| The quick brown fox jumps over the lazy dog. |
| I love programming in Python.                |
| Artificial Intelligence is the future.       |
| The weather today is sunny and bright.       |

**Output CSV (`augmented_dataset.csv`)**:

| text                                         | augmented_text                                |
|----------------------------------------------|-----------------------------------------------|
| The quick brown fox jumps over the lazy dog. | The quick brown fox leaps over the lazy hound.|
| I love programming in Python.                | I adore coding in Python.                     |
| Artificial Intelligence is the future.       | Artificial Intelligence is the destiny.       |
| The weather today is sunny and bright.       | The weather today is sunny and radiant.       |

## Notes

- Adjust the `top_k` and `swap_prob` parameters in the `augment_text` function to control the number of similar words considered for each swap and the probability of swapping each word.
- Ensure that the column name containing the text to augment matches the `text_column` parameter in the `process_csv` function.