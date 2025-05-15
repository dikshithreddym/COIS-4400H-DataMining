import nltk
import string
import pandas as pd
from collections import Counter
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize

# Ensure required resources are downloaded
nltk.download('gutenberg')  # Load Gutenberg dataset
nltk.download('stopwords')  # Load stopwords list
nltk.download('punkt')  # Load tokenizer for word tokenization

# Load English stopwords and punctuation set
stop_words = set(stopwords.words('english'))  # Common words like "the", "is", "and"
punctuation = set(string.punctuation)  # Special characters like ".", ",", "!", etc.

# Function to preprocess text
def preprocess_text(text):
    """
    Tokenizes text, converts to lowercase, removes stopwords & punctuation,
    and returns word frequency counts.
    """
    words = word_tokenize(text.lower())  # Convert text to lowercase and tokenize words
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Keep only words (no numbers or punctuation)
    return Counter(words)  # Return word frequency count

# Create a Term-Document Matrix
tdm = {}  # Dictionary to store most frequent words for each document

for file_id in gutenberg.fileids():
    text = gutenberg.raw(file_id)  # Read entire document as a string
    word_freqs = preprocess_text(text)  # Preprocess text (clean & tokenize)
    most_common_words = [word for word, _ in word_freqs.most_common(10)]  # Extract 10 most frequent words
    tdm[file_id] = most_common_words  # Store results in dictionary

# Convert dictionary to Pandas DataFrame
tdm_df = pd.DataFrame.from_dict(tdm, orient='index')
tdm_df.columns = [f'Word {i+1}' for i in range(10)]  # Label columns as "Word 1", "Word 2", etc.

# Print the first 10 rows of the Term-Document Matrix
print("First 10 rows of the Term-Document Matrix:")
print(tdm_df.head(10))  # Display first 10 rows

# Save matrix to CSV file (optional, for submission)
tdm_df.to_csv("term_document_matrix.csv", index=True)
