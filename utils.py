import string
import re
import numpy as np
import pandas as pd
from config import *

df = pd.read_csv(ABBREVIATIONS_CSV, header=None)
abbreviations, meanings = np.array(df[0]), np.array(df[1])

def preprocess_sentence(text):
    # Convert string into lowercase
    text = text.lower()
    # Replace URL by <link_spam>
    text = re.sub(r"(?P<url>https?://[^\s]+)", "link_spam", text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert abbreviation into its meaning
    for abbreviation, meaning in zip(abbreviations, meanings):
        text = re.sub(rf"\b{abbreviation}\b", meaning, text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Tokenize sentence, remove word with only 1 character
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 1]
    # Concatenate tokens to sentence
    return ' '.join(tokens)