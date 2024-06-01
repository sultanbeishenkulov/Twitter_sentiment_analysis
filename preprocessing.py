import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessing:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None

    def filter_non_string(self, df, column):
        """
        Filter out rows with non-string values in the specified column.
        Convert non-string values to strings.
        """
        df = df.dropna(subset=[column])
        df[column] = df[column].astype(str)
        return df

    def normalize_text(self, text):
        """Convert text to lowercase to ensure consistency across the corpus."""
        return text.lower()

    def remove_html_tags(self, text):
        """Remove HTML tags from the text."""
        return re.sub(r'<.*?>', '', text)

    def remove_urls(self, text):
        """Remove URLs or hyperlinks from the text."""
        return re.sub(r'http\S+|www\S+', '', text)

    def remove_numbers(self, text):
        """Exclude numerical digits from the text."""
        return re.sub(r'\d+', '', text)

    def remove_punctuation(self, text):
        """Remove punctuation marks from the text."""
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize_text(self, text):
        """Split the text into individual words or tokens."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Eliminate common stopwords from the tokenized text."""
        return [word for word in tokens if word not in self.stop_words]

    def remove_emojis(self, text):
        """Remove emojis from the text."""
        if isinstance(text, str):
            emoji_pattern = re.compile(
                "[" 
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)
        else:
            return text

    def vectorize_data(self, text_data):
        # Join the tokenized text into strings
        text_data_strings = [" ".join(tokens) for tokens in text_data]
        # Initialize TfidfVectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer()
            tfidf_vectors = self.tfidf_vectorizer.fit_transform(text_data_strings)
        else:
            tfidf_vectors = self.tfidf_vectorizer.transform(text_data_strings)
        return tfidf_vectors

    def preprocess_text(self, data):
        data = self.filter_non_string(data, 'Tweet')
        data['Tweet'] = data["Tweet"].apply(self.normalize_text)
        data['Tweet'] = data['Tweet'].apply(self.remove_html_tags)
        data['Tweet'] = data['Tweet'].apply(self.remove_urls)
        data['Tweet'] = data['Tweet'].apply(self.remove_numbers)
        data['Tweet'] = data['Tweet'].apply(self.remove_punctuation)
        data['Tweet'] = data['Tweet'].apply(self.tokenize_text)
        data['Tweet'] = data['Tweet'].apply(self.remove_stopwords)
        data['Tweet'] = data['Tweet'].apply(self.remove_emojis)
        return data

