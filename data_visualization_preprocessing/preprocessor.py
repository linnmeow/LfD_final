import pandas as pd
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download('punkt')  # download the NLTK tokenizer data

class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()
        self.stemmer = PorterStemmer()

    def load_data(self):
        # load the TSV file
        self.data = pd.read_csv(self.file_path, sep='\t', header=None)

    def preprocess_text(self, text: str):
        # convert text to lowercase, remove frequent words like @USER and URL,
        # remove punctuation, remove everything that isn't letters,
        # and convert emojis to their textual descriptions
        # split hashtag and converted emojis 
        assert isinstance(text, str), "This function only takes strings."
        
        text = re.sub(r'@USER\s*', '', text) # there is whitespce after @USER
        text = re.sub(r'URL\s*', '', text)
        text = emoji.demojize(text)  # convert emojis to text
        text = re.sub(r'[:_]+', ' ', text) # use underscore to split emojis
        text = re.sub(r'#([A-Z][a-z]+)([A-Z][a-z]+)+', r' \1 \2', text) # split token after hashtags
        text = re.sub('[^A-Za-z ]+', '', text)  # remove everything that isn't letters including punctuation
        
        text = text.lower() # lowercasing
        text = text.strip()
        return text
    
    @ staticmethod
    def remove_duplicate_tokens(text):
        tokens = text.split()
        new_tokens = []
        token_frequencies = {} 
        for token in tokens:
            if token_frequencies.get(token, 0) == 0:
                token_frequencies[token] = 1
                new_tokens.append(token)
        return ' '.join(new_tokens)
    
    def tokenize_text(self, text):
        # tokenize the text using NLTK's word_tokenize
        tokens = word_tokenize(text)
        return ' '.join(tokens)

    def stem_text(self, text):
        # tokenize the text
        tokens = word_tokenize(text)
        
        # apply stemming to each token
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def apply_preprocessing(self):
        # apply the preprocess_text function to all the tweets
        self.data[0] = self.data[0].apply(self.preprocess_text)

    def apply_remove_duplicate_tokens(self):
        # apply the remove_duplicate_tokens function to all tweets
        self.data[0] = self.data[0].apply(self.remove_duplicate_tokens)

    def apply_tokenization(self):
        # tokenize all the texts
        self.data[0] = self.data[0].apply(self.tokenize_text)
        
    def apply_stemming(self):
        # apply stemming to all the texts
        self.data[0] = self.data[0].apply(self.stem_text)
       
    def save_preprocessed_tsv(self, output_file):
        # save the preprocessed data to another TSV file with the second column as labels
        self.data.to_csv(output_file, sep='\t', header=False, index=False) 

if __name__ == "__main__":

    input_tsv_file = "dev.tsv"
    output_tsv_file = "dev_clean.tsv"

    preprocessor = Preprocessor(input_tsv_file)
    preprocessor.apply_preprocessing()
    preprocessor.apply_remove_duplicate_tokens()   
    preprocessor.apply_tokenization()
    preprocessor.apply_stemming()

    preprocessor.save_preprocessed_tsv(output_tsv_file)



