import pandas as pd
import re
import emoji
import string  

class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        # load the TSV file
        self.data = pd.read_csv(self.file_path, sep='\t', header=None)

    def preprocess_text(self, text: str):
        # convert text to lowercase, remove frequent words like @USER and URL,
        # remove punctuation, remove everything that isn't letters,
        # and convert emojis to their textual descriptions
        assert isinstance(text, str), "This function only takes strings."
        text = text.lower()
        text = re.sub(r'@user\s*', '', text) # there is whitespce after @USER, input already been lowercased
        text = text.replace('URL', '')
        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # remove punctuation
        text = re.sub('[^A-Za-z ]+', '', text)  # remove everything that isn't letters
        text = emoji.demojize(text)  # convert emojis to text
        text = text.strip()
        return text

    def remove_frequent_words(self):
        # apply the preprocess_text function to the 'text' column
        self.data[0] = self.data[0].apply(self.preprocess_text)
    
    def save_preprocessed_tsv(self, output_file):
        # save the preprocessed data to another TSV file with the second column as labels
        self.data.to_csv(output_file, sep='\t', header=False, index=False) 


input_tsv_file = "dev.tsv"
output_tsv_file = "dev_clean.tsv"

preprocessor = Preprocessor(input_tsv_file)
preprocessor.remove_frequent_words()
preprocessor.save_preprocessed_tsv(output_tsv_file)

