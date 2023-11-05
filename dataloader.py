from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import nltk

nltk.download('vader_lexicon')

class SentimentDataLoader:
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.string_documents = []
        self.tokenized_documents = []
        self.labels = []
        self.labels_bin = []
        self.sentiment_labels = []
        self.sentiment_scores = [] 
    
    @ staticmethod
    def class_to_int(label):
        if label == 'NOT':
            return 0
        elif label == 'OFF':
            return 1
        else:
            raise ValueError("Invalid label.")
    
    def load_data(self):
        with open(self.corpus_file, encoding='utf-8') as in_file:

            for line in in_file:
                # split the line into tokens, get the tokenized text without labels
                tokens = line.strip().split()[:-1]

                # get the label from the line
                label = line.strip().split()[-1]
                self.labels.append(label)

                # apply the class_to_int function to convert the labels to numerical values
                label_numeric = self.class_to_int(label)
                self.labels_bin.append(label_numeric)

                # get string texts for character information extraction
                text = ' '.join(tokens)

                # calculate the sentiment score
                compound_score = self.sentiment_analyzer.polarity_scores(text)['compound']

                # append the sentiment score 
                self.sentiment_scores.append(compound_score)

                # pos labeled as 0 and neg labeled as 1
                sentiment_label = 0 if compound_score >= 0 else 1 
                # append the sentiment label
                self.sentiment_labels.append(sentiment_label)

                self.string_documents.append(text)
                self.tokenized_documents.append(tokens)  

    def calculate_label_overlap(self):
        if len(self.labels_bin) != len(self.sentiment_labels):
            raise ValueError("The lengths of labels and sentiment_labels must be the same.")

        overlap_count = sum(1 for label, sentiment_label in zip(self.labels_bin, self.sentiment_labels) if label == sentiment_label)
        overlap_percentage = (overlap_count / len(self.labels)) * 100

        return overlap_count, overlap_percentage             

    def get_data(self):
        return self.string_documents, self.tokenized_documents, self.labels, self.labels_bin, self.sentiment_labels, self.sentiment_scores
    

if __name__ == "__main__":
    
    data_loader = SentimentDataLoader('test_clean.tsv')
    data_loader.load_data()
    string_documents, tokenized_documents, labels, labels_bin, sentiment_labels, sentiment_scores = data_loader.get_data()
    print(sentiment_labels[:3])

    overlap_count, overlap_percentage = data_loader.calculate_label_overlap()
    print(f"Overlap count: {overlap_count}")
    print(f"Overlap percentage: {overlap_percentage:.2f}%")
