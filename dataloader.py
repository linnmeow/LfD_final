from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')

class SentimentDataLoader:
    def __init__(self, corpus_file, use_sentiment_score=False, use_sentiment_label=False):
        self.corpus_file = corpus_file
        self.use_sentiment_score = use_sentiment_score
        self.use_sentiment_label = use_sentiment_label
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.string_documents = []
        self.tokenized_documents = []
        self.labels = [] 
    
    def load_data(self):
        with open(self.corpus_file, encoding='utf-8') as in_file:

            for line in in_file:
                # split the line into tokens, get the tokenized text without labels
                tokens = line.strip().split()[:-1]

                # append the labels to the list
                self.labels.append(line.strip().split()[-1])

                # get string texts for character information extraction
                text = ' '.join(tokens)

                if self.use_sentiment_score:
                    # calculate the sentiment score
                    compound_score = self.sentiment_analyzer.polarity_scores(text)['compound']

                    # append the compound score as a token
                    tokens.append(str(compound_score))
                    text = text + ' ' + str(compound_score)

                if self.use_sentiment_label:
                    # calculate the sentiment label (binary pos or neg)
                    compound_score = self.sentiment_analyzer.polarity_scores(text)['compound']
                    sentiment_label = 'pos' if compound_score >= 0 else 'neg'

                    # append the sentiment label as a token
                    tokens.append(sentiment_label)
                    text = text + ' ' + sentiment_label

                # append both the string and tokenized documents
                self.string_documents.append(text)
                self.tokenized_documents.append(tokens)               

    def get_data(self):
        return self.string_documents, self.tokenized_documents, self.labels

if __name__ == "__main__":
    
    data_loader = SentimentDataLoader('dev_clean.tsv', use_sentiment_label=True, use_sentiment_score=True)
    data_loader.load_data()
    string_documents, tokenized_documents, labels = data_loader.get_data()

    print(string_documents[:3])
