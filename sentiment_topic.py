import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import nltk

nltk.download('vader_lexicon')

class TopicSentimentAnalyzer:
    def __init__(self, file_path, n_topics=5):
        self.file_path = file_path
        self.load_data()
        self.n_topics = n_topics
        self.lda_model = None
        self.count_vectorizer = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        nltk.download('vader_lexicon')

    def load_data(self):
        # load the TSV file
        self.data = pd.read_csv(self.file_path, sep='\t', header=None)

    def analyze_sentiment(self, text):
        # analyze sentiment using NLTK's VADER
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment['compound']

        if compound_score >= 0.05:
            sentiment_label = 'POS'
        elif compound_score <= -0.05:
            sentiment_label = 'NEG'
        else:
            sentiment_label = 'NEU'

        return sentiment_label
    
    def analyze_sentiment_and_store(self):
        # add a new column 'Sentiment' to the DataFrame
        self.data['Sentiment'] = self.data[0].apply(self.analyze_sentiment)


if __name__ == "__main__":
    analyzer = TopicSentimentAnalyzer('train_clean.tsv', n_topics=5)
    analyzer.analyze_sentiment_and_store()
    
    # save the DataFrame with the 'Sentiment' column to a new TSV file
    analyzer.data.to_csv('train_with_sentiment.tsv', sep='\t', index=False)

