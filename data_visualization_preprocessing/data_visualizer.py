import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class DataVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        # load the TSV file 
        self.data = pd.read_csv(self.file_path, sep='\t', header=None)

    def visualize_word_cloud(self):
        # visualize data using wordcloud to get first impression
        text_data = self.data[0].values  # text is in the first column
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(text_data))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def visualize_word_frequencies(self):
        text_data = self.data[0].values
        word_freq = pd.Series(' '.join(text_data).split()).value_counts()[:20]
        word_freq.plot(kind='bar')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.show()

    def visualize_label_distribution(self):
        sentiment_distribution = self.data[1].value_counts() # labels in the second column
        print(sentiment_distribution)
        sentiment_distribution.plot(kind='bar')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.show()
    
if __name__ == "__main__":

    file_path = 'dev.tsv'
    visualizer = DataVisualizer(file_path)
    visualizer.visualize_word_cloud()
    visualizer.visualize_word_frequencies()
    visualizer.visualize_label_distribution()
