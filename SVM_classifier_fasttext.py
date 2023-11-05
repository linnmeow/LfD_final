import pickle
import numpy as np
import json
from sklearn.svm import SVC
from dataloader import SentimentDataLoader

class TextClassifierFastText:
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.classifier = None

    def get_sentence_embeddings(self, text_data):
        embeddings = []
        for text in text_data:
            words = text.split()
            sentence_vector = np.mean([self.embeddings_dict.get(word, np.zeros(300)) for word in words], axis=0)
            embeddings.append(sentence_vector)
        return np.array(embeddings)

    def train_classifier(self, train_embeddings, train_labels):
        self.classifier = SVC(kernel='linear')
        self.classifier.fit(train_embeddings, train_labels)

    def save_model(self, model_path):
        if self.classifier is not None:
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.classifier, model_file)
        else:
            print("Classifier has not been trained yet.")   

if __name__ == "__main__":
    # load word embeddings from file
    with open('fasttext_embeddings.json', 'r') as embeddings_file:
        embeddings_dict = json.load(embeddings_file)

    classifier = TextClassifierFastText(embeddings_dict)

    # load training data
    train_loader = SentimentDataLoader('train.tsv')
    train_loader.load_data()
    train_string_documents, _, train_labels, _, _, _ = train_loader.get_data()

    # get sentence embeddings
    train_embeddings = classifier.get_sentence_embeddings(train_string_documents)

    # train the classifier
    classifier.train_classifier(train_embeddings, train_labels)

    # save the trained model
    classifier.save_model('svm_fasttext_model.pkl')
