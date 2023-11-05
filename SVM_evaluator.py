from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from dataloader import SentimentDataLoader
from SVM_classifier_fasttext import TextClassifierFastText
from SVM_classifier import identity
import json


class Evaluator:
    def __init__(self, evaluate_fasttext_classifier=True):
        self.classifier = None
        self.test_data = None
        self.test_labels = None
        self.evaluate_fasttext_classifier = evaluate_fasttext_classifier

    def load_classifier(self, classifier_path):
        # load a saved classifier from a file
        with open(classifier_path, 'rb') as cls:
            self.classifier = pickle.load(cls)

    def set_test_data(self, test_data, test_labels):
        # set the test data and corresponding labels for evaluation
        self.test_data = test_data
        self.test_labels = test_labels

    def evaluate_classifier(self):
        if self.evaluate_fasttext_classifier:
                # if evaluating a FastText-based classifier
                with open('fasttext_embeddings.json', 'r') as embeddings_file:
                    embeddings_dict = json.load(embeddings_file)
                
                classifier = TextClassifierFastText(embeddings_dict)
                test_embeddings = classifier.get_sentence_embeddings(self.test_data)
                predictions = self.classifier.predict(test_embeddings)

        else:
            # if evaluating a different classifier
            predictions = self.classifier.predict(self.test_data)

        # calculate various evaluation metrics
        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(self.test_labels, predictions, average='macro')
        recall = recall_score(self.test_labels, predictions, average='macro')
        f1 = f1_score(self.test_labels, predictions, average='macro')
        conf_matrix = confusion_matrix(self.test_labels, predictions)

        print("Overall Metrics:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:")
        print(conf_matrix)
  

if __name__ == "__main__":
    # create an Evaluator instance
    evaluator = Evaluator(evaluate_fasttext_classifier=True)

    # load the saved model
    evaluator.load_classifier('SVM_fasttext_model.pkl')  

    # load test data
    test_loader = SentimentDataLoader('test.tsv')
    test_loader.load_data()  
    test_string_documents, test_tokenized_documents, test_labels, labels_bin, sentiment_labels, sentiment_scores = test_loader.get_data()

    evaluator.set_test_data(test_string_documents, test_labels)

    # evaluate the classifier using the Evaluator, Class 0 and 1 are NOT and OFF respectively
    evaluator.evaluate_classifier()


