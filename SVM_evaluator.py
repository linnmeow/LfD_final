from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import fasttext
from dataloader import SentimentDataLoader
from classifier_fasttext import TextClassifierFastText
from classifier import identity


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
            model_path = 'crawl-300d-2M-subword.bin'
            model = fasttext.load_model(model_path)
            classifier = TextClassifierFastText(model_path)
            test_embeddings = classifier.get_sentence_embeddings(self.test_data, model)
            predictions = self.classifier.predict(test_embeddings)

        else:
            # if evaluating a different classifier
            predictions = self.classifier.predict(self.test_data)

        # calculate various evaluation metrics
        accuracy = accuracy_score(self.test_labels, predictions)
        precision = precision_score(self.test_labels, predictions, average='weighted')
        recall = recall_score(self.test_labels, predictions, average='weighted')
        f1 = f1_score(self.test_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(self.test_labels, predictions)

        print("Overall Metrics:")
        print("Accuracy:", accuracy)
        print("Precision (weighted):", precision)
        print("Recall (weighted):", recall)
        print("F1 Score (weighted):", f1)
        print("Confusion Matrix:")
        print(conf_matrix)

        # calculate and print metrics for individual classes (if applicable)
        class_precision = precision_score(self.test_labels, predictions, average=None)
        class_recall = recall_score(self.test_labels, predictions, average=None)
        class_f1 = f1_score(self.test_labels, predictions, average=None)

        num_classes = len(class_precision)
        for i in range(num_classes):
            print(f"Metrics for Class {i}:")
            print("Precision:", class_precision[i])
            print("Recall:", class_recall[i])
            print("F1 Score:", class_f1[i])
       

if __name__ == "__main__":
    # create an Evaluator instance
    evaluator = Evaluator(evaluate_fasttext_classifier=False)

    # load the saved model
    evaluator.load_classifier('TF-IDF_word_Linear SVM.pkl')  

    # load test data
    test_loader = SentimentDataLoader('dev_clean.tsv')
    test_loader.load_data()  
    test_string_documents, test_tokenized_documents, test_labels, sentiment_labels, sentiment_scores = test_loader.get_data()
    #print(test_tokenized_documents)

    evaluator.set_test_data(test_string_documents, test_labels)

    # evaluate the classifier using the Evaluator, Class 0 and 1 are NOT and OFF respectively
    evaluator.evaluate_classifier()


