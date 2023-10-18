'''TODO: add high-level description of this Python script'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix 
from numpy import ndarray
import wandb


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from (default train.tsv)")
    parser.add_argument("-df", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on (default dev.tsv)")
    parser.add_argument("-svm1", "--svm1", action="store_true",
                        help="Use baseline Linear SVM")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''TODO: add function description'''
    '''TODO: Change this to fit our current data'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()

            documents.append(tokens[:-1])
            labels.append(tokens[-1])

    return documents, labels

def identity(inp):
    '''Dummy function that just returns the input'''
    return inp




if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    # Bag of Words vectorizer
    vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
    vec_name = "CountVectorizer"

        # create an empty list to store the classifiers (Lin 12.09.23)

    classifiers = []
    if args.svm1:
        classifiers.append(("Linear SVM", LinearSVC()))
    
    # Iterate through the list of classifiers and apply each one with the selected vectorizer (Lin 12.09.23)
    for classifier_name, classifier_model in classifiers:
        # create seperate pipelines 
        classifier = Pipeline([("vec", vec), ("cls", classifier_model)])
        print(f"\n Vectorizer: {vec_name}, Classifier: {classifier_name} \n")

        pred = cross_val_predict(classifier, X_train, Y_train, cv=5)
        print(classification_report(Y_train, pred, digits=3))
        conf_matrix = confusion_matrix(Y_train,pred)
        print(conf_matrix)

