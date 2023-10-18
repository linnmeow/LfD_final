#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix 
from numpy import ndarray
import wandb

def class_to_int(label):
    if label == 'books':
        return 0
    elif label == 'camera':
        return 1
    elif label == 'dvd':
        return 2
    elif label == 'health':
        return 3
    elif label == 'music':
        return 4
    elif label == 'software':
        return 5
    else:
        raise IndexError("This is not one of the labels.")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-df", "--dev_file", default='dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer")
    parser.add_argument("-c", "--countvec", action="store_true",
                        help= "Use the Countvectorizer")
    parser.add_argument("-u", "--union", action="store_true",
                        help= "Use the Unionvectorizer")
    parser.add_argument("-n", "--nbayes", action="store_true",
                        help="Use Naive Bayes classifier")
    parser.add_argument("-d", "--tree", action="store_true",
                        help="Use Decision Tree Classifier")
    parser.add_argument("-r", "--forest", action="store_true",
                        help="Use Random Forest Classifier")
    parser.add_argument("-knn", "--knearest", action="store_true",
                        help="Use K Nearest Neighbors Classifier")
    parser.add_argument("-svm1", "--svm1", action="store_true",
                        help="Use normal SVM")
    parser.add_argument("-svm2", "--svm2", action="store_true",
                        help="Use linear SVM")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    '''TODO: add function description'''
    ## MF 11.9.23
    ## The function takes a .txt file in the format of 'reviews.txt' as input, and returns two separate lists:
    ## 1) The documents and 2) the corresponding labels.
    ## The raw data contains two labels for each document - one for the binary sentiment classification task,
    ## and one for the 6-class topic classification task. You can specify which task you wish to use
    ## with the 'use_sentiment' argument.
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # TODO: comment
    # Comments by Maria, 10.9.23
    # read_corpus takes the input file as a command line argument. It splits it into input data and labels.
    # The default train file name is 'train.txt'.
    # The default dev file name is 'dev.txt'. 
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "TF-IDF"
    elif args.countvec:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "CountVectorizer"
    else: 
        count = CountVectorizer(preprocessor=identity, tokenizer=identity)
        tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "UnionVectorizer"
        vec = FeatureUnion([("count", count), ("tf", tf_idf)])

    # create an empty list to store the classifiers (Lin 12.09.23)

    classifiers = []
    if args.nbayes:
        classifiers.append(("Naive Bayes", MultinomialNB()))
    if args.tree:
        classifiers.append(("Decision Tree", DecisionTreeClassifier()))
    if args.forest:
        classifiers.append(("Random Forest", RandomForestClassifier()))
    if args.knearest:
        classifiers.append(("K Nearest Neighbors", KNeighborsClassifier()))
    if args.svm1:
        classifiers.append(("Normal SVM", SVC()))
    if args.svm2:
        classifiers.append(("Linear SVM", LinearSVC()))
    
    # Iterate through the list of classifiers and apply each one with the selected vectorizer (Lin 12.09.23)
    for classifier_name, classifier_model in classifiers:
        # create seperate pipelines 
        classifier = Pipeline([("vec", vec), ("cls", classifier_model)])
        print(f"\n Vectorizer: {vec_name}, Classifier: {classifier_name} \n")

        # TODO: comment this
        # MF 11.09.23
        # The method classifier.fit fits (trains) our classifier model on the training data
        # specified in the arguments.
        # The classifier model is defined above.

        # classifier.fit(X_train, Y_train)

        # TODO: comment this
        # MF 11.09.23
        # Our classifier is now trained on the training data. Now, we can use it to make predictions 
        # on new documents. The method classifier.predict takes documents and returns the models predictions
        # on these documents.

        #Y_pred = classifier.predict(X_test)

        # TODO: comment this
        pred = cross_val_predict(classifier, X_train, Y_train, cv=5)
        print(classification_report(Y_train, pred, digits=3))
        conf_matrix = confusion_matrix(Y_train,pred)
        print(conf_matrix)


        ### Track results on WandB
        wandb.login()

        wandb.init(
        # set the wandb project where this run will be logged
        project="LfD_Assignment_1",

        # track hyperparameters and run metadata
        config={
            "Model": classifier_name,
            "Vectorizer": vec_name
            }
        )

        # create lists where the labels are replaced by corresponding numbers
        Y_train_wandb = [class_to_int(item) for item in Y_train]
        pred_wandb = [class_to_int(item) for item in pred.tolist()]
        
        class_names = ["books","camera","dvd","health","music","software"]
        wandb.log({"confusion matrix" : wandb.plot.confusion_matrix(probs=None,
                        y_true=Y_train_wandb, preds=pred_wandb,
                        class_names=class_names)})

        accuracy = accuracy_score(y_true=Y_train_wandb, y_pred=pred_wandb)
        wandb.log({"accuracy": accuracy})

        wandb.finish()

