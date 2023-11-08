from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from dataloader import SentimentDataLoader
import pickle
import argparse
import numpy as np 


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.tsv', type=str,
                        help="Train file to learn from (default train.txt)")
    
    parser.add_argument("-c", "--count", action="store_true",
                        help="Use word level count vectorizer")
    parser.add_argument("-t", "--word_level_tfidf", action="store_true",
                        help="Use word level TF-IDF vectorizer")
    parser.add_argument("-ch", "--char_level_tfidf", action="store_true",
                        help="Use character level TF-IDF vectorizer")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Use Sentiment Features")
    
    parser.add_argument("-svm1", "--svm1", action="store_true",
                        help="Use normal SVM")
    parser.add_argument("-svm2", "--svm2", action="store_true",
                        help="Use linear SVM")
    args = parser.parse_args()
    return args

def identity(inp):
    return inp

def save_model(model_name, classifier):
    if classifier is not None:
        # Save the trained classifier to a file using pickle
        with open(model_name, 'wb') as model_file:
            pickle.dump(classifier, model_file)
        print(f"Saved model to {model_name}")
    else:
        print("Classifier has not been trained yet.")


if __name__ == "__main__":
    args = create_arg_parser()

    # load training data 
    train_loader = SentimentDataLoader(args.train_file)
    train_loader.load_data()
    train_string_documents, train_tokenized_documents, train_labels, labels_bin, sentiment_labels, sentiment_scores = train_loader.get_data()

    if args.word_level_tfidf:
        # word level tfidf vectorizer
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "TF-IDF_word"

    elif args.char_level_tfidf:
        # char level tfidf vectorizer
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1)
        vec_name = "TF-IDF_char"

    elif args.count:
        # bag of words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
        vec_name = "CountVectorizer"

    elif args.sentiment: 
        # use sentiment features 
        sentiment_features = np.column_stack((sentiment_labels, sentiment_scores))
        if args.word_level_tfidf:
            # if word level
            word_vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
            vec =  np.column_stack((word_vec, sentiment_features))
            vec_name = "Word level tfidf"
        
        if args.char_level_tfidf:
            # if char level
            char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), min_df=1)
            vec =  np.column_stack((char_vec, sentiment_features))
            vec_name = "Character level tfidf"


    # create an empty list to store the classifiers 
    classifiers = []

    if args.svm1:
        classifiers.append(("Normal SVM", SVC()))
    if args.svm2:
        classifiers.append(("Linear SVM", LinearSVC()))
    
    # iterate through the list of classifiers and apply each one with the selected vectorizer
    for classifier_name, classifier_model in classifiers:

        classifier = Pipeline([("vec", vec), ("cls", classifier_model)])
        print(f"\n Vectorizer: {vec_name}, Classifier: {classifier_name} \n")

        # train the classifier on the training data and labels
        if args.char_level_tfidf: 
            classifier.fit(train_string_documents, train_labels)
        if args.word_level_tfidf:
            classifier.fit(train_tokenized_documents, train_labels)

        # Save the trained model
        model_name = f"{vec_name}_{classifier_name}.pkl"
        save_model(model_name, classifier)
    