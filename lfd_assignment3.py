#!/usr/bin/env python


'''TODO: add high-level description of this Python script'''


import json
import argparse
import random as py_random
import wandb
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# Random seed for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)
py_random.seed(1234)


def create_arg_parser() -> argparse.Namespace:
    """Converts command line arguments to Namespace object and returns it."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--train_file", default="train.txt", type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default="dev.txt",
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default="glove_reviews.json", type=str,
                        help="Embedding file we are using (default glove_reviews.json)")

    args = parser.parse_args()

    return args


def read_corpus(corpus_file: str) -> tuple[list[str], list[str]]:
    """Reads review data set and returns docs and labels."""

    documents, labels = [], []

    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])

    return documents, labels


def read_embeddings(embeddings_file: str) -> dict[str, np.ndarray]:
    """Reads in word embeddings from file and save as numpy array."""

    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb) -> np.ndarray[float]:
    """Creates embedding matrix given vocab and the embeddings."""

    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train: list[str], emb_matrix: np.ndarray[float]) -> Sequential:
    """Creates the bidirectional LSTM model."""

    # Hyperparameters
    HIDDEN_UNITS = 64
    LEARNING_RATE = 0.01
    LOSS_FUNCTION = "categorical_crossentropy"
    OPTIM = Adam(learning_rate=LEARNING_RATE)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))

    model = Sequential()
    model.add(
        Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False)
    )
    model.add(
        Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=False, recurrent_dropout=0.1))
    )
    model.add(Dropout(0.1))
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))

    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIM, metrics=["accuracy"])

    return model


def train_model(model: Sequential,
                X_train: list[str], Y_train: list[str],
                X_dev: list[str], Y_dev: list[str]) -> Sequential:
    """Trains the LSTM model."""

    VERBOSE = 1
    BATCH_SIZE = 25
    EPOCHS = 50

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    model.fit(
        X_train,
        Y_train,
        verbose=VERBOSE,
        epochs=EPOCHS,
        callbacks=[callback],
        batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev)
    )
    test_set_predict(model, X_dev, Y_dev, "dev")

    return model


def test_set_predict(model: Sequential, X_test: list[str], Y_test: list[str], ident: str) -> None:
    """Measure accuracy on own test set, which is a subset of the training data."""

    Y_pred = np.argmax(model.predict(X_test), axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    formatted_acc = round(accuracy_score(Y_test, Y_pred), 3)
    print(f"Accuracy on own {ident} set: {formatted_acc}")


def main() -> None:
    """Main function to train and test neural network given cmd line arguments."""

    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test")

        y_prediction = model.predict(X_test_vect)
        print('test')
        print(confusion_matrix(Y_test, y_prediction, normalize='pred'))


if __name__ == "__main__":
    main()
