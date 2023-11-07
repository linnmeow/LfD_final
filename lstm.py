import json
import argparse
import pickle
import random as py_random

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.initializers import Constant
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataloader import SentimentDataLoader


np.random.seed(1234)
tf.random.set_seed(1234)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
py_random = np.random.RandomState(1234)

def create_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--train_file", default="train.tsv", type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default="dev.tsv",
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default="glove_embeddings.json", type=str,
                        help="Using GloVe embeddings")

    args = parser.parse_args()

    return args

def read_embeddings(embeddings_file):
    # read the filtered pretrained embeddings    
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    # old function from assignment 3
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    embedding_dim = len(emb["the"])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(emb_matrix, hidden_units, loss_function, optim, recurrent_dropout, dropout):
    # take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0]) 
    num_tokens = len(emb_matrix)

    # model architecture
    model = Sequential()
    model.add(
        Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False)
    )

    model.add(
        LSTM(units=hidden_units, return_sequences=False, recurrent_dropout=recurrent_dropout)
    )
        
    model.add(Dropout(dropout))

    model.add(Dense(input_dim=10, units=1, activation='sigmoid'))

    model.compile(loss=loss_function, optimizer=optim, metrics=[tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)])

    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, verbose, batch_size, epochs):
    
    callback = EarlyStopping(monitor="val_loss", patience=3)
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=epochs,
        callbacks=[callback],
        batch_size=batch_size,
        validation_data=(X_dev, Y_dev)
    )
    test_set_predict(model, X_dev, Y_dev, "dev")

    return model

def test_set_predict(model, X_test, Y_test, ident):

    # if >= 0.5 return 1, else 0
    Y_pred = list(map(lambda value: 1 if value >= 0.5 else 0, model.predict(X_test)))

    f1_result = round(f1_score(Y_test, Y_pred, average='macro'), 3)
    print(f"F1 on {ident} set: {f1_result}")

    if ident == "test":
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        print('Confusion Matrix:')
        print(conf_matrix)
        with open('confusion_matrix.pkl', 'wb') as file:
            pickle.dump(conf_matrix, file)

def main():
    
    args = create_arg_parser()

    # set the hyperparameters for creating a model
    hidden_units = 64
    learning_rate = 0.005
    loss_function = 'binary_crossentropy'
    optim = Adam(learning_rate=learning_rate)
    recurrment_dropout = 0.6
    dropout = 0.1
    # set the hyperparameters for training the model
    verbose = 1
    batch_size = 16
    epochs = 20

    # read in the training data using dataloader
    train_loader = SentimentDataLoader(args.train_file)
    train_loader.load_data()
    X_train, _, Y_train, _, _, _ = train_loader.get_data()

    # read in the dev data using dataloader
    dev_loader = SentimentDataLoader(args.dev_file)
    dev_loader.load_data()
    X_dev, _, Y_dev, _, _, _ = dev_loader.get_data()

    embeddings = read_embeddings(args.embeddings)

    # transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # create vocab 
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)

    # dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train) 
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # create model
    model = create_model(emb_matrix, hidden_units=hidden_units, loss_function=loss_function, optim=optim, recurrent_dropout=recurrment_dropout, dropout=dropout)

    # transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, verbose=verbose, batch_size=batch_size, epochs=epochs)

    # determine the model name based on the argument
    if "fasttext" in args.embeddings:
        model_name = 'fasttext'
    elif "glove" in args.embeddings:
        model_name = 'glove'
    else:
        model_name = 'vanilla'

    # save the trained model with the appropriate name
    saved_model_name = f'{model_name}_model.h5'
    model.save(f'{model_name}_model.h5')
    print(f"Model saved as: {saved_model_name}")

    # do predictions on specified test set
    if args.test_file:

        # read in the training data using dataloader
        test_loader = SentimentDataLoader(args.test_file)
        test_loader.load_data()
        X_test, _, Y_test, _, _, _ = test_loader.get_data()

        Y_test_bin = encoder.transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        
        test_set_predict(model, X_test_vect, Y_test_bin, "test")

if __name__ == "__main__":
    main()
