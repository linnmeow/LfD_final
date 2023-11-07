# LfD_final

## Dependencies


The following dependencies are required to use the SVM modules:

- [fasttext] - Version 0.9.2
- [scikit-learn](https://scikit-learn.org/stable/) - Version 1.3.2
- [nltk](https://www.nltk.org) - Version 3.8.1
- [numpy](https://numpy.org) - Version 1.26.1
- [scipy](https://scipy.org) - Version 1.11.3

The following dependencies are required to use lstm.py:
- [tensorflow](https://www.tensorflow.org/api_docs/python/tf) - Version 2.12.0
- [scikit-learn](https://scikit-learn.org/stable/) - Version 1.3.0
- [transformers](https://pypi.org/project/transformers/) - Version 4.35.0
- [numpy](https://numpy.org) - Version 1.23.5
- [scipy](https://pypi.org/project/scipy/) - Version 1.11.3

The following dependencies are required to use bert.py:
- [tensorflow](https://www.tensorflow.org/api_docs/python/tf) - Version 2.14.0
- [scikit-learn](https://scikit-learn.org/stable/) - Version 1.2.2
- [transformers](https://pypi.org/project/transformers/) - Version 4.35.0
- [torch](https://pypi.org/project/torch/) - Version 2.1.0+cu118
- [spacy](https://pypi.org/project/spacy/) - Version 3.6.1
- [scipy](https://pypi.org/project/scipy/) - Version 1.11.3
- (I trained the models on own device, clusters, google colab respectively, that is why the dependencies are slightly different.)

## Training the models
### SVM
- to train the SVM baseline model, run python SVM_classifier.py -ch --svm2 --train_file train.tsv
- to evaluate the model, run the SVM_evaluator and provide the correct model name (set evaluate_fasttext_classifier to False)
### LSTM
- example usage for using lstm.py: python lstm.py -i train.tsv -d dev.tsv -e glove_embeddings
### BERT
- example usage for using bert.py: python bert.py -m bert-base-uncased -i train.tsv -d dev.tsv



    - How to **train** all models on the data
    - Output files for each experiment on which you report results in the paper
    - File that runs evaluation of input and output file
    - All relevant code that you used to train/evaluate/analyse models
