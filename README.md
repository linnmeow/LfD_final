# LfD_final

## Dependencies


The following dependencies are required to use the SVM modules:

- [fasttext](https://fasttext.cc) - Version 0.9.2
- [emoji](https://pypi.org/project/emoji/) - Version 2.8.0
- [scikit-learn](https://scikit-learn.org/stable/) - Version 1.3.2
- [nltk](https://www.nltk.org) - Version 3.8.1
- [numpy](https://numpy.org) - Version 1.26.1
- [pandas](https://pandas.pydata.org) - Version 2.1.1
- [scipy](https://scipy.org) - Version 1.11.3


### Other Dependencies

The SVM modules may also depend on the following libraries to support its functionality. While they are not directly required for basic usage, they may be used in the underlying code:

- [smart-open](https://pypi.org/project/smart-open/) - Version 6.4.0
- [filelock](https://pypi.org/project/filelock/) - Version 3.12.4
- [fsspec](https://pypi.org/project/fsspec/) - Version 2023.10.0
- [Jinja2](https://pypi.org/project/Jinja2/) - Version 3.1.2
- [joblib](https://pypi.org/project/joblib/) - Version 1.3.2
- [MarkupSafe](https://pypi.org/project/MarkupSafe/) - Version 2.1.3
- [mpmath](https://pypi.org/project/mpmath/) - Version 1.3.0
- [networkx](https://pypi.org/project/networkx/) - Version 3.2

## Training the models
### SVM
- to train the SVM baseline model, run python SVM_classifier.py -ch --svm2 --train_file train.tsv
- to evaluate the model, run the SVM_evaluator and provide the correct model name
### LSTM



    - How to **train** all models on the data
    - Output files for each experiment on which you report results in the paper
    - File that runs evaluation of input and output file
    - All relevant code that you used to train/evaluate/analyse models
