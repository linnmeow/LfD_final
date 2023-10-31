# LfD_final

## SVM
### Dependencies

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


## LSTM

### Dependencies

The following dependencies are required to use the LSTM module:

- [numpy](https://numpy.org) - Version 1.26.1
- [tensorflow](https://www.tensorflow.org/install) - Version 2.13.0
- [keras](https://keras.io/getting_started/) - Version 2.13.1
- [sklearn](https://scikit-learn.org/stable/) - Version 1.3.2

### Usage

Run the lstm.py file using the following command:

```
python lstm.py
``` 

This command will use the default training and development files, as well as print out training and validation performance metrics.
The default training and development files are data/train_clean.tsv and data/dev_clean.tsv, respectively.


#### Optional Arguments

You can specify the input training and development files using the following optional command-line arguments:

* -i or --input: To specify the training file, use the -i argument followed by the path to your training file. For example:

```
python lstm.py -i data/my_training_data.tsv
```

* -d or --dev: To specify the development file, use the -d argument followed by the path to your development file. For example:
```
python lstm.py -d data/my_dev_data.tsv
```
* -s or --sentiment: To train with sentiment analysis as a feature, use the -s argument followed by 'True', as well as specifying a path to training and development datasets that include sentiment in the correct format. We have included such files in the 'data' folder.
```
python lstm.py -i 'data/train_with_sentiment.tsv' -d 'data/dev_with_sentiment.tsv' -s True
```
#### Testing

If you want to train the model and evaluate it on a test set, you can use the -t argument followed by the path to your test set file. For example:

```
python lstm.py -t data/my_test_data.tsv
```
By specifying the test set, the code will train the model on the training data, evaluate it on the development data, and finally test it on the provided test set.

#### Output

The code will output performance metrics and model evaluation results to the console

    - How to **train** all models on the data
    - Output files for each experiment on which you report results in the paper
    - File that runs evaluation of input and output file
    - All relevant code that you used to train/evaluate/analyse models
