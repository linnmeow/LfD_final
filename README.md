# Offensive Language Detection with Machine Learning

This repository contains the implementation and experiments for detecting offensive language on Twitter using various machine learning models, including **SVM**, **LSTM**, **BERT**, and **Electra**. The project leverages the **Offensive Language Identification Dataset (OLID)** and explores the impact of data preprocessing and pretrained word embeddings on model performance.

---

## Key Features
- **Dataset**: OLID dataset with 14,100 tweets labeled as `Offensive` or `Not Offensive`.
- **Preprocessing**:
  - Token removal (`@USER`, `URL`).
  - Emoji-to-text conversion.
  - Hashtag segmentation.
  - Punctuation and numerical removal.
  - Text lowercasing.
  - Tokenization and stemming.
  - Removal of duplicate words.
- **Models**:
  - **SVM**: Baseline with character-level TF-IDF and FastText embeddings.
  - **LSTM**: Vanilla, GloVe, and FastText-augmented variants.
  - **Transformer Models**: BERT-uncased and Electra.
- **Evaluation**: Macro-F1 score to handle class imbalance.

---

## Results
### Original Dataset
| Model               | Validation F1 | Test F1     |
|---------------------|---------------|-------------|
| SVM Baseline        | 73.0%         | 70.6%       |
| LSTM (FastText)     | 72.9%         | 74.4%       |
| BERT-uncased        | 78.7%         | 79.5%       |
| Electra             | 79.0%         | **80.5%**   |

### Preprocessed Dataset
| Model               | Validation F1 | Test F1     |
|---------------------|---------------|-------------|
| SVM Baseline        | 73.6%         | 74.7%       |
| LSTM (FastText)     | 74.5%         | 76.7%       |
| BERT-uncased        | 76.7%         | 77.9%       |
| Electra             | 77.8%         | **79.6%**   |

---

## Key Findings
1. **Preprocessing Benefits**:
   - Improved SVM and LSTM performance by **0.6% to 4.7%**.
   - Balanced misclassification rates (e.g., LSTM confusion matrices).
2. **Transformer Models**:
   - Electra outperformed BERT on both original and preprocessed data.
   - Preprocessing slightly degraded transformer performance (1-2% drop).
3. **FastText Embeddings**:
   - Enhanced LSTM performance by capturing subword information.

---

## Usage

### Dependencies
#### For SVM:
- `fasttext==0.9.2`
- `scikit-learn==1.3.2`
- `nltk==3.8.1`
- `numpy==1.26.1`
- `scipy==1.11.3`

#### For LSTM:
- `tensorflow==2.12.0`
- `scikit-learn==1.3.0`
- `transformers==4.35.0`
- `numpy==1.23.5`
- `scipy==1.11.3`

#### For BERT:
- `tensorflow==2.14.0`
- `scikit-learn==1.2.2`
- `transformers==4.35.0`
- `torch==2.1.0+cu118`
- `spacy==3.6.1`
- `scipy==1.11.3`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Preprocessing
To preprocess the data, run:
```bash
python scripts/preprocessor.py --input data/olid.csv --output data/olid_preprocessed.csv
```

### Training
#### SVM:
- Train:
  ```bash
  python SVM_classifier.py -ch --svm2 --train_file train.tsv
  ```
- Evaluate:
  ```bash
  python SVM_evaluator.py --model_name <model_name> --evaluate_fasttext_classifier False
  ```

#### LSTM:
- Train:
  ```bash
  python lstm.py -i train.tsv -d dev.tsv -e glove_embeddings
  ```

#### BERT:
- Train:
  ```bash
  python bert.py -m bert-base-uncased -i train.tsv -d dev.tsv
  ```

---

## Future Work
- Explore multilingual offensive language detection.
- Fine-tune transformer models on domain-specific data.
- Incorporate additional features (e.g., sentiment scores).
