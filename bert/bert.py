import argparse
import numpy as np
import torch
import os
import pandas as pd
from dataloader_bert import BertDataLoader
from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--train_file", default='train.tsv', type=str, 
                        help="Input file to train from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.tsv', 
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", default='test.tsv', type=str, 
                        help="Test file for model evaluation (default test.tsv)")
    
    parser.add_argument("-m", "--model_name", type=str, default='bert-base-uncased', 
                        help="Pre-trained model name (e.g., 'bert-base-uncased')")
    args = parser.parse_args()
    return args

# load and process the data 
def load_and_process_data(train_loader):
    X_train, Y_train = [], []
    for sample in train_loader:
        X_train.append(sample["text"])
        Y_train.append(sample["label"])
    return X_train, Y_train

# check whether gpu is available
def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device


def train_model(model, train_loader, dev_loader, optimizer, loss_function, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0 # initialize total loss
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_function(outputs.logits, labels)
          
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # calculate the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {average_loss:.4f}")

        best_macro_f1 = 0.0
        best_model_epoch = 0

        # evaluate the model on the dev set
        model.eval()
        with torch.no_grad():
            all_true_labels = []
            all_predicted_labels = []
            for batch in dev_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # collect true labels and predicted labels for F1 score calculation
                true_labels = labels.cpu().numpy()
                predicted_labels = (torch.sigmoid(outputs.logits) > 0.5).cpu().numpy()
                all_true_labels.extend(true_labels)
                all_predicted_labels.extend(predicted_labels)

            # calculate the macro F1 score for the epoch
            macro_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {average_loss:.4f}, Macro F1: {macro_f1:.4f}")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model_epoch = epoch + 1
                # best_model_save_path = f'best_model.pth'
                # torch.save(model.state_dict(), best_model_save_path)
                # print(f"Saved best model to {best_model_save_path}")

            class_report = classification_report(all_true_labels, all_predicted_labels, target_names=['class_0', 'class_1'])
            print(class_report)

    print(f"Best model based on Macro F1 Score: Epoch {best_model_epoch}, Macro F1 Score: {best_macro_f1:.4f}")
  
def test_model(model, test_loader, device,  model_name, data_file_name):
    model.to(device)
    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # collect true labels and predicted labels for evaluation
            true_labels = labels.cpu().numpy()
            predicted_labels = (torch.sigmoid(outputs.logits) > 0.5).cpu().numpy()
            all_true_labels.extend(true_labels)
            all_predicted_labels.extend(predicted_labels)

    # calculate evaluation metrics 
    macro_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    class_report = classification_report(all_true_labels, all_predicted_labels, target_names=['class_0', 'class_1'])

    print(f"Macro F1 on test set: {macro_f1:.4f}")
    print(class_report)

    # create the output directory if it doesn't exist
    output_directory = 'predicted_labels_google'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # save the predicted labels to an output file with model name and data file name
    output_file = f"predicted_labels_{model_name}_{data_file_name}.csv"
    pd.DataFrame(all_predicted_labels, columns=['Predicted_Label']).to_csv(output_file, index=False)
    print(f"Predicted labels saved to {output_file}")


def main():
    args = create_arg_parser()

    # initialize BertDataLoader for training data
    train_loader = BertDataLoader(args.train_file)
    dev_loader = BertDataLoader(args.dev_file)

    X_train, Y_train = load_and_process_data(train_loader)
    X_dev, Y_dev = load_and_process_data(dev_loader)

    # define tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # tokenize data
    tokens_train = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False)
    tokens_dev = tokenizer(X_dev, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False)

    # convert labels to numeric format 
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)

    # define data loaders
    batch_size = 64
    train_dataset = TensorDataset(tokens_train["input_ids"], tokens_train["attention_mask"], torch.tensor(Y_train_bin, dtype=torch.float32))
    dev_dataset = TensorDataset(tokens_dev["input_ids"], tokens_dev["attention_mask"], torch.tensor(Y_dev_bin, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_function = torch.nn.BCEWithLogitsLoss()

    # define other training parameters like num_epochs and device
    num_epochs = 3
    device = check_gpu()  # implement the check_gpu function to choose CPU or GPU

    # train model
    train_model(model, train_loader, dev_loader, optimizer, loss_function, num_epochs, device)

    if args.test_file:
        # load and process the test data
        test_loader = BertDataLoader(args.test_file)
        X_test, Y_test = load_and_process_data(test_loader)

        # tokenize the test data
        tokens_test = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False)

        # convert labels to numeric format 
        Y_test_bin = encoder.transform(Y_test)

        # create a test data loader
        test_dataset = TensorDataset(tokens_test["input_ids"], tokens_test["attention_mask"], torch.tensor(Y_test_bin, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # call test_model and save labels
        test_model(model, test_loader, device, model_name, args.test_file)

if __name__ == '__main__':
    main()


