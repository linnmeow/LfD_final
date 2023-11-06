import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class BertDataLoader(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.texts = [item[0] for item in self.data]
        self.labels = [item[1] for item in self.data]
        
        # encode labels to integers
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:  
                    data.append((line[0], line[1]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "text": self.texts[idx],
            "label": self.labels[idx]
        }
        return sample

# create a DataLoader
def create_data_loader(data_path, batch_size):
    dataset = BertDataLoader(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
