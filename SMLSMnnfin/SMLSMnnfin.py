import numpy as np
import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from google.colab import drive 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv

drive.mount("/content/gdrive")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['AR'].tolist()
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-tone')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output.view(-1, 1).squeeze(1)

def train(model, train_data, learning_rate, epochs):

    train = Dataset(train_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.float())
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .10f}')
            

def evaluate(model, test_data):

    test = Dataset(test_data)

    prediction = []

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              prediction.append(output.tolist())
    
    return sum(prediction, [])

def split_months(dt):
    return [dt[dt["ordered_month"] == y] for y in dt["ordered_month"].unique()]

data = pd.read_csv("gdrive/My Drive/Thesis/processed data/CAR_regression/datasets_final/data_whole_woScAR.csv", index_col = False)
print(data.columns)

data_onlytext = data[data["word_count"] != 0]
data_onlytext["Date"] = pd.to_datetime(data_onlytext["Date"])
data_onlytext["Year"] = [x.year for x in data_onlytext["Date"]]
data_onlytext["Month"] = [x.month for x in data_onlytext["Date"]]
data_onlytext["ordered_month"] = [((x[1]["Year"]-2015)*12 + x[1]["Month"]) for x in data_onlytext.iterrows()]


tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
EPOCHS = 100
LR = 1e-5
             
data_splt_months = split_months(data_onlytext)

i = 51

np.random.seed(9000)
for _, month in enumerate(data_splt_months):

        i += 1

        data_train = pd.concat([data_splt_months[i], data_splt_months[i+1], data_splt_months[i+2]])
        data_test = data_splt_months[i+3]

        model = BertClassifier()
        train(model, data_train, LR, EPOCHS)
        pred = evaluate(model, data_test)

        print(mean_squared_error(data_test["AR"], pred))

        if i+1 < len(data_splt_months):

            data_splt_months[i+3]["AR_FinNN"] = pred

            with open("gdrive/My Drive/Thesis/processed data/CAR_regression/FinBERTNN_sentiment/" + str(i+1) + ".csv", "w") as csv_file:
                  
                  writer = csv.writer(csv_file)
                  writer.writerow(
                      ['Date', 'Ticker', 'Nasdaq', 'Turnover', 'Size', 'BTM', 'pref_alpha','Text', 'word_count', 'AR', 'Return', 'AR_FinNN'])
                  for index, row in data_splt_months[i+3].iterrows():
                      writer.writerow([row["Date"], row["Ticker"], row["Nasdaq"], row["Turnover"], row["Size"], row["BTM"], row["pref_alpha"], row["Text"], row["word_count"], row["AR"], row["Return"], row["AR_FinNN"]])

