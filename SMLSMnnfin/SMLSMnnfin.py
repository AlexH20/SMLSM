import numpy as np
import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd

#The following code uses FinBERT as an encoder, and uses the pooler output as input for the dropout & dense layer 
#Dataset class, FinBERT class and get_pooleroutput function based on Ruben Winastwan (2021)

#Class to tokenize text dataset and prepare inputs in form of batches for FinBERT
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['AR'].tolist()
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Text_unprocessed']]

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

#FinBERT class. Extract hidden state of CLS token further processed by Linear Layer and tanh activiation function and then passed to Dropout & Dense Layer
class FinBERT(nn.Module):

    def __init__(self, dropout=0.1):

        super(FinBERT, self).__init__()

        #FinBERT model pre-trained and additionally fine-tuned on 10,000 manually annotated sentences from analyst reports. Without classification layer
        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output.view(-1, 1).squeeze(1)

#Train model. For hyperparameter tuning, insert train test split before Dataset class call and create validation set
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
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f}')
            
#Out-of-sample prediction with trained model
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

#Function to split dataset based on months
def split_months(dt):
    return [dt[dt["ordered_month"] == y] for y in dt["ordered_month"].unique()]

#Text tokenizer 
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
EPOCHS = 18
LR = 1e-4
             
data_splt_months = split_months(data_onlytext)

i = -1

np.random.seed(9000)
#Sliding window approach
for _, month in enumerate(data_splt_months):

        i += 1

        data_train = pd.concat([data_splt_months[i], data_splt_months[i+1], data_splt_months[i+2]])
        data_test = data_splt_months[i+3]

        model = FinBERT()
        train(model, data_train, LR, EPOCHS)
        pred = evaluate(model, data_test)

#References:
#Ruben Winastwan (2021) https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f, accessed 18.06.2022 
#Huggingface FinBERT model: https://huggingface.co/yiyanghkust/finbert-tone, accessed 18.06.2022
#Yang, Y., Uy, M. C. S., and Huang, A. (2020). Finbert: A pretrained language model for financial communications. arXiv preprint arXiv:2006.08097.


