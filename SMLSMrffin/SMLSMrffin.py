import numpy as np
import torch
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from tqdm import tqdm
import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor

#The following code uses FinBERT as an encoder, and uses the pooler output as input for the Random Forest algorithm. 
#Dataset class, FinBERT class and get_pooleroutput function based on Ruben Winastwan (2021)

#Class to tokenize text dataset and prepare inputs in form of batches for FinBERT
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

#FinBERT class. Extract hidden state of CLS token further processed by Linear Layer and tanh activation function (pooler output)
class FinBert(nn.Module):

    def __init__(self, dropout=0.1):

        super(FinBERT, self).__init__()

        #FinBERT model pre-trained and additionally fine-tuned on 10,000 manually annotated sentences from analyst reports. Without classification layer
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone")

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        return pooled_output.tolist()

#Gets pooler output of train and test data
def get_pooleroutput(model, train_data):

    train = Dataset(train_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

            model = model.cuda()

    frames = []

    for train_input, train_label in tqdm(train_dataloader):

        train_label = train_label.to(device)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        df = pd.DataFrame(model(input_id, mask))

        frames.append(df)

    df_final = pd.concat(frames)
    df_final = df_final.reset_index()

    return df_final

#Function to split dataset based on months
def split_months(dt):
    return [dt[dt["ordered_month"] == y] for y in dt["ordered_month"].unique()]

data = pd.read_csv("DATA FILEPATH")

#Get rid of all observations without news articles
data_onlytext = data[data["word_count"] != 0]

#Prepare data for sliding window approach. Ordered Month dataframe column has value 1 for first month of dataset and e.g. 20 for 20th month of dataset
data_onlytext["Date"] = pd.to_datetime(data_onlytext["Date"])
data_onlytext["Year"] = [x.year for x in data_onlytext["Date"]]
data_onlytext["Month"] = [x.month for x in data_onlytext["Date"]]
data_onlytext["ordered_month"] = [((x[1]["Year"]-2015)*12 + x[1]["Month"]) for x in data_onlytext.iterrows()]

#Text tokenizer 
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
  
data_splt_months = split_months(data_onlytext)

i = -1

#Sliding window approach
for _, month in enumerate(data_splt_months):
    
        np.random.seed(9000)

        i += 1

        data_train = pd.concat([data_splt_months[i], data_splt_months[i+1], data_splt_months[i+2]])
        data_test = data_splt_months[i+3]

        model = FinBERT()
        X_train_hidden = get_pooleroutput(model, data_train)
        X_test_hidden = get_pooleroutput(model, data_test)

        y_train = data_train["AR"]

        #Random Forest
        rf = RandomForestRegressor(n_estimators=1000, n_jobs = -1, max_features = 2/3)
        rf = rf.fit(X_train_hidden, y_train)
        pred = rf.predict(X_test_hidden).tolist()
        
#References:
#Ruben Winastwan (2021) https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f, accessed 18.06.2022 
#Huggingface FinBERT model: https://huggingface.co/yiyanghkust/finbert-tone, accessed 18.06.2022
#Yang, Y., Uy, M. C. S., and Huang, A. (2020). Finbert: A pretrained language model for financial communications. arXiv preprint arXiv:2006.08097.
