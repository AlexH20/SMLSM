import pandas as pd
import pysentiment2 as ps
import re

#The following code extracts the net tone sentiment measures based on the Harvard and LM dictionary. Both dictionaries were accessed through the
#package Pysentiment2

#Function that counts words of text article documents
def count_words(Text):
    word_count = len(re.findall(r'\w+', Text))
    return word_count

data = pd.read_csv("DATA FILEPATH")
data_onlytext = data[data["word_count"] != 0]

#Initial dictionary methods. Harvard and LM dms will be used to assess text sentiment
hiv4 = ps.HIV4()
lm = ps.LM()

data_fill = []

for index, row in data.iterrows():

    #Use texts in csv file to tokenize and get counts of positive and negative words according to HIV4 and LM
    text = row["Text"]
    word_count = row["word_count"]

    tokens_hiv4 = hiv4.tokenize(text)
    tokens_lm = lm.tokenize(text)
    score_hiv4 = hiv4.get_score(tokens_hiv4)
    score_lm = lm.get_score(tokens_lm)

    hiv4_pos = score_hiv4["Positive"]
    hiv4_neg = score_hiv4["Negative"]
    hiv4_pos_tone = hiv4_pos / word_count
    hiv4_neg_tone = hiv4_neg / word_count
    hiv4_net_tone = (hiv4_pos - hiv4_neg) / (hiv4_pos + hiv4_neg)

    lm_pos = score_lm["Positive"]
    lm_neg = score_lm["Negative"]
    lm_pos_tone = lm_pos / word_count
    lm_neg_tone = lm_neg / word_count
    lm_net_tone = (lm_pos - lm_neg) / (lm_pos + lm_neg)
    
    data_fill.append([hiv4_pos_tone, hiv4_neg_tone, hiv4_net_tone, lm_pos_tone, lm_neg_tone, lm_net_tone])
    
dm_sentiment = pd.DataFrame(data_fill, columns = ["HIV4_pos", "HIV4_neg", "HIV4_tone", "LM_pos", "LM_neg", "LM_tone"])

#References:
#DeRobertis, N. (2020). Pysentiment2 0.1.1.
