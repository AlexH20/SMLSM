import pandas as pd
from google.colab import drive
import csv
import numpy as np
import matplotlib.pyplot as plt

#Part of processed data
data_aapl = pd.read_csv("data_whole_AAPL2016.csv")

#DM sentiment measures
sentiment_measures = ["HIV4 TONE", "LM TONE"]
colors = {"HIV4 TONE":"black", "LM TONE":"green"}

fig1, ax1 = plt.subplots(figsize = (10, 10))

for sentiment in sentiment_measures:
  ax1.plot(data_aapl["Date"], data_aapl[sentiment], c = colors[sentiment], linewidth = 0.7)

plt.xticks(fontsize = 20,rotation = 45)
plt.yticks(fontsize = 20)
plt.show()

#RF sentiment measures
sentiment_measures = ["r_{ab} RF12", "r_{ab} RFFin"]
colors = {"r_{ab} RF12":"red", "r_{ab} RFFin":"blue"}

fig1, ax1 = plt.subplots(figsize = (10, 10))

for sentiment in sentiment_measures:
  ax1.plot(data_aapl["Date"], data_aapl[sentiment], c = colors[sentiment], linewidth = 0.7)

plt.xticks(fontsize = 20,rotation = 45)
plt.yticks(fontsize = 20)
plt.show()

#All SML sentiment measures and abnormal returns
sentiment_measures = ["r_{ab}", "r_{ab} RF12", "r_{ab} RFFin", "r_{ab} FinNN"]
colors = {"r_{ab}":"gray", "r_{ab} RF12":"red", "r_{ab} RFFin":"blue", "r_{ab} FinNN": "aquamarine"}

fig1, ax1 = plt.subplots(figsize = (10, 10))

for sentiment in sentiment_measures:
  ax1.plot(data_aapl["Date"], data_aapl[sentiment], c = colors[sentiment],label = sentiment)

plt.xticks(fontsize = 20,rotation = 45)
plt.yticks(fontsize = 20)
plt.show()
