import pandas as pd
import csv
from datetime import datetime, timedelta, time
from google.colab import drive 
import sys
import re
import math
import holidays
import pandas_market_calendars as mcal

#This code merges the output files of SMLSMgetpanel.py and SMLSMprocessraw.py

#Function to concatenate text
def concatenate_text(g):
    return ' '.join(g.Text)
  
#Function to count words of text using regex
def count_words(Text):
    word_count = len(re.findall(r'\w+', Text))
    return word_count

# Function to check whether date is on a non-trading day or news article is published later than 4pm. If so, then return next trading day.
holidaysUS = holidays.US()
nyse = mcal.get_calendar('NYSE')
stock_holidays = nyse.holidays()

stock_holidays = list(pd.to_datetime(stock_holidays.holidays))
stock_holidays = [x.date() for x in stock_holidays]

def check_tradingdayhour(day):
    trading_day = day
    if trading_day.hour >= 16:
        trading_day += timedelta(1)
    #Check if news article published on weekend. If so return monday. If day on holiday return next trading day
    while trading_day.weekday() in holidays.WEEKEND or trading_day in stock_holidays:
        trading_day += timedelta(1)
    return trading_day

data_text = pd.read_csv("DATA FILE PATH OF SMLSMprocessraw.py output")

df = pd.read_csv("DATA FILE PATH OF SMLSMgetpanel.py output")

data_text["Date"] = pd.to_datetime(data_text["Date"])
data_text["Date"] = [check_tradingdayhour(x) for x in data_text["Date"]]
data_text["Date"] = [x.strftime("%Y-%m-%d") for x in data_text["Date"]]

df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = [check_tradingdayhour(x) for x in df["Date"]]
df["Date"] = [x.strftime("%Y-%m-%d") for x in df["Date"]]

#Merge rows with same ticker symbol and date
#This can be done either for the processed text or unprocessed text. Adjust variable in concatenate_text function
new_df = data_text.groupby(["Date", "Ticker"]).apply(concatenate_text).to_frame(name = "c_Text")
df = df.merge(new_df, how= "left", left_on = ["Date", "Ticker"], right_on = ["Date", "Ticker"])

df["c_Text"] = df["c_Text"].fillna(" ")

df["word_count"] = [count_words(x) for x in df.c_Text]

df.drop_duplicates(inplace = True)
df = df.rename(columns={'c_Text': 'Text'})
df_columns = ["Date", "Ticker", "Nasdaq", "Turnover", "Size", "BTM","pref_alpha", "Text", "word_count", "AR", "Return"] 
df = df[df_columns]
