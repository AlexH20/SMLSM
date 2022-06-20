import pandas as pd
import json
import string
from nltk.stem.porter import *
import wrds
from datetime import datetime
from google.colab import drive 
stemmer = PorterStemmer()
exclude = set(string.punctuation)

#The following code cleans and processes the raw NASDAQ news article text dataset
#Code to clean phrases and words is based on Frankel, Jennings and Lee (2021)

db = wrds.Connection(wrds_username="YOUR USERNAME")

json_file_path = "JSON DATA FILE PATH"

#Choose between removing stopwords or not
#stopwords = ['a','able','across','after','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','let','like','likely','me','my','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']
stopwords = []

stopwords_dict={}
for stopword in stopwords:
    stopwords_dict[stopword]=0

def fix_phrases(section):
    section = re.sub('(\d+)\.(\d+)','\g<1>\g<2>',section) # Remove periods from numbers -- 4.55 --> 455
    section = section.replace(".com", "com")
    section = section.replace("-", " ")
    section = section.replace('. .', '.')
    section = section.replace('.', 'XXYYZZ1')
    section = ''.join(ch for ch in section if ch not in exclude) #Delete all punctuation except periods
    section = section.replace('XXYYZZ1', '.')
    section = section.lower()
    section = re.sub(' +',' ',section) #Remove multiple spaces
    if section == '.': section = ''
    return section

def fixword(word, portstem = True):
        word = word.replace('\n',' ')
        if re.search('[0-9]',word) != None:
            word = '00NUMBER00' # Replace numbers with 000NUMBER000
        try:
            test = stopwords_dict[word]
            word = '_' # Replace stop words with _
        except Exception:
            donothing = 1
        #Variable if stemming or not
        if portstem:
            try:
                word = stemmer.stem(word)  # Stemp words
            except Exception:
                word = ''
        return word

def split_years(dt):
    dt["Year"] = dt["Date"].dt.year
    return [dt[dt["Year"] == y] for y in dt["Year"].unique()]

def split_weeks(dt):
    dt["Week"] = dt["Date"].dt.isocalendar().week
    return [dt[dt["Week"] == y] for y in dt["Week"].unique()]

with open(json_file_path, 'r') as json_file:

    data_fill = []

    count_ticker1_only = 0
    count_permno_match = 0

    for i, line in enumerate(json_file):

        data = json.loads(line)

        try:
            date = data["article_time"]["$date"]
        except KeyError:
            continue

        try:
            ticker = data["symbols"]
        except KeyError:
            continue

        try:
            text = data["article_content"]
        except KeyError:
            continue

        if len(ticker.split(sep=",")) == 1 and ticker != "":

            count_ticker1_only += 1

            # Change date to proper format without T and Z
            date = list(date)
            date = [w.replace("T", " ") for w in date]
            date = [w.replace("Z", " ") for w in date]
            date = "".join(date)
            date_str = pd.to_datetime(date.split(" ")[0])
            date = pd.to_datetime(date)
            date_str = date_str.strftime("%m/%d/%Y")

            #Keep text unprocessed for BERT 
            text_unprocessed = text

            #Pre-processing text to reduce noise
            text = fix_phrases(text)
            sentences = text.split('.')

            count_words = 0

            for v, sentence in enumerate(sentences):

                sentences[v] = sentences[v].replace(".", "").strip()

                allwords = sentences[v].split(" ")

                for w, word in enumerate(allwords):
                    allwords[w] = fixword(allwords[w], portstem=False)
                    if allwords[w].strip() != '.' and allwords[w].strip() != '':
                        count_words += 1

                sentences[v] = " ".join(allwords)

            text = ".".join(sentences)
            text = text.replace(".", ". ")

            #Get permno of company for fetching data from wrds. If no PERMNO match then observation will not be used in sentiment analysis.

            permno = db.raw_sql("""select permno
                                            from crsp.dse
                                            where TICKER = '{}'
                                            and date  <= '{}'""".format(ticker, date_str))

            try:
                permno["permno"].iloc[-1]
                count_permno_match += 1
            except IndexError:
                continue

            data_fill.append([date, ticker, text, text_unprocessed ,count_words, permno["permno"].iloc[-1]])


    data = pd.DataFrame(data_fill, columns = ["Date", "Ticker", "Text", "Text_unprocessed","Word Count", "Permno"])

    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data.sort_values(by="Date", inplace=True)
    data.drop_duplicates(inplace=True)
    dropped_duplicates = len(data)

    data_splt_years = split_years(data)

    sorteddflist = sorted(data_splt_years, key=lambda x: x["Year"].min(axis=0))

    sorted_final = []

    for year in sorteddflist:
        year = split_weeks(year)
        year_sorted = sorted(year, key=lambda x: x["Week"].min(axis=0))
        sorted_final.append(pd.concat(year_sorted))

    df = pd.concat(sorted_final)
    
#References:
#Frankel, R., Jennings, J., and Lee, J. (2021). Disclosure sentiment: Machine learning vs. dictionary methods. Management Science

