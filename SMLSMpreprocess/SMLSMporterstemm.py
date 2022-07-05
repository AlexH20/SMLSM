import pandas as pd
import string
from nltk.stem.porter import *
stemmer = PorterStemmer()
exclude = set(string.punctuation)

#The following code porter stems words based on the porter stemm algorithm and removes stopwords (Porter 1980)
#Code from or based on online appendix by Frankel, Jennings and Lee (2021)

data = pd.read_csv("gdrive/My Drive/Thesis/processed data/CAR_regression/datasets_final/data_whole_woScAR.csv", index_col = False)
stopwords = ['a','able','across','after','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','let','like','likely','me','my','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

stopwords_dict={}
for stopword in stopwords:
    stopwords_dict[stopword]=0

def fixword(word, portstem = True):
        word = word.replace('\n','')
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

            
for i, text in enumerate(data.Text):
      print(i)
      
      sentences = text.split('.')

      for v, sentence in enumerate(sentences):

            sentences[v] = sentences[v].replace(".", "").strip()

            allwords = sentences[v].split(" ")

            for w, word in enumerate(allwords):
                allwords[w] = fixword(allwords[w], portstem=True)

            sentences[v] = " ".join(allwords)

      data.Text[i] = ".".join(sentences)
      
#References:
#Porter, M. F. (1980). An algorithm for suffix stripping. Program.
#Frankel, R., Jennings, J., and Lee, J. (2021). Disclosure sentiment: Machine learning vs. dictionary methods. Management Science.
