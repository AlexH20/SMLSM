import csv
import sys
from scipy.sparse import csc_matrix, vstack
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

#Create a memory efficient sparse matrix format. Input is the dictionary returned by n_grams function. 
#Code from paper Frankel, Jennings and Lee (2021) (modified for own needs)

def sparse_mat(data):
    row1 = []
    col1 = []
    data1 = []

    #Iterate through dictionary from get_sparsematrix_and_car to create sparse matrix (value in interation are all ngram counts of observation 'key').

    for key, value in data.items():

        value_n = list(value.values())

        for e, elem in enumerate(value_n):
            colnum = e
            value = elem

            row1.append(key)
            col1.append(colnum)
            data1.append(value)

    X = csc_matrix((data1, (row1, col1)))  # Sparse matrix of rows (observations) and columns (independent variables)

    return X

#Function to get one- and two-gram counts of text documents. This function will iterate over all text items in the dataset and return all unique one- and twograms.
#Code from paper Frankel, Jennings and Lee (2021) (modified for own needs)

def get_ngrams(data):

    onegrams = []
    twograms = []

    for index, row in data.iterrows():

        sentences = row["Text"].split('.')

        #### EXTRACT ALL ONE AND TWO WORD PHRASES #### Frankel, Jennings and Lee (2021)

        for sentence in sentences:

            sentence = sentence.replace('.', '').strip()

            allwords = sentence.split(' ')

            for w, word in enumerate(allwords):
                word0 = allwords[w]
                try:
                    word1 = allwords[w + 1]
                except Exception:
                    word1 = ''

                if word0.strip() != '.' and word0.strip() != '':
                    onegrams.append(word0)

                    if word1.strip() != '.' and word1.strip() != '':
                        twogram = word0 + ' ' + word1
                        twograms.append(twogram)

    n_grams_dict = {}

    uniqueonegrams = list(set(onegrams))
    uniqueonegrams = sorted(uniqueonegrams)

    uniquetwograms = list(set(twograms))
    uniquetwograms = sorted(uniquetwograms)

    ngrams = uniqueonegrams + uniquetwograms

    return ngrams

#The function get_sparsematrix_and_car 


#The function get_sparsematrix_and_car uses the get_ngrams and sparse_mat function to create inputs for the Random Forest algorithm
#The inputs are either twice the training dataset or once the training dataset and once the test dataset. 
#The function first extracts all one- and two-grams of the training dataset via the get_ngrams function. Next, the function iterates over each text document of the second input data file to count occurences of one- and two-grams found in the first input file. 

def get_sparsematrix_and_car(data_train, data_test):

    #Get one- and two-grams of data_train
    ngram_list = get_ngrams(data_train)

    wrd_list = ngram_list

    wrd_list = sorted(wrd_list)
    wrd_list = tuple(wrd_list)

    #Initialize dependent variable list (CAR)
    car = []

    #Initialize dictionary with pre-determined number of txt files. Later used for memory reasons.
    wrd_dictionary = dict.fromkeys(range(67))

    i = 0
    j = 0

    for index, row in data_test.iterrows():

        car.append(row["AR"])

        sentences = row["Text"].split('.')

        print(j)

        # Initialize dictionary within dictionary with keys according to all ngrams found in training dataset.
        wrd_dictionary[i] = dict.fromkeys(wrd_list, 0)

        #Count one- and two-grams found in data_train within data_test
        for sentence in sentences:

            sentence = sentence.replace('.', '').strip()
            allwords = sentence.split(' ')

            for w, word in enumerate(allwords):
                word0 = allwords[w]
                try:
                    word1 = allwords[w + 1]
                except Exception:
                    word1 = ''

                # Add count of found ngrams occurence to dictionary
                if word0.strip() != '.' and word0.strip() != '':
                    if word0 in wrd_dictionary[i].keys():
                        wrd_dictionary[i][word0] = wrd_dictionary[i][word0] + 1

                    if word1.strip() != '.' and word1.strip() != '':
                        if word0 + ' ' + word1 in wrd_dictionary[i].keys():
                            wrd_dictionary[i][word0 + ' ' + word1] = wrd_dictionary[i][word0 + ' ' + word1] + 1

        i += 1
        j += 1
        
        #The following code creates a memory-efficient sparse matrix format every 67 observation and then vstacks them to already exisiting sparse matrices.
        #i and j necessary due to RAM overload. i serves as marker to create a sparse matrix every n observations.
        #j serves as marker to concatenate the sparse matrices and to stop the for loop when iterated of all items in data_test. In last iteration all empty key-value pairs of the dictionary need to be deleted

        if j == len(data_test):
            keys_to_remove = (j % 67)
            for key in range(keys_to_remove, 67):
                del wrd_dictionary[key]
            spar_mat_i = sparse_mat(wrd_dictionary)
            spar_mat = vstack((spar_mat, spar_mat_i))
            break
        
        if i % 67 == 0:
            spar_mat_i = sparse_mat(wrd_dictionary)
            if j != 67:
              spar_mat = vstack((spar_mat, spar_mat_i))
              wrd_dictionary = dict.fromkeys(range(67))
              i = 0
            else:
              spar_mat = spar_mat_i
              wrd_dictionary = dict.fromkeys(range(67))
              i = 0

    return spar_mat, car

#Function to split dataset based on months.
def split_months(dt):
    return [dt[dt["ordered_month"] == y] for y in dt["ordered_month"].unique()]

data = pd.read_csv("DATA FILEPATH")

#Get rid of all observations without news articles
data_onlytext = data.dropna()

#Prepare data for sliding window approach. Ordered Month dataframe column has value 1 for first month of dataset and e.g. 20 for 20th month of dataset
data_onlytext["Date"] = pd.to_datetime(data_onlytext["Date"])
data_onlytext["Year"] = [x.year for x in data_onlytext["Date"]]
data_onlytext["Month"] = [x.month for x in data_onlytext["Date"]]
data_onlytext["ordered_month"] = [((x[1]["Year"]-2015)*12 + x[1]["Month"]) for x in data_onlytext.iterrows()]

data_splt_months = split_months(data_onlytext)

i = -1

np.random.seed(9000)
#Sliding window approach
for _, month in enumerate(data_splt_months):

        i += 1

        data_train = pd.concat([data_splt_months[i], data_splt_months[i+1], data_splt_months[i+2]])
        data_test = data_splt_months[i+3]

        #Get sparse matrices as input for ML models
        X_train, y_train = get_sparsematrix_and_car(data_train, data_train)
        X_test, y_test = get_sparsematrix_and_car(data_train, data_test)

        #Random Forest algorithm
        rf = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs=-1)
        rf = rf.fit(X_train, y_train)
        pred = rf.predict(X_test).tolist()
