import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm  # progress bar
import copy  # perform deep copyong rather than referencing in python
import multiprocessing  # for threading of word2vec model process
from sklearn.preprocessing import MinMaxScaler
# importing classes helpfull for text processing

import matplotlib.pyplot as plt  # data visualization
from nltk.corpus import stopwords  # to delete the stopwords
from nltk import word_tokenize  # for the tokenization
from tqdm._tqdm_notebook import tqdm_notebook
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import mean_squared_error
# split data into test and train datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import pickle
from sklearn.metrics import classification_report
import numpy as np

np.random.seed(0)
import datetime


class stocks_sentiments_correlation:

    # calculate daily sentiment score using average
    @staticmethod
    def calculate_day_sentiment_average(data):
        dates = pd.DataFrame()
        dates['dates'] = data.Date.unique()
        dates['sentiment'] = np.zeros(len(dates.dates))
        sentiment_score = 0
        numb_row = 0
        sentiment_so = []
        for p in dates['dates']:
            for i in range(len(data.Date)):
                if data.Date[i] == p:
                    sentiment_score += int(data.sentiment[i])
                    numb_row = numb_row + 1

            sentiment_so.append(sentiment_score / numb_row)
            numb_row = 0
            sentiment_score = 0

        for s in sentiment_so:
            for i in range(len(sentiment_so)):
                dates['sentiment'][i] = sentiment_so[i]

        return dates

    # calculate daily sentiment score
    @staticmethod
    def calculate_daily_sentiment(data):
        dates = pd.DataFrame()
        dates['dates'] = data.Date.unique()
        dates['sentiment'] = np.zeros(len(dates['dates']))
        sentiment_score = 0

        sentiment_so = []
        for p in dates['dates']:
            for i in range(len(data.Date)):
                if data.Date[i] == p:
                    sentiment_score += int(data.sentiment[i])

            sentiment_so.append(sentiment_score)
            sentiment_score = 0

        for s in sentiment_so:
            for i in range(len(sentiment_so)):
                dates['sentiment'][i] = sentiment_so[i]

        return dates

    # compare the current day with the next day and add a column for that (never used)
    @staticmethod
    def add_updown_label(stocks_data):
        stocks_data['status'] = np.zeros(len(stocks_data['Date']))
        for i in range(len(stocks_data['Date']) - 1):
            if float(stocks_data['Close'][i]) < float(stocks_data['Close'][i + 1]):
                stocks_data['status'][i + 1] = 1
            elif float(stocks_data['Close'][i]) > float(stocks_data['Close'][i + 1]):
                stocks_data['status'][i + 1] = -1
            else:
                stocks_data['status'][i] = 0

        return stocks_data

    # compare the next day with the current day if the Close value for the next day is higher than the current day add a value of 1 otherwise -1 (used for the training)
    @staticmethod
    def add_updown_label_tomorrow(stocks_data):
        stocks_data['status_tomorrow'] = np.zeros(len(stocks_data['Date']))
        for i in range(len(stocks_data['Date']) - 1):
            if float(stocks_data['Close'][i + 1]) < float(stocks_data['Close'][i]):
                stocks_data['status_tomorrow'][i] = -1
            elif float(stocks_data['Close'][i + 1]) > float(stocks_data['Close'][i]):
                stocks_data['status_tomorrow'][i] = 1
            else:
                stocks_data['status_tomorrow'][i] = 0

        return stocks_data

    # add sentiment score to stocks prices in a separete column (NB: the first date of sentiments should be the same as stock prices)
    # using normal approach
    @staticmethod
    def add_stocks_to_sentiments_score(sentiment_score_dt, stocks_prices_dt):
        stocks_prices_dt['sentiment_score'] = np.zeros(len(stocks_prices_dt['Date']))
        sentiment_score_dt = stocks_sentiments_correlation.calculate_daily_sentiment(sentiment_score_dt)
        # sentiment_score_dt['dates'] = pd.to_datetime(sentiment_score_dt['dates'])
        # stocks_prices_dt['Date'] = pd.to_datetime(stocks_prices_dt['Date'])
        i = 0
        for date in stocks_prices_dt['Date']:
            temp_index = sentiment_score_dt['dates'].loc[(sentiment_score_dt['dates'] == date) == True]
            temp_index.dropna(inplace=True)
            stocks_prices_dt['sentiment_score'][i] = sentiment_score_dt['sentiment'][temp_index.index.values[0]]
            i = i + 1

        return stocks_prices_dt

    # using normal average approach
    @staticmethod
    def add_stocks_to_sentiments_score_average(sentiment_score_dt, stocks_prices_dt):
        stocks_prices_dt['sentiment_score'] = np.zeros(len(stocks_prices_dt['Date']))
        sentiment_score_dt = stocks_sentiments_correlation.calculate_day_sentiment_average(sentiment_score_dt)
        # sentiment_score_dt['dates'] = pd.to_datetime(sentiment_score_dt['dates'])
        # stocks_prices_dt['Date'] = pd.to_datetime(stocks_prices_dt['Date'])
        i = 0
        for date in stocks_prices_dt['Date']:
            temp_index = sentiment_score_dt['dates'].loc[(sentiment_score_dt['dates'] == date) == True]
            temp_index.dropna(inplace=True)
            stocks_prices_dt['sentiment_score'][i] = sentiment_score_dt['sentiment'][temp_index.index.values[0]]
            i = i + 1

        return stocks_prices_dt
