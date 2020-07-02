import datetime
import json
import pickle
import sqlite3
from opencage.geocoder import OpenCageGeocode
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.backend import set_session

from Sentiments_Stocks import settings as sett
import tweepy
import numpy as np
from django.http import JsonResponse
from rest_framework import viewsets

from rest_framework.decorators import api_view
from datetime import date, datetime, timedelta

from Sentiments_Stocks.api import stocks_sentiments_correlation
from .serializers import TwitterUserSerializer, TweetsSerializer, AppUserSerializer
from .models import TwitterUser, Tweets, AppUser
import pandas as pd
from . import sentiment_analysis_class
# 
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


# save the uploaded file
def handle_uploaded_file(f):
    with open('Sentiments_Stocks/file_uploaded.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return chunk


# API to upload a file (csv)
@api_view(["POST"])
def upload_file_test(file):
    # data = pd.read_csv(filepath_or_buffer=file.FILES, sep=',', header=None, engine='python', encoding='latin-1')
    print(file.FILES['file'])
    handle_uploaded_file(file.FILES['file'])
    data = pd.read_csv('Sentiments_Stocks/file_uploaded.csv', engine='python')
    print(data)
    return JsonResponse({"uploaded": "True"})


# API to predict sentiment in a csv file of tweets
@api_view(["POST"])
def getsentimentstweets(file):
    handle_uploaded_file(file.FILES['file'])
    data = pd.read_csv('Sentiments_Stocks/file_uploaded.csv', engine='python', delimiter=',')
    # data = data.mask(data.eq('None')).dropna()
    # data = data.reset_index()
    start = datetime.now()
    model = sett.model
    tokenizer = sett.tokenizer
    print(data)
    data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data)
    pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
        tokenizer, data_test)
    end = datetime.now()
    print('time to make a load for the model is :' + str(end - start))
    start = datetime.now()
    labels = ['1', '-1', '0']
    graph = sett.graph
    sess = sett.sess
    with graph.as_default():
        set_session(sess)
        predicted = model.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])
    end = datetime.now()
    print('time to make a prediction is :' + str(end - start))
    data_test['sentiment'] = prediction_labels
    data_predicted = data_test[['Date', 'cleaned_tweets', 'sentiment']]
    print(str(data_predicted))
    tweets = []
    print('length is : ' + str(len(data_predicted['cleaned_tweets'])))
    for i in range(len(data_predicted['cleaned_tweets'])):
        temp = {}
        temp["Date"] = data_predicted["Date"][i]
        temp["cleaned_tweets"] = data_predicted["cleaned_tweets"][i]
        temp["sentiment"] = data_predicted["sentiment"][i]
        tweets.append(temp)

    print(json.dumps(temp))
    return JsonResponse(json.dumps(tweets), safe=False)


# get stocks values for a stock market
@api_view(["GET"])
def stock_values(request, stockname):
    # add an if for the holidays 2 1
    start_date = str(date.today() - timedelta(days=1))
    end_date = str(date.today())
    data = yf.download(stockname, start=start_date, end=end_date)
    data = data.reset_index()
    data['Date'] = data['Date'].astype(str)
    data['Open'] = data['Open'].astype(str)
    data['High'] = data['High'].astype(str)
    data['Low'] = data['Low'].astype(str)
    data['Close'] = data['Close'].astype(str)
    data['Adj Close'] = data['Adj Close'].astype(str)
    data['Volume'] = data['Volume'].astype(str)
    print(data)
    print(start_date)
    print(end_date)
    tweets = []

    # for i in range(len(data['Date'])):
    #    if data['Date'][i] == str(date.today() - timedelta(days=2)):
    temp = {}
    temp["Date"] = data["Date"][0]
    temp["Open"] = round(float(data["Open"][0]), 4)
    temp["High"] = round(float(data["High"][0]), 4)
    temp["Low"] = round(float(data["Low"][0]), 4)
    temp["Close"] = round(float(data["Close"][0]), 4)
    temp["Adj Close"] = round(float(data["Adj Close"][0]), 4)
    temp["Volume"] = data["Volume"][0]
    tweets.append(temp)
    print('---------------------------------------------')
    print(tweets)
    return JsonResponse(json.dumps(tweets), safe=False)


@api_view(["GET"])
def select_users(request):
    listuser = list(AppUser.objects.values())
    return JsonResponse(listuser, safe=False)


# calculate sentiment score
@api_view(["POST"])
def calculate_sentiment_score(file1):
    with open('Sentiments_Stocks/sentiment_uploaded.csv', 'wb+') as destination:
        for chunk in file1.FILES['file1'].chunks():
            destination.write(chunk)

    with open('Sentiments_Stocks/stocks_uploaded.csv', 'wb+') as destination:
        for chunk in file1.FILES['file2'].chunks():
            destination.write(chunk)

    data = pd.read_csv('Sentiments_Stocks/sentiment_uploaded.csv', engine='python')
    data.dropna(inplace=True)
    stocks_data = pd.read_csv('Sentiments_Stocks/stocks_uploaded.csv', engine='python')
    print('-------------------------------------------')
    print(data)
    print('-------------------------------------------')
    print(stocks_data)
    start = datetime.now()
    model = sett.model
    tokenizer = sett.tokenizer
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data['Date'] = data['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))

    stocks_data['Date'] = pd.to_datetime(stocks_data['Date'], format='%Y/%m/%d')
    stocks_data['Date'] = stocks_data['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))

    data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data)
    pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
        tokenizer, data_test)
    end = datetime.now()
    print('time to make a load for the model is :' + str(end - start))
    start = datetime.now()
    labels = ['1', '-1', '0']
    graph = sett.graph
    sess = sett.sess
    pickle_model = sett.random_forrest_model
    # start sentiment prediction
    with graph.as_default():
        set_session(sess)
        predicted = model.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])
    end = datetime.now()
    print('time to make a prediction is :' + str(end - start))
    data_test['sentiment'] = prediction_labels
    data_predicted = data_test[['Date', 'cleaned_tweets', 'sentiment']]
    data_predicted.to_csv('Sentiments_Stocks/msft.csv')
    start1 = datetime.now()
    stocks_data = stocks_sentiments_correlation.stocks_sentiments_correlation.add_stocks_to_sentiments_score(
        data_predicted, stocks_data)
    end1 = datetime.now()
    print(' time for adding sentiment score to stock price is ' + str(end1 - start1))

    print('hello world')
    print(str(stocks_data['Date']))
    test_cols_x = ["Open", "High", "Low", "Close", "Adj Close", "sentiment_score"]
    x_test = stocks_data.loc[:, test_cols_x].values
    scaler = MinMaxScaler()
    with graph.as_default():
        set_session(sess)
        x_test_ml_scaled = scaler.fit_transform(x_test)
        rf_prediction = pickle_model.predict(x_test_ml_scaled)

    print(str(rf_prediction[0]))
    if str(rf_prediction[0]) == '1.0':
        state = 'rise'
    elif str(rf_prediction[0]) == '-1.0':
        state = 'down'
    elif str(rf_prediction[0]) == '0.0':
        state = 'same'
    return JsonResponse({"results": state})

# method for sentiment distribution un a map
@api_view(["GET"])
def map_sentiment(request, keyword):
    print('map_sentiment start')
    start = datetime.now()
    '''geocoder = OpenCageGeocode(key)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    graph = sett.graph
    sess = sett.sess
    model = sett.model
    tokenizer = sett.tokenizer
    tweets = pd.DataFrame()
    for tweet in tweepy.Cursor(api.search, q=keyword + ' -filter:retweets', lang='en', wait_on_rate_limit=True,
                               count=400, include_rts=False).items(100):
        tweets1 = pd.DataFrame(np.zeros(1))
        query = tweet.user.location
        results = geocoder.geocode(query)
        tweets1["username"] = tweet.user.screen_name
        temp_string = str(tweet.text).replace('\n', '')
        temp_string = str(temp_string).replace('"', '')
        tweets1["cleaned_tweets"] = str(temp_string).replace(',', '')
        tweets1["Date"] = tweet.created_at
        tweets1["Date"] = tweets1["Date"].astype(str)
        if results:
            # print(results[0]['components']['country_code'])
            tweets1["location"] = results[0]['components']['country_code']
            results = ''
        else:
            results = ''
        tweets = tweets.append(tweets1)
    
    data = tweets[['username', 'cleaned_tweets', 'location', 'Date']]

    labels = ['1', '-1', '0']

    data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data)
    pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
        tokenizer, data_test)
    # start sentiment prediction
    with graph.as_default():
        set_session(sess)
        predicted = model.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])

    data_test['sentiment'] = prediction_labels
    data_predicted = data_test[['Date', 'cleaned_tweets', 'location', 'sentiment']]
    data_predicted.to_csv('data_location.csv')'''
    if keyword == 'MSFT':
        data_predicted = pd.read_csv('data_location_MSFT.csv')
    elif keyword == 'AAPL':
        data_predicted = pd.read_csv('data_location_Apple.csv')
    elif keyword == 'GOOGL':
        data_predicted = pd.read_csv('data_location_Google.csv')
    elif keyword == 'TSLA':
        data_predicted = pd.read_csv('data_location_TSLA.csv')

    print(data_predicted)
    # data_predicted.to_csv('data_location.csv')
    end = datetime.now()
    print('the time needed for this operation is' + str(end - start))

    '''data_predicted['Date'] = pd.to_datetime(data_predicted['Date'], format='%Y-%m-%d')
    data_predicted['Date'] = data_predicted['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))
    data_predicted['Date'] = pd.to_datetime(data_predicted['Date'], format='%d/%m/%Y')'''
    # print(data_predicted['Date'])
    temp1 = []
    # print(len(data_predicted['Date'].unique()))
    data_predicted = data_predicted.dropna()

    locations = data_predicted['location'].unique()
    locations = locations
    for i in range(len(locations)):
        l = {}
        temp = data_predicted.loc[data_predicted['location'] == locations[i]]
        # print(temp)
        # print(str(date.today() - timedelta(days=i)))
        # l['Date'] = str(date.today() - timedelta(days=i))
        daily_sentiment_net = 0
        daily_sentiment_neg = 0
        daily_sentiment_pos = 0
        for p in temp['sentiment']:
            if p == 1:
                daily_sentiment_pos = daily_sentiment_pos + 1
            elif p == -1:
                daily_sentiment_neg = daily_sentiment_neg + 1
            elif p == 0:
                daily_sentiment_net = daily_sentiment_net + 1
        l['location'] = locations[i]
        l['negative_score'] = daily_sentiment_neg
        l['positive_score'] = daily_sentiment_pos
        l['neutre_score'] = daily_sentiment_net
        temp1.append(l)
        # print(temp1)
    print('map_sentiment done ' + keyword)
    return JsonResponse(json.dumps(temp1), safe=False)

# method for a chart of sentiment score for a week
@api_view(["GET"])
def chart_sentiment_analysis(request, keyword, period):
    # Twitter Authentification
    geocoder = OpenCageGeocode(key)
    start = datetime.now()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    # since_date = '2020-04-25'
    # until_date = '2020-04-25' until=str(date.today())
    start_date = str(date.today() - timedelta(days=period))
    print(start_date)
    item = 3000
    if period == 6:
        item = 10000
    graph = sett.graph
    sess = sett.sess
    model = sett.model
    tokenizer = sett.tokenizer
    tweets = pd.DataFrame()
    print('the desired items' + str(item))
    for tweet in tweepy.Cursor(api.search, q=keyword + ' -filter:retweets', lang='en', wait_on_rate_limit=True,
                               count=400, since=start_date, include_rts=False).items(item):
        tweets1 = pd.DataFrame(np.zeros(1))
        query = tweet.user.location
        # results = geocoder.geocode(query)
        tweets1["username"] = tweet.user.screen_name
        temp_string = str(tweet.text).replace('\n', '')
        temp_string = str(temp_string).replace('"', '')
        tweets1["cleaned_tweets"] = str(temp_string).replace(',', '')
        tweets1["Date"] = tweet.created_at
        tweets1["Date"] = tweets1["Date"].astype(str)
        '''if results:
            # print(results[0]['components']['country_code'])
            tweets1["location"] = results[0]['components']['country_code']
            results = ''
        else:
            results = 'us'
            tweets1["location"] = results
        '''
        tweets = tweets.append(tweets1)

    data = tweets[['username', 'cleaned_tweets', 'Date']]
    print(data)
    labels = ['1', '-1', '0']

    data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data)
    pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
        tokenizer, data_test)
    # start sentiment prediction
    with graph.as_default():
        set_session(sess)
        predicted = model.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])

    data_test['sentiment'] = prediction_labels
    data_predicted = data_test[['Date', 'cleaned_tweets','sentiment']]
    # data_predicted.to_csv('data_location2.csv')
    # data_predicted = pd.read_csv('data_location.csv')
    print(data_predicted)
    end = datetime.now()
    print('the time needed for this operation is' + str(end - start))
    # str_now = datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))
    # datetime_object = datetime.strptime(str_now, "%Y-%m-%d %H:%M:%S")
    # data_cal = data_predicted.loc[data_predicted['Date'] >= str(datetime_object - timedelta(days=1))]

    data_predicted['Date'] = pd.to_datetime(data_predicted['Date'], format='%Y-%m-%d')
    data_predicted['Date'] = data_predicted['Date'].apply(lambda x: x.strftime("%d/%m/%Y"))
    data_predicted['Date'] = pd.to_datetime(data_predicted['Date'], format='%d/%m/%Y')
    # print(data_predicted['Date'])
    temp1 = []
    print(len(data_predicted['Date'].unique()))
    for i in range(len(data_predicted['Date'].unique())):
        l = {}
        temp = data_predicted.loc[data_predicted['Date'] == str(date.today() - timedelta(days=i))]
        # print(str(date.today() - timedelta(days=i)))
        l['Date'] = str(date.today() - timedelta(days=i))
        daily_sentiment_net = 0
        daily_sentiment_neg = 0
        daily_sentiment_pos = 0
        for p in temp['sentiment']:
            if p == '1':
                daily_sentiment_pos = daily_sentiment_pos + 1
            elif p == '-1':
                daily_sentiment_neg = daily_sentiment_neg + 1
            elif p == '0':
                daily_sentiment_net = daily_sentiment_net + 1

        l['negative_score'] = daily_sentiment_neg
        l['positive_score'] = daily_sentiment_pos
        l['neutre_score'] = daily_sentiment_net
        temp1.append(l)
        print(temp1)
    return JsonResponse(json.dumps(temp1), safe=False)


class AppUserViewSet(viewsets.ModelViewSet):
    queryset = AppUser.objects.all()
    serializer_class = AppUserSerializer


class TwitterUserViewSet(viewsets.ModelViewSet):
    queryset = TwitterUser.objects.all()
    serializer_class = TwitterUserSerializer


class TweetsViewSet(viewsets.ModelViewSet):
    queryset = Tweets.objects.all()
    serializer_class = TweetsSerializer


MAX_SEQUENCE_LENGTH = 50


class GetTweets(viewsets.ModelViewSet):
    # extract tweest from twitter api
    @staticmethod
    @api_view(["GET"])
    def gettweets(request, keyword):
        # Twitter Authentification
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

        graph = sett.graph
        sess = sett.sess
        model = sett.model
        tokenizer = sett.tokenizer
        since_date = '2020-04-25'
        until_date = '2020-04-25'
        print(type(keyword))
        tweets = pd.DataFrame()
        #3 2
        for tweet in tweepy.Cursor(api.search, q=keyword + ' -filter:retweets', lang='en', wait_on_rate_limit=True,
                                   count=400, include_rts=False, since=str(date.today() - timedelta(days=2)), until=str(date.today() - timedelta(days=1))).items(500):
            tweets1 = pd.DataFrame(np.zeros(1))
            #print(tweet.user.locations.country_code)
            tweets1["username"] = tweet.user.screen_name
            temp_string = str(tweet.text).replace('\n', '')
            temp_string = str(temp_string).replace('"', '')
            tweets1["cleaned_tweets"] = str(temp_string).replace(',', '')
            tweets1["Date"] = tweet.created_at
            tweets1["Date"] = tweets1["Date"].astype(str)
            tweets = tweets.append(tweets1)

        data = tweets[['username', 'cleaned_tweets', 'Date']]
        # data['cleaned_tweets'] = data['cleaned_tweets'].apply(lambda x: sentiment_analysis_class.sentiment_analysis_class.remove_punct(x))
        data = data.reset_index()
        data1 = data[['username', 'cleaned_tweets', 'Date']]
        labels = ['1', '-1', '0']

        data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data1)
        pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
            tokenizer, data_test)
        # start sentiment prediction
        with graph.as_default():
            set_session(sess)
            predicted = model.predict(pads_testing_data)
            prediction_labels = []
            for p in predicted:
                prediction_labels.append(labels[np.argmax(p)])

        data_test['sentiment'] = prediction_labels
        print(data_test)
        tweets = []
        print('length is : ' + str(len(data_test['cleaned_tweets'])))
        for i in range(len(data_test['cleaned_tweets'])):
            temp = {}
            temp["username"] = data_test["username"][i]
            temp["cleaned_tweets"] = data_test["cleaned_tweets"][i]
            temp["Date"] = data_test["Date"][i]
            temp["sentiment"] = data_test["sentiment"][i]
            tweets.append(temp)

        # print(json.dumps(temp))
        return JsonResponse(json.dumps(tweets), safe=False)


class PredictSentiment(viewsets.ModelViewSet):
    # predict sentiment for a single tweet
    @staticmethod
    @api_view(["GET"])
    def predict_sentiments(request, model_name, tweets):
        data = pd.DataFrame(np.zeros(1))
        data['cleaned_tweets'] = tweets
        start = datetime.now()

        model = sett.model
        tokenizer = sett.tokenizer
        data_test = sentiment_analysis_class.sentiment_analysis_class.data_preparation(data)
        pads_testing_data, test_word_index = sentiment_analysis_class.sentiment_analysis_class.tokenize_pad_sequences(
            tokenizer, data_test)
        end = datetime.now()
        print('time to make a load for the model is :' + str(end - start))
        start = datetime.now()
        labels = [1, -1, 0]
        graph = sett.graph
        sess = sett.sess
        with graph.as_default():
            set_session(sess)
            predicted = model.predict(pads_testing_data)
            prediction_labels = []
            for p in predicted:
                prediction_labels.append(labels[np.argmax(p)])
        end = datetime.now()
        print('time to make a prediction is :' + str(end - start))
        data_test['sentiment'] = prediction_labels
        # prediction = sentiment_analysis_class.sentiment_analysis_class.prediction_function(model_trained=model, data_test=data)
        print(str(data_test.sentiment[0]) + ' is the sentiment predicted')
        state = ''
        # call the sentiment analysis class
        if data_test.sentiment[0] == 1:
            state = 'positive'
        elif data_test.sentiment[0] == -1:
            state = 'negative'
        elif data_test.sentiment[0] == 0:
            state = 'neutral'

        return JsonResponse({"results": state})
