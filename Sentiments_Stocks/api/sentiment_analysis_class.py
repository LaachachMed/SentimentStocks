# importing the dependencies
# !pip install simplejson
import simplejson as json
import io
import keras
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# importing classes helpfull for text processing
import nltk  # general NLP

#nltk.download('punkt')
import re  # regular expressions
import string
import gensim.models.word2vec as w2v  # word2vec model

import matplotlib.pyplot as plt  # data visualization
# lemmatizing
from nltk.stem import WordNetLemmatizer
# stemmer
from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords  # to delete the stopwords
from nltk import word_tokenize  # for the tokenization
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential


from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.vis_utils import plot_model

import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

np.random.seed(0)
# nltk.download('stopwords')
# nltk.download('wordnet')
stoplist = stopwords.words('english')

# Load Google News Word2Vec model
from gensim import models

word2vec_path = 'Sentiments_Stocks/GoogleNews-vectors-negative300.bin.gz'
word2vec = "models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True,limit=100)"

# with open(str('Sentiments_Stocks/') + 'tokenizer.json') as f:
#     data = json.load(f)
#     tokenizer = tokenizer_from_json(data)


# number of epochs
num_epochs = 6
batch_size = 32

stoplist.remove("not")
stoplist.remove("very")
stoplist.remove("don't")
stoplist.remove("haven't")
stoplist.remove("hasn't")
stoplist.remove("weren't")
stoplist.remove("wasn't")
stoplist.remove("didn't")

wordnet_lemmatizer = WordNetLemmatizer()


class sentiment_analysis_class:
    # define methods for reading and cleaning the training data and the unlabelled data
    @staticmethod
    def read_data_csv(link_training):
        data = pd.read_csv(link_training, encoding='latin-1', engine='python')
        return data

    # add more labels
    @staticmethod
    def orginize_data(data):
        # create a new labels
        pos = []
        neg = []
        neut = []

        for l in data.sentiment:
            if l == 0:
                pos.append(0)
                neg.append(0)
                neut.append(1)
            elif l == 1:
                pos.append(1)
                neg.append(0)
                neut.append(0)
            elif l == -1:
                pos.append(0)
                neg.append(1)
                neut.append(0)

        data['Pos'] = pos
        data['Neg'] = neg
        data['Neut'] = neut

        return data

    # define cleaning data function
    # remove punc
    @staticmethod
    def remove_punct(text):
        text_nopunct = ''
        text_nopunct = re.sub(r'[' + string.punctuation + ']', ' ', str(text), flags=re.MULTILINE)
        return text_nopunct

    # function to remove one character words
    @staticmethod
    def remove_one_word(text):
        text_more_words = ''
        text_more_words = re.sub(r'\b\w{1,1}\b', '', str(text), flags=re.MULTILINE)
        return text_more_words

    # function to remove links
    @staticmethod
    def remove_links(text):
        text_no_links = ''
        text_no_links = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', str(text), flags=re.MULTILINE)
        return text_no_links

    # we lower case the data
    @staticmethod
    def lower_token(tokens):
        return [w.lower() for w in tokens]

    # function to remove stopwords
    @staticmethod
    def removeStopWords(tokens):
        return [word for word in tokens if word not in stoplist]

    # define a method for lematization
    @staticmethod
    def lem_list(row):
        wordnet_lemmatizer = WordNetLemmatizer()
        my_list = row['tokens']
        lem_list = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in my_list]
        return (lem_list)

    # define a method for stemming
    @staticmethod
    def stem_list(row):
        stemmer = SnowballStemmer(language='english')
        my_list = row['lemmatized']
        stemmed_list = [stemmer.stem(word) for word in my_list]
        return (stemmed_list)

    # lemmatization
    @staticmethod
    def lemmatization(data):
        # lemmatization of the training dataset
        data['lemmatized'] = data.apply(sentiment_analysis_class.lem_list, axis=1)
        return data

    # Stemming function
    @staticmethod
    def stemming(data):
        # stemming training dataset
        data['stemmed'] = data.apply(sentiment_analysis_class.stem_list, axis=1)
        return data

    # function to PreProcess the training data
    @staticmethod
    def data_preparation(data_toprocess):
        # clean the training dataset
        data_toprocess['text_clean'] = data_toprocess['cleaned_tweets'].apply(lambda x: sentiment_analysis_class.remove_links(x))

        data_toprocess['text_clean'] = data_toprocess['text_clean'].apply(lambda x: sentiment_analysis_class.remove_punct(x))

        data_toprocess['text_clean'] = data_toprocess['text_clean'].apply(lambda x: sentiment_analysis_class.remove_one_word(x))

        # Tokenize the training data set
        tokens = [word_tokenize(sen) for sen in data_toprocess['text_clean']]
        lower_tokens = [sentiment_analysis_class.lower_token(token) for token in tokens]


        # delete stop words from training dataset
        filtered_words = [sentiment_analysis_class.removeStopWords(sen) for sen in lower_tokens]
        data_toprocess['Text_Final'] = [' '.join(sen) for sen in filtered_words]


        data_toprocess['tokens'] = filtered_words
        data_toprocess = sentiment_analysis_class.lemmatization(data_toprocess)

        return data_toprocess

        # split training and testing data

    @staticmethod
    def split_train_test_data(data_tosplit):
        data_train, data_test = train_test_split(data_tosplit, test_size=0.2)
        return data_train, data_test

    # create a vocab for our data
    @staticmethod
    def create_vocab(data_voc):
        data_words = [word for tokens in data_voc["lemmatized"] for word in tokens]
        training_sentence_lengths = [len(tokens) for tokens in data_voc["lemmatized"]]
        # create a liste of words contain all words in the training dataset
        data_vocab = sorted(list(set(data_words)))
        print("%s words total, with a vocabulary size of %s" % (len(data_words), len(data_vocab)))
        print("Max sentence length is %s" % max(training_sentence_lengths))
        return data_vocab

    # function to impute the messing values of the words' vectors using mean approach (k=300 EMBEDDING DIMENSION)
    @staticmethod
    def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
        if len(tokens_list) < 1:
            return np.zeros(k)
        if generate_missing:
            vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
        else:
            vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        length = len(vectorized)
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
        return averaged

    # generate embeddings
    @staticmethod
    def get_word2vec_embeddings(vectors, clean_tokens, generate_missing=False):
        embeddings = clean_tokens['lemmatized'].apply(
            lambda x: sentiment_analysis_class.get_average_word2vec(x, vectors, generate_missing=generate_missing))
        return list(embeddings)

    # train embeddings
    @staticmethod
    def train_embeddings(word2vec, data_train):
        training_embeddings = sentiment_analysis_class.get_word2vec_embeddings(word2vec, data_train,
                                                                               generate_missing=True)
        return training_embeddings

    # fited tokenization model on training data
    @staticmethod
    def tokenization_model(train_vocab, train_data):
        tokenizer = Tokenizer(num_words=len(train_vocab), lower=True, char_level=False)
        tokenizer.fit_on_texts(train_data["lemmatized"].tolist())
        return tokenizer

    # Tokenize and Pad sequences to make it all in the same length
    # Tokenize and pad sequences of the dataset
    @staticmethod
    def tokenize_pad_sequences(tokenize_model, data_toc):
        MAX_SEQUENCE_LENGTH = 50
        training_sequence = tokenize_model.texts_to_sequences(data_toc["lemmatized"].tolist())
        word_index = tokenize_model.word_index
        print('Found %s unique tokens.' % len(word_index))
        pads_data = pad_sequences(training_sequence, maxlen=MAX_SEQUENCE_LENGTH)
        return pads_data, word_index

    # Train embeddings weights for the dataset
    @staticmethod
    def train_embeddings_index(train_word_index, EMBEDDING_DIM):
        embedding_weight = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
        for word, index in train_word_index.items():
            embedding_weight[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
        print(embedding_weight.shape)
        return embedding_weight

    # link the Y of the dataset with the preprocessed one
    @staticmethod
    def data_combining(data_train, data_preprocessed):
        label_name = ['Pos', 'Neg', 'Neut']
        y_train = data_train[label_name].values
        x_train = data_preprocessed
        return x_train, y_train

    # define a CNN function (model)
    @staticmethod
    def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):

        embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[embeddings],
                                    input_length=max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(max_sequence_length,), dtype='float64')
        embedded_sequences = embedding_layer(sequence_input)
        convs = []
        filter_sizes = [2, 3, 4, 5, 5]
        for filter_size in filter_sizes:
            l_conv = Conv1D(filters=80,
                            kernel_size=filter_size,
                            activation='sigmoid')(embedded_sequences)
            l_pool = GlobalMaxPooling1D()(l_conv)
            convs.append(l_pool)
        l_merge = concatenate(convs, axis=1)
        x = Dropout(0.1)(l_merge)
        x = Dense(128, activation='sigmoid')(x)
        x = Dropout(0.2)(x)
        preds = Dense(labels_index, activation='sigmoid')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        print('Structure of CNN Model')
        print('-----------------------------------------------------------------------------')
        model.summary()
        print('-----------------------------------------------------------------------------')
        return model

    # define a RNN function (model)
    @staticmethod
    def recurrent_nn(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):

        embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[embeddings],
                                    input_length=max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(max_sequence_length,), dtype='float64')
        embedded_sequences = embedding_layer(sequence_input)

        # lstm = LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedded_sequences)
        lstm = LSTM(128)(embedded_sequences)

        x = Dense(128, activation='softmax')(lstm)
        x = Dropout(0.2)(x)
        preds = Dense(labels_index, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        print('Structure of LSTM Model')
        print('-----------------------------------------------------------------------------')
        model.summary()
        print('-----------------------------------------------------------------------------')
        return model

    # define a RNN function (model)
    @staticmethod
    def CNN_LSTM_Model(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
        model = Sequential()
        embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[embeddings],
                                    input_length=max_sequence_length,
                                    trainable=False)
        model.add(embedding_layer)
        model.add(Conv1D(filters=80, kernel_size=3, padding='same', activation='softmax'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Structure of CNN+LSTM Model')
        print('-----------------------------------------------------------------------------')
        print(model.summary())
        print('-----------------------------------------------------------------------------')
        return model

    # save the text tokenizer file
    @staticmethod
    def txt_tokenizer(tokenize, path):
        json_tokenezer = tokenize.to_json()
        # save the model of tokenizer
        with io.open(str(path) + 'tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_tokenezer, ensure_ascii=False))

        print('text tokenizer saved in : ' + str(path) + 'tokenizer.json')

    # open the text tokenizer file
    @staticmethod
    def open_txt_tokeznizer(path):
        # open the tokenizer
        with open(str(path) + 'tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        return tokenizer

    # training function
    @staticmethod
    def training_function(model_name, data):
        num_epochs = 6
        batch_size = 32
        MAX_SEQUENCE_LENGTH = 50
        EMBEDDING_DIM = 300
        data = sentiment_analysis_class.orginize_data(data)
        data = sentiment_analysis_class.data_preparation(data)
        data = data[['Text_Final', 'lemmatized', 'sentiment', 'Pos', 'Neg', 'Neut']]
        train, test = sentiment_analysis_class.split_train_test_data(data)
        train_vocab = sentiment_analysis_class.create_vocab(train)
        test_vocab = sentiment_analysis_class.create_vocab(test)
        training_embedding = sentiment_analysis_class.train_embeddings(word2vec, train)
        tokenize = sentiment_analysis_class.tokenization_model(train_vocab, train)
        # save the trained tokenizer
        sentiment_analysis_class.txt_tokenizer(tokenize, '')
        pads_training_data, training_word_index = sentiment_analysis_class.tokenize_pad_sequences(tokenize, train)
        training_embedding_weight = sentiment_analysis_class.train_embeddings_index(training_word_index, 300)
        # create a dataset test sequences
        pads_testing_data, test_word_index = sentiment_analysis_class.tokenize_pad_sequences(tokenize, test)
        y_test = test.sentiment
        x_train, y_train = sentiment_analysis_class.data_combining(train, pads_training_data)

        if model_name == 'CNN':
            print('CNN')
            model_to_train = sentiment_analysis_class.ConvNet(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                              len(training_word_index) + 1, EMBEDDING_DIM, 3)
        elif model_name == 'LSTM':
            print('LSTM')
            model_to_train = sentiment_analysis_class.recurrent_nn(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                                   len(training_word_index) + 1, EMBEDDING_DIM, 3)
        elif model_name == 'CNN_LSTM':
            print('CNN_LSTM')
            model_to_train = sentiment_analysis_class.CNN_LSTM_Model(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                                     len(training_word_index) + 1, EMBEDDING_DIM, 3)

        # train the models
        # feeding the  model
        start = datetime.datetime.now()
        model_to_train.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, shuffle=True,
                           batch_size=batch_size)
        end = datetime.datetime.now()
        print('time for training the ' + str(model_name) + ' model is : ' + str(end - start))
        # test the models
        sentiment_analysis_class.Model_Acc(model_to_train, model_name, pads_testing_data, y_test)
        return model_to_train

    # calculate the accuracy for a model
    @staticmethod
    def Model_Acc(model, model_name, test_data, y_test):
        predictions = model.predict(test_data)
        labels = [1, -1, 0]
        prediction_labels = []
        for p in predictions:
            prediction_labels.append(labels[np.argmax(p)])

        print('Accuracy for ' + str(model_name) + ' Model is {:0.3f}%'.format(
            accuracy_score(prediction_labels, y_test) * 100))
        print(classification_report(y_test, prediction_labels))

    # test function
    @staticmethod
    def prediction_function(model_trained, data_test):
        data_test = sentiment_analysis_class.data_preparation(data_test)
        path_tokenizer_trained = ""
        tokenize = sentiment_analysis_class.open_txt_tokeznizer(path_tokenizer_trained)
        pads_testing_data, test_word_index = sentiment_analysis_class.tokenize_pad_sequences(tokenize, data_test)

        labels = [1, -1, 0]
        predicted = model_trained.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])

        data_test['sentiment'] = prediction_labels
        data_test['cleaned_tweets'] = data_test['Text_Final']

        return data_test


    # plot the models schema
    @staticmethod
    def Plot_Models(model_deepl, model_name):
        plot_model(model_deepl, to_file='model_' + str(model_name) + '.png', show_shapes=True, show_layer_names=True)

    # save a pre trained model
    @staticmethod
    def save_pretrained_model(modelpretrained, path):
        modelpretrained.save(str(path) + 'pretrained_model.h5')

    # load a pre trained model
    @staticmethod
    def load_pretrained_model(path):
        model_trained = keras.models.load_model(path)
        return model_trained

    # pseudo-labelling
    @staticmethod
    def pseudo_lebelling(model_name, big_data, data_to_label):
        num_epochs = 6
        batch_size = 32
        MAX_SEQUENCE_LENGTH = 50
        EMBEDDING_DIM = 300
        big_data = sentiment_analysis_class.orginize_data(big_data)
        big_data = sentiment_analysis_class.data_preparation(big_data)
        big_data = big_data[['Text_Final', 'lemmatized', 'sentiment', 'Pos', 'Neg', 'Neut']]
        train_vocab = sentiment_analysis_class.create_vocab(big_data)
        tokenize = sentiment_analysis_class.tokenization_model(train_vocab, big_data)
        # save the trained tokenizer
        sentiment_analysis_class.txt_tokenizer(tokenize, '')
        pads_training_data, training_word_index = sentiment_analysis_class.tokenize_pad_sequences(tokenize, big_data)
        training_embedding_weight = sentiment_analysis_class.train_embeddings_index(training_word_index, 300)
        x_train, y_train = sentiment_analysis_class.data_combining(big_data, pads_training_data)

        if model_name == 'CNN':
            print('CNN')
            model_to_train = sentiment_analysis_class.ConvNet(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                              len(training_word_index) + 1, EMBEDDING_DIM, 3)
        elif model_name == 'LSTM':
            print('LSTM')
            model_to_train = sentiment_analysis_class.recurrent_nn(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                                   len(training_word_index) + 1, EMBEDDING_DIM, 3)
        elif model_name == 'CNN_LSTM':
            print('CNN_LSTM')
            model_to_train = sentiment_analysis_class.CNN_LSTM_Model(training_embedding_weight, MAX_SEQUENCE_LENGTH,
                                                                     len(training_word_index) + 1, EMBEDDING_DIM, 3)

        # train the models
        # feeding the  model
        start = datetime.datetime.now()
        model_to_train.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, shuffle=True,
                           batch_size=batch_size)
        end = datetime.datetime.now()
        print('time for training the ' + str(model_name) + ' model is : ' + str(end - start))

        data_to_label = sentiment_analysis_class.data_preparation(data_to_label)
        pads_testing_data, test_word_index = sentiment_analysis_class.tokenize_pad_sequences(tokenize, data_to_label)

        labels = [1, -1, 0]
        predicted = model_to_train.predict(pads_testing_data)
        prediction_labels = []
        for p in predicted:
            prediction_labels.append(labels[np.argmax(p)])

        data_to_label['sentiment'] = prediction_labels
        data_to_label['cleaned_tweets'] = data_to_label['Text_Final']
        data_to_label = data_to_label[['cleaned_tweets', 'sentiment']]

        big_data['cleaned_tweets'] = big_data['Text_Final']
        big_data = big_data[['cleaned_tweets', 'sentiment']]
        big_data = big_data.append(data_to_label)
        return big_data