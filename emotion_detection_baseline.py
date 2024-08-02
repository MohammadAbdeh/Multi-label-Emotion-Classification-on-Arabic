import gensim
import re
import numpy as np
from nltk import ngrams
from utilities import * # import utilities.py module

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score
import logging
import pandas as p
import re 
import string
import demoji
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import tensorflow as tf
from transformers import AutoConfig, BertForSequenceClassification, AutoTokenizer, AutoModel
from transformers.data.processors import SingleSentenceClassificationProcessor
from transformers import Trainer , TrainingArguments
from arabert.preprocess import ArabertPreprocessor
from transformers import TFBertModel,  BertConfig, BertTokenizerFast

names = [ 'id','tweet','anger','anticipation', 'disgust','fear','joy','love','optimism','pessimism', 'sadness','surprise','trust']

data = p.read_csv("E-c/2018-E-c-Ar-train.txt", names = names , sep = '\t', lineterminator= '\n', header=0)
dev_data = p.read_csv("E-c/2018-E-c-Ar-dev.txt", names = names , sep = '\t', lineterminator= '\n', header=0)
test_data = p.read_csv("E-c/2018-E-c-Ar-test-gold.txt", names = names , sep = '\t', lineterminator= '\n', header=0)

line = ""
lines = []
for index, row in data.iterrows():
    line = re.sub(r'#', '', str(row["tweet"]))
    line = re.sub(r'https?:\/\/.*[\r\n]*', 'URL', line)
    line = re.sub("@[A-Za-z0-9-_]+", "@USER",line)
    line = row["tweet"].replace('\r', ' ')
    line = row["tweet"].replace('\n', ' ')
    lines.append(line)

data["cleaned_tweet"] = lines
del data['tweet']
file_name = "preprocessed.csv"
data.to_csv(file_name, sep='\t', encoding='utf-8')

line = ""
lines = []
for index, row in dev_data.iterrows():
    line = re.sub(r'#', '', str(row["tweet"]))
    line = re.sub(r'https?:\/\/.*[\r\n]*', 'URL', line)
    line = re.sub("@[A-Za-z0-9-_]+", "@USER",line)
    line = row["tweet"].replace('\r', ' ')
    line = row["tweet"].replace('\n', ' ')
    lines.append(line)

dev_data["cleaned_tweet"] = lines
del dev_data['tweet']
file_name = "dev_preprocessed.csv"
dev_data.to_csv(file_name, sep='\t', encoding='utf-8')

line = ""
lines = []
for index, row in test_data.iterrows():
    line = re.sub(r'#', '', str(row["tweet"]))
    line = re.sub(r'https?:\/\/.*[\r\n]*', 'URL', line)
    line = re.sub("@[A-Za-z0-9-_]+", "@USER",line)
    line = row["tweet"].replace('\r', ' ')
    line = row["tweet"].replace('\n', ' ')
    lines.append(line)

test_data["cleaned_tweet"] = lines
del test_data['tweet']
file_name = "test_preprocessed.csv"
test_data.to_csv(file_name, sep='\t', encoding='utf-8')


data['anger_label'] = p.Categorical(data['anger'])
data['anticipation_label'] = p.Categorical(data['anticipation'])
data['disgust_label'] = p.Categorical(data['disgust'])
data['fear_label'] = p.Categorical(data['fear'])
data['joy_label'] = p.Categorical(data['joy'])
data['love_label'] = p.Categorical(data['love'])
data['optimism_label'] = p.Categorical(data['optimism'])
data['pessimism_label'] = p.Categorical(data['pessimism'])
data['sadness_label'] = p.Categorical(data['sadness'])
data['surprise_label'] = p.Categorical(data['surprise'])
data['trust_label'] = p.Categorical(data['trust'])

dev_data['anger_label'] = p.Categorical(dev_data['anger'])
dev_data['anticipation_label'] = p.Categorical(dev_data['anticipation'])
dev_data['disgust_label'] = p.Categorical(dev_data['disgust'])
dev_data['fear_label'] = p.Categorical(dev_data['fear'])
dev_data['joy_label'] = p.Categorical(dev_data['joy'])
dev_data['love_label'] = p.Categorical(dev_data['love'])
dev_data['optimism_label'] = p.Categorical(dev_data['optimism'])
dev_data['pessimism_label'] = p.Categorical(dev_data['pessimism'])
dev_data['sadness_label'] = p.Categorical(dev_data['sadness'])
dev_data['surprise_label'] = p.Categorical(dev_data['surprise'])
dev_data['trust_label'] = p.Categorical(dev_data['trust'])

test_data['anger_label'] = p.Categorical(test_data['anger'])
test_data['anticipation_label'] = p.Categorical(test_data['anticipation'])
test_data['disgust_label'] = p.Categorical(test_data['disgust'])
test_data['fear_label'] = p.Categorical(test_data['fear'])
test_data['joy_label'] = p.Categorical(test_data['joy'])
test_data['love_label'] = p.Categorical(test_data['love'])
test_data['optimism_label'] = p.Categorical(test_data['optimism'])
test_data['pessimism_label'] = p.Categorical(test_data['pessimism'])
test_data['sadness_label'] = p.Categorical(test_data['sadness'])
test_data['surprise_label'] = p.Categorical(test_data['surprise'])
test_data['trust_label'] = p.Categorical(test_data['trust'])


data['anger'] = data['anger_label'].cat.codes
data['anticipation'] = data['anticipation_label'].cat.codes
data['disgust'] = data['disgust_label'].cat.codes
data['fear'] = data['fear_label'].cat.codes
data['joy'] = data['joy_label'].cat.codes
data['love'] = data['love_label'].cat.codes
data['optimism'] = data['optimism_label'].cat.codes
data['pessimism'] = data['pessimism_label'].cat.codes
data['sadness'] = data['sadness_label'].cat.codes
data['surprise'] = data['surprise_label'].cat.codes
data['trust'] = data['trust_label'].cat.codes


dev_data['anger'] = dev_data['anger_label'].cat.codes
dev_data['anticipation'] = dev_data['anticipation_label'].cat.codes
dev_data['disgust'] = dev_data['disgust_label'].cat.codes
dev_data['fear'] = dev_data['fear_label'].cat.codes
dev_data['joy'] = dev_data['joy_label'].cat.codes
dev_data['love'] = dev_data['love_label'].cat.codes
dev_data['optimism'] = dev_data['optimism_label'].cat.codes
dev_data['pessimism'] = dev_data['pessimism_label'].cat.codes
dev_data['sadness'] = dev_data['sadness_label'].cat.codes
dev_data['surprise'] = dev_data['surprise_label'].cat.codes
dev_data['trust'] = dev_data['trust_label'].cat.codes

test_data['anger'] = test_data['anger_label'].cat.codes
test_data['anticipation'] = test_data['anticipation_label'].cat.codes
test_data['disgust'] = test_data['disgust_label'].cat.codes
test_data['fear'] = test_data['fear_label'].cat.codes
test_data['joy'] = test_data['joy_label'].cat.codes
test_data['love'] = test_data['love_label'].cat.codes
test_data['optimism'] = test_data['optimism_label'].cat.codes
test_data['pessimism'] = test_data['pessimism_label'].cat.codes
test_data['sadness'] = test_data['sadness_label'].cat.codes
test_data['surprise'] = test_data['surprise_label'].cat.codes
test_data['trust'] = test_data['trust_label'].cat.codes


del data["anticipation"]
del data["disgust"]
del data["love"]
del data["optimism"]
del data["pessimism"]
del data["surprise"]
del data["trust"]

del dev_data["anticipation"]
del dev_data["disgust"]
del dev_data["love"]
del dev_data["optimism"]
del dev_data["pessimism"]
del dev_data["surprise"]
del dev_data["trust"]

del test_data["anticipation"]
del test_data["disgust"]
del test_data["love"]
del test_data["optimism"]
del test_data["pessimism"]
del test_data["surprise"]
del test_data["trust"]
# stemmins
# model_name = "aubmindlab/bert-base-arabertv2"
# arabert_prep = ArabertPreprocessor(model_name=model_name)

# data['cleaned_tweet'] = data['cleaned_tweet'].apply(lambda x: arabert_prep.preprocess(x))


# dev_data['cleaned_tweet'] = dev_data['cleaned_tweet'].apply(lambda x: arabert_prep.preprocess(x))

##### stemming #####
from nltk.stem import arlstem
stemmer = arlstem.ARLSTem()
stemmed_matrix = []
tweet_matrix = [x.split(" ") for x in data["cleaned_tweet"]]
for tweet in tweet_matrix:
    stemmed_tweet = []
    for word in tweet:
        word = clean_str(word)
        stemmed_tweet.append(stemmer.stem(word)) 
    stemmed_matrix.append(stemmed_tweet)


# print(data["cleaned_tweet"][0])
t_model = gensim.models.Word2Vec.load('../aravec/full_grams_cbow_300_twitter.mdl')

print ("Building tfidf")
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(stemmed_matrix)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        word = clean_str(word)
        try:
            vec += t_model.wv[word].reshape((1, size)) 
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            try:
                vec += t_model.wv[stemmer.stem(word)].reshape((1, size)) 
                count += 1.
            except KeyError:
                try:
                    closest = t_model.wv.most_similar(word, topn=10 )
                    average_vec = np.zeros(size)
                    for term in closest:
                        average_vec+= t_model.wv[term[0]]
                    average_vec/=len(closest)
                    vec += average_vec.reshape((1, size)) 
                    count += 1.
                except KeyError:
                    continue
                continue
            continue
    if count != 0:
        vec /= count
    return vec

# classes = {
#     'anger' : 0,
#     'anticipation' : 1,
#     'disgust' : 2,
#     'fear' : 3,
#     'joy' : 4,
#     'love' : 5,
#     'optimism' : 6,
#     'pessimism' : 7,
#     'sadness' : 8,
#     'surprise' : 9,
#     'trust' : 10,
# }  

classes = {
    'anger' : 0,
    'fear' : 1,
    'joy' : 2,
    'sadness' : 3,
}  

from sklearn.preprocessing import scale

train_vecs_w2v = np.concatenate([buildWordVector(z, 300) for z in tqdm(map(lambda x: x.split(" "), data["cleaned_tweet"]))])
train_vecs_w2v = scale(train_vecs_w2v)

dev_vecs_w2v = np.concatenate([buildWordVector(z, 300) for z in tqdm(map(lambda x: x.split(" "), dev_data["cleaned_tweet"]))])
dev_vecs_w2v = scale(dev_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, 300) for z in tqdm(map(lambda x: x.split(" "), test_data["cleaned_tweet"]))])
test_vecs_w2v = scale(test_vecs_w2v)

y = np.zeros((data.shape[0], 4))
for key in classes:
    y[:,classes[key]] = data.loc[:, key].values
    
dev_y = np.zeros((dev_data.shape[0], 4))
for key in classes:
    dev_y[:,classes[key]] = dev_data.loc[:, key].values

test_y = np.zeros((test_data.shape[0], 4))
for key in classes:
    test_y[:,classes[key]] = test_data.loc[:, key].values

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix

def printEmotionF1(y, y_pred, classes):
    for key in classes:
        f1 = f1_score(y[:,classes[key]], y_pred[:,classes[key]])
        print(key, ': ', f1*100, '%')
        cf_matrix = confusion_matrix(y[:,classes[key]], y_pred[:,classes[key]])
        print(cf_matrix)
        

classif = OneVsRestClassifier(LinearSVC(penalty='l1', dual = False, verbose=True, max_iter=15000))
classif.fit(train_vecs_w2v, y)

# Evaluating on dev
dev_predictions = classif.predict(dev_vecs_w2v)

dev_micro_accuracy = jaccard_score(dev_y, dev_predictions, average='micro')
dev_samples_accuracy = jaccard_score(dev_y, dev_predictions, average='samples')
dev_macro_accuracy =  jaccard_score(dev_y, dev_predictions, average='macro')
print("dev micro accuracy", dev_micro_accuracy)
print("dev samples accuracy", dev_samples_accuracy)
print("dev macro accuracy", dev_macro_accuracy)

dev_micro_f1 = f1_score(dev_y, dev_predictions, average='micro')
dev_samples_f1 = f1_score(dev_y, dev_predictions, average='samples')
dev_macro_f1 = f1_score(dev_y, dev_predictions, average='macro')

print("dev micro f1", dev_micro_f1)
print("dev samples f1", dev_samples_f1)
print("dev macro f1", dev_macro_f1)

dev_y_pred = dev_predictions
dev_y_pred = dev_y_pred >= 0.5
printEmotionF1(dev_y, dev_y_pred , classes)

# Evaluating on test
test_predictions = classif.predict(test_vecs_w2v)

test_micro_accuracy = jaccard_score(test_y, test_predictions, average='micro', zero_division= 1.0)
test_samples_accuracy = jaccard_score(test_y, test_predictions, average='samples', zero_division= 1.0)
test_macro_accuracy =  jaccard_score(test_y, test_predictions, average='macro', zero_division= 1.0)
print("test micro accuracy", test_micro_accuracy)
print("test samples accuracy", test_samples_accuracy)
print("test macro accuracy", test_macro_accuracy)

test_micro_f1 = f1_score(test_y, test_predictions, average='micro', zero_division= 1.0)
test_samples_f1 = f1_score(test_y, test_predictions, average='samples', zero_division= 1.0)
test_macro_f1 = f1_score(test_y, test_predictions, average='macro', zero_division= 1.0)

print("test micro f1", test_micro_f1)
print("test samples f1", test_samples_f1)
print("test macro f1", test_macro_f1)

test_y_pred = test_predictions
test_y_pred = test_y_pred >= 0.5
printEmotionF1(test_y, test_y_pred , classes)