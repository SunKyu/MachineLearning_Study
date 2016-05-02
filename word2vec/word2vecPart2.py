#-*- coding:utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_wordlist( review, remove_stopwords=False):

    review_text = BeautifulSoup(review).get_text()

    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stop = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    return (words)

def review_to_sentences( review, tokenizer, remove_stopwords=False) :
    raw_sentences = tokenizer.tokenize(review.decode("utf-8").strip())

    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter="\t", quoting = 3)

test = pd.read_csv( "testData.tsv", header = 0, delimiter = "\t", quoting = 3)

unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting = 3)

print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" %(train["review"].size, test["review"].size, unlabeled_train["review"].size)


sentences = []

print "Parsing sentences from training import set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabled import set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


import logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : $(message)s', level=logging.INFO)


#set values for various parameters

num_features = 300 #word vector dimensionality
min_word_count = 40 #Minumum word count
num_workers = 8 # Number of threads to run in parallel
context = 10 # context window size
downsampling = 1e-3 # Dwonsample setting for frequent words

#Initialize and train the model
print "Training model...."
model = word2vec.Word2Vec(sentences, workers=num_workers, size = num_features, min_count= min_word_count, window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)


