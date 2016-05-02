from gensim.models import Word2Vec
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data


model = Word2Vec.load("300features_40minwords_10context")

type(model.syn0)

model.syn0.shape


model["flower"]

def review_to_wordlist( review, remove_stopwords=False):

    review_text = BeautifulSoup(review).get_text()

    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    return (words)


import numpy as np

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.

    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add( featureVec, model[word])

    featureVec = np.divide( featureVec, nwords)
    return featureVec

def getAvgFeatureVecs( reviews, model, num_features):

    counter = 0.

    reviewFeatureVecs = np.zero(( len(reviews), num_features), dtype="float32")

    for review in reviews:

        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))

            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        counter = counter + 1.
    
    return reviewFeatureVecs

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter="\t", quoting = 3)

test = pd.read_csv( "testData.tsv", header = 0, delimiter = "\t", quoting = 3)

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features)

print "Creating average feature vecs for test reviews"
clean_train_reviews = []
for review in test["review"]:
    clean_train_reviews.append( review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100)

print "Fitting a random forest to labeld training data..."
forest = forest.fit( trainDataVecs, train["sentiment"])

result = forest.predict( testDataVecs)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result})
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3)
