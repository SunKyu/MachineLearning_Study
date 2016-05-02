import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy

CPU = -1

print "reading training set"
dataset = pd.read_csv("train.csv", header = 0)
print "Read all"
pd.core.frame.DataFrame.
target = []
train = []

data = dataset.get_values()

for i in xrange(0, len(data)):
    train.append(data[i][1])
    feature = data[i][2:9]
    feature.put(0, data[i][0])
    target.append(feature)
    if (i% 1000 ==0):
        print "the number of dataset is read is %d  %d" %(x)


print "Reading test set"

test = pd.read_csv("test.csv")

rf = RandomForestClassifier(n_estimators = 1000, n_jobs=CPU)

print "Fitting RF classifer"
rf.fit(train, target)

print "predicting test set"

result = rf.predict(test)

output = pd.DataFrame()

numpy.ndarray.
