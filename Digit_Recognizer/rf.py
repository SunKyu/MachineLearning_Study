from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import pandas as pd

CPU = 6

print "Readign traingng set"
dataset = genfromtxt(open('train.csv', 'r'), delimiter=',', dtype='int64')[1:]
print "read all"
num_target=len(dataset)
#target =  [x[0] for x in dataset]
target =[]
train = []
for x in xrange(0, num_target):
    target.append(dataset[x][0])
    train.append(dataset[x][1:])
    if (x % 1000 == 0):
        print "the number of dataset that read is %d" %(x)


print "Reading test set"
test = genfromtxt(open('test.csv', 'r'), delimiter=',', dtype = 'int64')[1:]

rf = RandomForestClassifier(n_estimators = 1000, n_jobs=CPU)

print "Fitting RF classifier"
rf.fit(train, target)

print "Predicting test set"

result = rf.predict(test)
output = pd.DataFrame(data={'ImageId':range(1,28001), 'Label'= result})
output.to_csv('submission-version-rf.csv', index = False)
savetxt('submission-version1.csv', rf.predict(test), delimiter='\n' ,fmt='%d')


