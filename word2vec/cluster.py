from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import time

model = Word2Vec.load("300features_40minwords_10context")

start = time.time()
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

kmeans_clustering = KMeans( n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict( word_vectors)

end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."

word_centroid_map = dict(zip( model.index2word, idx))


for cluster in xrange(0, 10):

    print "\ncluster %d" % cluster

    words = []
    for i in xrange(0, len(word_centroid_map.values())):
        if (word_centroid_map.values()[i] == cluster):
            words.append(word_centroid_map.keys()[i])
    print words

def create_bag_of_centriods( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1

    bag_of_centroids = np.zeros( num_centroids, dtype="float32")

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    retrun bag_of_centroids

train_centroids = np.zeros( ( train["review"].size, num_clusters), dtype="float32")

counter = 0

for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centriods( review, word_centroid_map)
    counter+= 1

test_centroids = np.zeros(( test["review"].size, num_clusters), dtype="floate32")
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review)
