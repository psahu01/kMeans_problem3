#######################################
# CIS-563 Introduction to Data Science
# Homowork2
# Problem 3
# Prateek Sahu (801311241)
#######################################

import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def removePunctuations(s):
    return re.sub(r'['+string.punctuation+']', ' ', s)

def removeBogusCharacters(s):
    return re.sub(r"br\s|\d", '', s)

# Returns the set of stopWords
def stopWordsSet():
    with open("../stopWords.txt", "r") as f:
        return set(removePunctuations(f.read()).split())
        #return set(f.read().splitlines())

# Read the foods file and filter out text but review/text
testFile = open("../foods.txt","r", encoding='latin-1')
reviewStr = ''
for line in testFile:
    if line.startswith('review/text: '):
        reviewStr = reviewStr + line.lstrip('review/text: ')
testFile.close()

#-------------------------------------------------------------------------
# Data Pre-processing(Removing punctuations, Bogus Characters and numbers)
#-------------------------------------------------------------------------
cleanReviewStr = removePunctuations(removeBogusCharacters(reviewStr.lower()))
reviewList = cleanReviewStr.splitlines();
reviewWordsList = cleanReviewStr.split()

L = set(reviewWordsList)
W = L - stopWordsSet();

reviewWordsFreq = Counter(reviewWordsList)
for n in list(stopWordsSet()):
    del reviewWordsFreq[n]

# Write top 500 words with its frequency in file
topWordsList = reviewWordsFreq.most_common(500)
topWordsDict = dict(topWordsList)
with open(r"../top500Words.txt", 'w') as file:
    file.write(str(topWordsDict))

#-------------
#Vectorization
#-------------
corpus = np.array(list(zip(*topWordsList)))[0]

df = pd.DataFrame(reviewList)
rawDoc = df.iloc[0:,0]

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
P = vectorizer.transform(rawDoc)
sparseMatrix = P.A

#-------------
# K-Means
#-------------
k = 10
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(sparseMatrix)

centroids = model.cluster_centers_.argsort()[:, ::-1]
#get the date in the cluster and sort them to get top 5 data values
terms = vectorizer.get_feature_names()

for i in range(k):
    print("Cluster {}:".format(i+1))
    for data in centroids[i, :5]:
        print("{}: {}".format(terms[data], model.cluster_centers_[i,data]))
    print("\n")