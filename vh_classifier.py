#!/usr/bin/env python

"""
Recreating van Halteren et al. - New Machine Learning Methods Demonstrate the Existence of a Human Stylome

"""

import pandas as pd
import numpy as np
import os, sys
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings("ignore")           # features from test text not found in the model raises UserWarning


#########################
# HELPER FUNCTIONS

def wordcount(wordlist):
    dict = {}
    for word in wordlist:
        if word not in dict: dict[word] = 1
        else: dict[word] += 1
    return dict

def token_in_textblock(text, token):        # returns number of blocks (consisting of 1/7th of the text)
    blocks = []                             # in which the current token is found, in 4 classes: 1, 2-3,4-6,7
    block_size = len(text)/7
    last = no_of_blocks = 0

    while last < len(text):
        blocks.append(text[int(last):int(last + block_size)])
        last += block_size

    for block in blocks:
        if token in block: no_of_blocks += 1

    if no_of_blocks == 1: occur_class = 1
    elif 2 <= no_of_blocks <= 3: occur_class = 2
    elif 4 <= no_of_blocks <= 6: occur_class = 3
    else: occur_class = 4

    return occur_class

def distance_to_previous(curr_tok_id, curr_sent_id, occurrences):
    # returns distance in sentences to the previous occurrence
    # of the current token (in 7 classes: NONE, SAME, 1, 2-3,4-7,8-15,16+

    occurrences = occurrences.reset_index()                             # add new index from 0 .. len(occurrences.index)

    current_key = occurrences[occurrences['TokenId'] == curr_tok_id].index[0]   # get row corresponding to curr_tok_id + its new index value

    if current_key > 0:                                                 # there is more than one && its not the first occurrence
        prev_sent_id = int(occurrences.iloc[current_key-1, 1])          # get previous sentence id based on that index

        dist = curr_sent_id - prev_sent_id

        if dist == 0: d_class = 2
        elif dist == 1: d_class = 3
        elif 2 <= dist <= 3: d_class = 4
        elif 4 <= dist <= 7: d_class = 5
        elif 8 <= dist <= 15: d_class = 6
        elif 16 <= dist: d_class = 7
    else:
        d_class = 1

    return d_class


#########################
# FEATURE SELECTION

def featureselect(text):
    columns = ['SentenceId', 'TokenId', 'Token', 'CPOS']
    columns_features = ['CurrToken', 'PrevToken', 'NextToken', 'TokenTags', 'LengthPosition', 'TagFreqOccur']

    csv = pd.read_csv(text, sep="\t")
    df = csv[columns]                               # create copy containing only the specified columns

    sent_max = df["SentenceId"].max()               # number of sentences in the text
    token_max = df["TokenId"].max()                 # number of tokens in the text

    text = list(df["Token"])
    word_freq = wordcount(text)                     # word frequencies

    features = pd.DataFrame(columns=columns_features, index=range(token_max+1))       # dataframe to hold the results

    for sent_id in range(sent_max+1):               # iterate through sentences
        sentence = df[df['SentenceId'] == sent_id]  # return rows corresponding to sent_id

        s_len = len(sentence.index)                 # length of the sentence
        if s_len == 1: s_class = 1                  # in 7 classes: 1, 2, 3, 4, 5-10,11-20 or 21+ tokens
        elif s_len == 2: s_class = 2
        elif s_len == 3: s_class = 3
        elif s_len == 4: s_class = 4
        elif 5 <= s_len <= 10: s_class = 5
        elif 11 <= s_len <= 20: s_class = 6
        elif 21 <= s_len: s_class = 7

        tok_count = 1
        for row in sentence.iterrows():
            tok_id = row[0]                         # row/dataframe index is the same as TokenId

            features.iat[tok_id, 0] = current_tok = row[1].get("Token")             # save current token
            tokentags = current_pos = row[1].get("CPOS")                            # get current pos tag

            if tok_id > 0:
                features.iat[tok_id, 1] = df.iloc[tok_id-1, 2]                      # save previous token
                tokentags += "-" + df.iloc[tok_id-1, 3]                             # get previous pos tag
            else:
                tokentags += "-NaN"

            if tok_id < token_max:
                features.iat[tok_id, 2] = df.iloc[tok_id+1, 2]                      # save next token
                tokentags += "-" + df.iloc[tok_id+1, 3]                             # get next pos tag
            else:
                tokentags += "-NaN"

            features.iat[tok_id, 3] = tokentags                         # save pos tags


            if tok_count <= 3: t_class = 1                              # position in the sentence
            elif (s_len-3) < tok_count <= s_len: t_class = 2            # in 3 classes: first three tokens, last three tokens, other
            else: t_class = 3

            features.iat[tok_id, 4] = str(s_class) + "-" + str(t_class) # save sentence length + token position


            tok_freq = word_freq[current_tok]                           # frequency of the current token in the text
            if tok_freq == 1: f_class = 1                               # in 5 classes: 1, 2-5, 6-10,11-20 or 21+
            elif 2 <= tok_freq <= 5: f_class = 2
            elif 6 <= tok_freq <= 10: f_class = 3
            elif 11 <= tok_freq <= 20: f_class = 4
            elif 21 <= tok_freq: f_class = 5

            block_occur = token_in_textblock(text, current_tok)

            occurrences = df[df['Token'] == current_tok]                # new dataframe containing all of curr_token's occurrences
            previous_distance = distance_to_previous(tok_id, sent_id, occurrences)

            features.iat[tok_id, 5] = current_pos + "-" + str(f_class) + "-" + str(block_occur) + "-" + str(previous_distance)

            tok_count += 1

    return features


#########################
# PRE-PROCESSING

def preprocessing(path, n):
    feats = []
    y = []

    print("processing files and randomly selecting {0} features each ...\n".format(n))

    for file in os.listdir(path=path):
        if not file.startswith("."):
            author = file.split("-")[0].replace("%20", " ")
            filepath = path+"/"+file
            print(filepath)

            for i in range(n): y.append(author)                     # add n labels to y

            with open(filepath, "r") as f:
                feat = featureselect(f)                             # perform feature selection
                rows = np.random.choice(feat.index.values, n)       # randomly select n observations
                feat_rand = feat.ix[rows]

                feats.append(feat_rand)
                f.close()

    data = pd.concat(feats, ignore_index=True)                      # merge into one dataframe


    print("\ndimensions of X: {0}".format(data.shape))
    print("dimensions of y: {0}\n".format(len(y)))


    print("vectorizing ...\n")
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(data.T.to_dict().values())
    #vec = DictVectorizer()
    #X = vec.fit_transform(data.T.to_dict().values()).toarray()
    print("dimensions of X after vectorization: {0}\n".format(X.shape))

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)    # replace NaN
    X = imp.fit_transform(X)

    """
    print("dimensionality reduction ...\n")
    svd = TruncatedSVD(n_components=2, random_state=42)
    X = svd.fit_transform(X)
    print("dimensions of X after SVD: {0}\n".format(X.shape))
    """

    return X, y, vec


#########################
# MAIN

n_obs = 700                                                         # no. of observations to select
n_trees = 30                                                        # no. of estimators in RandomForestClassifier

if len(sys.argv) < 3:
    print("usage: {0} [folder containing csv files for training] [csv file for testing]".format(sys.argv[0]))
    sys.exit(1)


# do feature selection, normalization, and vectorization
X, y, vec = preprocessing(sys.argv[1], n_obs)

# model training
print("training classifier ...\n")
clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1).fit(X, y)
print(clf,"\n")

# evaluation
print("performing cross validation (n_iter=5, test_size=0.125) ...")
cv = ShuffleSplit(X.shape[0], n_iter=5, test_size=0.125, random_state=4)
scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
print(scores)
print("mean accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))


#########################
# PREDICTION

print("predicting author for {0} ...\n".format(sys.argv[2]))


# feature selection and preprocessing for testfile
with open(sys.argv[2], "r") as f:
    feat = featureselect(f)                             # perform feature selection
    rows = np.random.choice(feat.index.values, n_obs)       # randomly select n observations
    feat = feat.ix[rows]

print("dimensions of X_test: {0}".format(feat.shape))

X_test = vec.transform(feat.T.to_dict().values())#.toarray()       # vec must be the same DictVectorizer object as generated by preprocessing()
print("dimensions of X_test after vectorization: {0}".format(X_test.shape))

imp = Imputer(missing_values='NaN', strategy='median', axis=0)    # replace NaN
X_test = imp.fit_transform(X_test)


# prediction
y_pred = clf.predict(X_test)

c = Counter(y_pred)
c_key = list(c.keys())
c_val = list(c.values())
print(c_key[0], c_val[0]/(sum(c.values())/100), "% - ",
      c_key[1], c_val[1]/(sum(c.values())/100), "%")
