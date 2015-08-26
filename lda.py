#!/usr/bin/env python

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, LdaModel
import pyLDAvis.gensim
import pandas as pd
import os
import sys

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#########################
# CONFIGURATION

no_of_topics = 20                                       # no. of topics to be generated
no_of_passes = 200                                      # no. of lda iterations

# csv reader
columns = ['ParagraphId', 'TokenId', 'Lemma', 'CPOS', 'NamedEntity']   # columns to read from csv file
pos_tags = ['ADJ', 'NN']   #['ADJ', 'NN', 'V']                           # parts-of-speech to include into the model

# document size
doc_size = 1000                                          # in words
doc_split = 0                                           # 1: on, 0: off -- uses ParagraphId

# stopwords
stopwordlist = ""                                       # "/path/to/txt"


#########################
# PRE-PROCESSING

def preprocessing(path, columns, pos_tags, doc_size, doc_split, stopwordlist):
    docs = []
    stopwords = ""

    print("reading files ...\n")

    try:
        with open(stopwordlist, 'r') as f: stopwords = f.read()
    except OSError:
        pass
    stopwords = sorted(set(stopwords.split("\n")))

    for file in os.listdir(path=path):
        if not file.startswith("."):
            filepath = path+"/"+file
            print(filepath)

            df = pd.read_csv(filepath, sep="\t")
            df = df[columns]
            df = df.groupby('CPOS')

            doc = pd.DataFrame()
            for p in pos_tags:                          # collect only the specified parts-of-speech
                doc = doc.append(df.get_group(p))

            """
            df = df.groupby('NamedEntity')              # add named entities to stopword list
            names = df.get_group('B-PER')['Lemma'].values.astype(str)
            names += df.get_group('I-PER')['Lemma'].values.astype(str)
            """
            names = df.get_group('NP')['Lemma'].values.astype(str)
            stopwords += names.tolist()

            # construct documents
            if doc_split:                               # size according to paragraph id
                doc = doc.groupby('ParagraphId')
                for para_id, para in doc:
                    docs.append(para['Lemma'].values.astype(str))
            else:                                       # size according to doc_size
                doc = doc.sort(columns='TokenId')
                while(doc_size < doc.shape[0]):
                    docs.append(doc[:doc_size]['Lemma'].values.astype(str))
                    doc = doc.drop(doc.index[:doc_size])        # drop doc_size rows
                docs.append(doc['Lemma'].values.astype(str))    # add the rest

    #for doc in docs: print(str(len(doc)))              # display resulting doc sizes
    #print(stopwords)

    print("\nnormalizing and vectorizing ...\n")        # cf. https://radimrehurek.com/gensim/tut1.html

    texts = [[word for word in doc if word not in stopwords] for doc in docs]       # remove stopwords

    all_tokens = sum(texts, [])                                                     # remove words that appear only once
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in texts]

    dictionary = Dictionary(texts)                      # vectorize
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus


#########################
# MAIN

if len(sys.argv) < 2:
    print("usage: {0} [folder containing csv files]\n"
          "parameters are set inside the script.".format(sys.argv[0]))
    sys.exit(1)

path = sys.argv[1]
foldername = path.split("/")[-1]

dictionary, corpus = preprocessing(path, columns, pos_tags, doc_size, doc_split, stopwordlist)

print("fitting the model ...\n")
model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=no_of_topics, passes=no_of_passes,
                 eval_every=1, chunksize=1)
#model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=no_of_topics, passes=no_of_passes,
#                 eval_every=1, chunksize=1, alpha='auto')
print(model)

# chunksize=mm.num_docs
# TODO: bessere values finden für update_every=1, chunksize=1

model.save(foldername+".lda")


#########################
# VISUALIZATION

# cf. https://pyldavis.readthedocs.org/en/latest/modules/API.html

print("displaying results ...\n")

vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)

pyLDAvis.save_html(vis, foldername+".html")
pyLDAvis.show(vis)
