#!/usr/bin/env python

from gensim.corpora import MmCorpus, Dictionary
from gensim.models import LdaMulticore, LdaModel
import pandas as pd
import os
import sys
import csv

#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#########################
# CONFIGURATION

# input
columns = ['ParagraphId', 'TokenId', 'Lemma', 'CPOS']   #, 'NamedEntity']   # columns to read from csv file
pos_tags = ['ADJ', 'NN', 'V']                        # parts-of-speech to include into the model, following dkpro's
                                            # coarse grained tagset: ADJ, ADV, ART, CARD, CONJ, N (NP, NN), O, PP, PR, V, PUNC

# stopwords
stopwordlist = "stopwords.txt"              # path to text file, e.g. stopwords.txt in the same directory as the script

# document size (in words)
#doc_size = 1000000                             # set to arbitrarily large value to use original doc size
doc_size = 1000                                 # the document size for LDA commonly ranges from 500-2000 words
doc_split = 0                                   # set to 1 to use the pipeline's ParagraphId feature instead of doc_size

# model parameters, cf. https://radimrehurek.com/gensim/models/ldamodel.html
no_of_topics = 20                               # no. of topics to be generated
no_of_passes = 100                              # no. of lda iterations - the more the better, but increases computing time

eval = 1                                        # perplexity estimation every n chunks - the smaller the better, but also increases computing time
chunk = 10                                      # documents to process at once

alpha = "auto"                             # "symmetric", "asymmetric", "auto", or array (default: a symmetric 1.0/num_topics prior)
                                                # affects sparsity of the document-topic (theta) distribution

# custom alpha may increase topic coherence, but may also produce more topics with zero probability
#alpha = np.array([ 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05,
#                   0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02])

eta = None                                      # can be a number (int/float), an array, or None
                                                # affects topic-word (lambda) distribution - not necessarily beneficial to topic coherence


#########################
# PRE-PROCESSING

def preprocessing(path, columns, pos_tags, doc_size, doc_split, stopwordlist):
    docs = []
    doc_labels = []
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

            df = pd.read_csv(filepath, sep="\t", quoting=csv.QUOTE_NONE)
            #df = pd.read_csv(filepath)
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
            #names = df.get_group('NP')['Lemma'].values.astype(str)
            #stopwords += names.tolist()

            # construct documents
            if doc_split:                               # size according to paragraph id
                doc = doc.groupby('ParagraphId')
                for para_id, para in doc:
                    docs.append(para['Lemma'].values.astype(str))
                    doc_labels.append(file.split(".")[0]+" #"+str(para_id))     # use filename + doc id as plot label
            else:                                       # size according to doc_size
                doc = doc.sort(columns='TokenId')
                i = 1
                while(doc_size < doc.shape[0]):
                    docs.append(doc[:doc_size]['Lemma'].values.astype(str))
                    doc_labels.append(file.split(".")[0]+" #"+str(i))
                    doc = doc.drop(doc.index[:doc_size])        # drop doc_size rows
                    i += 1
                docs.append(doc['Lemma'].values.astype(str))    # add the rest
                doc_labels.append(file.split(".")[0]+" #"+str(i))

    #for doc in docs: print(str(len(doc)))              # display resulting doc sizes
    #print(stopwords)

    print("\nnormalizing and vectorizing ...\n")        # cf. https://radimrehurek.com/gensim/tut1.html

    texts = [[word for word in doc if word not in stopwords] for doc in docs]       # remove stopwords

    all_tokens = sum(texts, [])                                                     # remove words that appear only once
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in texts]

    dictionary = Dictionary(texts)                      # vectorize
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus, doc_labels


#########################
# MAIN

if len(sys.argv) < 2:
    print("usage: {0} [folder containing csv files]\n"
          "parameters are set inside the script.".format(sys.argv[0]))
    sys.exit(1)

path = sys.argv[1]
foldername = path.split("/")[-1]

dictionary, corpus, doc_labels = preprocessing(path, columns, pos_tags, doc_size, doc_split, stopwordlist)


print("fitting the model ...\n")

model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=no_of_topics, passes=no_of_passes,
                 eval_every=eval, chunksize=chunk, alpha=alpha, eta=eta)

#model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=no_of_topics, passes=no_of_passes,
#                 eval_every=eval, chunksize=chunk, alpha=alpha, eta=eta)

print(model, "\n")

topics = model.show_topics(num_topics=no_of_topics)

for item, i in zip(topics, enumerate(topics)):
    print("topic #"+str(i[0])+": "+str(item)+"\n")


print("saving ...\n")

if not os.path.exists("out"): os.makedirs("out")

with open("out/"+foldername+"_doclabels.txt", "w") as f:
    for item in doc_labels: f.write(item+"\n")

with open("out/"+foldername+"_topics.txt", "w") as f:
    for item, i in zip(topics, enumerate(topics)):
        f.write("topic #"+str(i[0])+": "+str(item)+"\n")

dictionary.save("out/"+foldername+".dict")
MmCorpus.serialize("out/"+foldername+".mm", corpus)
model.save("out/"+foldername+".lda")
