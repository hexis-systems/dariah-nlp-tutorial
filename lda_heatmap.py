#!/usr/bin/env python

from gensim.corpora import MmCorpus
from gensim.models import LdaModel
import numpy as np
import matplotlib.pyplot as plt
import sys, os


if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]


# load model

doc_labels = []
with open(path+"/"+corpusname+"_doclabels.txt", "r") as f:
    for line in f: doc_labels.append(line)

corpus = MmCorpus(path+"/"+corpusname+".mm")
model = LdaModel.load(sys.argv[1])

no_of_topics = model.num_topics
no_of_docs = len(doc_labels)


# get doc-topic matrix

doc_topic = np.zeros((no_of_docs, no_of_topics))

for doc, i in zip(corpus, range(no_of_docs)):           # use document bow from corpus
    topic_dist = model.__getitem__(doc)                 # to get topic distribution from model
    for topic in topic_dist:                            # topic_dist is a list of tuples (topic_id, topic_prob)
        doc_topic[i][topic[0]] = topic[1]               # save topic probability

# get plot labels

topic_labels = []
for i in range(no_of_topics):
    topic_terms = [x[0] for x in model.show_topic(i, topn=3)]           # show_topic() returns tuples (word_prob, word)
    topic_labels.append(" ".join(topic_terms))

#print(doc_topic)
#print(doc_topic.shape)


# cf. https://de.dariah.eu/tatom/topic_model_visualization.html

if no_of_docs > 20 or no_of_topics > 20: plt.figure(figsize=(20,20))    # if many items, enlarge figure
plt.pcolor(doc_topic, norm=None, cmap='Reds')
plt.yticks(np.arange(doc_topic.shape[0])+1.0, doc_labels)
plt.xticks(np.arange(doc_topic.shape[1])+0.5, topic_labels, rotation='90')
plt.gca().invert_yaxis()
plt.colorbar(cmap='Reds')
plt.tight_layout()

plt.savefig(path+"/"+corpusname+"_heatmap.png") #, dpi=80)
#plt.show()
