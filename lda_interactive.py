#!/usr/bin/env python

from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary
import sys, os
import pyLDAvis.gensim


if len(sys.argv) < 2:
    print("usage: {0} [path to model.lda]\n".format(sys.argv[0]))
    sys.exit(1)


path, file = os.path.split(sys.argv[1])
corpusname = file.split(".")[0]

dictionary = Dictionary.load(path+"/"+corpusname+".dict")
corpus = MmCorpus(path+"/"+corpusname+".mm")
model = LdaModel.load(sys.argv[1])


##############
# cf. https://pyldavis.readthedocs.org/en/latest/modules/API.html

vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)

pyLDAvis.save_html(vis, path+"/"+corpusname+"_interactive.html")
pyLDAvis.show(vis)