#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
from math import log, exp
from functools import partial
import re
import os
import random
import pickle
import pylab
import sys
import cPickle

#handle=open("trained","rb")
#sums, positive, negative =pickle.load(handle)


def processTweet(tweet):
     tweet=tweet.lower()
     tweet=re.sub('((www\.[\s]+)|(http?://[^\s]+))','URL',tweet)
     tweet=re.sub('@[^\s]+',' ',tweet)
     tweet=re.sub('[\s]+',' ',tweet)
     tweet=re.sub(r'#([^\s]+)',' ',tweet)
     tweet=tweet.strip('\'"')
     return tweet


class MyDict(dict):
      def __getitem__(self, key):
          if key in self:
              return self.get(key)
          return 0

pos=MyDict()
neg=MyDict()
features=set()
totals=[0,0]
delchars=''.join(c for c in map(chr, range(128)) if not c.isalnum())
CDATA_FILE="countdata.pickle"
FDATA_FILE="reduceddata.pickle"

def negate_sequence1(text):
    negation=False
    delims="?.,!:;"
    result=[]
    words=text.split()
    prev=None
    pprev=None
    for word in words:
       stripped=word.strip(delims).lower()
       negated="not_" + stripped if negation else stripped
       result.append(negated)
       if prev:
          bigram=prev+" "+negated
          result.append(bigram)
          if pprev:
               trigram=pprev+" "+bigram
               result.append(trigram)
          pprev=prev
       prev=negated

       if any(neg in word for neg in ["not","n't","no"]):
            negation= not negation

       if any(c in word for c in delims):
            negation=False
    return result

def train():
     global pos, neg, totals
     dir1=os.environ['dir1']
     dir2=os.environ['dir2']
     dir3=os.environ['dir3']
     dir4=os.environ['dir4']

     retrain=False

     if not retrain and os.path.isfile(CDATA_FILE):
          pos, neg, totals=cPickle.load(open(CDATA_FILE))
          return

     limit=12500
     for file in os.listdir(dir1)[:limit]:
          for word in set(negate_sequence1(open(dir2+file).read())):
             pos[word]+=1
             neg['not_'+word]+=1
     for file in os.listdir(dir3)[:limit]:
          for word in set(negate_sequence1(open(dir4+file).read())):
             neg[word]+=1
             pos['not_'+word]+=1
     prune_features()

     totals[0]=sum(pos.values())
     totals[1]=sum(neg.values())

     countdata=(pos,neg,totals)
     cPickle.dump(countdata, open(CDATA_FILE, 'w'))

def MI(word):
    T=totals[0]+totals[1]
    W=pos[word]+neg[word]
    I=0
    if W==0:
       return 0
    if neg[word]>0:
      I+=(totals[1]-neg[word])/T*log((totals[1]-neg[word])*T/(T-W)/totals[1])
      I+=neg[word]/T*log(neg[word]*T/W/totals[1])
    if pos[word]>0:
      I+=(totals[0]-pos[word])/T*log((totals[0]-pos[word])*T/(T-W)/totals[0])
      I+=pos[word]/T*log(pos[word]*T/W/totals[0])
    return I
def get_relevant_features():
    pos_dump=MyDict({k:pos[k] for k in pos if k in features})
    neg_dump=MyDict({k:neg[k] for k in neg if k in features})
    totals_dump=[sum(pos_dump.values()), sum (neg_dump.values())]
    return (pos_dump, neg_dump, totals_dump)


def prune_features():
    global pos, neg
    for k in pos.keys():
        if pos[k] <= 1 and neg[k]<=1:
             del pos[k]
    for k in neg.keys():
        if neg[k]<=1 and pos[k]<=1:
             del neg[k]

def feature_selection_trials():
    global pos, neg, totals, features
    dir5=os.environ['dir5']
    dir6=os.environ['dir6']
    dir7=os.environ['dir7']
    dir8=os.environ['dir8']

    retrain=True
    if not retrain and os.path.isfile(FDATA_FILE):
        pos, neg,totals=cPickle.load(open(FDATA_FILE))
        return
    words=list(set(pos.keys()+neg.keys()))
  #  print "Total no of features:", len(words)
    words.sort(key=lambda w: -MI(w))
    num_features, accuracy =[], []
    bestk=0
    limit=500
    step=500
    start=20000
    best_accuracy=0.0
    for w in words[:start]:
        features.add(w)
    for k in xrange(start, 40000, step):
        for w in words[k:k+step]:
            features.add(w)
        correct=0
        size=0

        for file in os.listdir(dir5)[:limit]:
            correct+=classify(open(dir6 +file).read())==True
            size+=1
        for file in os.listdir(dir7)[:limit]:
            correct +=classify(open(dir8+file).read())==False
            size+=1
        num_features.append(k+step)
        accuracy.append(correct/size)
        if (correct/size)>best_accuracy:
            bestk=k
 #       print k+step, correct/size
    features=set(words[:bestk])
    cPickle.dump(get_relevant_features(),open(FDATA_FILE, 'w'))


def tokenize(text):
    return re.findall("\w+",text)

def negate_sequence(text):
    negation=False
    delims= "?.,!:;"
    result=[]
    words=text.split()
    for word in words:
       stripped=word.strip(delims).lower()
       result.append("not_" + stripped if negation else stripped)

       if any(neg in word for neg in frozenset(["not","'t", "no"])):
             negation=not negation

       if any(c in word for c in delims):
             negation= False
    return result

def get_positive_prob(word):
      return 1.0 *(pos[word]+ 1)/(2 *totals[0])

def get_negative_prob(word):
      return 1.0 *(neg[word]+1)/(2*totals[1])

def classify(text, pneg=0.5, preprocessor=negate_sequence):
       words=preprocessor(text)
       pscore, nscore=0,0

       for word in words:
          pscore=log(get_positive_prob(word))
          nscore=log(get_negative_prob(word))

       return pscore>nscore

def classify_demo(text):
    words=negate_sequence(text)
    pscore, nscore=0,0

    for word in words:
        pdelta=log(get_positive_prob(word))
        ndelta=log(get_negative_prob(word))
        pscore+=pdelta
        nscore+=ndelta
#        print "%25s,pos=(%lf, %10d) \t\t neg=(%10lf, %10d)" % (word, pdelta, pos[word], ndelta, neg[word])
#    print "\nPositive" if pscore > nscore else "Negative"
#    print "Cofidence: %lf" % exp(abs(pscore-nscore))
#    return pscore> nscore, pscore, nscore
    return  "Positive" if pscore > nscore else "Negative",  exp(abs(pscore-nscore))

 

#if __name__=='__main__'
#train()
#feature_selection_trials()
#f=open('classify_demo.pickle','wb')
#pickle.dump(classify_demo,f)
#f.close()
#      value=classify_demo(processTweet("I like this movie a lot"))
#      print value[0]


for line in sys.stdin:
     keyval=line.strip().split("\t")
     key,val=(keyval[0],keyval[1])
     train()
     feature_selection_trials()
     value=classify_demo(processTweet(val))[0]
     print "%s\t%s" % (key,value)






