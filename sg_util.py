import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf


def maybe_download(url='http://mattmahoney.net/dc/',filename='text8.zip',expected_bytes=31344016):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo=os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('veryfied')
    else :
        raise Exception(
            "size not match")
    return filename

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words,vocabulary_size):
    '''
    :param words: list of tokens ['this', 'is','dataset','for','training',....]
    :return: lists of tokens converted to their ids [1,4,2,5,7,....]
    '''
    count=[['UNK',0]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    data=[]
    cnt=0
    wordidx=dict()
    for word,_ in count:
        wordidx[word]=len(wordidx)
    for word in words:
        if not word in wordidx:
            data.append(0)
            cnt+=1
        else:
            data.append(wordidx[word])
    count[0][1] = cnt
    inv_wordidx=dict(zip(wordidx.values(),wordidx.keys()))
    return data,count,wordidx,inv_wordidx


def closest_words(sess,embeddings,inverse_dic,valid_examples,top_k=5):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_examples)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    nearests = tf.nn.top_k(similarity,top_k)[1]
    for word,nearest_words in enumerate(nearests.eval()):
        print('nearests to ',inverse_dic[valid_examples[word]],' :',end=' ')
        for nearest_word in nearest_words[1:]:
            print(inverse_dic[nearest_word],end=', ')
        print()

    return normalized_embeddings
