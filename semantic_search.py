# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:36:56 2020

@author: Akush
"""
from typing import Dict, Tuple, List
from collections import defaultdict
import subprocess, os, sys, codecs, getopt
import numpy as np
import mmap
import contextlib
from tqdm import tqdm
from scipy.spatial.distance import braycurtis, cosine, canberra, chebyshev, cityblock, euclidean, hamming, minkowski

class Embeddings:
    def __init__(self):
        self.embedding = defaultdict(dict)

    def load(self, path):
        try:
            with open(path, 'r') as f:
                with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                    for line in tqdm(iter(m.readline, b'')):
                         token = line.decode('utf-8').rstrip()
                         token = token.split(' ')
                         self.embedding[token[0]] = token[1:]      
            print("Embeddings Loaded..")
            return self
        except Exception as e:
            print("Below Given Error Occured during loading:\n",str(e))

class SearchEngine:    
    def __init__(self, document_vector=None, embedding=None):
        self.document_vec = document_vector
        self.embedding = embedding if embedding else defaultdict(dict)    
        self.method = "default"
    
    @classmethod
    def update_weights(cls, document_vector, embedding):
        return cls(document_vector, embedding) 
    
    def tokenizer(self, documents: List, vocab=False):
        vocabulary = set() 
        sentence_tokens = defaultdict(list)
        for i,sentence in enumerate(documents):
            sentence_tokens[i] = sentence.strip().split(' ')
            if vocab:
                vocabulary.update(sentence_tokens[i])
        return sentence_tokens, vocabulary   
    
    def document_vectors(self, corpus: List):
        sentence_tokens, _ = self.tokenizer(corpus)
        n_row = len(sentence_tokens)
        n_col = 300      
        
        self.document_vec = np.zeros((n_row, n_col), dtype='float32')
        
        try:
            for i, sentence_token in tqdm(sentence_tokens.items()):
                for word in sentence_token:
                    word_vector = self.embedding.get(word)
                    if word_vector:
                        word_vector = np.asarray(word_vector, dtype='float32')
                        self.document_vec[i] = self.document_vec[i] + word_vector/np.linalg.norm(word_vector)
                    else:
                        word_vector = np.zeros((1, n_col), dtype='float32')
                        self.document_vec[i] = self.document_vec[i] + word_vector

            for index in range(len(self.document_vec)):
                self.document_vec[index] = self.document_vec[index]/np.linalg.norm(self.document_vec[index])
            print("Corpus successfully converted to Vector..")
        except Exception as e:
            print("Below Given Error Occured during Ops:\n",str(e))
            
    def update_similarity_method(self, method_name: str):
        self.method = method_name
        return self
           
    def query_vectorizer(self, query: str):
        sentence_tokens, _ = self.tokenizer([query])
        n_row = len(sentence_tokens)
        n_col = 300      
        query_vec = np.zeros((n_row, n_col), dtype='float32')

        try:
            for i, sentence_token in sentence_tokens.items():
                for word in sentence_token:
                    word_vector = self.embedding.get(word)
                    if word_vector:
                        word_vector = np.asarray(word_vector, dtype='float32')
                        query_vec[i] = query_vec[i] + word_vector/np.linalg.norm(word_vector)
                    else:
                        print("log: word not found")
            if query_vec.all():
                for index in range(len(query_vec)):
                    query_vec[index] = query_vec[index]/np.linalg.norm(query_vec[index])
            print("Query successfully converted to Vector..")
            return query_vec
        
        except Exception as e:
            print("Below Given Error Occured during ops:\n",str(e))

        
    def select_method(self):
        if self.method == "braycurtis":
            return braycurtis            
        if self.method == "cosine":
            return cosine
        if self.method == "canberra":
            return canberra
        if self.method == "chebyshev":
            return chebyshev
        if self.method == "cityblock":
            return cityblock
        if self.method == "euclidean":
            return euclidean
        if self.method == "hamming":
            return hamming
        if self.method == "minkowski":
            return minkowski
                
    def similarity_vector(self, query_vector):
        similarity = []
        try:
            if self.method == "default":
                similarity = self.document_vec.dot(query_vector.T)
                
            else:
                method = self.select_method()
                for d_vec in self.document_vec:
                    similarity_score = 1 - method(d_vec,query_vector)
                    similarity.append(similarity_score)
                similarity = np.asarray(similarity, dtype='float32')
            return similarity
        except Exception as e:
            print("Below Given Error Occured during ops:\n",str(e))
    
    def search(self, query):
        query_vec = self.query_vectorizer(query)
        similarity = self.similarity_vector(query_vec)
        return similarity  #starting with index 0

    @staticmethod
    def test(query):
        pass
    
def read_document(path):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        sentences = [x.strip() for x in sentences]
    return sentences

if __name__=='__main__':
    try:
        EMBEDDING_PATH = None
        DOCUMENT_PATH = None
        SEARCH_METHOD = "default"
        method_list = ["braycurtis", "cosine", "canberra", "chebyshev", "cityblock", "euclidean", "hamming", "minkowski"]
        # Keep all but the first
        argument_list = sys.argv[1:]
        short_options = "e:d:"
        long_options = ["embedding", "documents"]
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
        for current_argument, current_value in arguments:
            if current_argument in ("-e", "--embedding"):
                EMBEDDING_PATH = current_value
            elif current_argument in ("-d", "--documents"):
                DOCUMENT_PATH = current_value
        
        doc = read_document(DOCUMENT_PATH)  
        
        emb = Embeddings()
        emb.load(EMBEDDING_PATH)
        
        engine = SearchEngine(emb.embedding)
        engine.document_vectors(doc)
        
        while True:
            query = str(input())
            if query != "quit":
                SEARCH_METHOD = input("Enter search method: ")
                SEARCH_METHOD = SEARCH_METHOD if SEARCH_METHOD in method_list else "default"
                engine.update_similarity_method(SEARCH_METHOD)
                print(engine.search(query))        
                
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)
        
        
        
        
        
        
        
        
        
