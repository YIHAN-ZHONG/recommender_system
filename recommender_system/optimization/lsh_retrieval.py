# -*- coding:utf-8 -*-
"""
@file: lsh_retrieval.py
@time: 07/10/2022 16:30
@desc: retrieval using lsh
@author: Echo
"""
import time

from ext import preprocessing
from recommender_system.evaluation import Evaluation
from recommender_system import MovieRecommenderSystem
import numpy as np


class LSHRetrieval(MovieRecommenderSystem):
    def __init__(self):
        self.tables_num = 10
        self.a = 5
        self.R = np.random.random([20, self.tables_num])
        self.b = np.random.uniform(0, self.a, [1, self.tables_num])
        self.hash_tables = [dict() for _ in range(self.tables_num)]
        from ext import movies_embeddings
        self.movies_matrix_dict = {str(v): k for k, v in movies_embeddings.items()}
        self.docs = movies_embeddings

    def _hash(self, inputs):
        """
        mapping vectors to hash_table index
        :param inputs: input vectors
        :return: Each row represents all indices of a vector output, and each column represents an index in a hash_table
        """
        """H(V) = |V·R + b| / a，R is the random variable，a is the band width，b is a random variable uniformly 
        distributed between [0,a]"""
        hash_val = np.floor(np.abs(np.matmul(inputs, self.R) + self.b) / self.a)
        return hash_val

    def build_dataset(self):
        data = []
        for item in self.docs.items():
            data.append(item[1])
        data = np.array(data)
        return data

    def insert(self):
        """
        Map the vector to the index of the corresponding hash_table and insert into all hash_tables
        :param inputs:
        :return:
        """
        """Convert the inputs to a 2D vector"""
        inputs = self.build_dataset()
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        hash_index = self._hash(inputs)
        for inputs_one, indexs in zip(inputs, hash_index):
            for i, key in enumerate(indexs):
                """i is the ith hash_table, and key is the index position of the current hash_table, 
                inputs_one is the current vector"""
                self.hash_tables[i].setdefault(key, []).append(tuple(inputs_one))

    def query(self, user_id, top_k=100):
        """
        Query vectors that are similar to inputs and output nums with the highest similarity
        :param user_id: input vectors
        :param top_k:
        :return:
        """
        query_movie_id = list(preprocessing.get_users_movies_ratings(user_id).keys())
        # inputs_vectors = []
        # for movie_id in query_movie_id:
        #     inputs_vectors.append(self.docs[movie_id])
        inputs_vectors=[self.docs[query_movie_id[0]]]
        movies_id = []
        hash_val = self._hash(inputs_vectors).ravel()
        candidates = set()
        s = time.time()
        """Add a vector at the same index position to the candidate set"""
        for i, key in enumerate(hash_val):
            candidates.update(self.hash_tables[i][key])
        e = time.time()
        print("fff", e - s)
        """Sort by vector distance"""
        candidates = sorted(candidates, key=lambda x: Evaluation().get_cosine_similarity([x], [inputs_vectors[0]]))[
                     : int(top_k)]

        for val in candidates:
            try:
                movies_id.append(
                    self.movies_matrix_dict[str(list(val)).replace("0.0,", "0,").replace("0.0]", "0]")])
            except Exception:
                continue

        return list(set(movies_id))


# if __name__ == '__main__':
#     import time
#     query = 1
#     lsh = LSHRetrieval()
#     lsh.insert()
#     start = time.time()
#     res = lsh.query(1, 100)
#     end = time.time()
#     print(end - start)
#     res = np.array(res)
#     # print(res)
#     print("len", len(res))