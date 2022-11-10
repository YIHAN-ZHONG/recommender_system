# -*- coding:utf-8 -*-
"""
@file: faiss_retrieval.py
@time: 01/11/2022 23:37
@desc: optimization using faiss as retrieval
@author: Echo
"""
import time
from ext import preprocessing
from recommender_system import MovieRecommenderSystem
import numpy as np
import faiss
from operator import itemgetter


np.random.seed(42)


class FaissRetrieval(MovieRecommenderSystem):
    def __init__(self):
        self.dims = 20
        from ext import movies_embeddings
        self.docs = movies_embeddings
        self.index = self.insert_data()
        self.movies_dict_ids = preprocessing.get_movies_ids()


    def build_dataset(self):
        """
        build the dataset using the embeddings of movies
        :return:
        """
        data = []
        for item in self.docs.items():
            data.append(item[1])
        data = np.array(data).astype('float32')
        return data

    def insert_data(self):
        """
        insert data into vector spaces
        :return:
        """
        """build index and insert data"""
        index = faiss.IndexFlatL2(self.dims)
        data = self.build_dataset()
        """add vectors to the index"""
        index.add(data)
        return index

    def query(self, user_id: int, k: int) -> list:
        """
        query the most similar k movies according to the movies that the user has watched.
        :param user_id: user id
        :param k: the number of top_k similar items
        :return:
        """

        query_movie_id = list(preprocessing.get_users_movies_ratings(user_id).keys())

        query_docs = []
        for movie_id in query_movie_id:
            query_docs.append(self.docs[movie_id])
        query = np.array(query_docs).astype('float32')



        distance, movie_ids = self.index.search(query, k)


        top_k_movie_ids = itemgetter(*movie_ids[0])(self.movies_dict_ids)

        """filter the movies that the user has already seen"""
        output = list(set(top_k_movie_ids).difference(set(query_movie_id)))

        return output


start = time.time()
print(FaissRetrieval().query(1, 100))
end = time.time()
print(end-start)