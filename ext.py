# -*- coding:utf-8 -*-
"""
@file: ext.py
@time: 10/10/2022 10:28
@desc: ext tools for initialization
@author: Echo
"""
import os
import pickle
from config import config
from data_processing.preprocessing import Preprocessing


root_path = os.path.dirname(os.path.abspath(__file__))

movie_genres_list = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime', 'Sci-Fi', 'Horror',
                     'Fantasy', 'Children', 'Animation', 'Mystery', 'Documentary', 'War', 'Musical', 'Western', 'IMAX',
                     'Film-Noir', 'other']


def load_embeddings(path):
    embeddings = pickle.load(open(path, "rb"))
    return embeddings


movies_embeddings = load_embeddings(root_path + config["embeddings"]["moviesTfidf"])
rating_matrix_embeddings = load_embeddings(root_path + config["embeddings"]["userRatingsMatrix"])
preprocessing = Preprocessing()
