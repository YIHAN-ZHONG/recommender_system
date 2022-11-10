# -*- coding:utf-8 -*-
"""
@file: embeddings.py
@time: 10/10/2022 13:02
@desc: 
@author: Echo
"""
import pickle

from recommender_system.collaborative_filtering import CollaborativeFilteringUsersBased
from recommender_system.content_based import ContentBasedRecommender


def save_movie_tfidf_embeddings(path):
    movies_tfidf_matrix = ContentBasedRecommender().get_movie_matrix()
    with open(path, 'wb') as pkl:
        pickle.dump(movies_tfidf_matrix, pkl)


def save_ratings_matrix_embeddings(path):
    ratings_matrix = CollaborativeFilteringUsersBased().get_ratings_matrix()
    with open(path, 'wb') as pkl:
        pickle.dump(ratings_matrix, pkl)

