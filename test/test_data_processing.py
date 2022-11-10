# -*- coding:utf-8 -*-
"""
@file: test_data_processing.py
@time: 10/10/2022 12:58
@desc: 
@author: Echo
"""
import pytest
from data_processing.embeddings import save_movie_tfidf_embeddings, save_ratings_matrix_embeddings
import os
from config import config
from ext import movie_genres_list, load_embeddings, preprocessing

root_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(root_path)


class TestDataProcessing(object):

    def test_save_movie_tfidf_embeddings(self):
        filepath = project_path + config["embeddings"]["moviesTfidf"]
        save_movie_tfidf_embeddings(filepath)
        assert os.path.exists(filepath) == True

    def test_save_ratings_matrix_embeddings(self):
        filepath = project_path + config["embeddings"]["userRatingsMatrix"]
        save_ratings_matrix_embeddings(filepath)
        assert os.path.exists(filepath) == True

    def test_load_embeddings(self):
        filepath = project_path + config["embeddings"]["moviesTfidf"]
        embeddings = load_embeddings(filepath)
        assert len(embeddings) == len(preprocessing.get_items_list())
        assert len(embeddings[1]) == len(movie_genres_list)

    def test_get_users_movies(self):
        users_movies_dict = preprocessing.get_users_movies_ratings(1)
        assert users_movies_dict[1] == 4.0

    def test_get_user_unseen_movies_list(self):
        unseen_movies_list = preprocessing.get_user_unseen_movies_list(1, 0)
        assert len(unseen_movies_list) == 9492

    def test_get_one_user_ratings_with_movies_ids(self):
        movies_ratings_dict = preprocessing.get_one_user_ratings_with_movies_ids(1, [1, 3, 6, 47, 50])
        assert movies_ratings_dict == {1: 4.0, 3: 4.0, 6: 4.0, 47: 5.0, 50: 5.0}

    def test_dataset_split(self):
        training_movies_ids, test_movies_ids = preprocessing.dataset_split(1, 0.2)
        assert len(training_movies_ids) == 186
        assert len(test_movies_ids) == 46

    def test_get_movies_ids(self):
        movies_dict = preprocessing.get_movies_ids()
        assert movies_dict[9741] == 98296

    def test_get_movies_names_by_ids(self):
        movie_ids_list = [1, 2, 193609]
        movies_names_list = preprocessing.get_movies_names_by_ids(movie_ids_list)
        assert movies_names_list == ['Toy Story (1995)', 'Jumanji (1995)', 'Andrew Dice Clay: Dice Rules (1991)']


if __name__ == "__main__":
    pytest.main()
