# -*- coding:utf-8 -*-
"""
@file: test_content_based.py
@time: 10/10/2022 09:52
@desc: unit tests for content based class
@author: Echo
"""
import pytest
from ext import movie_genres_list, preprocessing
from recommender_system.content_based import ContentBasedRecommender


class TestContentBased(object):

    def test_tf_idf(self):
        word = "Comedy"
        count = 1
        genre_count_total = 5
        tf_idf = ContentBasedRecommender().tf_idf(word, count, genre_count_total)
        assert tf_idf == 0.0827613860344733

    def test_get_movie_matrix(self):
        movie_matrix_dict = ContentBasedRecommender().get_movie_matrix()
        assert len(movie_matrix_dict) == len(preprocessing.get_items_list())
        assert len(movie_matrix_dict[1]) == len(movie_genres_list)

    def test_get_user_profile(self):
        user_profile_matrix = ContentBasedRecommender().get_user_profile(2)
        assert len(user_profile_matrix) == len(movie_genres_list)

    def test_get_ordered_rankings(self):
        get_ordered_rankings = ContentBasedRecommender().get_ordered_rankings(1)
        assert len(get_ordered_rankings) == 9492

    def test_get_user_input_ratings_matrix(self):
        movie_ids, ratings = ContentBasedRecommender().get_user_input_ratings_matrix(2)
        assert len(movie_ids) == 29
        assert len(ratings) == 29


if __name__ == "__main__":
    pytest.main()