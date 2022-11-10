# -*- coding:utf-8 -*-
"""
@file: test_collaborative_filtering.py
@time: 07/10/2022 16:29
@desc: unit tests for Collaborative Filtering class
@author: Echo
"""
import pytest
from ext import rating_matrix_embeddings, preprocessing
from recommender_system.collaborative_filtering import CollaborativeFilteringUsersBased


class TestCollaborativeFiltering(object):
    def test_jaccard_similarity(self):
        pass

    def test_get_users_similarity_matrix(self):
        users_cos_sim = CollaborativeFilteringUsersBased().get_users_similarity_matrix(1)
        assert len(users_cos_sim) == len(rating_matrix_embeddings)

    def test_generate_ratings_matrix_subset(self):
        recommendation_users_ratings_matrix = CollaborativeFilteringUsersBased().generate_ratings_matrix_subset(2, 0.2)
        unseen_movies_list = preprocessing.get_user_unseen_movies_list(2, 0.2)
        assert len(recommendation_users_ratings_matrix) == len(rating_matrix_embeddings)
        assert len(recommendation_users_ratings_matrix[0]) == len(unseen_movies_list)

    def test_predicting_ordered_ratings(self):
        predicting_ordered_ratings = CollaborativeFilteringUsersBased().predicting_ordered_ratings(2, 0.2)
        print(predicting_ordered_ratings)


if __name__ == "__main__":
    pytest.main()
