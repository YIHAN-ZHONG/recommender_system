# -*- coding:utf-8 -*-
"""
@file: collaborative_filtering.py
@time: 07/10/2022 16:30
@desc:  collaborative filtering class
@author: Echo
"""
from operator import itemgetter
import numpy as np
from ext import rating_matrix_embeddings, preprocessing
from recommender_system import MovieRecommenderSystem
from recommender_system.content_based import ContentBasedRecommender
from recommender_system.evaluation import Evaluation

"""
1. generate users ratings matrix based on the movies that different users have watched.
2. calculate the similarity matrix between different users based on the movies that they have watched.
3. get the subset of ratings matrix from the users who have watched recommended candidates movies.
4. multiple the subset ratings matrix and similarity matrix, to get the weighted ratings matrix.
5. sum the weighted ratings for recommended candidates movies according to different users.
6. normalize the values by diving the sum with the sum of cosine similarity from different users.
"""


class CollaborativeFilteringUsersBased(MovieRecommenderSystem):
    def __init__(self):
        self.movie_ids_and_index = preprocessing.get_movies_ids()
        self.users_ids = preprocessing.get_user_ids()

    def get_ratings_matrix(self):
        """
        get ratings matrix, the number of rows is the number of users; the number of columns is the number of movies;
        each value is the rating that one user gives to one movie.
        :return:
        """
        ratings_matrix = [[0] * len(self.movie_ids_and_index) for _ in range(len(self.users_ids))]
        """switch the index to value and value to index"""
        movie_index_ids = dict(zip(self.movie_ids_and_index.values(), self.movie_ids_and_index.keys()))
        """ get rating matrix according to the ratings one user gives to all the movies they watched"""
        for user_id in self.users_ids:
            movie_ids, ratings = ContentBasedRecommender().get_user_input_ratings_matrix(user_id)
            for i, movie_id in enumerate(movie_ids):
                if movie_id in movie_index_ids.values():
                    ratings_matrix[user_id - 1][movie_index_ids[movie_id]] = ratings[i]
        return ratings_matrix

    def generate_ratings_matrix_subset(self, user_id, test_size):
        """
        generate the subset rating matrix, which can be used to predict the unseen movies for the target user
        according to the similar users.
        """
        """get the unseen movies list of the target user"""
        unseen_movies_list = preprocessing.get_user_unseen_movies_list(user_id, test_size)
        movie_index_ids = dict(zip(self.movie_ids_and_index.values(), self.movie_ids_and_index.keys()))
        unseen_movies_ids_index = list(itemgetter(*unseen_movies_list)(movie_index_ids))

        """get the subset ratings matrix from the users who have watched these movies"""
        subset_ratings_matrix = np.array(rating_matrix_embeddings)[:, unseen_movies_ids_index]
        return subset_ratings_matrix

    def get_users_similarity_matrix(self, user_id) -> np.ndarray:
        """
        get the cosine similarity matrix between one user and the rest of the users.
        :param user_id:
        :return:
        """
        """note: we need to minus 1 here, as the user_id starts from 1 instead 0."""
        matrix_a = [rating_matrix_embeddings[user_id - 1]]
        matrix_b = rating_matrix_embeddings
        users_cos_sim = Evaluation().get_cosine_similarity(matrix_a, matrix_b)
        return users_cos_sim

    def predicting_ordered_ratings(self, user_id, test_size):
        """

        :param user_id:
        :return:
        """
        """generate weighted rating matrix by using ratings matrix subset * similarity matrix """
        subset_ratings_matrix = self.generate_ratings_matrix_subset(user_id, test_size)
        users_cos_sim = self.get_users_similarity_matrix(user_id)
        weighted_ratings_matrix = np.multiply(users_cos_sim.T, subset_ratings_matrix.T).T

        """sum the weighted ratings according to different movies, the column of matrix is the number of movies"""
        sum_weighted_ratings_matrix = np.sum(weighted_ratings_matrix, axis=0)
        """normalize the values by diving the sum with the sum of cosine similarity from different users"""
        weight_sum_list = []
        for index_movie in range(len(weighted_ratings_matrix[0])):
            rated_users_index = np.nonzero(weighted_ratings_matrix[:, index_movie])
            weight_sum = np.sum(users_cos_sim[rated_users_index])
            weight_sum_list.append(weight_sum)
        """handling the 0 values for denominators."""
        recommendation_rankings = (sum_weighted_ratings_matrix.T / np.array(weight_sum_list).T).T
        recommendation_rankings[np.isnan(recommendation_rankings)] = 0
        unseen_movies_list = preprocessing.get_user_unseen_movies_list(user_id, test_size)

        recommendation_rankings, unseen_movies_list = zip(
            *sorted(zip(recommendation_rankings, unseen_movies_list), reverse=True))
        movies_id_with_recommendation_rankings = dict(zip(unseen_movies_list, recommendation_rankings))

        return movies_id_with_recommendation_rankings
