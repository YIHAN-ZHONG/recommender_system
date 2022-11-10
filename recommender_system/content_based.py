# -*- coding:utf-8 -*-
"""
@file: content_based.py
@time: 07/10/2022 16:30
@desc: A class of content based recommender system
@author: Echo
"""
import math
import numpy as np
from ext import movie_genres_list, movies_embeddings, preprocessing
from recommender_system import MovieRecommenderSystem
from operator import itemgetter
from recommender_system.evaluation import Evaluation
from recommender_system.manager import faiss_retrieval, lsh_retrieval

"""
1. generate the movie matrix based on the genres using tf-idf
2. generate the users ratings towards different movies, to get a weighted genre matrix 
based on the movies that the user has watched
3. aggregate the weighted genres and then normalize them to get the user profile
4. get the movies that have not been watched by the user, get the candidate movies matrix
5. multiply user's profile with the movie matrix to get the weighted movies matrix 
and recommendation matrix which is the recommendation list, return the movies with higher scores.

"""


class ContentBasedRecommender(MovieRecommenderSystem):
    def __init__(self):
        super().__init__()
        self.movies_genres = preprocessing.get_genres_frequency()
        self.all_items_dict = preprocessing.get_items_list()

    def get_user_input_ratings_matrix(self, user_id) -> (list, list):
        """
        get users ratings matrix.
        :param user_id:
        :return:
        """
        user_input_ratings = preprocessing.get_users_movies_ratings(user_id)
        ratings = list(user_input_ratings.values())
        movie_ids = list(user_input_ratings.keys())
        return movie_ids, ratings

    def get_user_profile(self, user_id) -> list:
        """
        get normalized user profile matrix based on movies genres.
        :param user_id: user id.
        :return norm_user_profile_matrix: 1-d vector, normalized user profile matrix,
                    len(norm_user_profile_matrix) == len(movie_genres_list)
        """
        movie_ids, ratings = self.get_user_input_ratings_matrix(user_id)
        users_movies_ratings = itemgetter(*movie_ids)(movies_embeddings)
        """input user ratings multiple corresponding movies matrix"""
        user_profile_matrix = np.dot(ratings, users_movies_ratings)
        """normalization the matrix"""
        norm_user_profile_matrix = (user_profile_matrix - user_profile_matrix.min(axis=0)) \
                                   / (user_profile_matrix.max(axis=0) - user_profile_matrix.min(axis=0))
        return norm_user_profile_matrix

    def get_movie_matrix(self) -> dict:
        """
        generate the movie matrix based on the genres using tf-idf
        :return movie_matrix:
        """
        """
        generate the movie matrix based on the genres using tf-idf
        :return movie_matrix: the number of rows are the number of items,
        the columns are corresponed to the different genres
        """

        length_genres = len(movie_genres_list)
        movie_matrix_dict = {}
        for i, items_properties in enumerate(self.all_items_dict.values()):
            movie_matrix = [0] * length_genres
            for j, genre in enumerate(movie_genres_list):
                if genre in items_properties:
                    genre_count = items_properties.count(genre)
                    genre_count_total = len(items_properties)
                    movie_matrix[j] = self.tf_idf(genre, genre_count, genre_count_total)
            movie_matrix_dict[list(self.all_items_dict.keys())[i]] = movie_matrix

        return movie_matrix_dict

    def get_ordered_rankings(self, user_id, test_size, retrieval_method=None, top_k=10) -> dict:
        """

        :param user_id: user id.
        :param test_size: test dataset size, if it's 0, then use all the data to train.
        :param retrieval_method:
        :param top_k:
        :return:
        """
        if retrieval_method == "faiss":
            unseen_movies_list = faiss_retrieval.query(user_id, top_k)
        elif retrieval_method == "lsh":
            unseen_movies_list = lsh_retrieval.query(user_id, top_k)
        else:
            unseen_movies_list = preprocessing.get_user_unseen_movies_list(user_id, test_size)
        print("unseen_movies_list", unseen_movies_list)
        """get unseen movies embeddings"""
        unseen_movies_embeddings = itemgetter(*unseen_movies_list)(movies_embeddings)
        """get user profile"""
        user_profile_matrix = self.get_user_profile(user_id)
        """get cosine similarities between users and items, then sorted the recommender rankings"""
        cosine_similarities = Evaluation().get_cosine_similarity([user_profile_matrix],
                                                                 np.array(list(unseen_movies_embeddings)))

        cosine_similarities, unseen_movies_list = zip(
            *sorted(zip(cosine_similarities, unseen_movies_list), reverse=True))
        movies_id_with_cosine_similarities = dict(zip(unseen_movies_list, cosine_similarities))
        return movies_id_with_cosine_similarities

    def tf_idf(self, word: str, count: int, genre_count_total: int) -> float:
        """
        compute the tf-idf value for each item

            tf: it shows how often a given word appears in the file. This number is normalized to the number of words
                to prevent it from skewing towards long files. The numerator is the number of frequency of the word in
                the document, and the denominator is the sum of the frequency of all the words in the document.

            idf: is a measure of the universal importance of a word. The idf for a particular term can be obtained by
                 by the total number of documents divided by the number of documents containing the term.
                 Then take the base 10 logarithm of the obtained quotient to get it.

        :param word: the word that needs to be calculated.
        :param count: the number of frequency of the word in the document.
        :param genre_count_total: the sum of the frequency of all the words in the document.
        :return: tf-idf value.
        """
        """compute tf"""
        tf = count / genre_count_total
        """compute how many items have the word"""
        idf_n = 0
        for items in self.all_items_dict.values():
            if word in items:
                idf_n += 1

        """compute idf"""
        idf = math.log(len(self.all_items_dict) / (idf_n + 1), 10)
        """compute tf-idf"""
        tf_idf = tf * idf
        return tf_idf
