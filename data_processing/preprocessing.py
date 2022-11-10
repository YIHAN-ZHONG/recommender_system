# -*- coding:utf-8 -*-
"""
@file: preprocessing.py
@time: 09/10/2022 18:47
@desc: data preprocessing for the input files
@author: Echo
"""
import pandas as pd
from config import config
import os

root_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(root_path)


class Preprocessing:
    def __init__(self):
        self.movies_data = self.load_data(project_path + config["dataset"]["moviesPath"])
        self.ratings_data = self.load_data(project_path + config["dataset"]["ratingPath"])

    def load_data(self, path) -> pd.DataFrame:
        """
        load data
        :param path: file path
        :return: data in DataFrame type.
        """
        data = pd.read_csv(path)
        return data

    def get_movies_ids(self) -> dict:
        """
        get all movies ids
        :return: movies ids dict. Values are movies ids, keys are the index of each movie.
        """
        data = self.movies_data
        movies_ids = data["movieId"]
        index = range(len(set(movies_ids)))
        movies_dict = zip(index, list(set(movies_ids)))
        return dict(movies_dict)

    def get_movies_names_by_ids(self, movie_ids_list: list) -> list:
        """
        get movies names by corresponding ids
        :param movie_ids_list:
        :return movies_names:
        """
        data = self.movies_data
        data = data.set_index("movieId")
        movies_names_list = list(data.loc[movie_ids_list]["title"])
        return movies_names_list

    def get_user_ids(self) -> list:
        """
        get all users ids
        :return: users ids list
        """
        data = self.ratings_data
        user_ids = data["userId"]
        return list(set(user_ids))

    def get_items_list(self) -> dict:
        """
        get the movies for different genres. Each item is one movie with the corresponding genres.
        :return: an item dict which the keys are movie_ids and the values are corresponding genres.
        """
        data = self.movies_data
        all_items_dict = dict((data.genres.str.split('|')))
        result = {data.movieId[i]: v for i, (k, v) in enumerate(all_items_dict.items())}
        return result

    def get_genres_frequency(self) -> dict:
        """
        get the frequency of different genres
        :return: a frequency dict which the keys are genres and the values are frequency
        """
        data = self.movies_data
        movies_genres = dict(data.genres.str.split('|').explode().value_counts())
        return movies_genres

    def get_one_user_ratings_with_movies_ids(self, user_id, movies_ids_list) -> dict:
        """

        :param user_id:
        :param movies_ids_list:
        :return:
        """
        data = self.ratings_data
        data = data.set_index("movieId")
        movies_ratings_dict = data[data["userId"] == user_id].loc[movies_ids_list]["rating"]
        return dict(movies_ratings_dict)

    def get_users_movies_ratings(self, user_id: int) -> dict:
        """
        get the movies and corresponding ratings by a user
        :param user_id: user id
        :return: a ratings' dict which the keys are movie_ids and the values are ratings
        """
        data = self.ratings_data
        movie_ids = data[data["userId"] == user_id]["movieId"]
        ratings_list = data[data["userId"] == user_id]["rating"]
        users_movies_dict = dict(zip(movie_ids, ratings_list))
        return users_movies_dict

    def dataset_split(self, user_id, test_size) -> (list, list):
        data = self.ratings_data
        movie_ids = data[data["userId"] == user_id]["movieId"]
        timestamp = data[data["userId"] == user_id]["timestamp"]
        """sorted movieID according to the timestamp"""
        timestamp, movie_ids = zip(*sorted(zip(timestamp, movie_ids)))
        training_set_length = round(len(movie_ids) * (1 - test_size))
        training_movies_ids = movie_ids[:training_set_length]
        test_movies_ids = movie_ids[training_set_length:]
        return training_movies_ids, test_movies_ids

    def get_user_unseen_movies_list(self, user_id, test_size) -> list:
        """
        get the movies ids that the user has not watched.
        :param user_id: user id.
        :param test_size: test dataset size, if it's not 0, then we take this as an evaluation mode,
                         we will return the unseen_movies_list with the test dataset that the user has already watched,
                         just for the sake of evaluation.
                        If it's 0, we will we take this as an recommender mode, and use all movies to do the recommendation.
        use all the data to train.
        :return: movies ids list that the user has not watched.
        """
        data = self.ratings_data
        all_movies = data["movieId"]
        """this is the recommender mode"""

        if test_size == 0:
            user_seen_movies, _ = self.dataset_split(user_id, test_size)
            unseen_movies_list = list(set(all_movies).difference(set(user_seen_movies)))
        else:
            """this is the evaluation mode"""
            user_seen_movies_train, user_seen_movies_test = self.dataset_split(user_id, test_size)
            unseen_movies_list = user_seen_movies_test
        return unseen_movies_list
