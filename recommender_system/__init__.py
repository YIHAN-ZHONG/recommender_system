# -*- coding:utf-8 -*-
"""
@file: __init__.py.py
@time: 09/10/2022 18:47
@desc: 
@author: Echo
"""
import abc


class MovieRecommenderSystem(metaclass=abc.ABCMeta):

    def __init__(self):
        # self.movie_names = self.load_movies_names()
        pass
    #
    # @abc.abstractmethod
    # def pre_processing(self, input_data: list) -> dict:
    #     pass

    # @abc.abstractmethod
    # def load_dataset(self):
    #     """load the dataset"""
    #     pass

    # @abc.abstractmethod
    # def load_movies_names(self):
    #     pass
    #
    # @abc.abstractmethod
    # def get_similarity_matrix(self):
    #     pass
    #
    # @abc.abstractmethod
    # def accuracy_evaluation(self):
    #     pass