# -*- coding:utf-8 -*-
"""
@file: evaluation.py
@time: 09/10/2022 20:24
@desc: 
@author: Echo
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


class Evaluation:
    def __int__(self):
        pass

    def root_mean_square_error(self, y_true, y_pred) -> float:
        """
        evaluation with mean square error
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        mse = mean_squared_error(y_pred, y_true)
        return sqrt(mse)

    def precision_top_k(self, y_true, y_pred, k) -> float:
        """
        Precision @ k = (  # of recommended items @k that are relevant) / (# of recommended items @k)

        :param y_true:
        :param y_pred:
        :return:
        """
        """we calculate the relevant list by getting the intersection set form the recommended items and true items 
        that the user watched and liked """
        true_positives_list = list(set(y_pred).intersection(set(y_true)))
        p_top_k = len(true_positives_list) / k
        print(true_positives_list)
        return p_top_k

    # def precision_top_k(y_true, y_score, k, pos_label=1):
    #     # from sklearn.utils import column_or_1d
    #     # from sklearn.utils.multiclass import type_of_target
    #
    #     # y_true_type = type_of_target(y_true)
    #     # if not (y_true_type == "binary"):
    #     #     raise ValueError("y_true must be a binary column.")
    #
    #     # Makes this compatible with various array types
    #     # y_true_arr = column_or_1d(y_true)
    #     # y_score_arr = column_or_1d(y_score)
    #
    #     y_true_arr = y_true_arr == pos_label
    #
    #     desc_sort_order = np.argsort(y_score_arr)[::-1]
    #     y_true_sorted = y_true_arr[desc_sort_order]
    #     y_score_sorted = y_score_arr[desc_sort_order]
    #
    #     true_positives = y_true_sorted[:k].sum()
    #
    #     return true_positives / k


    def get_cosine_similarity(self, matrix_a, matrix_b) -> np.ndarray:
        """
        get the cosine similarity between matrix_a and matrix_b
        :param matrix_a: 2d matrix
        :param matrix_b: 2d matrix
        :return: cosine similarity
        """
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)
        cos_sim = np.sum(matrix_a * matrix_b, axis=1) / (
                np.linalg.norm(matrix_a, axis=1) * np.linalg.norm(matrix_b, axis=1))
        return cos_sim
