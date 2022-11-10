'''
Author: echo accounts_5f16e5f68294980020af7ee4@mail.teambition.com
Date: 2022-10-09 20:29:04
LastEditors: echo accounts_5f16e5f68294980020af7ee4@mail.teambition.com
LastEditTime: 2022-11-03 15:00:08
FilePath: /recommender_system/recommender_system/optimization/dimensionality_reduction.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding:utf-8 -*-
"""
@file: dimensionality_reduction.py
@time: 09/10/2022 20:29
@desc: optimization using svd
@author: Echo
"""
from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


class DimensionReduce():
    def __init__(self, data):
        self.reader = Reader(rating_scale=(1, 5), line_format='user item rating timestamp')
        self.data = Dataset.load_from_df(data, self.reader)
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2)

    def svd(self):
        algo = SVD()
        algo.fit(self.trainset)
        return algo

    def predict_rmse(self, algo):
        predictions = algo.test(self.testset)
        rmse = accuracy.rmse(predictions)
        return rmse
