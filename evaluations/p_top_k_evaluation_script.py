# -*- coding:utf-8 -*-
"""
@file: p_top_k_evaluation_script.py
@time: 31/10/2022 17:41
@desc: 
@author: Echo
"""
from ext import preprocessing
from recommender_system.content_based import ContentBasedRecommender
from recommender_system.evaluation import Evaluation


def evaluation_p_top_k(user_id, test_size, k,retrieval_method=None):
    _, true_test_movies = preprocessing.dataset_split(user_id, test_size)

    """content based recommendation"""
    content_recommender = ContentBasedRecommender()
    predict_movies_ids = list(
        content_recommender.get_ordered_rankings(user_id, test_size, retrieval_method=retrieval_method, top_k=k).keys())

    true_test_movies = list(true_test_movies)
    true_test_movies_ratings = preprocessing.get_one_user_ratings_with_movies_ids(user_id, true_test_movies)

    true_test_movies_ids = []
    for item in true_test_movies_ratings.items():
        if item[1] >= 3.5:
            true_test_movies_ids.append(item[0])


    p_top_k = Evaluation().precision_top_k(true_test_movies_ids, predict_movies_ids[:k], k)

    return p_top_k


if __name__ == '__main__':
    user_id = 610
    test_size = 0.5
    k_list = [5,10,15,50,100]

    method =['lsh','faiss',None]
    for k in k_list:
        for m in method:
            p_top_k = evaluation_p_top_k(user_id, test_size, k,m)
            print(f'p_top_k for {k} with {m} is {p_top_k}')
    # mse_list = []
    # for user_id in range(1, 610):
    #     mse = evaluation_with_rmse(user_id, test_size)
    #     mse_list.append(mse)
    # print(sum(mse_list)/len(mse_list))
