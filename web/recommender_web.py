# -*- coding:utf-8 -*-
"""
@file: recommender_web.py
@time: 29/10/2022 09:19
@desc: web demo entry
@author: Echo
"""
from pywebio.input import input, NUMBER, TEXT
from pywebio.output import put_text

from ext import preprocessing
from recommender_system.collaborative_filtering import CollaborativeFilteringUsersBased
from recommender_system.content_based import ContentBasedRecommender


def recommender():
    while True:
        user_id = input("The user_id you want to predict：", type=NUMBER)
        recommender_methods = input(
            "The recommender methods you want to use. You can choose 'content_based' or 'collacorative_filtering'",
            type=TEXT)
        number_k = input("number k: The size of output recommendation list you want to see", type=NUMBER)
        test_size = 0

        if recommender_methods == "content_based":
            """content based recommendation"""
            content_recommender = ContentBasedRecommender()
            predict_movies = list(content_recommender.get_ordered_rankings(user_id, test_size).keys())[:number_k]
        else:
            """CollaborativeFilteringUsersBased recommendation"""
            cf_recommender = CollaborativeFilteringUsersBased()
            predict_movies = list(cf_recommender.predicting_ordered_ratings(user_id, test_size).keys())[:number_k]

        movies_names = preprocessing.get_movies_names_by_ids(predict_movies)
        _, true_test_movies = preprocessing.dataset_split(user_id, test_size)

        put_text('The movies that user %d may want to watch according to %s recommender system is ' % (user_id, recommender_methods))
        for movie in movies_names:
            put_text(movie)

        put_text('\n')

        is_continued = input("Do you want to continue?'yes' or 'no'. When you choose 'no', the program will stop."
                             "If you want to start it again, you have to rerun the server :)", type=TEXT)
        if is_continued == "yes":
            continue
        else:
            break

