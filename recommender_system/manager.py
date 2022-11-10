# -*- coding:utf-8 -*-
"""
@file: manager.py
@time: 03/11/2022 16:47
@desc: 
@author: Echo
"""
from recommender_system.optimization.faiss_retrieval import FaissRetrieval
from recommender_system.optimization.lsh_retrieval import LSHRetrieval

faiss_retrieval = FaissRetrieval()
lsh_retrieval = LSHRetrieval()
lsh_retrieval.insert()
