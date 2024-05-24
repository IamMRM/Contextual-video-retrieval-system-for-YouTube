# Semantic Search:
# 1. take documents and convert them into vectors (text embeddings)
# 2. take a search query and convert it into a vector
# 3. compare the vector of the search query to the vectors of the documents (to find the closest documents to eachother)
# it is called semantic search because the meaning of the query is taken into account

import polars as pl
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import DistanceMetric
import numpy as np
import matplotlib.pyplot as plt


print("Loading the model")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Model loaded")