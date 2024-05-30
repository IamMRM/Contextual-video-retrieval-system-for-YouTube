# Semantic Search:
# 1. take documents and convert them into vectors (text embeddings)
# 2. take a search query and convert it into a vector
# 3. compare the vector of the search query to the vectors of the documents (to find the closest documents to eachother)
# it is called semantic search because the meaning of the query is taken into account

# our dataset parguqet file contains VideoId, Title, Transcript. Title can be used as a summary

import polars as pl
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import DistanceMetric
import numpy as np
import matplotlib.pyplot as plt


def returnVideoID_index(df: pl.dataframe.frame.DataFrame, df_eval: pl.dataframe.frame.DataFrame, query_n: int) -> int:
    """
        Function to return the index of a dataframe corresponding to the nth row in evaluation dataframe
    """

    return [i for i in range(len(df)) if df['video_id'][i]==df_eval['video_id'][query_n]][0]


def evalTrueRankings(dist_arr_isorted: np.ndarray, df: pl.dataframe.frame.DataFrame,
                     df_eval: pl.dataframe.frame.DataFrame) -> np.ndarray:
    """
        Function to return "true" video ID rankings for each evaluation query
    """

    # intialize array to store rankings of "correct" search result
    true_rank_arr = np.empty((1, dist_arr_isorted.shape[1]))

    # evaluate ranking of correct result for each query
    for query_n in range(dist_arr_isorted.shape[1]):
        # return "true" video ID's in df
        video_id_idx = returnVideoID_index(df, df_eval, query_n)

        # evaluate the ranking of the "true" video ID
        true_rank = np.argwhere(dist_arr_isorted[:, query_n] == video_id_idx)[0][0]

        # store the "true" video ID's ranking in array
        true_rank_arr[0, query_n] = true_rank

    return true_rank_arr


# Experimentation with the models
df = pl.read_parquet('data/video-transcripts.parquet')
df_eval = pl.read_csv('data/eval-raw.csv')
# print(df_eval.head())


column_to_embed_list = ['title', 'transcript']
# models taken from https://sbert.net/docs/pretrained_models.html
#"all-mpnet-base-v2", "all-MiniLM-L12-v2"
model_name_list = ["all-MiniLM-L6-v2", "multi-qa-distilbert-cos-v1", "multi-qa-mpnet-base-dot-v1"]
# initialize dict to keep track of all text embeddings
text_embedding_dict = {}

for model_name in model_name_list:

    model = SentenceTransformer(model_name)

    for column_name in column_to_embed_list:

        key_name = model_name + "_" + column_name
        print(key_name)
        embedding_arr = model.encode(df[column_name].to_list())
        text_embedding_dict[key_name] = embedding_arr

# same embedding but for evaluation dataset
query_embedding_dict = {}

for model_name in model_name_list:
    model = SentenceTransformer(model_name)
    embedding_arr = model.encode(df_eval['query'].to_list())
    query_embedding_dict[model_name] = embedding_arr

# initialize distance metrics to experiment
dist_name_list = ['euclidean', 'manhattan', 'chebyshev']
sim_name_list = ['cos_sim', 'dot_score']

eval_results = []

for model_name in model_name_list:
    query_embedding = query_embedding_dict[model_name]
    for column_name in column_to_embed_list:
        embedding_arr = text_embedding_dict[model_name + '_' + column_name]
        for dist_name in dist_name_list:
            dist = DistanceMetric.get_metric(dist_name)
            dist_arr = dist.pairwise(embedding_arr, query_embedding)
            dist_arr_isorted = np.argsort(dist_arr, axis=0)
            method_name = "_".join([model_name, column_name, dist_name])
            true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)
            eval_list = [method_name] + true_rank_arr.tolist()[0]
            eval_results.append(eval_list)

        # loop through sbert similarity scores from huggingface
        for sim_name in sim_name_list:
            #  minus because it gives difference and we want similarity
            cmd = "dist_arr = -util." + sim_name + "(embedding_arr, query_embedding)"
            exec(cmd)

            # sort indexes of distance array (notice minus sign in front of cosine similarity)
            dist_arr_isorted = np.argsort(dist_arr, axis=0)

            method_name = "_".join([model_name, column_name, sim_name.replace("_", "-")])

            # evaluate the ranking of the ground truth
            true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

            # store results
            eval_list = [method_name] + true_rank_arr.tolist()[0]
            eval_results.append(eval_list)



# compute rankings for title + transcripts embedding
for model_name in model_name_list:

    # generate embeddings
    embedding_arr1 = text_embedding_dict[model_name + '_title']
    embedding_arr2 = text_embedding_dict[model_name + '_transcript']
    query_embedding = query_embedding_dict[model_name]

    for dist_name in dist_name_list:
        # compute distance between video text and query
        dist = DistanceMetric.get_metric(dist_name)
        dist_arr = dist.pairwise(embedding_arr1, query_embedding) + dist.pairwise(embedding_arr2, query_embedding)

        # sort indexes of distance array
        dist_arr_isorted = np.argsort(dist_arr, axis=0)

        # define label for search method
        method_name = "_".join([model_name, "title-transcript", dist_name])

        # evaluate the ranking of the ground truth
        true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

        # store results
        eval_list = [method_name] + true_rank_arr.tolist()[0]
        eval_results.append(eval_list)

    # loop through sbert similarity scores
    for sim_name in sim_name_list:
        # apply similarity score from sbert
        cmd = "dist_arr = -util." + sim_name + "(embedding_arr1, query_embedding) - util." + sim_name + "(embedding_arr2, query_embedding)"
        exec(cmd)

        # sort indexes of distance array (notice minus sign in front of cosine similarity)
        dist_arr_isorted = np.argsort(dist_arr, axis=0)

        # define label for search method
        method_name = "_".join([model_name, "title-transcript", sim_name.replace("_", "-")])

        # evaluate the ranking of the ground truth
        true_rank_arr = evalTrueRankings(dist_arr_isorted, df, df_eval)

        # store results
        eval_list = [method_name] + true_rank_arr.tolist()[0]
        eval_results.append(eval_list)

# define schema for results dataframe
schema_dict = {'method_name':str}
for i in range(len(eval_results[0])-1):
    schema_dict['rank_query-'+str(i)] = float

# store results in dataframe
df_results = pl.DataFrame(eval_results, schema=schema_dict)
print(df_results.head())

# compute mean rankings of ground truth search result
df_results = df_results.with_columns(new_col=pl.mean_horizontal(df_results.columns[1:])).rename({"new_col": "rank_query-mean"})

# compute number of ground truth results which appear in top 3
for i in [1,3]:
    df_results = df_results.with_columns(new_col=pl.sum_horizontal(df_results[:,1:-1]<i)).rename({"new_col": "num_in_top-"+str(i)})

# results
df_summary = df_results[['method_name', "rank_query-mean", "num_in_top-1", "num_in_top-3"]]
print(df_summary.sort('rank_query-mean').head())
print(df_summary.sort("num_in_top-1", descending=True).head())
print(df_summary.sort("num_in_top-3", descending=True).head())
for i in range(4):
    print(df_summary.sort("num_in_top-3", descending=True)['method_name'][i])

# Conclusion
model_name = 'all-MiniLM-L6-v2'
column_name_list = ['title', 'transcript']
model = SentenceTransformer(model_name)

for column_name in column_name_list:
    # generate embeddings
    embedding_arr = model.encode(df[column_name].to_list())

    # store embeddings in a dataframe
    schema_dict = {column_name+'_embedding-'+str(i): float for i in range(embedding_arr.shape[1])}
    df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

    # append embeddings to video index
    df = pl.concat([df, df_embedding], how='horizontal')

df.write_parquet('data/video-index.parquet')
