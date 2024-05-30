import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
import gradio as gr

model_name = 'all-MiniLM-L6-v2'
dist_name = 'manhattan'
df = pl.scan_parquet('data/video-index.parquet')
model = SentenceTransformer(model_name)
dist = DistanceMetric.get_metric(dist_name)


def returnSearchResults(query: str, index: pl.lazyframe.frame.LazyFrame) -> np.ndarray:
    """
        Function to return indexes of top search results
    """

    # embed query
    query_embedding = model.encode(query).reshape(1, -1)

    # compute distances between query and titles/transcripts
    dist_arr = dist.pairwise(df.select(df.columns[4:388]).collect(), query_embedding) + dist.pairwise(
        df.select(df.columns[388:]).collect(), query_embedding)

    # search paramaters
    threshold = 40  # eye balled threshold for manhatten distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # return indexes of search results
    return idx_below_threshold[idx_sorted][:top_k]

query = "LLM"
idx_result = returnSearchResults(query, df)

print(df.select(['video_id', 'title']).collect()[idx_result])
print(df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False))

# GUI

def pseudoSearchAPI(query: str):
    # return top 5 search results
    idx_result = returnSearchResults(query, df)
    response = df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)

    return response


def formatResultText(title: str, video_id: str):
    text = markdown_text = f"""<br> <br>
# {title}<br>

🔗 [Video Link](https://youtu.be/{video_id})"""

    return text


def formatVideoEmbed(video_id: str):
    # other options
    # embed = '<iframe width="640" height="360" src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" </iframe>'
    # embed = '<a href="https://youtu.be/'+ video_id +'"> <img src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" style="width:576;height:324;"></a>'
    # embed = '<a href="www.youtube.com/watch?v='+ video_id +'"> <img src="https://img.youtube.com/vi/'+ video_id +'/0.jpg" style="width:576;height:324;"></a>'

    return '<iframe width="576" height="324" src="https://www.youtube.com/embed/' + video_id + '"></iframe>'


def searchResults(query):
    # pseudo API call
    response = pseudoSearchAPI(query)

    # format search results

    # initialize list of outputs
    output_list = []

    # compute number of null search results (out of 5)
    num_empty_results = 5 - len(response['title'])

    # display search results
    for i in range(len(response['title'])):
        video_id = response['video_id'][i]
        title = response['title'][i]

        embed = gr.HTML(value=formatVideoEmbed(video_id), visible=True)
        text = gr.Markdown(value=formatResultText(title, video_id), visible=True)

        output_list.append(embed)
        output_list.append(text)

    # make null search result slots invisible
    for i in range(num_empty_results):

        # if no search results display "No results." text
        if num_empty_results == 5 and i == 0:
            embed = gr.HTML(visible=False)
            text = gr.Markdown(value="No results. Try rephrasing your query.", visible=True)

            output_list.append(embed)
            output_list.append(text)
            continue

        embed = gr.HTML(visible=False)
        text = gr.Markdown(visible=False)

        output_list.append(embed)
        output_list.append(text)

    return output_list


output_list = []

with gr.Blocks() as demo:
    gr.Markdown("# YouTube Search")

    with gr.Row():
        inp = gr.Textbox(placeholder="What are you looking for?", label="Query", scale=3)
        btn = gr.Button("Search")
        btn.click(fn=searchResults, inputs=inp, outputs=output_list)

    for i in range(5):
        with gr.Row():
            output_list.append(gr.HTML())
            output_list.append(gr.Markdown())

    inp.submit(fn=searchResults, inputs=inp, outputs=output_list)

demo.launch()