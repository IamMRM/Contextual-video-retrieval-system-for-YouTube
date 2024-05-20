import requests
import json
import polars as pl
import os
from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt
import difflib

def get_channel_id(channel_name, my_key):
    # Define the parameters for the request
    params = {
        "key": my_key,
        "q": channel_name,
        "part": "snippet",
        "type": "channel"
    }

    # URL for the YouTube Data API v3 search endpoint
    url = 'https://www.googleapis.com/youtube/v3/search'

    # Make the GET request
    response = requests.get(url, params=params)

    # Parse the JSON response
    data = response.json()

    if 'items' in data and len(data['items']) > 0:
        channel_titles = [item['snippet']['title'] for item in data['items']]
        closest_match = difflib.get_close_matches(channel_name, channel_titles, n=1, cutoff=0.1)

        if closest_match:
            for item in data['items']:
                if item['snippet']['title'] == closest_match[0]:
                    return item['snippet']['channelId']
    return None


def _getVideoRecords(response: requests.models.Response) -> list:
    """
        Function to extract YouTube video data from GET request response
    """

    video_record_list = []

    for raw_item in json.loads(response.text)['items']:

        # only execute for youtube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue

        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']
        video_record_list.append(video_record)

    return video_record_list


def getVideoRecord(my_key, channel_id, url):
    video_record_list = []
    page_token = None
    while page_token != 0:
        # define parameters for API call
        params = {"key": my_key, 'channelId': channel_id, 'part': ["snippet", "id"], 'order': "date", 'maxResults': 50,
                  'pageToken': page_token}
        # make get request
        response = requests.get(url, params=params)

        # append video records to list
        video_record_list += _getVideoRecords(response)

        try:
            # grab next page token
            page_token = json.loads(response.text)['nextPageToken']
        except:
            # if no next page token kill while loop
            page_token = 0

    # write data to file
    pl.DataFrame(video_record_list).write_parquet('data/video-ids.parquet')
    pl.DataFrame(video_record_list).write_csv('data/video-ids.csv')


########################################################################################
def extract_text(transcript: list) -> str:
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)


def _getTranscript(df):
    transcript_text_list = []
    for i in range(len(df)):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
            transcript_text = extract_text(transcript)
        # if not available set as n/a
        except:
            transcript_text = "n/a"

        transcript_text_list.append(transcript_text)
    return transcript_text_list


def getTranscript():
    df = pl.read_parquet('data/video-ids.parquet')
    print(df.head())
    print(df.shape)
    transcript_text_list = _getTranscript(df)
    df = df.with_columns(pl.Series(name="transcript", values=transcript_text_list))
    print(df.head())
    df.write_parquet('data/video-transcripts.parquet')
    df.write_csv('data/video-transcripts.csv')

########################################################################################


def transform_data():
    df = pl.read_parquet('data/video-transcripts.parquet')
    # shape + unique values
    print("shape:", df.shape)
    print("n unique rows:", df.n_unique())
    for j in range(df.shape[1]):
        print("n unique elements (" + df.columns[j] + "):", df[:, j].n_unique())
    print("Total number of title characters:", sum(len(df['title'][i]) for i in range(len(df))))
    print("Total number of transcript characters:", sum(len(df['transcript'][i]) for i in range(len(df))))
    # change datetime to Datetime dtype
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))
    print(df.head())
    # lengths/character counts
    plt.hist(df['title'].str.len_chars())
    plt.hist(df['transcript'].str.len_chars())
    print(df['title'][3])
    print(df['transcript'][3])
    special_strings = ['&#39;', '&amp;']
    special_string_replacements = ["'", "&"]

    for i in range(len(special_strings)):
        df = df.with_columns(df['title'].str.replace(special_strings[i], special_string_replacements[i]).alias('title'))
        df = df.with_columns(
            df['transcript'].str.replace(special_strings[i], special_string_replacements[i]).alias('transcript'))
    print(df['title'][3])
    print(df['transcript'][3])
    # write data to file
    df.write_parquet('data/video-transcripts.parquet')
    df.write_csv('data/video-transcripts.csv')


channel_id = None  # 'UCa9gErQ9AE5jT2DZLjXBIdA'
channel_name = 'aiexplained-official'
url = 'https://www.googleapis.com/youtube/v3/search'
my_key = os.getenv('my_key')

if channel_id is None:
    channel_id = get_channel_id(channel_name, my_key)

print("Starting the extraction of video records from YouTube...")
getVideoRecord(my_key, channel_id, url)
print("Extraction of video records from YouTube completed!")
print("Starting the extraction of video transcripts from YouTube...")
getTranscript()
print("Extraction of video transcripts from YouTube completed!")
print("Starting the transformation of video transcripts...")
transform_data()
print("Transformation of video transcripts completed!")
