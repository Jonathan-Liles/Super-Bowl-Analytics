import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

!pip install wordcloud matplotlib numpy

# final tweets - corrected version
url = 'https://www.dropbox.com/scl/fi/h1ebhbfuifvvdyy1o6ryq/final_tweets_corrected_2024.csv?rlkey=pg0jpdpc0979daqj67gpdpdi3&dl=0'
modified_url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "?dl=1")
df = pd.read_csv(modified_url, dtype=str, encoding='utf-8')
