#Let's set the working directory YOURPATH

import os os.chdir('YOURPATH')    

os. getcwd()

#and import the key libraries

import chart_studio
import re
import string
import collections
import ipywidgets
import cufflinks
import nltk.tokenize

import pandas as pd
import datetime
import seaborn as sns
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import nltk
import gensim
import yfinance as yf

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from tqdm.notebook import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#read and copy the input Kaggle dataset
#https://www.kaggle.com/code/khalidryder777/comprehensive-analysis-of-500k-tweets-on-chatgpt/input

tweet_df = pd.read_csv('TwitterJanMar23.csv')

df = tweet_df.copy(deep = True)

#Remove missing values
df = df.dropna()
print("Shape: ",df.shape)

#Shape:  (499974, 6)

#edit the date column

df['date'] = pd.to_datetime(df['date'])

df['date'] = df['date'].dt.date

df['date'] = pd.to_datetime(df['date'])

#Checking range of dates
print("Start Date: " ,df['date'].min())
print("End Date: " ,df['date'].max())

#Start Date:  2023-01-04 00:00:00
#End Date:  2023-03-29 00:00:00

#Let's introduce the preprocessing function
def pre_process(text):
# Remove links
text = re.sub('http://\S+|https://\S+', '', text)
text = re.sub('http[s]?://\S+', '', text)
text = re.sub(r"http\S+", "", text)

text = re.sub('&amp', 'and', text)
text = re.sub('&lt', '<', text)
text = re.sub('&gt', '>', text)

# Remove new line characters
text = re.sub('[\r\n]+', ' ', text)

text = re.sub(r'@\w+', '', text)
text = re.sub(r'#\w+', '', text)
# text = re.sub(r'@\w+', lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.group(0)), text) #Keeps the character trailing @
# text = re.sub(r'#\w+', lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.group(0)), text) #Keeps the character trailing #

# Remove multiple space characters
text = re.sub('\s+',' ', text)

# Convert to lowercase
text = text.lower()
return text

df['processed_content'] = df['content'].apply(pre_process)

#Let's sort the data frame and keep only the first tweet copy with max likes

df_sorted = df.sort_values(by='like_count', ascending=False)

df_cleaned = df_sorted.drop_duplicates(subset='processed_content', keep='first')

df_final = df_cleaned.sort_index()
df = df_final
print (df.shape)

#(458210, 7)

df.columns

#Index(['date', 'id', 'content', 'username', 'like_count', 'retweet_count',
#       'processed_content'],
#      dtype='object')

#NLP EDA

