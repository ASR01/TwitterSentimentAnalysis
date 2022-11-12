import os
import pandas as pd
import streamlit as st
import requests
from transformers import pipeline
import tweepy
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

##################### Variables #############################
#Tweepy
MAX_RESULTS = 10 #number of twitter results
# Keep secret from Github or aliens
MY_TWEEPY_TOKEN = "AAAAAAAAAAAAAAAAAAAAANrDjAEAAAAAPO39Plw0Plu4XlOMh22F7EkdZoc%3Di7SICjE77yoVgOWwKfyyBWgS9qMB8LRM9REA8DIogDnUHOEUK4"

max_date = datetime.today()
min_date = max_date - timedelta(days=6)


#HF Model
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_MODEL_ID = "andersab/tweet_model_sentiment_andersab"
HF_API_TOKEN = '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'

API_URL = st.sidebar.text_input("API URL", HF_API_URL)
MODEL_NAME = st.sidebar.text_input("MODEL NAME", HF_MODEL_ID)

map_result = {'LABEL_0':'Negative',
              'LABEL_1':'Neutral',
              'LABEL_2':'Positive'}

tweets_found = False



##################### Functions #############################

def get_tweets(query, start_time, end_time, max_results):

	tweets = client.search_recent_tweets(query=query,
									 start_time=start_time,
									 end_time=end_time,
									 tweet_fields = ["created_at", "text", "source"],
									 user_fields = ["name", "username", "location", "verified", "description", 'public_metrics'],
									 max_results = max_results,
									 expansions='author_id'
									 )

	return tweets



def model_query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def raise_on_not200(response):
	if response.status_code != 200:
		st.write("There was an error!")
		st.write(response)

def sanitize_input(text):
	clean = str(text)
	return clean

def predict(model, model_input):
	res = client.post(API_URL + model, json=model_input)
	return res.json()


##################### Initialise #############################


client = tweepy.Client(MY_TWEEPY_TOKEN)


##################### Start #############################

#Check
#st.write(f"API endpoint: {API_URL}{MODEL_NAME}")

# Get tweets

st.header("Get Tweeter Mood about a subject")

col1, col2 = st.columns([1, 1])
with col1:
	startdate = st.date_input('Pick a start date', min_date)
	username = st.text_input('Twitter username without the @ sign', 'elonmusk')
with col2:
	enddate = st.date_input('Pick an end date', max_date)
	number_tweets = st.slider('Number fo tweets to analyze (max = 50)', 0, 50, 10 )

issue = st.text_input("Insert the search words ( # hashtags allowed)")

startdate = datetime.combine(startdate, datetime.min.time())
enddate = datetime.combine(enddate, datetime.min.time())

request_text = 'from:' + username + ' ' + issue

tweets = get_tweets(request_text, startdate, enddate, number_tweets)
if tweets.data != None:
   
	# create a list of records
	tweet_texts = []
	# iterate over each tweet and corresponding user details
	for tweet in tweets.data:
		tweet_texts.append(tweet.text)
	
	tweets_df = pd.DataFrame(tweet_texts)

	datalist = []
	for text in tweet_texts:
		data = model_query(text, HF_MODEL_ID, HF_API_TOKEN)
		if data is None:
			tweets_found = False
		else:
			tweets_found = True
			dict = {}
			for e in data[0]:
				# print(e)
				dict[e['label']] = e['score']
			datalist.append(dict)
 
	df = pd.DataFrame(datalist).round(3)
	df.rename(columns = map_result, inplace = True)
	result = df.mean(axis=0)
 
	

	# Fixing random state for reproducibility
	np.random.seed(19680801)


	plt.rcdefaults()
	fig, ax = plt.subplots()

	# Example data
	results = (result.index)
	y_pos = np.arange(len(results))
	percentage = result
	color=['green', 'blue', 'red']
	ax.barh(y_pos, percentage, color = color, align='center')
	ax.set_yticks(y_pos, labels=results)
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_xlabel('Mean of added percentages')
	plt.xlim(0,1)

	st.pyplot(fig)

	tt = pd.DataFrame(tweet_texts, columns = ['Text'])

	tab = pd.concat([tt, df], axis=1)
	st.text("Here is the text of the original tuits")
	st.table(tab.iloc[:,0:10])



	
else:
   	st.text('No tweets found, with the parameters requested')
	