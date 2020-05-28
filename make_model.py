
# coding: utf-8

# Python version: 3.5+


from flask import Flask, render_template, flash, request
from flask import jsonify

#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import nltk, string
import pickle
import pandas as pd

import numpy as np


import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

import re

# In[3]:


#############################
nltk.download('wordnet') # first-time use only
nltk.download('punkt') # first-time use only

###########################

# Define preprocessing  i.e stemming, lematization, tokenization, stopwords removal, TFIDF vector conversion

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


stemmer = nltk.stem.porter.PorterStemmer()
def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')


# Define functions of cleaning the data and performing cosine similarity between vectors


def fn_clean_text(text):
    text = str(text)
    text = re.sub("[^a-zA-Z0-9().,?']", " ", text)
    text = re.sub(r'\.+', ".", text)
    text = text.split()
    text = " ".join(text)
    return text

def fn_cosine_sim(a, b):

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# Read Data


df = pd.read_csv('data/similar-staff-picks-challenge-clips.csv', sep=',')

# Number of similar posts to return
nbr_similar_posts = 10


# Clean the data by calling above defined functions and append in a new clean dataFrame


title = []
caption = []
combined_text = []
for i in range (0,len(df)):
        a = fn_clean_text(df.iloc[i]['title'])
        b = fn_clean_text(df.iloc[i]['caption'])
        #c = fn_clean_text(a) + fn_clean_text(b)
        title.append(a)
        caption.append(b)
        combined_text.append(a + ' ' + b)
        



clean_df = pd.DataFrame()
clean_df['id'] = df['id']
clean_df['title'] = title
clean_df['caption'] = caption
clean_df['combined_text'] = combined_text


#Transform the data based on title and caption into TFIDF vectors


db_tfidf = TfidfVec.fit_transform(clean_df['combined_text'])


# Save TFIDF matrix for all data for future reference, if required (not necessary)



pickle.dump( db_tfidf, open( "database/db_tfidf.pickle", "wb" ) )
pickle.dump( TfidfVec, open( "database/TfidfVec.pickle", "wb" ) )




# Initialize Flask based restful API

DEBUG = False
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
#class ReusableForm(Form):
 #   name = TextField('Name:', validators=[validators.required()])
# Render HTML template to take input
@app.route("/")
def input_fun():
    
    return render_template('query_db.html')

 
# Render Restful API to display jason results 
@app.route("/", methods=['POST'])
def query_db():
    
 
    if request.method == 'POST':
        #take input from form
        query_id=int(request.form['text'])
 
        
        # Get index of the ID that was posted by user
        query_index = clean_df.index[clean_df['id'] == query_id].tolist()
        # Get TFIDF vector of the query ID
        query_tfidf = db_tfidf.toarray()[query_index]
        #Perform Cosine similarity with all TFIDF vectors
        cos_sim = []
        for vector in db_tfidf.toarray():
            cos_sim.append(fn_cosine_sim(query_tfidf, vector))
        cos_sim = np.array(cos_sim)
        cos_sim = cos_sim[~np.isnan(cos_sim)]
        if len(cos_sim) > 0:
            # Get index of sorted cosine similarities

            prediction_idx = np.argsort(1 - cos_sim)
            top_k = []
            top_k_similarity = []
            # Get top k (10 in this case) posts
            for j in range(0, nbr_similar_posts):
                    sent_idx = prediction_idx[j]
                    top_k.append(clean_df.iloc[sent_idx])
                    top_k_similarity.append(cos_sim[sent_idx])

        # Convert the results into dataFrame and into json format for display
        top_k_df = pd.DataFrame(top_k)
        top_k_df = top_k_df.drop(['combined_text'], axis=1)
        top_k_df = top_k_df.reset_index()
        top_k_df['similairty'] = top_k_similarity
        top_k_dic = top_k_df.to_dict(orient='id')
        #result_json = json.dumps(tmp)
            
 
    return (jsonify(top_k_dic))
if __name__ == "__main__":
    app.run()
