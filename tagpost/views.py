from flask import Flask, render_template, request

import pandas as pd
import gensim
from joblib import load

from gensim.models import TfidfModel
from bs4 import BeautifulSoup

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



app = Flask(__name__)

@app.route('/requete',methods=['GET'])
def dashboard():

    return render_template('dashboard.html')

@app.route('/resultat',methods=['POST'])
def resultat():

    # Récupération des variables du formulaire
    result=request.form
    title_in = result['Title_post']
    post_in = result['Post_content']

    dict_bow = load('dict_bow.pkl')
    data_for_tfidf = load ('data_for_tfidf.pkl')
    dict_tag = load('dict_tag_100.pkl')


    title_body_api = pd.DataFrame()
    post_out = suppr_balises_html(post_in)
    title_out = tokenize_text(title_in)
    post_out = tokenize_text(post_out)
    title_out = suppr_stopwords(title_out)
    post_out = suppr_stopwords(post_out)
    title_out = lemmatize(title_out)
    post_out = lemmatize(post_out)
    title_body_api['Title_body'] = [title_out + post_out]
    texts = title_body_api['Title_body']
    texts = pd.concat([texts, data_for_tfidf['Title_body']])

    bow_corpus = [dict_bow.doc2bow(text) for text in texts]
    tfidf = TfidfModel(bow_corpus)
    bow_tv_ft_ttb = [tfidf[text] for text in bow_corpus]
    bow_tv_ft_ttb_test = [bow_tv_ft_ttb[0]]

    X_test = gensim.matutils.corpus2csc(bow_tv_ft_ttb_test, num_terms=len(dict_bow))
    X_test = X_test.T.toarray()
    scaler = load('standardscaler')
    X_test_std = scaler.transform(X_test)
    clf = load('modele_reg_log')
    y_pred_test_proba = clf.predict_proba(X_test_std)

    # Identification des 5 tags prédits par post
    y_pred_test_proba_w = y_pred_test_proba.copy()
    tag_predit = pd.DataFrame()
    for k in range(len(y_pred_test_proba_w)):
        for i in range(5):
            val_max = 0
            ind = -1
            for j in range(100):
                if y_pred_test_proba_w[k, j] > val_max:
                    val_max = y_pred_test_proba_w[k, j]
                    ind = j
            if ind > -1:
                tag_predit.loc[k, i] = dict_tag[ind]
                y_pred_test_proba_w[k, ind] = 0
    tag_predit['List_tags'] = tag_predit[0] + ' ' + tag_predit[1] + \
                              ' ' + tag_predit[2] + ' ' + tag_predit[3] + ' ' + tag_predit[4]
    Tags_out = tag_predit.loc[0]['List_tags']
    return render_template('resultat.html', Title=title_in, Post=post_in,
                           Tags=Tags_out)

def suppr_balises_html(text):
    soup = BeautifulSoup(text, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()

    text_out = ' '.join(soup.stripped_strings)
    return text_out

def tokenize_text(text):
    words = list(gensim.utils.tokenize(text, lowercase=True))
    return words

def suppr_stopwords(list):
    mystopwords = set(stopwords.words('english'))
    words = [x for x in list if x not in mystopwords and len(x) > 1]
    return words

def lemmatize(list):
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(x) for x in list]
    return words














if __name__ == "__main__":
        app.run()
