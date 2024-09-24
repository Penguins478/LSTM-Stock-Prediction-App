import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re
import math
from nltk.tokenize import word_tokenize
import nltk

# URL to scrape
# Sentiment rating may influence Apple (NASDAQ: AAPL) stock prices
url = "https://www.forbes.com/sites/ewanspence/2024/09/23/apple-iphone-16-pro-review-good-bad-specs-camera-control-apple-intelligence-new-iphone/"
max_sentences_in_summary = 10

def scrape_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article = ' '.join([p.get_text() for p in paragraphs])
        return article.strip()
    else:
        return ""

def clean_text(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()

    article = filedata.split(". ")
    sentences = []

    for sentence in article:
        sentence = re.sub(r'[^a-zA-Z.,!:;?|\’\'\"\-\_—\s]', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        sentences.append(sentence)

    return sentences

# def clean_text(file_name):
#     file = open(file_name, "r")
#     filedata = file.readlines()
#     article = filedata[0].split(". ")
#     sentences = []
#     # removing special characters and extra whitespaces
#     for sentence in article:
#         sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
#         sentence = re.sub('[\s+]', ' ', sentence)
#         sentences.append(sentence)
#     sentences.pop()
#     display = " ".join(sentences)
#     print('Initial Text: ')
#     print(display)
#     print('\n')
#     return sentences


def count_words(sent):
    cnt = 0
    words = word_tokenize(sent)
    for word in words:
        cnt = cnt + 1
    return cnt



def count_in_sent(sentences):
    txt_data = []
    i = 0
    for sent in sentences:
        i = i + 1
        cnt = count_words(sent)
        temp = {'id': i, 'word_cnt': cnt}
        txt_data.append(temp)
    return txt_data


def freq_dict(sentences):
    i = 0
    freq_list = []
    for sent in sentences:
        i = i + 1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] = freq_dict[word] + 1
            else:
                freq_dict[word] = 1
        temp = {'id': i, 'freq_dict': freq_dict}
        freq_list.append(temp)
    return freq_list


def calc_TF(text_data, freq_list):
    tf_scores = []
    for item in freq_list:
        ID = item['id']
        for k in item['freq_dict']:
            temp = {
                'id': item['id'],
                'tf_score': item['freq_dict'][k]/text_data[ID-1]['word_cnt'],
                'key': k
            }
            tf_scores.append(temp)
    return tf_scores


def calc_IDF(text_data, freq_list):
    idf_scores =[]
    cnt = 0
    for item in freq_list:
        cnt = cnt + 1
        for k in item['freq_dict']:
            val = sum([k in it['freq_dict'] for it in freq_list])
            temp = {
                'id': cnt,
                'idf_score': math.log(len(text_data)/(val+1)),
                'key': k}
            idf_scores.append(temp)
    return idf_scores


def calc_TFIDF(tf_scores, idf_scores):
    tfidf_scores = []
    for j in idf_scores:
        for i in tf_scores:
            if j['key'] == i['key'] and j['id'] == i['id']:
                temp = {
                    'id': j['id'],
                    'tfidf_score': j['idf_score'] * i['tf_score'],
                    'key': j['key']
                }
                tfidf_scores.append(temp)
    return tfidf_scores


def sent_scores(tfidf_scores, sentences, text_data):
    sent_data = []
    for txt in text_data:
        score = 0
        for i in range(0, len(tfidf_scores)):
            t_dict = tfidf_scores[i]
            if txt['id'] == t_dict['id']:
                score = score + t_dict['tfidf_score']
        temp = {
            'id': txt['id'],
            'score': score,
            'sentence': sentences[txt['id']-1]}
        sent_data.append(temp)
    return sent_data

def summary(sent_data):
    cnt = 0
    summary = []
    for t_dict in sent_data:
        cnt = cnt + t_dict['score']
    avg = cnt / len(sent_data)
    appended = 0
    for sent in sent_data:
        if sent['score'] >= (avg * 0.999):
            summary.append(sent['sentence'])
            appended += 1
            if appended >= max_sentences_in_summary:
                break
    txt = ". ".join(summary)
    return txt, summary


scraped_text = scrape_article(url)
with open("nlp/article.txt", "w") as f:
    f.write(str(scraped_text))


sentences = clean_text('nlp/article.txt')
text_data = count_in_sent(sentences)

freq_list = freq_dict(sentences)
tf_scores = calc_TF(text_data, freq_list)
idf_scores = calc_IDF(text_data, freq_list)

tfidf_scores = calc_TFIDF(tf_scores, idf_scores)

sent_data = sent_scores(tfidf_scores, sentences, text_data)
result_txt, summary = summary(sent_data)

with open("nlp/summary.txt", "w") as file:
    # file.write(result_txt)
    for s in summary:
        file.write(s + "\n")

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
sentiment = sentiment_pipeline(sentences)

'''
LABELS
----------
"LABEL_0": "Negative"
"LABEL_1": "Neutral"
"LABEL_2": "Positive"
'''

final_score = 0

for res in sentiment:
    label = res['label']
    score = res['score']
    # neutral tag does not contribute
    if label == 'LABEL_2':
        final_score += score
    elif label == 'LABEL_0':
        final_score -= score

len_factor = 25
neutral_threshold = len(sentences) / len_factor

if final_score >= neutral_threshold:
    print("Sentiment: Positive")
elif final_score <= -neutral_threshold:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")

print(f"Final Sentiment Score: {final_score:.2f}")
print("***\tSummary located in summary.txt\t***")

print("Done!")