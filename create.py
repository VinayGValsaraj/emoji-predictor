import csv
import os
from emoji import UNICODE_EMOJI
import nltk
from utils import Indexer
emoji_set = set(UNICODE_EMOJI)

def process_text(text):
    text = text.replace("RT ", "")
    text = text.replace("\n", " ")
    words = text.split(" ")

    for word in words:
        if word.startswith("@") or word.startswith("http"):
            text = text.replace(word, "")
    
    return text.strip()

def get_emojis_in_text(text, emoji_set):
    char_set = set(text)
    #print(char_set)
    all_emojis = emoji_set & char_set
    
    return all_emojis

def clean_text(tweet):
    cleaned_str = tweet['processed_text']
    for emoji in tweet['emojis_in_text']:
        cleaned_str = tweet['processed_text'].replace(emoji, "")
        
    return cleaned_str

tweets = []
with open('twitter_october.csv', 'r') as csv_file:
    print("reading csv...")
    csv_reader = csv.DictReader(csv_file)
    print("done")
    print("processing text")
    count = 0
    for row in csv_reader:
        if row['text'] == 'text':
            continue
        row['processed_text'] = process_text(row['text'])
        row['emojis_in_text'] = get_emojis_in_text(row['processed_text'], emoji_set)
        tweets.append(row)
        count += 1
        if count % 500000 == 0:
            print("processed", count, "tweets")

from collections import Counter

def get_labels(tweet, indexer):
    
    all_emojis = tweet['emojis_in_text']
    emojis = {}
    c = Counter()
    for emoji in all_emojis:
        indexer.add_and_get_index(emoji)
        c[emoji] += tweet['processed_text'].count(emoji)
    
    #print(all_emojis)
    #print("txt", tweet['processed_text'])
    max_score = max(c.values())
    labels = [indexer.index_of(k) for k in c if c[k] == max_score]
    
    return labels

class DataPoint():
    def __init__(self, text, label):
        self.text = text
        self.label = label

def create_dataset(tweets, indexer, label_counter):
    dataset = []
    count = 0
    for idx, tweet in enumerate(tweets):
        #print(idx, tweet['processed_text'])
        try:
            labels = get_labels(tweet, indexer)
        except:
            continue
        cleaned_text = clean_text(tweet)
        for label in labels:
            if indexer.get_object(label) is not '♀' and indexer.get_object(label) is not '♂':
                datapoint = DataPoint(cleaned_text, label)
                dataset.append(datapoint)
                label_counter[indexer.get_object(label)] += 1
                
        count += 1
        if count % 500000 == 0:
            print("created", count, "datapoints")
            
    return dataset

indexer = Indexer()
label_counter = Counter()
dataset = create_dataset(tweets, indexer, label_counter)


