from collections import Counter
import csv
import os
from emoji import UNICODE_EMOJI, demojize, unicode_codes
import html
import nltk
import re
from sklearn.utils import shuffle
from utils import Indexer
import string
import sys
# emoji_set = set(['😂', '😭', '❤', '😍', '🔥', '🤣', '💕', '🙏', '✨', '😩', '💜', '😊', '💀', '🤷', '👀', '💙', '🎃', '🤔', '😘', '👏', '🙌', '🎉', '💯', '🤦', '👍', '👉', '💖', '🙄', '😎', '😁'])
# emoji_set = set(['😂', '😭', '❤', '😍', '🔥', '💕', '🙏', '😩', '😊', '💀', '👀', '🎃', '😘', '💯', '🙄'])
punc = string.punctuation.replace("'", "…")

if sys.argv[1] == "15":
    emoji_set = set(['😂', '😭', '❤', '😍', '🔥', '💕', '🙏', '😩', '😊', '💀', '👀', '🎃', '😘', '💯', '🙄'])
else:
    emoji_set = set(['😂', '😭', '❤', '😍', '🔥', '🤣', '💕', '🙏', '✨', '😩', '💜', '😊', '💀', '🤷', '👀', '💙', '🎃', '🤔', '😘', '👏', '🙌', '🎉', '💯', '🤦', '👍', '👉', '💖', '🙄', '😎', '😁'])


def process_text(text):
    text = text.replace("RT ", "")
    text = text.replace("\n", " ").lower()
    #text = re.sub(r'[^\w\s]','',text)
    words = text.split(" ")

    for word in words:
        if "@" in word or word.startswith("http"):
            text = text.replace(word, "")

    text = text.translate(str.maketrans(" ", " ", punc))
    text = html.unescape(text)
    return text.strip()

def get_emojis_in_text(text, emoji_set):
    char_set = set(text)
    #print(char_set)
    all_emojis = set(emoji_set) & char_set

    return all_emojis

def clean_tweet(tweet):
    delimiters=(":",":")
    _DEFAULT_DELIMITER = ":"
    pattern = re.compile(u'(%s[a-zA-Z0-9\+\-_&.ô’Åéãíç()!#*]+%s)' % delimiters)
    def replace(match):
        mg = match.group(1).replace(delimiters[0], _DEFAULT_DELIMITER).replace(delimiters[1], _DEFAULT_DELIMITER)
        return ""
    demojized_tweet = demojize(tweet['processed_text'])
    clean_text = pattern.sub(replace, demojized_tweet)
    return ' '.join(clean_text.split())

tweets = []
unique_tweet_ids = set()
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
        if not row['emojis_in_text'] or row['id'] in unique_tweet_ids:
            continue
        tweets.append(row)
        unique_tweet_ids.add(row['id'])
        count += 1
        if count % 200000 == 0:
            print("processed", count, "tweets")

print ("length of tweets: ", len(tweets))

def get_labels(tweet, indexer):

    all_emojis = tweet['emojis_in_text']
    emojis = {}
    c = Counter()
    for emoji in all_emojis:
        indexer.add_and_get_index(emoji)
        c[emoji] += tweet['processed_text'].count(emoji)

    max_score = max(c.values())
    labels = [indexer.index_of(k) for k in c if c[k] == max_score]
    emojis = [k for k in c if c[k] == max_score]
    return labels, emojis

def get_most_recent_label(tweet, emojis, indexer):
    emojis_in_text = tweet['emojis_in_text']
    for i in reversed(tweet['processed_text']):
        if i in emojis:
            label_emoji = i
            break
    return indexer.index_of(label_emoji)

class DataPoint():
    def __init__(self, text, label):
        self.text = text
        self.label = label

def create_dataset(tweets, indexer, label_counter):
    dataset = []
    count = 0
    for idx, tweet in enumerate(tweets):
        try:
            labels, emoji_labels = get_labels(tweet, indexer)
            label = get_most_recent_label(tweet, emoji_labels, indexer)
        except:
            continue

        cleaned_text = clean_tweet(tweet)
        datapoint = DataPoint(cleaned_text, label)
        dataset.append(datapoint)
        label_counter[indexer.get_object(label)] += 1
        count += 1
        if count % 500000 == 0:
            print("created", count, "datapoints")

    return dataset

indexer = Indexer()
label_counter = Counter()
dataset = create_dataset(tweets, indexer, label_counter

print ("length of dataset: ", len(dataset))

from tokenizer import tokenizer as vinay
v = vinay.TweetTokenizer(regularize=True, preserve_len=False)


word_cnts = Counter()
def count_words(text):
    words = v.tokenize(text)
    for word in words:
        word_cnts[word] += 1

for dp in dataset:
    count_words(dp.text)

new_dataset = []
count_of_bad = 0
for i in range(0, len(dataset)):
    data = dataset[i]
    temp = data.text.replace("’", "'")
    temp = temp.replace("“", "")
    temp = temp.replace("”", "")
    words = v.tokenize(temp)
    for idx, word in enumerate(words):
        if word_cnts[word] <= 10:
            words[idx] = ""
    line = " ".join(word for word in words)
    if line.isspace() or line == "":
        count_of_bad += 1
    else:
        data.text = line
        new_dataset.append(data)
    if i % 200000 == 0:
            print("iterated", i, "cleaned datapoints")

emoji_sample_count = {}

for dp in new_dataset:
    emoji_sample_count[indexer.get_object(dp.label)] = emoji_sample_count.get(indexer.get_object(dp.label), 0) + 1

print ("emoji_sample count ", emoji_sample_count)
shuffle(new_dataset)

sample_dataset = []
emoji_sample_counter = {}

if sys.argv[2] == "Sample":
    for dp in new_dataset:
        if dp.label in emoji_sample_counter:
            if emoji_sample_counter[dp.label] < 50000:
                sample_dataset.append(dp)
                emoji_sample_counter[dp.label] += 1
        else:
            emoji_sample_counter[dp.label] = 1
            sample_dataset.append(dp)
else:
    for dp in new_dataset:
        if dp.label in emoji_sample_counter:
            if emoji_sample_counter[dp.label] < 700000:
                sample_dataset.append(dp)
                emoji_sample_counter[dp.label] += 1
        else:
            emoji_sample_counter[dp.label] = 1
            sample_dataset.append(dp)

print ("emoji_sample_counter ", emoji_sample_counter)
print ("length of sample dataset: ", len(sample_dataset))

shuffle(sample_dataset)
train_split = int(round(0.8*len(sample_dataset)))
dev_split = train_split + int(round(0.5 * (len(sample_dataset) - train_split)))

train_data = sample_dataset[:train_split]
dev_data = sample_dataset[train_split:dev_split]
test_data = sample_dataset[dev_split:]

train_file = "data/train_" + sys.argv[3] + ".csv"
dev_file = "data/dev_" + sys.argv[3] + ".csv"
test_file = "data/test_" + sys.argv[3] + ".csv"
indexer_file = "data/indexer_" + sys.argv[3] + ".csv"

index_dict = indexer.objs_to_ints
with open('indexer.csv', 'w') as f:
    for key in index_dict.keys():
        f.write("%s|%s\n"%(key,index_dict[key]))


with open(train_file, 'w') as f:
    for datapoint in train_data:
        text = re.sub(r'[^\w\s]','', datapoint.text)
        f.write("%s|%s\n"%(str(datapoint.label),datapoint.text))

with open(dev_file, 'w') as f:
    for datapoint in dev_data:
        text = re.sub(r'[^\w\s]','', datapoint.text)
        f.write("%s|%s\n"%(str(datapoint.label),datapoint.text))

with open(test_file, 'w') as f:
    for datapoint in test_data:
        text = re.sub(r'[^\w\s]','', datapoint.text)
        f.write("%s|%s\n"%(str(datapoint.label),datapoint.text))
