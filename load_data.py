import json
import csv
from emoji import UNICODE_EMOJI
import os

emoji_set = set(UNICODE_EMOJI)

csvfile = open('twitter_october.csv', 'a')
csvwriter = csv.writer(csvfile)

file_count = 0

for json_file in os.listdir('data/twitter/'):
    count = 0
    file_count += 1
    if json_file.endswith('.json'):
        path = os.path.join('data/twitter', json_file)
        with open(path, 'r') as file:
            contents = file.readlines()
            for line in contents:
                tweet = json.loads(line)
                if 'lang' in tweet:
                    if tweet['lang'] == 'en':
                        text = tweet['text']
                        chars = set(text)

                        if bool(chars & emoji_set):
                            tweet_dic = {}

                            tweet_dic['created_at'] = tweet['created_at']
                            tweet_dic['id'] = tweet['id']
                            tweet_dic['id_str'] = tweet['id_str']
                            tweet_dic['text'] = tweet['text']
                            tweet_dic['in_reply_to_status_id'] = tweet['in_reply_to_status_id']
                            tweet_dic['user_id'] = tweet['user']['id']
                            tweet_dic['user_id_str'] = tweet['user']['id_str']
                            tweet_dic['user_screen_name'] = tweet['user']['screen_name']

                            #print(tweet_dic)
                            #if count == 0:
                            #    header = tweet_dic.keys()
                            #    csvwriter.writerow(header)
                            #    count += 1

                            csvwriter.writerow(tweet_dic.values())
    #exit()
    if file_count % 50 == 0:
        print('processed files:', file_count)
                    
csvfile.close()
