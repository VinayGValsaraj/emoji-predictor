import csv
from utils import Indexer, get_indexer, get_data_from_csv, DataPoint

indexer = get_indexer()
dataset = get_data_from_csv()

for data in dataset[0:5]:
    print(data.label, data.text)