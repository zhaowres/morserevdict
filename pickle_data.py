import pickle
import json

# read json dictionary data from 
f = open('data_full.json')
data = json.load(f)

words = []
for point in data:
    words.append(point['word'])

# write word index to
with open(f'words.pkl', "wb") as fOut:
    pickle.dump({'words': words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)