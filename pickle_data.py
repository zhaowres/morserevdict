import pickle
import json

f = open('../../data/data_full.json')
data = json.load(f)

words = []
for point in data:
    words.append(point['word'])

with open(f'../../transformer_embeddings/words.pkl', "wb") as fOut:
    pickle.dump({'words': words}, fOut, protocol=pickle.HIGHEST_PROTOCOL)