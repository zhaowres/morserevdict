import pickle
from sentence_transformers import SentenceTransformer, util
import json
import faiss
import time


# other models: all-mpnet-base-v2 multi-qa-mpnet-base-dot-v1  all-distilroberta-v1 all-MiniLM-L12-v2 multi-qa-distilbert-cos-v1 all-MiniLM-L6-v2
model_name = 'multi-qa-mpnet-base-dot-v1'
model = SentenceTransformer(model_name)

# Load the Faiss index from this location
dindex = faiss.read_index('index.faiss')  

# Load word index from this location
with open(f'words.pkl', "rb") as fIn:
    wordlist = pickle.load(fIn)
    words = wordlist['words']

def search(query):
    t=time.time()
    query_vector = model.encode([query])
    k = 50 # top k number of entries
    top_k = dindex.search(query_vector, k)
    predictions = []
    distance = []
    for i,_id in enumerate(top_k[1].tolist()[0]):
        if (words[_id] not in predictions): # only keep duplicates
            predictions.append(words[_id])
            distance.append(top_k[0].tolist()[0][i])

    output = zip(predictions, distance)
    output = [(i+1, *tpl) for i, tpl in enumerate(output)][:25] # display top 25
    timespent = time.time()-t
    timespent = (float(f'{timespent:.5f}'))
    return output, timespent

while True:
    try:
        query = input("> ")
        index_output, _ = search(query)
        index_words = [tupleObj[1] for tupleObj in index_output]
        print(f"Closest words:", ', '.join(index_words))
    except KeyboardInterrupt as e:
	    print()
	    exit()



