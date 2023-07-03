import pickle
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
# import torch
import os
import faiss
import time

app = Flask(__name__)

# torch.cuda.empty_cache()
# other models: all-mpnet-base-v2 multi-qa-mpnet-base-dot-v1  all-distilroberta-v1 all-MiniLM-L12-v2 multi-qa-distilbert-cos-v1 all-MiniLM-L6-v2
model_name = 'multi-qa-mpnet-base-dot-v1'
model = SentenceTransformer(model_name)

# Load the Faiss index
dindex = faiss.read_index('index.faiss')  # Replace 'index.faiss' with the path to your saved Faiss index
print(dindex)

# Load word index
with open(f'words.pkl', "rb") as fIn:
    wordlist = pickle.load(fIn)
    words = wordlist['words']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        # Perform semantic similarity search using the embeddings index
        results, timespent = search(query)
        return render_template('index.html', results=results, query=query, timespent = timespent)
    return render_template('index.html')

def search(query):
    t=time.time()
    query_vector = model.encode([query])
    k = 50
    top_k = dindex.search(query_vector, k)
    predictions = []
    distance = []
    for i,_id in enumerate(top_k[1].tolist()[0]):
        if (words[_id] not in predictions):
            predictions.append(words[_id])
            distance.append(top_k[0].tolist()[0][i])

    output = zip(predictions, distance)
    output = [(i+1, *tpl) for i, tpl in enumerate(output)][:25]
    timespent = time.time()-t
    return output, timespent

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


