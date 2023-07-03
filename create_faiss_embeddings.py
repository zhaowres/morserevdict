import faiss
from sentence_transformers import SentenceTransformer
import json

# read json dictionary data from 
f = open('data_full.json')
data = json.load(f)

sentences = []
words = []
for point in data:
    sentences.append(point['definitions'])
    words.append(point['word'])

# other models: all-mpnet-base-v2 multi-qa-mpnet-base-dot-v1  all-distilroberta-v1 all-MiniLM-L12-v2 multi-qa-distilbert-cos-v1 all-MiniLM-L6-v2
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
encoded_data = model.encode(sentences)

index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(sentences))))

# write index to 
faiss.write_index(index, 'index.faiss')
