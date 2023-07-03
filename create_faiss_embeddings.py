import faiss
from sentence_transformers import SentenceTransformer
import json

f = open('../../data/data_full.json')
data = json.load(f)

sentences = []
words = []
for point in data:
    sentences.append(point['definitions'])
    words.append(point['word'])


model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
encoded_data = model.encode(sentences)

index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(sentences))))

faiss.write_index(index, '../../webapp/index.faiss')
