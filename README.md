# Reverse dictionary with semantic search

## First create indexes
- 900k dictionary definition-word dataset: [Google Drive Link](https://drive.google.com/file/d/1XbiQJidncJmvhr-hm9ai_9zNJ6J75i9g/view?usp=sharing)
- create FAISS index with create_faiss_embeddings.py
- create pickle word index with pickle_data.py
  
## Query index
- query the created indexes with search_index.py
```
python search_index.py
```

## Webapp
```
python app.py
```
