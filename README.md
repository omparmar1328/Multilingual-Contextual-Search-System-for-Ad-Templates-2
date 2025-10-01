## Multilingual Contextual Search (Python)

This project exposes a FastAPI endpoint for multilingual, embedding-based contextual search over simple ad templates. It supports optional Elasticsearch indexing.

### Quickstart (Windows PowerShell)

```bash
python -m venv search_env
search_env\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` and try the `/search` endpoint.

### Example Request

```json
{
  "query": "Zapatos para correr",
  "language": "es"
}
```

### Notes
- Translation uses `Helsinki-NLP/opus-mt-mul-en` (downloaded on first run).
- Embeddings use `all-MiniLM-L6-v2` from `sentence-transformers`.
- If you enable Elasticsearch, set `ELASTICSEARCH_URL` and run the indexing utility in `app/es_utils.py`.

### Optional: Elasticsearch (Local)
- Install and run Elasticsearch (you can use Docker Desktop on Windows):

```bash
# Example with Docker
docker run -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false docker.elastic.co/elasticsearch/elasticsearch:8.15.1
```

- Set env var `ELASTICSEARCH_URL=http://localhost:9200` and use functions in `app/es_utils.py` to create the index and index documents.

### Project Structure
- `app/main.py`: FastAPI app with `/search` endpoint
- `app/data.py`: Mock ad templates
- `app/pipeline.py`: Translation, language detection, embeddings, and search
- `app/es_utils.py`: Optional Elasticsearch helpers
- `requirements.txt`: Python dependencies

