import os
from typing import List, Dict, Any

from elasticsearch import Elasticsearch

INDEX_NAME = "ad_templates"


def get_es_client() -> Elasticsearch:
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    return Elasticsearch(url)


def ensure_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=INDEX_NAME):
        return
    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "template_id": {"type": "keyword"},
                    "description": {"type": "text"},
                    "category": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                }
            }
        },
    )


def index_templates(es: Elasticsearch, templates: List[Dict[str, Any]]) -> None:
    for t in templates:
        es.index(index=INDEX_NAME, id=t["template_id"], document=t)
