"""
Note: Prior to running this script, you need to run Elasticsearch Docker container.

docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
"""

import numpy as np
import pandas as pd
import requests

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

EMBEDDING_MODEL = SentenceTransformer("multi-qa-distilbert-cos-v1")
USER_QUESTION = "I just discovered the course. Can I still join it?"
QUESTION_VECTOR = EMBEDDING_MODEL.encode(USER_QUESTION)


# Prepare data ######################################
def build_dataset_url(relative_url):
    """Reads in a JSON file from a LLM Zoomcamp repo"""
    base_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main"
    return f"{base_url}/{relative_url}?raw=1"


def read_json_course_data(dataset_url, course_name):
    docs_response = requests.get(dataset_url)
    documents = docs_response.json()

    filtered_docs = []

    for doc in documents:
        if doc["course"] == course_name:
            filtered_docs.append(doc)
    return filtered_docs


def read_csv_course_data(dataset_url, course_name):
    docs_response = pd.read_csv(dataset_url)
    filtered_docs = docs_response[docs_response.course == course_name]
    return filtered_docs.to_dict(orient="records")


def generate_document_embeddings(docs):
    """Uses SentenceTransformer model to embed the documents"""
    embeddings = []
    for doc in docs:
        qa_text = f"{doc['question']} {doc['text']}"
        qa_text_vectors = EMBEDDING_MODEL.encode(qa_text)
        embeddings.append(qa_text_vectors)
    return embeddings


# Search index ###########################################
class VectorSearchEngine:
    """Searches an in-memory index for a match"""

    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


def set_elastic_search_index(es_client, index_name, documents):
    """Saves documents to an index
    NOTE: Drops the index each run
    """
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "id": {"type": "keyword"},
            }
        },
    }

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)


def elastic_search(query, course):
    """Conducts a search of an index stored in Elasticsearch"""
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": course}},
            }
        },
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in response["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


# Evaluate search engine ##########################################
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def evaluate(ground_truth):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q["document"]
        doc_question = EMBEDDING_MODEL.encode(q["question"])
        results = search_engine.search(doc_question)
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        # 'mrr': mrr(relevance_total),
    }


def evaluate_with_elasticsearch(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q["document"]
        results = search_function(q)
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        # 'mrr': mrr(relevance_total),
    }


if __name__ == "__main__":
    # Get answers to the homework questions
    dataset_url = build_dataset_url("03-vector-search/eval/documents-with-ids.json")
    mlz_docs = read_json_course_data(dataset_url, "machine-learning-zoomcamp")
    print(len(mlz_docs))

    mlz_embeddings = generate_document_embeddings(mlz_docs)
    X = np.array(mlz_embeddings)
    print(X.shape)

    scores = X.dot(QUESTION_VECTOR)
    print(sorted(scores))

    search_engine = VectorSearchEngine(documents=mlz_docs, embeddings=X)
    results = search_engine.search(QUESTION_VECTOR, num_results=5)
    print(results)

    ground_truth_dataset_url = build_dataset_url(
        "03-vector-search/eval/ground-truth-data.csv"
    )

    mlz_ground_truth_docs = read_csv_course_data(
        ground_truth_dataset_url, "machine-learning-zoomcamp"
    )
    print(mlz_ground_truth_docs[0])

    eval_hit_rate = evaluate(mlz_ground_truth_docs)
    print(eval_hit_rate)

    es_client = Elasticsearch("http://localhost:9200")

    index_name = "course-questions"
    set_elastic_search_index(es_client, index_name, mlz_docs)

    result = elastic_search(
        query="I just discovered the course. Can I still join it?",
        course="machine-learning-zoomcamp",
    )
    print(result)

    eval_hit_rate = evaluate_with_elasticsearch(
        mlz_ground_truth_docs,
        lambda q: elastic_search(q["question"], "machine-learning-zoomcamp"),
    )
    print(eval_hit_rate)
