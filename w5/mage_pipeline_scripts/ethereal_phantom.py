from elasticsearch import exceptions
from typing import Dict, List, Union

import numpy as np
from elasticsearch import Elasticsearch

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import tes


@data_loader
def search(*args, **kwargs) -> List[Dict]:
    """
    query_embedding: Union[List[int], np.ndarray]
    """

    connection_string = kwargs.get("connection_string", "http://localhost:9200")
    index_name = kwargs.get("index_name", "documents")

    script_query = {
        "query": {
            "function_score": {
                "query": {"match": {"text": "When is it?"}},
            }
        }
    }

    es_client = Elasticsearch(connection_string)

    try:
        response = es_client.search(
            index=index_name,
            body=script_query,
        )

        print("Raw response from Elasticsearch:", response)

        hits = response["hits"]["hits"]
        document_ids = [hit["_id"] for hit in hits]
        print("Document IDs:", document_ids)
    except exceptions.BadRequestError as e:
        print(f"BadRequestError: {e.info}")
        return []

    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
