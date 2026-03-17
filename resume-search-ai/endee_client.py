"""
endee_client.py — Interact with Endee Vector Database
"""

import traceback
from endee import Endee, Precision
from endee.schema import VectorItem

INDEX_NAME = "resumes"
EMBEDDING_DIM = 384
SPACE_TYPE = "cosine"
PRECISION = Precision.FLOAT32


def _get_client(base_url, auth_token=""):
    client = Endee(auth_token if auth_token else "")
    client.set_base_url(base_url)
    return client


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_resume_index(base_url, auth_token=""):
    try:
        client = _get_client(base_url, auth_token)
        try:
            client.delete_index(INDEX_NAME)
        except Exception:
            pass
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type=SPACE_TYPE,
            precision=PRECISION,
        )
        return True, "Index 'resumes' created successfully!"
    except Exception as e:
        return False, f"Failed to create index: {str(e)}"


def get_index_stats(base_url, auth_token=""):
    try:
        client = _get_client(base_url, auth_token)
        index = client.get_index(name=INDEX_NAME)
        stats = index.describe()
        if isinstance(stats, dict):
            return stats
        elif hasattr(stats, '__dict__'):
            return vars(stats)
        return {"info": str(stats)}
    except Exception as e:
        return {"error": str(e)}


def upsert_resume(base_url, auth_token, resume_id, embedding, metadata, full_text=""):
    try:
        client = _get_client(base_url, auth_token)
        index = client.get_index(name=INDEX_NAME)

        vector_data = {
            "id": resume_id,
            "vector": embedding,
            "meta": metadata,
            "filter": {},
        }

        index.upsert([vector_data])
        return True, f"Resume '{resume_id}' stored successfully!"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Upsert error: {str(e)} | Trace: {tb[-300:]}"


def search_resumes(base_url, auth_token, query_embedding, top_k=5):
    try:
        client = _get_client(base_url, auth_token)
        index = client.get_index(name=INDEX_NAME)
        results = index.query(vector=query_embedding, top_k=top_k)
        if results is None:
            return None
        output = []
        for r in results:
            item = {
                "id": _get_attr(r, "id", "unknown"),
                "similarity": _get_attr(r, "similarity", 0.0),
                "meta": _get_attr(r, "meta", {}),
            }
            output.append(item)
        return output
    except Exception as e:
        print(f"Search error: {e}")
        return None


def delete_resume(base_url, auth_token, resume_id):
    try:
        client = _get_client(base_url, auth_token)
        index = client.get_index(name=INDEX_NAME)
        index.delete_vector(resume_id)
        return True, f"Resume '{resume_id}' deleted."
    except Exception as e:
        return False, f"Failed to delete: {str(e)}"


def list_all_resumes(base_url, auth_token):
    try:
        client = _get_client(base_url, auth_token)
        index = client.get_index(name=INDEX_NAME)
        zero_vector = [0.0] * EMBEDDING_DIM
        results = index.query(vector=zero_vector, top_k=100)
        if results is None:
            return []
        output = []
        for r in results:
            item = {
                "id": _get_attr(r, "id", "unknown"),
                "similarity": _get_attr(r, "similarity", 0.0),
                "meta": _get_attr(r, "meta", {}),
            }
            output.append(item)
        return output
    except Exception as e:
        print(f"Error listing resumes: {e}")
        return None