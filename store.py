"""
read file knowledge.json, include list of metadata and vector
and save vector to the milvus
"""

from typing import Dict, List, Any
import pandas as pd

from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    exceptions,
)
from settings import COLLECTION_NAME


def create_collection(
    _client: MilvusClient,
    name: str,
    dims: int,
):
    """
    create milvus collection with name and data
    """
    vector_size = dims
    # create collection schema
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=vector_size,
        ),
        FieldSchema(
            name="document",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
    ]
    schema = CollectionSchema(
        fields,
        "knowledge base",
        auto_id=True,
        enable_dynamic_field=True,
    )
    _client.create_collection(
        name,
        schema=schema,
    )
    index_params = MilvusClient.prepare_index_params()

    # creat collection index
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="embedding_index",
        params={"nlist": vector_size},
    )

    _client.create_index(
        collection_name=name,
        index_params=index_params,
    )


def drop_collection(
    _client: MilvusClient,
    name: str,
):
    """
    drop_collection drops collection with given name
    """
    try:
        _client.drop_collection(collection_name=name)
    except exceptions.MilvusException as e:
        print("cannot drop collection", str(e))


def reset_collection(
    _client: MilvusClient,
    name: str,
):
    """
    reset_collection drop collection if exist,
    """
    try:
        drop_collection(_client, name)
    except exceptions.MilvusException as e:
        print("collection not found", str(e))


def insert_data(
    _client: MilvusClient,
    name: str,
    data: Dict[str, List[float]],
):
    """
    insert_data check if the collection is existed, if not create collection.
    then insert data into collection
    """
    try:
        _ = _client.get_collection_stats(name)
    except exceptions.MilvusException:
        vector_size = len(list(data.values())[0])
        create_collection(_client, name, vector_size)

    # insert data to collection
    enities = [
        {
            "embedding": embedding,
            "document": document,
        }
        for document, embedding in data.items()
    ]
    try:
        response: Dict[Any, Any] = _client.insert(
            collection_name=name,
            data=enities,
        )
    except exceptions.MilvusException as e:
        print("cannot insert data", str(e))
        print(data)
        return
    print(response, "inserted into vector db")


pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

pd.options.display.float_format = "{:,.2f}".format
if __name__ == "__main__":
    client = MilvusClient(uri="http://localhost:19530")

    # read jsonl file in chunk with pd, each line of the jsonl is a dictionary
    df = pd.read_json(
        "aws_knowledge_v1.jsonl",
        lines=True,
        chunksize=100,
        typ="series",
    )

    # loop in chunk, each line of the json is a dictionary
    # (which is a dict with key is string and value is list of float)
    for chunk in df:
        chunk_dict: Dict[str, List[float]] = {}
        for val in chunk.values:
            knowledge: Dict[str, List[float]] = val
            chunk_dict.update(knowledge)
        insert_data(client, COLLECTION_NAME, chunk_dict)
