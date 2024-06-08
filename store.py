"""
read file knowledge.json, include list of metadata and vector
and save vector to the milvus
"""

from typing import Dict, List
import json

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def create_collection(
    name: str,
    data: Dict[str, List[float]],
) -> Collection:
    """
    create milvus collection with name and data
    """
    vector_size = len(list(data.values())[0])
    # create collection schema
    fields = [
        FieldSchema(
            name="key",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=vector_size,
        ),
        FieldSchema(
            name="document",
            dtype=DataType.STRING,
        ),
    ]
    schema = CollectionSchema(fields, "knowledge base")
    collection = Collection(name, schema)

    # creat collection index
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": vector_size},
    }
    collection.create_index(field_name="embedding", index_params=index)

    # insert data to collection
    enities = [
        list(range(len(data))),  # key
        list(data.values()),  # embedding
        list(data.keys()),  # document
    ]
    collection.insert(enities)

    return collection


if __name__ == "__main__":
    connections.connect(
        "local",
        host="localhost",
        port="19530",
    )
    with open("knowledge.json", "r", encoding="utf-8") as file:
        knowledge: Dict[str, List[float]] = json.load(file)
        collection = create_collection("knowledge_collection", knowledge)
        print("Done")
