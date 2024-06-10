"""
check collection stats
"""

from pymilvus import MilvusClient, exceptions
from settings import COLLECTION_NAME

if __name__ == "__main__":
    try:
        with open("knowledge.json", "r", encoding="utf-8") as file:
            client = MilvusClient(uri="http://localhost:19530")
            stat = client.get_collection_stats(collection_name=COLLECTION_NAME)
            print(stat)
    except exceptions.MilvusException as e:
        print("collection not found", str(e))
