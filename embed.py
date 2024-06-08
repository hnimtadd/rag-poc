"""
embed.py
"""

import json
from typing import Dict, List
import os

import dotenv
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding


if __name__ == "__main__":
    # open file output.json, which contains list of chapter, each chapter is
    # a json which keys are section name and values are section value
    dotenv.load_dotenv(".env")
    print(os.environ.get("OPENAI_API_KEY"))

    with open("output.json", "r", encoding="utf-8") as file:
        chapters: List[Dict[str, str]] = json.load(file)

    client: OpenAI = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    embeddings: Dict[str, List[float]] = {}

    for chapter in chapters:
        sections = list(chapter.values())
        response: CreateEmbeddingResponse = client.embeddings.create(
            input=sections,
            model="text-embedding-3-small",
        )

        response_embeds: List[Embedding] = response.data
        for section, embed in zip(sections, response_embeds):
            embeddings[section] = embed.embedding

    with open("knowledge.json", "w+", encoding="utf-8") as file:
        json.dump(embeddings, file, indent=4, ensure_ascii=False)

    print("done")
