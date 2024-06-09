"""
embed.py
"""

import json
from typing import Dict, List, Optional
import os

from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding

from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers import SentenceTransformer
from settings import MAX_VECTOR_DIMENSIONS


def get_default_openai_client() -> OpenAI:
    """
    get_default_openai_client return default openai client,
    with api key read from OPENAI_API_KEY variable
    """
    _client: OpenAI = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def get_embedding_vector(
    docs: List[str],
    _client: Optional[OpenAI] = None,
) -> Dict[str, List[float]]:
    """
    get_embedding_vector try embed list of document by openAI APIS,
    use text-embedding-3-small-model
    """
    if _client is None:
        _client = get_default_openai_client()

    _embeddings: Dict[str, List[float]] = {}

    response: CreateEmbeddingResponse = _client.embeddings.create(
        input=docs,
        model="text-embedding-3-small",
    )

    response_embeds: List[Embedding] = response.data
    for doc, _embed in zip(docs, response_embeds):
        _embeddings[doc] = _embed.embedding

    return _embeddings


def get_query_embedding_vector_with_transformer(
    query: str,
) -> List[float]:
    """
    get_query_embedding_vector_with_transformer try
    to embed query using transformer model
    """
    instruct = "Represent this sentence for searching relevant passages:"
    query_prefix = "Instruct: " + instruct + "\nSentence: "
    model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1",
        truncate_dim=MAX_VECTOR_DIMENSIONS,
    )

    _query_embedding = model.encode(
        query,
        prompt=query_prefix,
    )
    binary_embeddings = quantize_embeddings(
        [_query_embedding],  # type: ignore
        precision="ubinary",
    )
    return binary_embeddings.tolist()[0]


def get_embedding_vector_with_transformer(
    docs: List[str],
) -> Dict[str, List[float]]:
    """
    this function tries to usetransformer model to embed list of document
    """
    model = SentenceTransformer(
        "mixedbread-ai/mxbai-embed-large-v1",
        truncate_dim=MAX_VECTOR_DIMENSIONS,
    )

    doc_embeddings = model.encode(docs)
    _embeddings = quantize_embeddings(
        doc_embeddings,  # type: ignore
        precision="ubinary",
    ).tolist()

    return dict(zip(docs, _embeddings))


if __name__ == "__main__":
    # open file output.json, which contains list of chapter, each chapter is
    # a json which keys are section name and values are section value
    with open("output.json", "r", encoding="utf-8") as file:
        chapters: List[Dict[str, str]] = json.load(file)

    embeddings: Dict[str, List[float]] = {}

    for chapter in chapters:
        sections = list(chapter.values())
        result = get_embedding_vector_with_transformer(sections)

        for section, embedding in result.items():
            embeddings[section] = embedding

    with open("knowledge.json", "w+", encoding="utf-8") as file:
        json.dump(embeddings, file, indent=4, ensure_ascii=False)

    print("done")
