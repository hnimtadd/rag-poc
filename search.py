"""
perform search query by embed search and get out document that relevant
to the query
"""

from typing import List, Tuple, Optional, Callable

from pymilvus import (
    MilvusClient,
)
from pymilvus.client.types import ExtraList

from llama_cpp import Llama

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from settings import COLLECTION_NAME
from load_dataset import embed_query

READER_LLM = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=35,
)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
RERANKER = AutoModelForSequenceClassification.from_pretrained(
    RERANKER_MODEL_NAME,
)
RERANKER.eval()


def rerank(question: str, documents: List[str]) -> List[str]:
    with torch.no_grad():
        pairs = [[question, doc] for doc in documents]
        inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )

        scores = (
            RERANKER(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )

        # return list of document string
        return [doc for _, doc in sorted(zip(scores, documents), reverse=True)]


RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and
relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer,
just say that you don't know the answer, and guide user to the right source.
<|enc|>
<|user|>
Context: {context}
----
Now here is the question you need to answer.

Question: {question}
<|assistant|>
"""


NORAG_PROMPT_TEMPLATE = """
<|system|>
You are the chat bot with respond to answer user question
Respond only to the question asked, response should be concise and
relevant to the question.
If the answer cannot be deduced from the context, do not give an answer,
just say that you don't know the answer, and guide user to the right source.
<|enc|>
<|user|>
Now here is the question you need to answer.

Question: {question}
<|assistant|>
"""


def search_relevant(
    _client: MilvusClient,
    _collection_name: str,
    _query: List[float],
    _limit: int = 10,
) -> List[str]:
    """
    search relevant document to the query
    """
    _client.load_collection(collection_name=_collection_name)
    # search by embedding
    search_params = {
        "metric_type": "L2",
        "params": {},
    }
    results = _client.search(
        collection_name=_collection_name,
        data=[_query],
        anns_field="embedding",
        search_params=search_params,
        limit=_limit,
        output_fields=["document"],
        timeout=10,
    )
    assert isinstance(results, ExtraList)

    # get document from key
    _documents: List[str] = []
    for result in results:
        for document in result:
            _entity = document.get("entity")
            if _entity is None or not isinstance(_entity, dict):
                continue
            _doc = _entity.get("document")
            if _doc is None or not isinstance(_doc, str):
                continue

            _documents.append(_doc)

    return _documents


def answer_with_rag(
    question: str,
    vector_client: MilvusClient,
    llm=READER_LLM,
    reranker: Optional[Callable] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    """
    try to answer without providing the context
    """
    # Gather documents with retriever
    embedding = embed_query(question)
    print("=> Retrieving documents...")
    # get_query_embedding_vector_with_transformer(query)
    assert embedding is not None, "should return embedding"

    relevant_docs = search_relevant(
        vector_client,
        COLLECTION_NAME,
        embedding,
        num_retrieved_docs,
    )

    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker(question, relevant_docs)
    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]  # noqa
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )
    print("=> Final prompt:", final_prompt)

    # Redact an answer
    print("=> Generating answer...")

    _anwser = llm(  # type: ignore
        final_prompt,
        stop=["<|end|>"],
        echo=False,  # Whether to echo the prompt
        max_tokens=2048,  # Generate up to 256 tokens
    )["choices"][0][
        "text"
    ]  # type: ignore

    return _anwser, relevant_docs


def answer_without_rag(
    question: str,
    llm=READER_LLM,
) -> str:
    """
    try to answer without rag
    """
    final_prompt = NORAG_PROMPT_TEMPLATE.format(
        question=question,
    )
    print("=> Final prompt:", final_prompt)

    # Redact an answer
    print("=> Generating answer...")
    # _anwser: str = llm(final_prompt)[0]["generated_text"]  # type: ignore
    _anwser = llm(  # type: ignore
        final_prompt,
        stop=["<|end|>"],
        max_tokens=2048,  # Generate up to 256 tokens
        echo=False,  # Whether to echo the prompt
    )["choices"][0][
        "text"
    ]  # type: ignore

    return _anwser


if __name__ == "__main__":
    client = MilvusClient(uri="http://localhost:19530")
    while True:
        query = input("Enter your query:")
        if query == "":
            continue
        answer, relevant = answer_with_rag(
            query,
            client,
            llm=READER_LLM,
            reranker=rerank,
        )
        answer_normal = answer_without_rag(query)
        print("==========================\nRelevant Documents:")
        for doc in relevant:
            print(doc)

        print(
            "==========================\n" "Answer with rag:  " + answer,
            "==========================",
        )

        print(
            "==========================\n" "Answer without rag:  " + answer,
            "==========================",
        )
