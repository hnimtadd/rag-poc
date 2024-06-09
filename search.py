"""
perform search query by embed search and get out document that relevant
to the query 
"""

from typing import List, Tuple

from pymilvus import (
    MilvusClient,
)
from pymilvus.client.types import ExtraList

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline,
)


from settings import COLLECTION_NAME
from load_dataset import embed_query

READER_MODEL_NAME: str = "HuggingFaceH4/zephyr-7b-beta"
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE: str = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)  # type: ignore


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
    llm: Pipeline,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
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

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    # Redact an answer
    print("=> Generating answer...")
    answer: str = llm(final_prompt)[0]["generated_text"]  # type: ignore

    return answer, relevant_docs


if __name__ == "__main__":
    query = input("Enter your query:")
    client = MilvusClient(uri="http://localhost:19530")
    answer, relevant = answer_with_rag(query, client, READER_LLM)
    print("==========================\nRelevant Documents:")
    for doc in relevant:
        print(doc)
    print("==========================\nAnswer:", answer)
