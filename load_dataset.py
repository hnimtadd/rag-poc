"""
load the dataset from the data folder
"""

import asyncio
import json
import os
import warnings
from typing import Dict, List

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)

# ref: https://paperswithcode.com/dataset/aws-documentation
DATA_FOLDER = "./datasets/aws-documentation/documents"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


async def load_datasets(
    data_folder: str = DATA_FOLDER,
):
    """
    read each folder in the data folder,
    in each folder, read all md files in the "doc_source" folder

    loop in sub folder of data folder
    loop in files of each sub folder
    read file content
    append content to a list
    return the list
    """
    sub_folder = os.listdir(data_folder)
    for folder in sub_folder:
        files_path = os.path.join(data_folder, folder, "doc_source")
        if not os.path.exists(files_path):
            continue

        files = os.listdir(files_path)
        # documents: List[LangchainDocument] = []
        print("loadding document from", files_path)
        for _file in files:
            if not _file.endswith(".md"):
                continue
            file_path = os.path.join(data_folder, folder, "doc_source", _file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                document = LangchainDocument(
                    page_content=content,
                )
                yield document
        print("loadded", files_path)


def get_splitter(
    chunk_size: int,
    tokenizer_name: str = EMBEDDING_MODEL_NAME,
):
    """
    split the documents into chunks of size chunk_size
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    def splitter(
        knowledge_base: List[LangchainDocument],
    ):
        docs_processed: List[LangchainDocument] = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents(knowledge_base)

        # Remove duplicates
        unique_texts: Dict[str, bool] = {}
        docs_processed_unique: List[LangchainDocument] = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    return splitter


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME):
    """
    get the embedder model
    """
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )

    async def embedder(
        docs: List[LangchainDocument],
    ):
        return embedding_model.embed_documents(
            [doc.page_content for doc in docs],
        )

    return embedder


def embed_query(
    query: str,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> List[float]:
    """
    get the embedder for user query
    """
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )
    query_vector = embedding_model.embed_query(query)
    return query_vector


async def pipe_line(
    chunk_size: int,
    data_folder: str = DATA_FOLDER,
    tokenizer_name: str = EMBEDDING_MODEL_NAME,
    model_name: str = EMBEDDING_MODEL_NAME,
    db_file: str = "knowledge.json",
):
    """
    loop in folder, get all md file and
    load the datasets and split them into chunks
    """
    print("loading dataset from", data_folder)
    files_path = os.path.join(data_folder, "doc_source")
    if not os.path.exists(files_path):
        return

    splitter = get_splitter(chunk_size, tokenizer_name)
    embedder = get_embedder(model_name)

    files = os.listdir(files_path)
    documents: List[LangchainDocument] = []
    for _file in files:
        if not _file.endswith(".md"):
            continue
        file_path = os.path.join(data_folder, "doc_source", _file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            document = LangchainDocument(
                page_content=content,
            )
            documents.append(document)

    for doc in documents:
        chunks = splitter([doc])
        embeddings = await embedder(chunks)
        docs = {
            chunk.page_content: embedding
            for (chunk, embedding) in zip(chunks, embeddings)  # noqa
        }
        with open(db_file, "a+", encoding="utf-8") as _file:
            json.dump(docs, _file)
            _file.write("\n")
    print("done", data_folder)


async def main():
    await asyncio.gather(
        *[
            pipe_line(
                chunk_size=512,
                data_folder=os.path.join(DATA_FOLDER, folder),
                tokenizer_name=EMBEDDING_MODEL_NAME,
                model_name=EMBEDDING_MODEL_NAME,
                db_file="sample-aws_knowledge.jsonl",
            )
            for folder in os.listdir(DATA_FOLDER)[0:1]
        ]
    )


if __name__ == "__main__":
    asyncio.run(main=main())
