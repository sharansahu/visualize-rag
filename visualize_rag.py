import argparse
import subprocess
import os
import hashlib
import json
import numpy as np
import pandas as pd
from typing import List
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from renumics import spotlight

def stable_hash(doc: Document) -> str:
    return hashlib.sha1(json.dumps(doc.metadata, sort_keys=True).encode()).hexdigest()

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata['source']}" for doc in docs
    )

def check_ollama():
    try:
        subprocess.run(["ollama", "--version"], check=True)
    except subprocess.CalledProcessError:
        install_ollama()

def install_ollama():
    user_response = input("Ollama is not installed. Do you want to download it? (yes/no): ")
    if user_response.lower() == "yes":
        os_type = platform.system().lower()
        if os_type == 'darwin':  # macOS
            download_url = "https://ollama.com/download/mac"
        elif os_type == 'linux':  # Linux
            download_url = "https://ollama.com/download/linux"
        elif os_type == 'windows':  # Windows
            download_url = "https://ollama.com/download/windows"
        else:
            print("Unsupported OS for automated Ollama installation. Please install it manually.")
            exit()
        
        installer_path = f"ollama-installer-{os_type}"
        if os_type == 'windows':
            subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri '{download_url}' -OutFile '{installer_path}.exe'"])
            subprocess.run([f"{installer_path}.exe"])
        else:
            subprocess.run(["curl", "-o", installer_path, download_url])
            subprocess.run(["chmod", "+x", installer_path])
            subprocess.run([f"./{installer_path}"])

def start_ollama_server(model_name):
    try:
        subprocess.run(["ollama", "serve", model_name], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ollama", "pull", model_name], check=True)
        subprocess.run(["ollama", "serve", model_name], check=True)

def main(args):
    if args.embeddings_model.startswith("ollama") or args.llm_model.startswith("ollama"):
        check_ollama()
        model_name = args.embeddings_model if args.embeddings_model.startswith("ollama") else args.llm_model
        start_ollama_server(model_name)

    if args.embeddings_model.startswith("openai") or args.llm_model.startswith("openai"):
        openai_api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Set up embeddings and vector store
    if args.embeddings_model.startswith("openai"):
        embeddings_model = OpenAIEmbeddings(model=args.embeddings_model.split(":")[1])
    elif args.embeddings_model.startswith("ollama"):
        # Assuming a hypothetical OllamaEmbeddings class
        embeddings_model = OllamaEmbeddings(model=args.embeddings_model.split(":")[1])

    docs_vectorstore = Chroma(
        collection_name="docs_store",
        embedding_function=embeddings_model,
        persist_directory=args.vectorstore_dir,
    )

    # Load documents
    loader = DirectoryLoader(
        args.docs_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        loader_kwargs={"open_encoding": "utf-8"},
        recursive=True,
        show_progress=True,
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    # Add documents to vector store
    split_ids = list(map(stable_hash, splits))
    docs_vectorstore.add_documents(splits, ids=split_ids)
    docs_vectorstore.persist()

    # Set up LLM and retriever
    if args.llm_model.startswith("openai"):
        llm = ChatOpenAI(model=args.llm_model.split(":")[1], temperature=0.0)
    elif args.llm_model.startswith("ollama"):
        # Assuming a hypothetical OllamaChat class
        llm = ChatOllama(model=args.llm_model.split(":")[1], temperature=0.0)

    retriever = docs_vectorstore.as_retriever(search_kwargs={"k": 20})

    # Set up prompt template
    template = """
    You are an assistant for question-answering tasks.
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    QUESTION: {question}
    =========
    {source_documents}
    =========
    FINAL ANSWER: """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain = RunnableParallel(
        {
            "source_documents": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    # Ask questions loop
    questions = []
    answers = []
    while True:
        question = input("Enter your question (or 'done' to finish): ")
        if question.lower() == 'done':
            break
        answer = rag_chain.invoke({"question": question})
        print("Answer:", answer)
        questions.append(question)
        answers.append(answer)

    # Visualize results
    response = docs_vectorstore.get(include=["metadatas", "documents", "embeddings"])
    df = pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source") for metadata in response["metadatas"]],
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": response["documents"],
            "embedding": response["embeddings"],
        }
    )

    for i, question in enumerate(questions):
        question_row = pd.DataFrame(
            {
                "id": [f"question_{i}"],
                "question": [question],
                "embedding": [embeddings_model.embed_query(question)],
            }
        )
        answer_row = pd.DataFrame(
            {
                "id": [f"answer_{i}"],
                "answer": [answers[i]],
                "embedding": [embeddings_model.embed_query(answers[i])],
            }
        )
        df = pd.concat([question_row, answer_row, df], ignore_index=True)

    spotlight.show(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document QA and Visualization Script")
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory of documents to load")
    parser.add_argument("--vectorstore_dir", type=str, required=True, help="Directory to store the vector database")
    parser.add_argument("--embeddings_model", type=str, required=True, help="Model for embeddings (e.g., openai:text-embedding-ada-002 or ollama:mistral)")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model for QA (e.g., openai:gpt-4 or ollama:mistral)")

    args = parser.parse_args()
    main(args)
