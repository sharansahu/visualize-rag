import argparse
import subprocess
import os
import hashlib
import platform 
import json
import numpy as np
import pandas as pd
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from renumics import spotlight
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

def start_ollama_server():
    """Start the Ollama server if it's not already running"""
    try:
        # Check if Ollama server is already running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("Ollama server is already running")
            return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    
    try:
        # Start Ollama server in the background
        print("Starting Ollama server...")
        # Use Popen to start server in background without blocking
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Verify server is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Ollama server started successfully")
            return True
        else:
            print("Failed to start Ollama server")
            return False
            
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return False

def ensure_model_available(model_name):
    """Ensure the specified model is available locally"""
    try:
        # Check if model is already available
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name in result.stdout:
            print(f"Model {model_name} is already available")
            return True
        else:
            print(f"Pulling model {model_name}...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Model {model_name} pulled successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error ensuring model {model_name} is available: {e}")
        return False

def create_ground_truth_answers(questions: List[str]) -> List[str]:
    """
    Create ground truth answers for evaluation.
    In a real scenario, you would have human-annotated answers.
    For demonstration, we'll use placeholder answers.
    """
    print("\n=== Ground Truth Collection ===")
    print("For RAGAS evaluation, some metrics (namely context_recall) require grouth truth annotations.")
    print("You can either:")
    print("1. Enter ground truth answers manually")
    print("2. Skip ground truth (context_recall won't be available)")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == "1":
        ground_truths = []
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}: {question}")
            ground_truth = input("Enter the ground truth answer: ").strip()
            ground_truths.append(ground_truth)
        return ground_truths
    else:
        return [""] * len(questions)  # Empty ground truths

def should_rebuild_vectorstore(docs_vectorstore, docs_dir, vectorstore_dir):
    """Determine if vector store should be rebuilt based on document changes"""
    try:
        # Create a hash of all PDF files in the directory
        pdf_files = []
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    filepath = os.path.join(root, file)
                    # Get file modification time and size for change detection
                    stat = os.stat(filepath)
                    pdf_files.append({
                        'name': file,
                        'size': stat.st_size,
                        'mtime': int(stat.st_mtime)
                    })
        
        # Sort for consistency
        pdf_files.sort(key=lambda x: x['name'])
        
        # Create hash of current document state
        current_docs_hash = hashlib.md5(
            json.dumps(pdf_files, sort_keys=True).encode()
        ).hexdigest()
        
        # Check if we have a stored hash
        hash_file = os.path.join(vectorstore_dir, 'docs_hash.txt')
        
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if stored_hash == current_docs_hash:
                # Check if vector store actually has content
                try:
                    # Use the correct method to check if collection has documents
                    existing_response = docs_vectorstore.get()
                    existing_count = len(existing_response.get("ids", []))
                    if existing_count > 0:
                        print(f"Documents unchanged, using existing {existing_count} chunks")
                        return False
                except Exception as e:
                    print(f"Could not check existing documents: {e}")
                    # If we can't check, assume we need to rebuild
                    pass
        
        # Store the new hash
        os.makedirs(vectorstore_dir, exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(current_docs_hash)
        
        return True
        
    except Exception as e:
        print(f"Could not check document changes: {e}")
        # Default to checking existing chunks
        try:
            existing_response = docs_vectorstore.get()
            existing_count = len(existing_response.get("ids", []))
            return existing_count == 0  # Only rebuild if empty
        except:
            return True  # Rebuild if we can't check

def main(args):
    ollama_models = []
    
    if args.embeddings_model.startswith("ollama"):
        ollama_models.append(args.embeddings_model.split(":")[1])
    if args.llm_model.startswith("ollama"):
        ollama_models.append(args.llm_model.split(":")[1])
    
    if ollama_models:
        check_ollama()

        if not start_ollama_server():
            print("Failed to start Ollama server. Exiting.")
            return
        
        for model_name in set(ollama_models): 
            if not ensure_model_available(model_name):
                print(f"Failed to ensure model {model_name} is available. Exiting.")
                return

    # Handle OpenAI API key if needed
    if args.embeddings_model.startswith("openai") or args.llm_model.startswith("openai"):
        if "OPENAI_API_KEY" not in os.environ:
            openai_api_key = input("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = openai_api_key

    if args.embeddings_model.startswith("openai"):
        embeddings_model = OpenAIEmbeddings(model=args.embeddings_model.split(":")[1])
    elif args.embeddings_model.startswith("ollama"):
        embeddings_model = OllamaEmbeddings(model=args.embeddings_model.split(":")[1])

    docs_vectorstore = Chroma(
        collection_name="docs_store",
        embedding_function=embeddings_model,
        persist_directory=args.vectorstore_dir
    )

    should_rebuild = should_rebuild_vectorstore(docs_vectorstore, args.docs_dir, args.vectorstore_dir)
    
    if should_rebuild:
        print("Rebuilding vector store due to document changes...")
        try:
            docs_vectorstore.delete_collection()
            docs_vectorstore = Chroma(
                collection_name="docs_store",
                embedding_function=embeddings_model,
                persist_directory=args.vectorstore_dir,
            )
        except Exception as e:
            print(f"Note: {e}")  # Collection might not exist yet

        # Only load and process documents if we're rebuilding
        loader = DirectoryLoader(
            args.docs_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            show_progress=True,
        )
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        # Add all documents (since we rebuilt, they're all new)
        split_ids = [stable_hash(doc) for doc in splits]
        docs_vectorstore.add_documents(splits, ids=split_ids)
        print(f"Added {len(splits)} chunks to Chroma.")
    else:
        print("Using existing vector store - no document processing needed.")

    # Set up LLM
    if args.llm_model.startswith("openai"):
        llm = ChatOpenAI(model=args.llm_model.split(":")[1], temperature=0.0)
    elif args.llm_model.startswith("ollama"):
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
    contexts = []
    while True:
        question = input("Enter your question (or 'done' to finish): ")
        if question.lower() == 'done':
            break
        response = rag_chain.invoke(question)['answer']
        print("Answer:", response)
        questions.append(question)
        answers.append(response)
        contexts.append([docs.page_content for docs in retriever.invoke(question)])
    
    if questions:
        ground_truths = create_ground_truth_answers(questions)
        formatted_contexts = []
        for context_list in contexts:
            if isinstance(context_list, list):
                formatted_contexts.append(context_list)
            else:
                formatted_contexts.append([str(context_list)])
        
        data = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": formatted_contexts,
            "reference": ground_truths
        }
        dataset = Dataset.from_dict(data)

        if args.llm_model.startswith("ollama"):
            ragas_llm = ChatOllama(
                model=args.llm_model.split(":")[1], 
                temperature=0.0,
                request_timeout=360,  # 6 minutes per request
            )
        else:
            ragas_llm = ChatOpenAI(
                model=args.llm_model.split(":")[1], 
                temperature=0.0,
                request_timeout=360,
            )
        
        if args.embeddings_model.startswith("ollama"):
            ragas_embeddings = OllamaEmbeddings(
                model=args.embeddings_model.split(":")[1]
            )
        else:
            ragas_embeddings = OpenAIEmbeddings(
                model=args.embeddings_model.split(":")[1]
            )
        
        individual_results = {}
        metrics_to_try = [
            ("faithfulness", faithfulness),
            ("answer_relevancy", answer_relevancy),
            ("context_precision", context_precision),
            ("context_recall", context_recall)
        ]
        
        print("Running RAGAS evaluation metrics individually...")
        
        for metric_name, metric in metrics_to_try:
            print(f"Evaluating {metric_name}...")
            try:
                if metric_name == "context_recall" and all(gt == "" for gt in ground_truths):
                    print(f"Skipping {metric_name} - no ground truth provided")
                    continue
                    
                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings
                )
                individual_results[metric_name] = result.to_pandas()[metric_name].tolist()
                print(f"✓ {metric_name} completed successfully")
                
            except Exception as e:
                print(f"✗ {metric_name} failed: {str(e)[:100]}...")
                individual_results[metric_name] = [None] * len(questions)
        
        if individual_results:
            combined_results = pd.DataFrame({
                "user_input": questions,
                "response": answers,
                "retrieved_contexts": formatted_contexts,
                "reference": ground_truths
            })
            
            for metric_name, values in individual_results.items():
                combined_results[metric_name] = values
            
            combined_results.to_csv("ragas_evaluation.csv", index=False)
            print(f"RAGAS evaluation completed with {len(individual_results)} metrics")
            print("Results saved to ragas_evaluation.csv")
            
            successful_metrics = [name for name, values in individual_results.items() if values[0] is not None]
            print(f"Successful metrics: {', '.join(successful_metrics)}")
            
        else:
            print("All RAGAS metrics failed - continuing with visualization...")
        
    response = docs_vectorstore.get(include=["metadatas", "documents", "embeddings"])

    embeddings_list = [list(emb) if emb is not None else [] for emb in response["embeddings"]]

    df = pd.DataFrame(
    {
        "id": response["ids"],
        "source": [metadata.get("source") for metadata in response["metadatas"]],
        "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
        "document": response["documents"],
        "embedding": embeddings_list, 
    }
)

    for i, question in enumerate(questions):
        question_embedding = embeddings_model.embed_query(question)
        question_row = pd.DataFrame(
            {
                "id": [f"question_{i}"],
                "question": [question],
                "embedding": [list(question_embedding)], 
            }
        )
        answer_embedding = embeddings_model.embed_query(answers[i])
        answer_row = pd.DataFrame(
            {
                "id": [f"answer_{i}"],
                "answer": [answers[i]],
                "embedding": [list(answer_embedding)], 
            }
        )
        df = pd.concat([question_row, answer_row, df], ignore_index=True)

        df["dist"] = df.apply(
            lambda row: np.linalg.norm(
                np.array(row["embedding"]) - np.array(question_embedding)
            ) if row["embedding"] else 0, 
            axis=1,
        )
        
    spotlight.show(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document QA and Visualization Script")
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory of documents to load")
    parser.add_argument("--vectorstore_dir", type=str, required=True, help="Directory to store the vector database")
    parser.add_argument("--embeddings_model", type=str, required=True, help="Model for embeddings (e.g., openai:text-embedding-ada-002 or ollama:mistral)")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model for QA (e.g., openai:gpt-4 or ollama:mistral)")

    args = parser.parse_args()
    main(args)
