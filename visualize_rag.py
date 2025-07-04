import argparse
import subprocess
import os
import hashlib
import platform 
import json
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from renumics.spotlight import Dataset as SpotDataset
from renumics.spotlight import show
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from umap import UMAP
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

    retriever = docs_vectorstore.as_retriever(search_kwargs={"k": 10})

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
    retrieved_sources = [] # Store retrieved sources for each question
    while True:
        question = input("Enter your question (or 'done' to finish): ")
        if question.lower() == 'done':
            break
        
        retrieved_docs = retriever.invoke(question)
        response = rag_chain.invoke(question)['answer']
        
        print("Answer:", response)
        questions.append(question)
        answers.append(response)
        contexts.append([docs.page_content for docs in retrieved_docs])
        retrieved_sources.append([docs.metadata.get('source') for docs in retrieved_docs])
    
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
                request_timeout=600,  # 10 minutes per request
            )
        else:
            ragas_llm = ChatOpenAI(
                model=args.llm_model.split(":")[1], 
                temperature=0.0,
                request_timeout=600,
            )
        
        if args.embeddings_model.startswith("ollama"):
            ragas_embeddings = OllamaEmbeddings(
                model=args.embeddings_model.split(":")[1]
            )
        else:
            ragas_embeddings = OpenAIEmbeddings(
                model=args.embeddings_model.split(":")[1]
            )
        
        run_cfg = RunConfig(
            timeout     = 600,   # up to 10 min per sample
            max_retries = 5,     # retry up to 5× on exception
            max_wait    = 30,    # max backoff wait = 30 s
            max_workers = 1,     # serialise to avoid rate‐limits
            log_tenacity=True,   # turn on retry logging
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
                    embeddings=ragas_embeddings,
                    run_config=run_cfg
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

    # Create the initial DataFrame for document chunks
    df_chunks = pd.DataFrame(
        {
            "id": response["ids"],
            "source": [metadata.get("source", "") for metadata in response["metadatas"]], # Handle missing source
            "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
            "document": [doc if doc is not None else "" for doc in response["documents"]], # Handle missing document
            "embedding": embeddings_list,
            "question": "",  # Initialize with empty string for display
            "answer": "",    # Initialize with empty string for display
            "dist": np.nan,
            "query_id": np.nan, # New column for sorting
            "type": "document"  # New column for sorting
        }
    )

    # Prepare lists for question and answer rows
    df_rows = []

    for i, question_text in enumerate(questions):
        question_embedding = embeddings_model.embed_query(question_text)
        answer_embedding = embeddings_model.embed_query(answers[i])

        # Question row
        question_row = {
            "id": f"question_{i}",
            "question": question_text,
            "embedding": list(question_embedding),
            "answer": "",
            "source": "",
            "page": np.nan,
            "document": "",
            "dist": 0.0, # Distance of question to itself is 0
            "query_id": i,
            "type": "question"
        }
        df_rows.append(question_row)

        # Answer row
        answer_row = {
            "id": f"answer_{i}",
            "answer": answers[i],
            "embedding": list(answer_embedding),
            "question": "",
            "source": "",
            "page": np.nan,
            "document": "",
            "dist": np.nan, # Distance for answer will be calculated later
            "query_id": i,
            "type": "answer"
        }
        df_rows.append(answer_row)

        # Add retrieved source documents for this question
        for doc_idx, retrieved_doc_content in enumerate(contexts[i]):
            # Find the original document ID for the retrieved content
            # This assumes that the content is unique enough to find the original id
            original_doc_id = next((df_chunks.loc[j, 'id'] for j, doc_chunk in df_chunks.iterrows() if doc_chunk['document'] == retrieved_doc_content), None)
            
            # If original_doc_id is found, retrieve other metadata. Otherwise, use what's available.
            retrieved_doc_source = retrieved_sources[i][doc_idx] if doc_idx < len(retrieved_sources[i]) else ""
            
            source_doc_row = {
                "id": original_doc_id if original_doc_id else f"retrieved_doc_{i}_{doc_idx}",
                "question": "",
                "answer": "",
                "source": retrieved_doc_source,
                "page": np.nan, # Can try to get page from original metadata if original_doc_id is found
                "document": retrieved_doc_content,
                "embedding": list(embeddings_model.embed_query(retrieved_doc_content)), # Re-embed if not found in df_chunks, or get from df_chunks
                "dist": np.nan, # Calculated later
                "query_id": i,
                "type": "source_document"
            }
            df_rows.append(source_doc_row)


    # Convert the list of question/answer/source dicts to a DataFrame
    if df_rows:
        df_qa_source = pd.DataFrame(df_rows)
        # Concatenate the QA and source documents DataFrame with the remaining chunks
        # Only include chunks not already present as "source_document"
        df_chunks_remaining = df_chunks[~df_chunks['id'].isin(df_qa_source['id'])].copy()
        
        # Ensure 'query_id' and 'type' are present in df_chunks_remaining
        df_chunks_remaining['query_id'] = np.nan
        df_chunks_remaining['type'] = 'document'

        df = pd.concat([df_qa_source, df_chunks_remaining], ignore_index=True)
    else:
        df = df_chunks # If no questions, just use the chunks dataframe

    # Calculate distances for all rows against each question
    for i, question_text in enumerate(questions):
        question_embedding = embeddings_model.embed_query(question_text)
        
        # Calculate distances for rows associated with this question
        # This will set the 'dist' for the answer and source documents related to this question
        for j, row in df.iterrows():
            if not pd.isna(row['query_id']) and row['query_id'] == i and \
               row['embedding'] is not None and len(row['embedding']) > 0:
                dist = np.linalg.norm(np.array(row["embedding"]) - np.array(question_embedding))
                # Only update if current dist is NaN or new dist is smaller (for multiple questions affecting dist)
                if pd.isna(df.loc[j, 'dist']) or dist < df.loc[j, 'dist']:
                    df.loc[j, 'dist'] = dist

    if questions:
        first_question_embedding = embeddings_model.embed_query(questions[0])
        for j, row in df.iterrows():
            if row['type'] == 'document' and pd.isna(row['dist']): # Only for chunks not related to a QA pair
                if row['embedding'] is not None and len(row['embedding']) > 0:
                    df.loc[j, 'dist'] = np.linalg.norm(np.array(row["embedding"]) - np.array(first_question_embedding))
    
    # Sort the DataFrame
    # Sort by query_id (questions first), then by type (question, answer, source_document, document), then by distance
    df['type_order'] = df['type'].map({'question': 0, 'answer': 1, 'source_document': 2, 'document': 3})
    df = df.sort_values(by=['query_id', 'type_order', 'dist']).reset_index(drop=True)
    df = df.drop(columns=['type_order']) # Remove the temporary sorting column

    # Fill NaN values in string columns with empty strings for better display
    for col in ['question', 'answer', 'source', 'document']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # Force column order: question, answer, source, document, then the rest (including x/y later)
    first_cols = ["question", "answer", "source", "document"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    n_neighbors = min(20, len(df) - 1)
    # Handle the case where there's only one data point for UMAP
    if len(df) <= 1:
        df["x"] = 0.0
        df["y"] = 0.0
    else:
        umap = UMAP(n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42)
        # Ensure embeddings are not empty lists or None before UMAP
        valid_embeddings_df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if len(valid_embeddings_df) > 1:
            coords = umap.fit_transform(valid_embeddings_df["embedding"].tolist())
            df.loc[valid_embeddings_df.index, "x"] = coords[:, 0]
            df.loc[valid_embeddings_df.index, "y"] = coords[:, 1]
            df["x"] = df["x"].fillna(0.0) # Fill NaNs for rows without embeddings
            df["y"] = df["y"].fillna(0.0) # Fill NaNs for rows without embeddings
        else:
            df["x"] = 0.0
            df["y"] = 0.0
        
    if args.h5_name:
        file_name = args.h5_name if args.h5_name.endswith(".h5") else f"{args.h5_name}.h5"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"docs_store_{ts}.h5"

    out_dir = os.path.join(args.vectorstore_dir, "visualization_datastore")
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, file_name)

    print(f"Saving Spotlight dataset to {h5_path}...")
    with SpotDataset(h5_path, "w") as ds:
        ds.append_string_column("question", df["question"].tolist(), order=0)
        ds.append_string_column("answer",   df["answer"].tolist(),   order=1)
        ds.append_string_column("source",   df["source"].tolist(),   order=2)
        ds.append_string_column("document", df["document"].tolist(), order=3)

        idx = 4
        for col in df.columns:
            if col in ("question","answer","source","document","x","y"):
                continue
            series = df[col]
            if col == "embedding" or (isinstance(series.iloc[0], list)):
                ds.append_embedding_column(col, series.tolist(), order=idx)
            elif pd.api.types.is_integer_dtype(series):
                ds.append_int_column(col, series.tolist(), order=idx)
            elif pd.api.types.is_float_dtype(series):
                ds.append_float_column(col, series.tolist(), order=idx)
            else:
                ds.append_string_column(col, series.astype(str).tolist(), order=idx)
            idx += 1

        ds.append_float_column("x", df["x"].tolist(), order=idx)
        ds.append_float_column("y", df["y"].tolist(), order=idx+1)

    print("Saved HDF5 dataset successfully.")

    print("Launching Spotlight viewer (precomputed 2D)…")
    show(h5_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document QA and Visualization Script")
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory of documents to load")
    parser.add_argument("--vectorstore_dir", type=str, required=True, help="Directory to store the vector database")
    parser.add_argument("--embeddings_model", type=str, required=True, help="Model for embeddings (e.g., openai:text-embedding-3-small or ollama:mistral)")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model for QA (e.g., openai:gpt-4.1 or ollama:mistral)")
    parser.add_argument("--h5_name", type=str, default=None, help="Optional base name for the HDF5 file (without .h5 extension)")

    args = parser.parse_args()
    main(args)