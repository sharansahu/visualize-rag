# visualize-rag

This project allows you to load documents from a specified directory, create embeddings for these documents, store them in a vector database, and use a language model to answer questions about the documents. The results are then visualized using Spotlight.

## Features

- Load documents from a directory (supports PDF files).
- Create embeddings using OpenAI or Ollama models.
- Store document embeddings in a Chroma vector database.
- Ask questions and get answers using OpenAI or Ollama models.
- Visualize the documents and answers using Spotlight.

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- Windows 10 or later / macOS 11 Big Sur or later

### Virtual Environment

Create a virtual environment to containerize all requirements and avoid dependency conflicts

```sh
conda create --name visualize_rag_env python3=3.8.3
```

### Dependencies

Install the required dependencies using pip:

```sh
pip install -r requirements.txt
```

## Usage

### Command Line Arguments

- `--docs_dir` (required): Directory of documents to load (required).
- `--vectorstore_dir` (required): Directory to store the vector database (required).
- `--embeddings_model` (required): Model for embeddings (e.g., `openai:text-embedding-ada-002` or `ollama:mistral`) (required).
- `--llm_model` (required): LLM model for QA (e.g., `openai:gpt-4` or `ollama:mistral`) (required).
- `--h5_name` (optional): Base name for the HDF5 file (without .h5). If omitted, a timestamped name docs_store_YYYYMMDD_HHMMSS.h5 is generated.

### Running the Script

To run the script, use the following command:

```sh
python3 visualize_rag.py --docs_dir ./documents --vectorstore_dir ./vectorstore --embeddings_model openai:text-embedding-ada-002 --llm_model openai:gpt-4 --h5_name visualization_experiment_with_doc_1
```

For open source models, use the following command:

```sh
python3 visualize_rag.py --docs_dir ./documents --vectorstore_dir ./vectorstore --embeddings_model ollama:mistral --llm_model ollama:mistral --h5_name visualization_experiment_with_doc_1
```

### Interactive Question-Answer Session

After running the script, you will be prompted to enter your questions. Type your question and press Enter to get an answer. To finish the session, type `done`.

### Visualization

Once you finish the question-answer session, the script will visualize the results using Spotlight.

## Reload Visualization: launch_visualization.py
Use this helper script to load a previously saved .h5 dataset without rebuilding embeddings or re-running QA.

### Command Line Arguments
- `--h5_path` (required): Path to the saved HDF5 dataset (e.g., vectorstore/visualization_datastore/docs_store_20250619_103045.h5).

To run the script, use the following command:
```sh
python3 launch_visualization.py --h5_path ./vectorstore/visualization_datastore/my_docs_snapshot.h5
```
This will spin up Spotlight immediately on the saved dataset.

## Additional Information

### OpenAI API Key

If you choose to use OpenAI models for embeddings or the LLM, you will be prompted to enter your OpenAI API key.

### Ollama Installation

If you choose to use Ollama models and Ollama is not installed on your system, the script will prompt you to download and install it. Follow the on-screen instructions to complete the installation. If for some reason installation does not work, do the following steps to get Ollama running on your local machine.

- Go to ollama.com/download and install Ollama
- Once Ollama is installed, run the following command to download a open-source model: `ollama pull model_name (mistral, llama3, etc)`
- Once your models are downloaded, you can check if they are installed by running `ollama list`
- Finally, you can simply run `ollama serve` to launch an Ollama server
- To run the open source script command, you must install Mistral (`ollama pull mistral`) and Nomic embeddings (`ollama pull nomic-embed-text`)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
