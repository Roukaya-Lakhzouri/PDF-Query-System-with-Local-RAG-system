

## 🧠 **Project: PDF Query System with Local Retrieval-Augmented Generation**

This project implements a **PDF Query System** powered by **Local Retrieval-Augmented Generation (RAG)**.
It allows users to **ask natural language questions about their PDF documents** and receive **context-aware, grounded answers** — all **locally**, without relying on cloud-based APIs.

At its core, the system combines:

* **Information retrieval** (to find the most relevant parts of your documents)
* **Language generation** (to synthesize an accurate, human-readable answer)

The system automatically extracts text from PDF files, breaks it into manageable chunks, generates dense **semantic embeddings** using a **SentenceTransformer**, and stores them in a **FAISS** vector database for fast retrieval.
When a user submits a query, the system retrieves the top relevant document chunks and passes them to a **local Transformer-based language model** (or optionally OpenAI) to generate an answer based only on the retrieved content.

### 🔍 Key Idea

Instead of relying on pre-trained models’ general knowledge, this system **grounds answers in your own data**  ,enabling private, domain-specific, and explainable question answering from any collection of PDFs.

This project demonstrates how **RAG architectures** can be implemented **locally** for secure and explainable knowledge retrieval from private PDFs — a useful base for chatbots, assistants, or document intelligence systems.

## 🚀 Key Features
- **PDF and text ingestion** with `PyPDF2`  
- **Semantic chunking** for better retrieval coverage  
- **Dense embeddings** using `sentence-transformers/all-MiniLM-L6-v2`  
- **FAISS vector database** for scalable similarity search  
- **Prompt-based generation** using local or OpenAI models  
- **Simple `rag_answer()` interface** to query and generate grounded responses  
- Modular, interpretable codebase with clear workflow structure

---

## 🧩 Techniques and Components

| Component | Technique / Tool | Description |
|------------|------------------|--------------|
| Document Loading | `PyPDF2` | Reads and extracts text from PDF files. |
| Text Chunking | Sliding Window Segmentation | Splits documents into overlapping chunks for robust retrieval. |
| Embeddings | `SentenceTransformers` | Converts text chunks into numerical vector representations. |
| Vector Store | `FAISS` | Enables fast similarity search among document embeddings. |
| Generation | `transformers` or `openai` | Generates grounded text responses using retrieved context. |
| Prompt Engineering | Structured Template | Ensures concise, source-based answers. |

---

## 🏗️ Architecture

```text
User Query
   ↓
Retriever (FAISS)
   ↓
Top-K Relevant Chunks
   ↓
Prompt Construction
   ↓
Generator (LLM)
   ↓
Final Answer
````

---

## ⚙️ Installation

Run once to install all dependencies:

```bash
pip install -q -U sentence-transformers faiss-cpu transformers accelerate datasets tiktoken PyPDF2
```

If you plan to use OpenAI models:

```bash
pip install openai
```

---

## 📂 Project Structure

```
PDF-Query-System-with-Local-RAG.ipynb     # Main notebook
Data/                  # Folder containing your .pdf or .txt files
```

---

## 🧠 How It Works

1. **Load Documents**

   * Extracts text from local PDF or TXT files in `Data/`.
2. **Preprocess and Chunk**

   * Splits text into overlapping chunks to optimize retrieval.
3. **Embed Chunks**

   * Encodes each chunk into a dense vector using `all-MiniLM-L6-v2`.
4. **Index with FAISS**

   * Builds a searchable index of embeddings.
5. **Retrieve + Generate**

   * Finds top-k relevant chunks for a query.
   * Builds a context-aware prompt.
   * Generates an answer using a chosen LLM.

---

## 🧪 Usage Example

```python
res = rag_answer("Explain the principles of the game Ticket to Ride.", top_k=3)
print(res["answer"])
```

**Output:**

```
The game is based on building railway routes between cities. Players collect cards of various types of train cars and use them to claim routes on a map.
```

---

## 💡 Customization

* Replace the embedding model in config:

  ```python
  EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
  ```
* Swap generation model:

  ```python
  GENERATION_MODEL = 'gpt2'  # or your OpenAI / HF model
  ```
* Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` for different document lengths.

---

## 🔮 Possible Extensions

* Integrate **LangChain** or **LlamaIndex** for orchestration.
* Add **OpenAI GPT-4**, **Llama 3**, or **Mistral** for stronger generation.
* Introduce **evaluation metrics** (similarity, factuality, BLEU).
* Wrap into an API or Streamlit/Gradio  web interface.

---

## 🧑‍💻 Author

Developed as a practical exploration of **Retrieval-Augmented Generation (RAG)** using modern NLP tools for document-based question answering.

---

## 📜 License

This project is open for educational and research use.
Please check each model or dataset’s license before deployment.

