# SherlockRAG 🔍

A Retrieval-Augmented Generation (RAG) system that answers questions about Sherlock Holmes novels using source-grounded context rather than the model's pretrained knowledge alone. Ask anything about the stories, characters, or plot — and get accurate, hallucination-resistant answers traced back to the original text.

---

## How It Works

1. Sherlock Holmes novels stored as Markdown files are loaded and split into overlapping chunks
2. Each chunk is embedded into a ChromaDB vector store using OpenAI's `text-embedding-3-small` model
3. User queries in natural language are semantically matched against the vector store via similarity search
4. The most relevant passages are retrieved and supplied as context to GPT-4.1 Nano
5. The model answers based strictly on the retrieved text — not its general training data — preventing hallucinations

---

## Project Structure

```
├── data/
│   └── books/            # Sherlock Holmes novels as Markdown files
├── chroma/               # ChromaDB vector store (auto-generated)
├── createdb.py           # Ingests and embeds novels into ChromaDB
├── query.py              # Interactive query interface
├── .env                  # API key (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

**1. Clone the repository and create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key to `.env`**
```
OPENAI_API_KEY=your-key-here
```

**4. Build the vector store**
```bash
python createdb.py
```

**5. Run the assistant**
```bash
python query.py
```

---

## Example Queries

```
Where did Sherlock Holmes first meet Dr. Watson?

What was the role of Irene Adler in A Scandal in Bohemia?

How did Holmes deduce that Watson had been in Afghanistan?

Describe the events leading up to Holmes's confrontation with Moriarty at the Reichenbach Falls.
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain | Document loading, chunking, prompt orchestration |
| ChromaDB | Vector store and similarity search |
| OpenAI Embeddings (`text-embedding-3-small`) | Chunk vectorisation |
| GPT-4.1 Nano | Answer generation |
| python-dotenv | API key management |

---

## Why RAG?

A base LLM can answer general questions about Sherlock Holmes from its training data, but it cannot cite specific passages, struggles with obscure plot details, and can confidently hallucinate events that never occurred in the novels. By grounding responses in the actual text via semantic retrieval, SherlockRAG produces answers that are accurate, specific, and traceable to a real source.
