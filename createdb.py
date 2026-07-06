from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md")
    return loader.load()

def split_documents(document: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(document)
    print("Split into chunks:", len(chunks))
    document = chunks[10]
    print("Random chunk:", document.page_content)
    print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=CHROMA_PATH)
    db.persist()
    print("Saved", len(chunks), "chunks to ChromaDB")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
