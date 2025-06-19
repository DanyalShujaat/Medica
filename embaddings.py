from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv('.env')

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

def load_pdf_files(folder_path: str) -> list[Document]:
    documents = []
    pdf_folder = Path(folder_path)
    pdf_files = list(pdf_folder.glob("*.pdf"))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            documents.append(
                Document(
                    page_content=page.page_content,
                    metadata={
                        "source": str(pdf_file),
                        "page": int(page.metadata.get("page", 0)),
                    },
                )
            )
    return documents


def main():
    documents = load_pdf_files("Medical-Data")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    chunks = splitter.split_documents(documents)

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "medical-data"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Batch upload chunks (100 at a time)
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Pinecone"):
        batch = chunks[i:i+batch_size]
        vector_store.add_documents(documents=batch)


if __name__ == "__main__":
    main()
