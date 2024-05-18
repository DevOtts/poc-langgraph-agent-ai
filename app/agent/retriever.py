# retriever.py
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.embeddings import GPT4AllEmbeddings

class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorstore = self._create_vector_db(documents)

    def _create_vector_db(self, documents):
        return Chroma.from_documents(
            documents=documents,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings()
        )

    def retrieve(self) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever()
