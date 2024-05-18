# document_handler.py
from langchain_community.document_loaders import FireCrawlLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentHandler:
    def __init__(self, api_key):
        self.api_key = api_key

    def load_and_format_documents(self, urls):
        docs = [FireCrawlLoader(api_key=self.api_key, url=url, mode="scrape").load() for url in urls]
        
        docs_list = [item for sublist in docs for item in sublist]  # Flattening the list

        # Initializing the text splitter with specific parameters
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, 
            chunk_overlap=0
        )

        # Splitting documents into manageable pieces
        doc_splits = text_splitter.split_documents(docs_list)

        return self._filter_complex_metadata(doc_splits)

    def _filter_complex_metadata(self, docs):
        filtered_docs = []
        for doc in docs:
            if isinstance(doc, Document) and hasattr(doc, 'metadata'):
                clean_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
                filtered_docs.append(Document(page_content=doc.page_content, metadata=clean_metadata))
        return filtered_docs
