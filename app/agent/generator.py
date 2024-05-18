# generator.py
from app.agent.document_handler import DocumentHandler
from app.agent.language_model import LanguageModel
from app.agent.retriever import Retriever
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

class AnswerGenerator:
    def __init__(self, language_model : LanguageModel, retriever : Retriever):
        self.language_model = language_model
        self.retriever = retriever

    def generate_answer(self, question, urls):
        docs = self.retriever.retrieve().invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])        
        
        messages = [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("user", f"Question: {question}\nContext: {context}"),
            ("assistant", "Answer:")
        ]        

        return self.language_model.invoke(messages)
