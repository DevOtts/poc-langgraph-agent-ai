import os
from app.agent.graph_state import GraphState, Workflow
from app.agent.retrieval_grader import RetrievalGrader
from app.agent.language_model import LanguageModel
from app.agent.document_handler import DocumentHandler
from app.agent.retriever import Retriever
from app.agent.generator import AnswerGenerator

def main_manual():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
    firecrawl_api_key = os.environ["FIRECRAWL_API_KEY"]
    endpoint = 'https://api.smith.langchain.com'
    urls = [
        "https://hubcharge.pro/",
    ]

    lang_model = LanguageModel('gpt-3.5-turbo', openai_api_key, langsmith_api_key, endpoint)
         
    doc_handler = DocumentHandler(firecrawl_api_key)
    docs = doc_handler.load_and_format_documents(urls)
    
    print("\n\nDocument Content1:", docs, "\n\n") 
    
    retriever = Retriever(docs)
    
    # Grading the relevance of a document
    question = "Como posso alugar um goleiro?"
    grader = RetrievalGrader(lang_model)
    docs2 = retriever.retrieve().invoke(question) #do cs[1].page_content  # Assuming docs is non-empty and has at least two documents
    document_content = docs2[1].page_content
    print("\n\nDocument Content:", document_content, "\n\n")
    relevance_score = grader.grade_relevance(document_content, question)
    print("Document Relevance:", relevance_score)

    print("\n\n Start Answer \n\n")
    # return
    generator = AnswerGenerator(lang_model, retriever.retrieve())

    question = "Como posso alugar um goleiro?"
    print(generator.generate_answer(question, urls))