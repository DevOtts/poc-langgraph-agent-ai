import os
from app.agent.graph_state import Workflow
from app.agent.language_model import LanguageModel
from app.agent.document_handler import DocumentHandler

def main_graph():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
    firecrawl_api_key = os.environ["FIRECRAWL_API_KEY"]
    endpoint = 'https://api.smith.langchain.com'
    urls = [
        "https://hubcharge.pro/en",
    ]

    lang_model = LanguageModel('gpt-3.5-turbo', openai_api_key, langsmith_api_key, endpoint)
    
    doc_handler = DocumentHandler(firecrawl_api_key)
    docs = doc_handler.load_and_format_documents(urls)
    
    # Create the workflow instance with the language model and API key
    workflow = Workflow(lang_model, langsmith_api_key, docs)
    
    # Compile the workflow into an executable application
    app = workflow.graph.compile()

    # Define the initial inputs for the graph
    inputs = {"question": "How can I recharge my mobile phone?"}
    
    # Execute the compiled workflow with the inputs
    from pprint import pprint
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}")
        print(value.generation if hasattr(value, 'generation') else "No generation produced")
