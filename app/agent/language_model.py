# language_model.py
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class LanguageModel:
    def __init__(self, model_id, openai_api_key, langsmith_api_key, endpoint, format='json', temperature=0):
        self.model_id = model_id
        self.format = format
        self.temperature = temperature
        os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
        os.environ['LANGCHAIN_ENDPOINT'] = endpoint
        os.environ["OPENAI_API_KEY"] = openai_api_key

    def invoke(self, prompt):
        # Example using Langchain's Langchain connection
        #llm = langchain.llama(self.model_id, format=self.format, temperature=self.temperature)
        llm = ChatOpenAI(model=self.model_id, temperature=self.temperature)
        return llm.invoke(prompt)
    
    def invoke2(self, prompt):
        llm = ChatOpenAI(model=self.model_id, temperature=self.temperature)
        messages = [
            ("system", "You are a helpful assistant that translates English to French."),
            ("human", """Translate this sentence {prompt} from English to French. I love programming."""),
        ]
        print(self.invoke(messages))
        #print(llm.invoke(messages))
