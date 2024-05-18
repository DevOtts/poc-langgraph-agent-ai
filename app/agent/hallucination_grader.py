# hallucination_grader.py
from langchain.prompts import PromptTemplate
from app.agent.language_model import LanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

class HallucinationGrader:
    def __init__(self, language_model : LanguageModel):
        self.language_model = language_model               

    def grade_hallucination(self, documents, generation):
        # Render the prompt with the given documents and generation
        page_contents = '\n'.join([doc.page_content for doc in documents])
        
        messages = [
            ("system", """You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                        Here are the facts:
                        \n----------\n
                        {documents}
                        \n----------\n
                        Here is the answer: {generation}"""
             .format(documents=page_contents, generation=generation))
        ]        
        
        # Use the language model to assess hallucination
        json_response = self.language_model.invoke(messages)            
        
        # Parse the response to get the binary score
        parser = JsonOutputParser()
        print("\n HallucinationGrader:", parser.parse(json_response.content), "\n")
        return parser.parse(json_response.content)
    
