# retrieval_grader.py
from langchain.prompts import PromptTemplate
from app.agent.language_model import LanguageModel
from langchain_core.output_parsers import JsonOutputParser

class RetrievalGrader:
    def __init__(self, language_model : LanguageModel):
        self.language_model = language_model

    def grade_relevance(self, question, documents):
        # Construct the prompt using the predefined template
        
        messages = [
            ("system", """ You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
                            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                            Here is the retrieved document: \n\n{document} \n\n
                            Here is the user question: {question}"""
             .format(document=documents, question=question))
        ]
        # Use the language model to assess relevance
        json_response = self.language_model.invoke(messages)
        
        # Parse JSON response
        parser = JsonOutputParser()
        print("\n RetrievalGrader:", parser.parse(json_response.content), "\n")
        return parser.parse(json_response.content)
