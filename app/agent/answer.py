# answer_grader.py
from langchain.prompts import PromptTemplate
from app.agent.language_model import LanguageModel
from langchain_core.output_parsers import JsonOutputParser

class AnswerGrader:
    def __init__(self, language_model):
        self.language_model = language_model
        self.prompt_template = PromptTemplate(
            template="""<begin_of_text><start_header_id>system<end_header_id> You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                        Here is the answer:
                        \n----------\n
                        {generation}
                        \n----------\n
                        Here is the question: {question}<leot_id><start_header_id>assistant<end_header_id>""",
            input_variables=["generation", "question"],
        )

    def grade_answer(self, question, generation):
        # Render the prompt with the given question and answer (generation)
        messages = [
            ("system", """You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                        Here is the answer:
                        \n----------\n
                        {generation}
                        \n----------\n
                        Here is the question: {question}"""
             .format(generation=generation, question=question))
        ]
        # Use the language model to assess usefulness
        json_response = self.language_model.invoke(messages)
        # Parse the response to get the binary score
        parser = JsonOutputParser()
        #print("\n AnswerGrader:", parser.parse(json_response.content), "\n")
        return parser.parse(json_response.content)
