from langchain.schema import Document
from langgraph.graph import StateGraph, END
from app.agent.document_handler import DocumentHandler
from app.agent.retriever import Retriever
from app.agent.generator import AnswerGenerator
from app.agent.retrieval_grader import RetrievalGrader
from app.agent.hallucination_grader import HallucinationGrader
from app.agent.answer import AnswerGrader

class GraphState:
    def __init__(self, question='', generation='', web_search='No', documents=None, retry_count=0):
        self.question = question
        self.generation = generation
        self.web_search = web_search
        self.documents = documents or []
        self.retry_count = retry_count

class Workflow:
    def __init__(self, lang_model, api_key, docs):
        self.lang_model = lang_model
        self.document_handler = DocumentHandler(api_key)
        self.retriever = Retriever(docs)  # Assumes the retriever is already configured with a store
        self.answer_generator = AnswerGenerator(lang_model, self.retriever)
        self.retrieval_grader = RetrievalGrader(lang_model)
        self.hallucination_grader = HallucinationGrader(lang_model)
        self.answer_grader = AnswerGrader(lang_model)
        self.setup_graph()

    def setup_graph(self):
        self.graph = StateGraph(GraphState)
        self.retries = 0

        # Define the nodes in the workflow
        self.graph.add_node("retrieve", self.retrieve)
        self.graph.add_node("websearch", self.web_search)        
        self.graph.add_node("grade_documents", self.grade_documents)
        self.graph.add_node("generate", self.generate)

        # Set entry point
        self.graph.set_entry_point("retrieve")

        # Define direct edges
        self.graph.add_edge("retrieve", "grade_documents")
        self.graph.add_edge("websearch", "generate")

        # Define conditional edges
        self.graph.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            }
        )

        self.graph.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "force_end" : END
            }
        )
        
    def retrieve(self, state: GraphState) -> GraphState:        
        question = state.get('question')
        documents = self.retriever.retrieve().invoke(question)
        return GraphState(question=question, documents=documents)

    def grade_documents(self, state: GraphState) -> GraphState:
        print("step grade_documents")
        filtered_docs, web_search = [], "No"
        for d in state.documents:
            if self.retrieval_grader.grade_relevance(state.question, d.page_content) == "yes":
                filtered_docs.append(d)
            else:
                web_search = "Yes"
        return GraphState(question=state.question, documents=filtered_docs, web_search=web_search)

    def generate(self, state: GraphState) -> GraphState:
        print("step generate")
        generation = self.answer_generator.generate_answer(state.question, state.documents)
        return GraphState(question=state.question, documents=state.documents, generation=generation)

    def web_search(self, state: GraphState) -> GraphState:
        print("step web_search")
        # Example placeholder for web search functionality
        docs_content = "Example web search result content"
        web_results_doc = Document(page_content=docs_content)
        documents = state.documents + [web_results_doc]
        return GraphState(question=state.question, documents=documents)

    def decide_to_generate(self, state: GraphState) -> str:
        print("step decide_to_generate")
        return "websearch" if state.web_search == "Yes" else "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        print("step grade_generation_v_documents_and_question")
        if self.hallucination_grader.grade_hallucination(state.documents, state.generation) == "yes":
            if self.answer_grader.grade_answer(state.question, state.generation) == "yes":
                print("useful")
                return "useful"
            else:
                print("NOT useful:",self.retries)
                if self.retries >= 2:  
                    state.generation.content = "Sorry, I don't know the answer to this question."
                    return "force_end"
                else:
                    self.retries += 1  
                    return "not useful"
        else:            
            print("NOT supported:",self.retries)
            if self.retries >= 2:                  
                print("\n\nFINAL GENERATION: \n\n", state.generation.content, "\n\n")
                state.generation.content = "Sorry, I don't know the answer to this question."
                return "force_end"
            else:
                self.retries = self.retries +1
                new_state = GraphState(question=state.question, documents=state.documents, generation=state.generation, retry_count=state.retry_count + 1)
                return "not supported"
