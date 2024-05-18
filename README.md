# AI Agent using LangGraph

This project POC on creating an AI agent that uses a language graph (LangGraph) to perform various tasks. The agent is designed to handle documents, retrieve information, generate answers, and grade the quality of its own responses.

This follows the tutorial provided by [AI Jason](https://www.youtube.com/@AIJasonZ) in his video ["I want Llama3 to perform 10x with my private knowledge" - Local Agentic RAG w/ llama3](https://www.youtube.com/watch?v=u5Vcrwpzoz8). All credits to his awesome job!

## Project Structure

The project is structured as follows:

- [``app/agent/agent.py``](): This is the main entry point of the application. It sets up the environment variables, initializes the language model, and runs the workflow.
- [``app/agent/graph_state.py``](): This file contains the `Workflow` and `GraphState` classes. The `Workflow` class is responsible for setting up the graph and defining the steps of the workflow. The `GraphState` class represents the state of the graph at any given point in time.
- [``app/agent/answer.py``](): These files contain classes responsible for grading the quality of the agent's responses.
- [``app/agent/retriever.py``](): This file contains the Retriever class, which is responsible for retrieving information from documents.
- [``app/agent/generator.py``](): This file contains the AnswerGenerator class, which is responsible for generating answers based on the retrieved information and the current state of the graph.
- [``app/agent/document_handler.py``](): These files are responsible for handling documents, including loading and formatting documents.


## Workflow

The workflow of the AI agent is as follows:

1. The agent retrieves information based on the current state of the graph.
2. The agent grades the retrieved documents.
3. The agent generates an answer based on the current state of the graph.
4. The agent performs a web search based on the current state of the graph.
5. The agent decides whether to generate an answer or not.
6. The agent grades the generated answer against the documents and the question.

## Running the Project

First, install the packages using `pip install -r requirements.txt`. 
I recommend to doing this in a virtual environment like `venv`.

To run the project, execute `python root.py`. This function sets up the environment variables, initializes the language model, and runs the workflow.

## Dependencies

The project requires the following environment variables to be set:

- `OPENAI_API_KEY`: The API key for OpenAI.
- `LANGSMITH_API_KEY`: The API key for LangSmith.
- `FIRECRAWL_API_KEY`: The API key for FireCrawl.

The project also requires the following Python packages:

- [`langchain`](https://python.langchain.com/v0.1/docs/get_started/introduction/)
- [`langgraph`](https://python.langchain.com/v0.1/docs/langgraph/)
- [`firecrawl`](https://docs.firecrawl.dev/introduction)

Please refer to the [``requirements.txt``]() file for a full list of dependencies.

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.