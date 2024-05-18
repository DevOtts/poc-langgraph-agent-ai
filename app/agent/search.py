### Search
import os

# Setting the API key for the Tavily search service
os.environ['TAVILY_API_KEY'] = 'xxxxx'

from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize the web search tool with Tavily
web_search_tool = TavilySearchResults(k=3)
