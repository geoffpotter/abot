
import os
import pprint

from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain.pydantic_v1 import BaseModel, Field

from lib.AgentToolDefinition import AgentToolDefinition as toolDef
from lib.vectors import getVectorDbRetriver, getEmbeddings
from langchain.tools import BaseTool, StructuredTool, Tool


searxng_search = SearxSearchWrapper(searx_host=os.environ.get('SEARX_HOST'))



class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    num_results: int = Field(description="The number of results to return", default=3)


def web_search(query: str, num_results: int = 3):
    results = searxng_search.results(query, num_results=num_results,categories="general",
    time_range="year")
    # pprint.pp(results)
    output = "Search Results:\n\n"
    for r in results:
        output += f"{r['link']}:\n{r['title']}\n{r['snippet']}\n\n"
    return output

def getSearTool(name:str, description:str):
    # return StructuredTool.from_function(
    #     func=web_search,
    #     name=name,
    #     description=description,
    #     # args_schema=SearchInput,
    #     # return_direct=True,
    #     # coroutine= ... <- you can specify an async method if desired as well
    # )
    return Tool(name=name, 
                func=web_search, 
                description=description)



def getSearxSearchTool():
    
    return toolDef(
        name="Internet Search",
        description= (
            "Useful for when you need to search the internet for information. "
        ),
        examples="\n".join([
            "Question: What is the Langchain?",
            "Thought: I should search the internet for Langchain.",
            "Action: Internet Search",
            "Action Input: Langchain",
            "Observation: [A collection of internet results for Langchain.]"
        ]),
        createTool=getSearTool,
    )