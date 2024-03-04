
import os
import pprint
from bs4 import BeautifulSoup

from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.pydantic_v1 import BaseModel, Field

from lib.AgentToolDefinition import AgentToolDefinition as toolDef
from lib.url_crawler import url_crawler
from lib.vectors import getVectorDbRetriver, getEmbeddings
from langchain.tools import BaseTool, StructuredTool, Tool
import lib.html_parser as hp

parser = hp.html_parser([hp.absLinks()], False)


class ScrapeInput(BaseModel):
    url: str = Field(description="The URL to load")
    useSelenium: bool = Field(description="Whether to use Selenium to load the page, or just parse the HTML.  Some pages require Selenium to load properly, but it's slower.", default=False)


def loadAndParse(url:str, useSelenium:bool = False):
        parse_errors:bool=True
        parser.currentURL = url
        html_page, error = url_crawler.LoadURL(url, useSelenium)
        
        if error is not None:
            #parse the error anyway, so the LLM can see what happened
            #(not for when putting into the database, but for live agent searches)
            error = error
            print(f"Error loading url: {error}")
            if not parse_errors:
                return

        element = None
        #only parse the page if there was no error
        if error is None:
            try:
                page = BeautifulSoup(html_page, 'html.parser')
                try:
                    page = BeautifulSoup(page.find("body").prettify(), 'html.parser')
                    element = page.find("body")
                except Exception as e:
                    print("Error prettifying page, ignoring", url,  e)
            except Exception as e:
                print("unable to parse page, skipping", url, e)
                error = e
            try:
                #only follow links when no errors
                if element is not None:
                    links = parser.getLinks(element)
            except Exception as e:
                print("Error getting links, ignoring", url, e)
                

        if element is not None:
            markdown = parser.toMarkdown(element)

        if error is None:
            output = f"""Page Content:
{markdown}"""
        else:
            output = f"""Error:
{error}"""

        return output




def getScrapeTool(name:str, description:str):
    # return StructuredTool.from_function(
    #     func=loadAndParse,
    #     name=name,
    #     description=description,
    #     # args_schema=ScrapeInput,
    #     # return_direct=True,
    #     # coroutine= ... <- you can specify an async method if desired as well
    # )
    return Tool(name=name, 
        func=loadAndParse, 
        description=description)

def getWebScrapeTool():
    
    return toolDef(
        name="Browse the Web",
        description= (
            "Useful for when you want to load a web page from a URL."
        ),
        examples="\n".join([
            "Question: What is this webpage about? https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents",
            "Thought: I should load the url.",
            "Action: Internet Search",
            "Action Input: https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents, false",
            "Observation: [The text of the webpage], [links on the page], [any errors]",
        ]),
        createTool=getScrapeTool,
    )