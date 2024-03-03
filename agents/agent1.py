import os
import re
from typing import List, Union

from langchain import hub

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from lib.AgentToolDefinition import AgentToolDefinition

from tools.Arxiv import getArxivTool
from tools.Github import github_tools, github_tools_raw
from tools.PGVector import getPGVectorTool, getVectorDbRetriver, getEmbeddings
from tools.SearxngSearch import getSearxSearchTool
from tools.WebScrape import getWebScrapeTool

from lib.llm import getOpenAILLM, getOpenAIChatLLM, AlpacaLLM

from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools, create_openai_tools_agent, create_react_agent, create_structured_chat_agent
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate

from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.format_scratchpad import format_log_to_str


from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import BaseTool

from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


repl = PythonAstREPLTool(
    # locals={"df": df},
    name="python_repl",
    description="Runs code and returns the output of the final line",
    args_schema=PythonInputs,
)
replTool = Tool(repl.name, repl.run, repl.description)

arxiv = getArxivTool()
searx = getSearxSearchTool()
pgvector = getPGVectorTool()
web_scrape = getWebScrapeTool()

toolsDefs:list[AgentToolDefinition] = [arxiv, searx, pgvector, web_scrape]
tools = [replTool] + [tool.getTool() for tool in toolsDefs] + github_tools_raw

# toolsDefs:list[AgentToolDefinition] = [arxiv]
# tools = [replTool] + [tool.getTool() for tool in toolsDefs]# + github_tools_raw


llm = getOpenAILLM()
# llm = AlpacaLLM()

# # STRUCTURED_CHAT includes args_schema for each tool, helps tool args parsing errors.
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )
# print("Available tools:")
# for tool in tools:
#     print("\t" + tool.name)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
template = """Answer the following questions as best you can. 
You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Additional context about question:
{context}

Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]
    
    def format_messages(self, **kwargs) -> str:
        print("formatting prompt", kwargs)
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        if "intermediate_steps" in kwargs:
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
                
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
        else:
            print("No intermediate steps found. Setting agent_scratchpad to empty string.")
            kwargs["agent_scratchpad"] = ""
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    # input_variables=["input", "intermediate_steps", "context"]
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
)

hub_prompt = hub.pull("hwchase17/react")
print("\n\nhub prompt:", hub_prompt)


hub_prompt2 =  hub.pull("hwchase17/structured-chat-agent")
for k, v in enumerate(hub_prompt2):
    print(k, v)

for message in hub_prompt2.messages:
    print("\n", message, "\n\n")
# print("\n\nhub prompt2:", hub_prompt2)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        print("\nparsing output\n", llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            print("final answer found")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        print("action:", action, "action_input:", action_input)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    


embedings = getEmbeddings()
retriever = getVectorDbRetriver(embedings, top_k=2)

# # LLM chain consisting of the LLM and a prompt
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# print([convert_to_openai_function(t) for t in tools])
llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

llm_with_tools_and_stop = llm.bind(stop=["\nObservation"])

format_scratchpad = lambda x: format_log_to_str(x["intermediate_steps"])

def format_docs(docs):
    # return "\n".join([f"{doc.page_content}\nSource: {doc.metadata['url']}" for doc in docs])
    return "\n\n".join([f"{doc.page_content}" for doc in docs])
# RAG chain

tool_data = [convert_to_openai_function(t) for t in tools]
tool_names = [tool.name for tool in tools]
chain = (
    # RunnableParallel({"context": retriever, "question": RunnablePassthrough(), "agent_scratchpad": format_scratchpad})
    {
        "input": lambda x: x["input"],
        "context": lambda x: format_docs(retriever.invoke(x["input"])),
        "agent_scratchpad": lambda x: format_log_to_str(
            x["intermediate_steps"]
        ),
        "tools": lambda x: tool_data,
        "tool_names": lambda x: tool_names,
    }
    | prompt
    | llm_with_tools_and_stop
    | ReActSingleInputOutputParser()
)



# Using tools, the LLM chain and output_parser to make an agent


print("available tools:", tool_names)
# agent = LLMSingleActionAgent(
#     llm_chain=chain, 
#     output_parser=output_parser,
#     # We use "Observation" as our stop sequence so it will stop when it receives Tool output
#     # If you change your prompt template you'll need to adjust this as well
#     stop=["\nObservation:"], 
#     allowed_tools=tool_names
# )

class AgentInput(BaseModel):
    input: str

# Initiate the agent that will respond to our queries
# Set verbose=True to share the CoT reasoning the LLM goes through
# agent_executor = AgentExecutor.from_agent_and_tools(agent=chain, tools=tools, verbose=True, return_intermediate_steps=True, max_iterations=50).with_types(input_type=AgentInput)

agent = create_react_agent(llm, tools, hub_prompt)
# agent = create_structured_chat_agent(llm, tools, hub_prompt2)

agent_executor = AgentExecutor(agent=chain, tools=tools, max_iterations=50).with_types(input_type=AgentInput)

agent_executor = agent_executor | (lambda x: x["output"])