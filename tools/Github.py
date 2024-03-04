import os
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from lib.AgentToolDefinition import AgentToolDefinition as toolDef

github = GitHubAPIWrapper()
ai_branch = os.environ.get("GITHUB_BRANCH", "ai-code")
github.set_active_branch(ai_branch)
toolkit = GitHubToolkit.from_github_api_wrapper(github)

github_tools_raw = toolkit.get_tools()

github_tools = []
for tool in github_tools_raw:
    def getTool(name:str, description:str):
        return tool
    t = toolDef(name=tool.name, 
                description=tool.description,
                examples="",
                createTool=getTool)
    github_tools.append(t)


# return toolDef(
#         name="PGVector Search",
#         description= (
#             "A wrapper around a PGVector "
#             "filled with data about LangChain.  "
#             "Input should be a search query."
#         ),
#         examples="\n".join(
#             "Question: What is the Higgs boson?",
#             "Thought: First, I should search PGVector for Higgs boson, in case I already have store information in this subject.",
#             "Action: PGVector Search",
#             "Action Input: Higgs boson",
#             "Observation: [A collection of stored documents about the Higgs boson.]"
#         ),
#         createTool=toolDef.retrieverTool(retriever),
#     )