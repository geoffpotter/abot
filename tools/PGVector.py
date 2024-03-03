
from lib.AgentToolDefinition import AgentToolDefinition as toolDef
from lib.vectors import getVectorDbRetriver, getEmbeddings



def getPGVectorTool():
    retriever = getVectorDbRetriver(getEmbeddings())
    return toolDef(
        name="PGVector Search",
        description= (
            "A wrapper around a PGVector "
            "filled with data about LangChain.  "
            "Input should be a search query."
        ),
        examples="\n".join([
            "Question: What is the Higgs boson?",
            "Thought: First, I should search PGVector for Higgs boson, in case I already have store information in this subject.",
            "Action: PGVector Search",
            "Action Input: Higgs boson",
            "Observation: [A collection of stored documents about the Higgs boson.]"
        ]),
        createTool=toolDef.retrieverTool(retriever),
    )