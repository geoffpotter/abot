from typing import List, Tuple
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools.retriever import create_retriever_tool
from langchain_core.utils.function_calling import *
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from dotenv import load_dotenv
from langchain.schema import BaseRetriever, Document

from lib.AgentToolDefinition import AgentToolDefinition as toolDef

load_dotenv()


class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
    """`Arxiv` retriever.

    It wraps load() to get_relevant_documents().
    It uses all ArxivAPIWrapper arguments without any change.
    """

    get_full_documents: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        print("getting docs from arxiv")
        try:
            if self.is_arxiv_identifier(query):
                results = self.arxiv_search(
                    id_list=query.split(),
                    max_results=self.top_k_results,
                ).results()
            else:
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
        except self.arxiv_exceptions as ex:
            return [Document(page_content=f"Arxiv exception: {ex}")]
        docs = [
            Document(
                page_content=result.summary,
                metadata={
                    "Published": result.updated.date(),
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                },
            )
            for result in results
        ]
        return docs


def getArxivTool():
    return toolDef(
        name="Arxiv Search",
        description= (
            "A wrapper around Arxiv.org "
            "Useful for when you need to answer questions about Physics, Mathematics, "
            "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
            "Electrical Engineering, and Economics "
            "from scientific articles on arxiv.org. "
            "Input should be a search query."
        ),
        examples="\n".join([
            "Question: What is the Higgs boson?",
            "Thought: First, I should look for scientific articles about the Higgs boson.",
            "Action: Arxiv Search",
            "Action Input: What is the Higgs boson?",
            "Observation: [A list of scientific articles about the Higgs boson.]"
        ]),
        createTool=toolDef.retrieverTool(ArxivRetriever()),
    )