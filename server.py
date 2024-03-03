from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from agents.agent1 import agent_executor

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")




# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)
# add_routes(app, rag_chroma_private_chain, path="/rag-chroma-private")
# add_routes(app, mongo_parent_document_retrieval_chain, path="/mongo-parent-document-retrieval")
# add_routes(app, research_assistant_chain, path="/research-assistant")
# add_routes(app, retrieval_agent_chain, path="/retrieval-agent")

add_routes(app, agent_executor, path="/agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
