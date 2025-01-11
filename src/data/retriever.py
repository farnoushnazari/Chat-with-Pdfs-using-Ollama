from typing import List
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema import Document
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from src.core.prompt.retriever_prompt import (
    get_multi_retriever_prompt, 
    get_retriever_agent_prompt
)
from src.core.model.ollama_model import get_model


class HybridRetriever:
    def __init__(self, vector_db: Chroma):
        self.vector_db = vector_db

    def get_relevent_documents_by_agent(
            self, 
            query: str
            ) -> List[Document]:
        
        vector_index_retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1}
            )
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_index_retriever,
            llm=get_model(),
            prompt=get_multi_retriever_prompt()
            )

        simple_retriever = create_retriever_tool(
            retriever=vector_index_retriever,
            name="vector index retriever",
            description="Use this tool when the user's query is clear and unambiguous."
            )
        
        generative_retriever = create_retriever_tool(
            retriever=multi_query_retriever,
            name="multi query retriever",
            description="Use this tool when the user's query is ambiguous or unclear."
            )
        
        tools = [simple_retriever, generative_retriever]
        
        agent = create_react_agent(
            llm=get_model(),
            tools=tools,
            prompt=get_retriever_agent_prompt(),
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=[simple_retriever, generative_retriever],
            handle_parsing_errors=True,
            verbose=True
        )
        documents = agent_executor.invoke({'question':query})

        documents = [
            Document(page_content=documents['question']),
            Document(page_content=documents['output'])
            ]

        return documents