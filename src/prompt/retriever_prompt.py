from langchain.prompts import PromptTemplate

multi_retriever_prompt = """
You are an AI language model assistant. Your task is to generate five different versions 
of the given user question to retrieve relevant documents from a vector database. By 
generating multiple perspectives on the user question, your goal is to help the user 
overcome some of the limitations of the distance-based similarity search. Provide these 
alternative questions separated by newlines. Original question: {question}
"""
multi_retriever_prompt_template = PromptTemplate(
    input_types={"question": "str"}, 
    template=multi_retriever_prompt
    )

retriever_agent_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:{agent_scratchpad}
"""

retriever_agent_prompt_template = PromptTemplate(
    input_types={"question": "str"}, 
    template=retriever_agent_prompt
    )

def get_multi_retriever_prompt() -> PromptTemplate:
    return multi_retriever_prompt_template

def get_retriever_agent_prompt() -> PromptTemplate:
    return retriever_agent_prompt_template