from langchain.prompts import PromptTemplate

augment_prompt = """
Answer the question based only on the following context:
{context}
"""

augment_prompt_template = PromptTemplate(
    input_types={"context": "str"}, 
    template=augment_prompt
    )

def get_augment_prompt() -> PromptTemplate:
    return augment_prompt_template