import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.core.data.embedded_vector import get_pdf_text, get_text_chunk, get_vectorstore
from src.core.data.retriever import HybridRetriever
from src.core.prompt.augment_prompt import get_augment_prompt
from src.core.model.ollama_model import get_model

def main():
    st.header("✨ Chat with multiple PDF files ✨")
    
    pdf_files = st.file_uploader(
        "Upload your PDF files here", 
        accept_multiple_files=True
        )
    
    if pdf_files:
        user_input = st.text_input(
            "Ask a question about your PDFs:",
            key="user_input"
            )
    else:
        st.warning("Please upload at least one PDF file before proceeding.")
        user_input = None

    output_placeholder = st.empty()
    if user_input and st.button("Get Answer"):
        with st.spinner("Processing..."):
            output_placeholder.empty()

            txt = get_pdf_text(pdf_files)
            txt_chunks = get_text_chunk(txt)
            vector_db = get_vectorstore(txt_chunks)
            
            retriever_agent = HybridRetriever(vector_db=vector_db)
            documents = retriever_agent.get_relevent_documents_by_agent(user_input)

            chain = create_stuff_documents_chain(llm=get_model(), prompt=get_augment_prompt())
            response = chain.invoke({"context": documents})

            output_placeholder.write(
                f"Here is the answer to your question: \n{response}"
                )
            

if __name__ == "__main__":
    main()