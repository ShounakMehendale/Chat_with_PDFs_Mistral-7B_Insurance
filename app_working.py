import streamlit as st
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_database(text_chunks):
    embeddings=HuggingFaceEmbeddings()
    vectordb=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectordb

def get_conversation_chain(vectordb):
    llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.9, "max_tokens":1000}
)
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 5}),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    #final_answer=""
    
    #for i in range(9):
    #while(len(final_answer)<1000):
    response=st.session_state.conversation({"question":user_question})
        #final_answer+=response['answer']
        #user_question=final_answer
    
    
    st.write(response["answer"])




def main():
    #load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    st.header("Chat with your PDFs :))")
    user_question=st.text_input("Ask a question from these PDFs")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                #get_pdf_text
                raw_text=get_pdf_text(pdf_docs)
                

                #get_text_chunks
                text_chunks=get_text_chunks(raw_text)
                

                #get_vector database
                vectordb=get_vector_database(text_chunks)
                st.write("Processed Successfully !")

                #get conversation chain
                st.session_state.conversation=get_conversation_chain(vectordb)


if __name__=='__main__':
    main()
