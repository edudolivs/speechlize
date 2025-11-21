import streamlit as st
import os
import whisper
import json
from datetime import timedelta
from langchain_text_splitters import RecursiveCharacterTextSplitter           
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import ollama

def app():
    
    st.set_page_config("Speechlize", layout="wide")
    
    if "models" not in st.session_state:
        models = []
        models_list = ollama.list()
        if models_list and "models" in models_list:
            for model in models_list["models"]:
                models.append(model["model"])
            st.session_state["models"] = models
            

    st.write("# Speechlize")
    st.sidebar.header("Settings")
    TRANSCR = st.sidebar.selectbox(
        "Escolha um modelo de transcrição do whisper:",
        ["tiny", "base", "small", "medium", "large", "turbo"]
    )
    ENCODER = st.sidebar.selectbox("Escolha um modelo de embeddings:", st.session_state["models"])
    audio_file = st.sidebar.file_uploader("Escolha o arquivo de audio:", type="mp3")
    MODEL = st.sidebar.selectbox("Escolha um modelo de linguagem:", st.session_state["models"])
    st.session_state["model"] = OllamaLLM(model=MODEL)

    if audio_file:
        audio_path = "tmp/audio.mp3"

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        with open(audio_path, 'wb') as audio:
            audio.write(audio_file.getvalue())
        
        st.audio(audio_path)

        if "vectorstore" not in st.session_state or not st.session_state["vectorstore"]:

            model = whisper.load_model(TRANSCR)
            result = model.transcribe(audio_path)
            
            text = ''
            for segment in result["segments"]:
                text += f"({timedelta(seconds=segment["start"])}): " # type: ignore
                text += segment["text"] + "\n" # type: ignore

            txt_path = "tmp/transcript.txt"
            with open(txt_path, 'w') as f:
                f.write(text)
                
            loader = TextLoader(
                file_path=txt_path
            )

            doc = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            data = text_splitter.split_documents(doc)

            embeddings = OllamaEmbeddings(model=ENCODER)

            st.session_state["vectorstore"] = Chroma.from_documents(data, embeddings, persist_directory="tmp/chroma_db")

            system_prompt = """
                    Você é uma IA especialista em processos jurídicos.
                    Seu papel é interpretar e discutir de forma clara e concisa transcrições de áudios de tribunais.
                    Você deve responder as perguntas apenas com as informações da transcrição apresentada e interação anterior com o usuário.
                    Se uma resposta é desconhecida apenas avise e não faça especulações.
                    Indique as marcações de tempo de trechos que você mencionar nas suas respostas.

                    Previous conversations:
                    {history}

                    Document context:
                    {context}
                """

            st.session_state["prompt"] = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

        user_input = st.chat_input("Pergunte alguma coisa")

        if "messages" not in st.session_state or user_input == "/clear":
            st.session_state["messages"] = []
            st.session_state["history"] = []

        if user_input and user_input != "/clear":
            st.session_state["messages"].append((user_input, "user"))
            st.session_state["history"].append({"role": "user", "content": HumanMessage(user_input)})

            retriever = st.session_state["vectorstore"].as_retriever()
            
            relevant_segments = retriever.invoke(user_input)
            
            context_doc_str = "\n\n".join(doc.page_content for doc in relevant_segments)
            
            qa_prompt_local = st.session_state["prompt"].partial(
                history=st.session_state["history"],
                context=context_doc_str
            )
            
            llm_chain = { "input": RunnablePassthrough() } | qa_prompt_local  | st.session_state["model"]
            
            result = llm_chain.invoke(user_input)
            
            st.session_state["messages"].append((result, "assistant"))
            st.session_state["history"].append({"role": "assistant", "content": AIMessage(result)})

        for message, author in st.session_state["messages"]:
            with st.chat_message(name=author):
                st.write(message)
        
    else:
        st.session_state["vectorstore"] = None
        
            
if __name__ == "__main__":
    app()