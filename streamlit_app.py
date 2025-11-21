import streamlit as st
import os
import whisper
import json
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

def app():
    st.write("# RAG")

    audio_file = st.file_uploader("Escolha o arquivo e espere a análise", type="mp3")
    if audio_file:
        audio_path = "tmp/audio.mp3"

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        with open(audio_path, 'wb') as audio:
            audio.write(audio_file.getvalue())
        
        st.audio(audio_path)

        if "vectorstore" not in st.session_state or not st.session_state["vectorstore"]:

            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)

            json_path = "tmp/audio.json"
            with open(json_path, 'w') as f:
                f.write(json.dumps(result))
                
            def metadata_func(record, metadata):
                metadata["timestamp"] = record.get("start")
                return metadata
            
            loader = JSONLoader(
                file_path=json_path,
                jq_schema='.segments[]',
                content_key='text',
                text_content=False,
                metadata_func=metadata_func
            )
            
            data = loader.load()
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            st.session_state["vectorstore"] = Chroma.from_documents(data, embeddings, persist_directory="chroma_db")
                
            st.session_state["llm"] = OllamaLLM(model="gemma3")

            system_prompt = """
                    Você é uma IA especialista em processos jurídicos.
                    Seu papel é interpretar e discutir de forma clara e concisa transcrições de áudio de tribunais.
                    Você deve responder as perguntas apenas com as informações da transcrição apresentada e interação anterior com o usuário.
                    Você deve agir como expert sem mencionar qual é a sua função.
                    You must always respond as a legal expert and avoid disclaiming your expertise.
                    Se uma resposta é desconhecida apenas avise e não faça especulações.
                    Indique as timestamps de trechos que você mencionar nas suas respostas.

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
            
            llm_chain = { "input": RunnablePassthrough() } | qa_prompt_local  | st.session_state["llm"]
            
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