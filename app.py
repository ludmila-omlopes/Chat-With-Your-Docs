import streamlit as st
import logging
import sys
import os
from llama_index import GPTSimpleVectorIndex, GPTListIndex, GPTListIndex
from llama_index.composability import ComposableGraph
from langchain.llms import OpenAIChat
from langchain.agents import initialize_agent
from llama_index import GPTListIndex
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from langchain.agents import Tool

from streamlit_chat import message

#os.environ['OPENAI_API_KEY'] = ""

# Log
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def build_Messari_agent():
    # Construindo e consultando os index e graphs
    conv_index = GPTListIndex([])
    messari_index = GPTSimpleVectorIndex.load_from_disk('indexMessari.json')
    messari_graph = ComposableGraph.load_from_disk("crypto_index_graph.json")

    # Construindo as "tools" do Langchain a serem usadas pelo agent.
    # No momento só tem tool do report da Messari
    tools = [
        Tool(
            name = "Messari Index",
            func=lambda q: str(messari_graph.query(q)),
            description="useful for when you want to answer questions about crypto predictions for 2023. Always mention that this is Messari's point of view. The input to this tool should be a complete english sentence.",
            return_direct=True
        ),
    ]

    # Objeto de memória, que tb usa o objeto de index da LlamaIndex para guardar a memória do chat
    memory = GPTIndexChatMemory(
        index=conv_index,
        memory_key="chat_history", 
        query_kwargs={"response_mode": "compact"},
        # return_source returns source nodes instead of querying index
        return_source=True,
        # return_messages returns context in message format
        return_messages=True
    )

    #Definindo que o LLM usado é o OpenAIChat (preciso ver outros disponíveis)
    llm = OpenAIChat(model_name="gpt-3.5-turbo")

    #Construindo o agente do Langchain, passando o objeto de memoria, as ferramentas e o llm.
    #Verbose=true mostra o passo a passo do raciocinio da AI
    v_agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory, verbose=True)
    return v_agent_chain

# Inicializando os agentes de chat, e os inputs/outputs do chat
if 'messariAgent' not in st.session_state:
    st.session_state.messariAgent = build_Messari_agent()

if 'replies' not in st.session_state:
    st.session_state.replies = []

if 'prompts' not in st.session_state:
    st.session_state.prompts = []

st.title('Messari Report Chat\n')

with st.sidebar.form("chat_form"):
   prompt = st.text_area('Ask something about Messari´s report.')
   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state.prompts.append(prompt)
    reply = st.session_state.messariAgent.run(input=prompt)
    st.session_state.replies.append(reply)

if 'replies' in st.session_state:
    
    for i in range(len(st.session_state.replies)-1, -1, -1):
        message(st.session_state.replies[i], key=str(i))
        message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user')