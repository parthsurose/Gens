#STREAMLIT + LANGCHAIN + LLMs +OLLAMA (LLMs-gemma2:2b model)
#import required libraries 

import os
import streamlit as st

#imports python built in os modules

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#STEP-1 --> create prompt template
#this define how AI should behave and how it recieves user input
prompt = ChatPromptTemplate.from_messages(
    [
        #system message defines AI behaviour
        ("system","You are a input assistant. Please respond clearly to the question asked."),
        #user message contains placeholder {question}
        ("user","question : {q}")
    ]
)

#STEP-2 --> streamlit app UI

#app title
st.title("LangOllama")

#text input box for user question
in_text = st.text_input("What do you want to Ask?")


#STEP-3 - load ollama model 

#load local gemma model(gemma2:2b)
llm = Ollama(model="gemma2:2b")

#condition - convert output model to string
out_parser = StrOutputParser()

#create langchain pipeline(prompt --> model --> output parser)
chain = prompt | llm | out_parser

#STEP-4 --> run model when user inputs question
if in_text :
    response = chain.invoke({"q" : in_text})
    st.write(response)


