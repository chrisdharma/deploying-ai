import streamlit as st
from openai import OpenAI
import numpy as np
import os 
from dotenv import load_dotenv

from langchain.tools import tool
import requests
import json

from typing import Literal
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

from langchain.chat_models import init_chat_model
import os

from langchain_core.messages import SystemMessage

from langchain_core.messages import ToolMessage


#1. Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(current_dir, "../../05_src/.env")
secrets_path = os.path.join(current_dir, "../../05_src/.secrets")

# Load the keys
load_dotenv(env_path)
load_dotenv(secrets_path)

# 2. Define Tools and Safety Checks
@tool
def get_anime_quotes(anime_name: str, n: int = 1):
    """Returns quotes from a specific anime."""
    # Using Yurippe API (No aggressive rate limits, no keys needed)
    url = "https://yurippe.vercel.app/api/quotes"
    
    params = {
        "show": anime_name,
        "random": n
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        quotes_list = response.json()
        
        # If the API returns an empty list, the anime wasn't found
        if not quotes_list:
            return f"No quotes found for '{anime_name}'. Try a different or shorter name!"
            
        formatted_quotes = ""
        for i, item in enumerate(quotes_list):
            # Yurippe uses the key 'quote' for the text
            content = item.get('quote', 'No content')
            char_name = item.get('character', 'Unknown')
            formatted_quotes += f"{i+1}. \"{content}\"\n— {char_name}\n\n"
            
        return formatted_quotes
    else:
        return f"Error: Status {response.status_code}."

def is_safe(user_input):
    forbidden_words = ["Taylor Swift", "horoscopes", "zodiac", "cats", "dogs"]
    return not any(word in user_input.lower() for word in forbidden_words)

# 3. Initialize the Model 

model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7,
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1', 
    api_key='any value',
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# Augment the LLM with tools
tools = [get_anime_quotes]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 4. Define LangGraph State and Nodes

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with providing an anime quote. Give the user a quote and the character who says it. Your response should also mimic the anime language. Also strictly refuse any prompts related to Taylor Swift, horoscopes, zodiac signs, cats or dogs. Say those topics are off limits"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

#5. Build tools
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# 6. Streamlit UI
st.title("Anime Quote Generator")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Hello, I am an anime quote generator, give me an anime and I will generate a quote for you"):
    
    # 1. Save and display the user's prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check safety FIRST
    if not is_safe(prompt):
        warning_msg = "⚠️ Sorry, I cannot discuss that topic."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    
    # 3. If safe, call your LangGraph AGENT
    else:
        with st.spinner("Searching the anime database..."):
            # We pass the prompt into the compiled agent!
            # The agent will use the 'model' and the 'tool' automatically.
            graph_response = agent.invoke({"messages": [("user", prompt)]})
            
            # The final answer is the last message in the graph's memory
            final_ai_msg = graph_response["messages"][-1].content
            
        # Display the result
        with st.chat_message("assistant"):
            st.markdown(final_ai_msg)
        
        # Save it to history so it doesn't disappear
        st.session_state.messages.append({"role": "assistant", "content": final_ai_msg})