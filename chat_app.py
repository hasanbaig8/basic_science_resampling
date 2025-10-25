"""
Simple Streamlit chat interface for vLLM server.

Usage:
    streamlit run chat_app.py
"""

import streamlit as st
import requests
from typing import List, Dict

# Configuration
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8b"

st.set_page_config(page_title="Chat with Qwen3-8b", page_icon="💬")
st.title("💬 Chat with Qwen3-8b")

# Initialize chat history and system prompt
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# Display chat history (skip system message in display)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Prepare messages with system prompt if provided
            messages_to_send = []
            if st.session_state.system_prompt:
                messages_to_send.append({
                    "role": "system",
                    "content": st.session_state.system_prompt
                })
            messages_to_send.extend(st.session_state.messages)

            # Prepare request to vLLM
            response = requests.post(
                VLLM_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": messages_to_send,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                timeout=60
            )
            response.raise_for_status()

            # Extract assistant response
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]

            # Display response
            message_placeholder.markdown(assistant_message)

            # Add to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message}
            )

        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to vLLM server: {str(e)}"
            message_placeholder.error(error_msg)
            st.error("Make sure vLLM server is running on port 8000")

# Sidebar with controls
with st.sidebar:
    st.header("Settings")

    # System prompt input
    st.markdown("### System Prompt")
    new_system_prompt = st.text_area(
        "System prompt (optional)",
        value=st.session_state.system_prompt,
        height=150,
        placeholder="Enter a system prompt to guide the model's behavior...",
        help="The system prompt is prepended to every conversation"
    )

    # Update system prompt if changed
    if new_system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_system_prompt
        st.info("System prompt updated! It will apply to new messages.")

    st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Server Info")
    st.text(f"Model: {MODEL_NAME}")
    st.text(f"URL: {VLLM_URL}")

    st.markdown("---")
    st.markdown("### How to start vLLM server:")
    st.code("python vllm_chat_server.py", language="bash")

    st.markdown("---")
    st.markdown(f"**Messages in history:** {len(st.session_state.messages)}")
    if st.session_state.system_prompt:
        st.markdown("✅ **System prompt:** Active")
    else:
        st.markdown("⚪ **System prompt:** None")
