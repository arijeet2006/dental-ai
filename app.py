import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessageChunk

# Import the compiled graph from your agent file
from dental_agent.agent import dental_graph 

load_dotenv()

st.set_page_config(
    page_title="Dental Care Assistant",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 Dental Appointment Management System")
st.caption("Powered by LangGraph & Google Gemini")
st.markdown("---")

with st.sidebar:
    st.header("📋 Clinic Dashboard")
    
    csv_path = "doctor_availability.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            st.subheader("Filter Availability")
            status = st.radio("Show slots:", ["All", "Available Only", "Booked Only"])
            
            if status == "Available Only":
                filtered_df = df[df['is_available'] == True]
            elif status == "Booked Only":
                filtered_df = df[df['is_available'] == False]
            else:
                filtered_df = df
                
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    else:
        st.warning(f"Data file '{csv_path}' not found yet.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    
    initial_msg = "Hello! I am your AI Dental Assistant. How can I help you manage your appointments today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_msg})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask to book, reschedule, cancel or view slots..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            try:
                final_messages = None
                for event_type, data in dental_graph.stream(
                    {"messages": st.session_state.chat_history},
                    stream_mode=["messages", "values"],
                    config={"recursion_limit": 20},
                ):
                    if event_type == "messages":
                        chunk, meta = data
                        if (
                            isinstance(chunk, AIMessageChunk)
                            and chunk.content
                            and not getattr(chunk, "tool_calls", None)
                        ):
                            if isinstance(chunk.content, str):
                                full_response += chunk.content
                            elif isinstance(chunk.content, list):
                                for item in chunk.content:
                                    if isinstance(item, dict) and "text" in item:
                                        full_response += item["text"]
                                    elif isinstance(item, str):
                                        full_response += item
                            # -----------------------
                            
                            response_placeholder.markdown(full_response + "▌")
                            
                    elif event_type == "values":
                        final_messages = data.get("messages", [])

                # Remove the blinking cursor at the end
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                if final_messages:
                    st.session_state.chat_history = final_messages
                    
                st.rerun()

            except Exception as e:
                response_placeholder.error(f"An error occurred: {e}")
                if st.session_state.chat_history:
                    st.session_state.chat_history.pop()