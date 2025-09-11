import streamlit as st
from src.answer_generator import AnswerGenerator


MODELS = ['Gensyn/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'unsloth/gemma-3-1b-it']

st.set_page_config(
    page_title="RAG System",
    page_icon=":rocket:",
    layout="centered",
)
st.title(':rocket: RAG System')

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
if "ask_llm" not in st.session_state:
    st.session_state.ask_llm = False
if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = None

selected_model_name = st.selectbox(
    label='Choice LLM model',
    placeholder='Click to get selections',
    options=MODELS,
    index=None
)

if selected_model_name:
    start_chat = st.button('Initialize system')

    if start_chat or st.session_state.start_chat:
        st.session_state.start_chat = True

        with st.spinner("Initializing..."):
            if not st.session_state.answer_generator:
                st.session_state.answer_generator = AnswerGenerator(model_name=selected_model_name)

        template_questions = (
            "",
            "Tell me about a horror movie where a scientist turns into a taloned beast.",
            "What is the plot in the movie The Royal Tenenbaums from 2001?",
            "What was the name of the comedy movie with Dan Aykroyd and Tom Hanks in cast? It was released in 1987.",
            "A war veteran is able to travel 15 years into the future and stay there for a short time. "
            "Write out everything you know about this movie.",
        )

        text_template = st.selectbox(
            'Template questions',
            template_questions,
            index=None,
            placeholder="Click to select a template question",
        )

        question = st.text_area('**Your question**', text_template)

        ask_llm = st.button('**Ask model**')

        if ask_llm or st.session_state.ask_llm:
            if question and len(question) > 10:
                st.session_state.ask_llm = True
                with st.spinner("Getting model answer..."):
                    answer, founded_movies = st.session_state.answer_generator(question)
                st.markdown("**Answer:**")
                st.info(answer)
                with st.expander("Open to view information about the found movies"):
                    st.markdown(founded_movies)
                st.session_state.ask_llm = False
            else:
                st.warning('The entered question is missing or it is very short')
                ask_llm = False
