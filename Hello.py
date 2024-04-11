import streamlit as st
from streamlit_option_menu import option_menu
from app_utils import switch_page
from PIL import Image

st.set_page_config(page_title = "Interview Chat Bot", layout = "centered",page_icon="🤖")

home_title = "Interview Chat Bot"
home_introduction = "Welcome to Interview Chat Bot, empowering your interview preparation with generative AI."
st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
st.markdown("""\n""")
#st.markdown("#### Greetings")
st.markdown("Welcome to Interview Chat Bot! 👏 Interview Chat Bot is your personal interviewer powered by generative AI that conducts mock interviews."
            "You can upload your resume and enter job descriptions, and AI Interviewer will ask you customized questions. ")
st.markdown("""\n""")
st.markdown("#### Get started!")
st.markdown("Select one of the following screens to start your interview!")
selected = option_menu(
        menu_title= None,
        options=["Professional", "Resume"],
        icons = ["cast", "cloud-upload", "cast"],
        default_index=0,
        orientation="horizontal",
    )
if selected == 'Professional':
    st.info("""
        📚In this session, the AI Interviewer will assess your technical skills as they relate to the job description.""")
    if st.button("Start Interview!"):
        switch_page("Professional Page")
if selected == 'Resume':
    st.info("""
    📚In this session, the AI Interviewer will review your resume and discuss your past experiences."""
    )
    if st.button("Start Interview!"):
        switch_page("Resume Page")
st.markdown("""\n""")