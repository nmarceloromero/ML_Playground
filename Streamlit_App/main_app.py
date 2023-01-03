# Imports.
import streamlit as st
import model_fr as mf 
import model_sp as ms 

# Config page.
st.set_page_config(layout="wide")

# Title
st.title('My QA App')

# Formatting.
col1, col2, col3 = st.columns(spec=3, gap="large")

# Input. 
with col1 : 
    context  = st.text_area(label="Input your source of information:")
    question = st.text_input(label="Input your question:")

 # Button.
with col2 : 
    for i in range(4):
        st.write("")
    option = st.selectbox(  label='Which language do you use?', 
                            options=('Spanish', 'French'))
    button = st.button(label="Get answer!")

# Functionality.
if button: 
    with col3 :
        with st.spinner('Getting the answer...'):
            if option == "French":
                model, tokenizer = mf.get_model_and_tokenizer() # Load model.
                answer = mf.inference(context, question, model, tokenizer).capitalize() # Infer.
            else:
                model, tokenizer = ms.get_model_and_tokenizer() # Load model.
                answer = ms.inference(context, question, model, tokenizer).capitalize() # Infer.
        st.markdown("### Answer")
        st.markdown(answer)
        st.success('Done!')