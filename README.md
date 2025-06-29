# herbal_dashboard_streamlit.py

import streamlit as st
import pandas as pd
import plotly.express as px
import speech_recognition as sr
import base64
import os

# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv("Top_6_Indian_Herbal_Companies_Comparison.csv")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Indian Herbal Industry Dashboard", layout="wide")

# ------------------------------
# Utility Functions
# ------------------------------
def load_logo(company):
    try:
        return f"images/{company.lower().replace(' ', '_')}.png"
    except:
        return None

def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak the company name")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except:
        st.error("Could not understand audio")
        return None

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🔍 Voice Activated Search")
if st.sidebar.button("🎙️ Start Voice Search"):
    voice_company = voice_input()
    if voice_company:
        st.session_state.company = voice_company.title()

company_names = data["Company Name"].unique().tolist()
selected_company = st.sidebar.selectbox("Or select a company manually", company_names)
st.session_state.company = selected_company

# ------------------------------
# Landing Section
# ------------------------------
st.title("🌿 Indian Herbal Supplement Industry Dashboard")
st.markdown(f"### Company Overview: **{st.session_state.company}**")

company_data = data[data["Company Name"] == st.session_state.company].iloc[0]

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    logo_path = load_logo(st.session_state.company)
    if logo_path and os.path.exists(logo_path):
        st.image(logo_path, width=120)

with col2:
    st.metric("Revenue", company_data["Annual Revenue"])
    st.metric("Top Product", company_data["Top 3 Products"].split(",")[0])

with col3:
    st.markdown(f"**Website:** [{company_data['Website']}]({company_data['Website']})")
    st.markdown(f"**Tagline/Type:** {company_data['Type']}")

# ------------------------------
# Charts Section
# ------------------------------
st.subheader("📊 Interactive Charts")

col4, col5 = st.columns(2)
with col4:
    categories = company_data["Product Categories"].split(", ")
    fig1 = px.pie(names=categories, values=[1]*len(categories), title="Product Category Contribution")
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    growth = company_data["Growth Trend"]
    # Dummy parsing, replace with actual trend data for realism
    st.markdown(f"**Growth Info:** {growth}")

# ------------------------------
# Comparison Radar (Dummy)
# ------------------------------
st.subheader("📈 Comparative Radar Analysis")
# Needs structured numerical values (R&D, Revenue, etc.) for radar
# You can simulate values or extend dataset later

# ------------------------------
# Product-Level Details
# ------------------------------
st.subheader("🔬 Product-Level Details")
with st.expander("Top Products & Benefits"):
    st.markdown(f"**Top Products:** {company_data['Top 3 Products']}")
    st.markdown(f"**Ingredients:** {company_data['Ingredient Uniqueness']}")
    st.markdown(f"**Benefits/Health Issues:** {company_data['Health Issues Targeted']}")
    st.markdown(f"**Scientific Claims:** {company_data['Scientific Claims']}")

# ------------------------------
# Maps Section (To be expanded with geopandas/plotly mapbox)
# ------------------------------
st.subheader("🗺️ Geographical Presence")
st.markdown(company_data['Geographical Presence'])

# ------------------------------
# Sentiment Analysis (Static for Now)
# ------------------------------
st.subheader("🗣️ Customer Sentiment")
col6, col7 = st.columns(2)
with col6:
    st.markdown(f"**Rating:** {company_data['Avg Online Rating']}")
    st.markdown(f"**Sentiment:** {company_data['Sentiment Analysis']}")

# ------------------------------
# OmniActive Comparison
# ------------------------------
st.subheader("🆚 Comparison with OmniActive")
st.markdown(company_data['Compared to OmniActive'])

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Built with ❤️ by an AI-driven Full Stack Engineer. For research use only.")
