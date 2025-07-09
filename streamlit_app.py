# herbal_dashboard_streamlit.py

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
import re
import streamlit.components.v1 as components
from difflib import get_close_matches

# =====================
# GEMINI SETUP
# =====================
import google.generativeai as genai

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None

# ------------------------------
# Load Data
# ------------------------------
data = pd.read_csv("Top_6_Indian_Herbal_Companies_Comparison.csv")
if not os.path.exists("Top_6_Indian_Herbal_Companies_Comparison.csv"):
    st.error("Data file missing. Please upload 'Top_6_Indian_Herbal_Companies_Comparison.csv'.")
    st.stop()

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

# Utility function to parse revenue (handles INR, crore, cr, USD, etc.)
def parse_revenue(val):
    if pd.isna(val): return 0
    val = str(val).replace(",", "").replace(" ", "")
    # Handle INR crore
    if "₹" in val and ("cr" in val or "Cr" in val):
        num = re.search(r"[\d.]+", val)
        return float(num.group()) if num else 0
    # Handle USD (rough conversion)
    if "$" in val:
        num = re.search(r"[\d.]+", val)
        usd = float(num.group()) if num else 0
        return usd * 83 / 10**7  # Convert to crore INR (approx)
    # Handle just numbers
    num = re.search(r"[\d.]+", val)
    return float(num.group()) if num else 0

def parse_products(val):
    if pd.isna(val): return 0
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 0

def parse_growth(val):
    if pd.isna(val): return 0
    m = re.search(r"([+-]?\d+\.?\d*)%", str(val))
    return float(m.group(1)) if m else 0

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("🌿 Herbal Dashboard Assistant")

# --- Voice Search Section ---
st.sidebar.markdown("#### 🎙️ Voice or Text Search")
st.sidebar.info("Tip: Voice search works best in Google Chrome with microphone permissions enabled.")

voice_text = st.sidebar.text_input(
    "Say a company name or type it below:",
    value=st.session_state.get("company", ""),
    key="voice_input"
)

voice_js = '''
<script>
function startDictation() {
    if (window.hasOwnProperty('webkitSpeechRecognition')) {
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";
        recognition.start();
        recognition.onresult = function(e) {
            var result = e.results[0][0].transcript;
            window.parent.postMessage({isStreamlitMessage: true, voiceResult: result}, "*");
            recognition.stop();
        };
        recognition.onerror = function(e) {
            recognition.stop();
        }
    }
}
window.addEventListener("message", (event) => {
    if (event.data && event.data.isStreamlitMessage && event.data.voiceResult) {
        const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
        if (input) {
            input.value = event.data.voiceResult;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
});
</script>
<button onclick="startDictation()" style="margin-top: 5px; margin-bottom: 10px;">🎙 Speak</button>
'''
components.html(voice_js, height=50)

company_names = data["Company Name"].unique().tolist()
# Fuzzy match the voice/company text to the closest company name
if voice_text:
    match = get_close_matches(voice_text, company_names, n=1, cutoff=0.5)
    if match:
        st.session_state.company = match[0]
    else:
        st.session_state.company = company_names[0]
else:
    st.session_state.company = company_names[0]

selected_company = st.sidebar.selectbox(
    "Or select a company manually:",
    company_names,
    index=company_names.index(st.session_state.company),
    help="Pick a company to view its dashboard"
)
st.session_state.company = selected_company

st.sidebar.markdown("---")

# --- Gemini Chat Section ---
st.sidebar.markdown("#### 🤖 Ask Pharmabot")
user_input = st.sidebar.text_input(
    "Ask about the Indian herbal industry:",
    key="gemini_input",
    placeholder="e.g. What is the market share of Himalaya?"
)

if "last_gemini_input" not in st.session_state:
    st.session_state.last_gemini_input = ""
if "gemini_response_text" not in st.session_state:
    st.session_state.gemini_response_text = ""

if user_input and user_input != st.session_state.last_gemini_input and gemini_model:
    with st.sidebar:
        with st.spinner("Thinking..."):
            try:
                gemini_response = gemini_model.generate_content(f"""
                You are an expert assistant on Indian herbal supplement industry.
                Based on the following user query, give clear and concise information: "{user_input}"
                """)
                st.session_state.gemini_response_text = gemini_response.text
            except Exception as e:
                st.session_state.gemini_response_text = "Gemini failed. Please check your API key or connection."
    st.session_state.last_gemini_input = user_input

if user_input:
    st.sidebar.markdown("##### Response:")
    st.sidebar.write(st.session_state.gemini_response_text)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Indian Herbal Dashboard")

# ------------------------------
# Landing Section
# ------------------------------
st.title("🌿 Indian Herbal Supplement Industry Dashboard")
st.markdown(f"### Company Overview: *{st.session_state.company}*")

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
    st.markdown(f"*Website:* [{company_data['Website']}]({company_data['Website']})")
    st.markdown(f"*Tagline/Type:* {company_data['Type']}")

# ------------------------------
# Charts Section
# ------------------------------
st.subheader("📊 Interactive Charts")

col4, col5 = st.columns(2)
with col4:
    categories = company_data["Product Categories"].split(", ")
    fig1 = px.pie(names=categories, values=[1]*len(categories), title="Product Category Contribution", color_discrete_sequence=px.colors.sequential.Greens)
    st.plotly_chart(fig1, use_container_width=True, key="pie_chart")

with col5:
    # Simulated sales trend (bar graph)
    years = [2021, 2022, 2023, 2024]
    sales = [int(100 + i*20 + hash(company_data['Company Name'])%30) for i in range(len(years))]
    fig2 = px.bar(x=years, y=sales, labels={'x': 'Year', 'y': 'Sales (simulated)'}, title="Sales Trend Over Years", color_discrete_sequence=px.colors.sequential.Greens)
    st.plotly_chart(fig2, use_container_width=True, key="bar_chart")

# Bubble chart: Product success rate vs market reach (simulated)
st.markdown("#### Product Success vs Market Reach (Bubble Chart)")
products = [p.strip() for p in company_data['Top 3 Products'].split(",")]
success = [80, 70, 60][:len(products)]
reach = [60, 50, 40][:len(products)]
fig3 = px.scatter(x=products, y=success, size=reach, color=products, labels={'x': 'Product', 'y': 'Success Rate'}, title="Product Success vs Market Reach", color_discrete_sequence=px.colors.sequential.Greens)
st.plotly_chart(fig3, use_container_width=True, key="bubble_chart")

# Radar chart: Comparative analysis (simulated)
st.markdown("#### Comparative Radar Analysis")
radar_params = ['R&D', 'Product Range', 'Revenue', 'Market Presence', 'Innovation']
radar_values = [70, 80, 90, 85, 75]
fig4 = px.line_polar(r=radar_values, theta=radar_params, line_close=True, title="Key Parameters (Simulated)", color_discrete_sequence=px.colors.sequential.Rainbow)
fig4.update_traces(fill='toself')
st.plotly_chart(fig4, use_container_width=True, key="radar_chart_sim")

# ------------------------------
# Product-Level Details
# ------------------------------
st.subheader("🔬 Product-Level Details")
products = [p.strip() for p in company_data['Top 3 Products'].split(",")]
ingredients = company_data['Ingredient Uniqueness'].split(';') if ';' in company_data['Ingredient Uniqueness'] else [company_data['Ingredient Uniqueness']]*len(products)
benefits = company_data['Health Issues Targeted'].split(';') if ';' in company_data['Health Issues Targeted'] else [company_data['Health Issues Targeted']]*len(products)
claims = company_data['Scientific Claims'].split(';') if ';' in company_data['Scientific Claims'] else [company_data['Scientific Claims']]*len(products)

for i, prod in enumerate(products):
    with st.expander(f"{prod}"):
        st.markdown(f"*Ingredients:* {ingredients[i] if i < len(ingredients) else ingredients[0]}")
        st.markdown(f"*Benefits/Issues Addressed:* {benefits[i] if i < len(benefits) else benefits[0]}")
        st.markdown(f"*Scientific Claims:* {claims[i] if i < len(claims) else claims[0]}")
        st.markdown(f"*User Reviews Summary:* Simulated positive reviews for {prod}.")

# ------------------------------
# Maps Section (To be expanded with geopandas/plotly mapbox)
# ------------------------------
# st.subheader("🗺 Geographical Presence")
# st.markdown(company_data['Geographical Presence'])
# # Simulated map (India focus)
# import plotly.graph_objects as go
# fig_map = go.Figure(go.Scattergeo(
#     locationmode = 'country names',
#     locations = ['India'],
#     text = [company_data['Company Name']],
#     marker = dict(size = 30, color = 'green', line_width=0)
# ))
# fig_map.update_geos(fitbounds="locations", visible=False)
# fig_map.update_layout(title="Geographical Presence (Simulated)", geo=dict(bgcolor='rgba(0,0,0,0)'))
# st.plotly_chart(fig_map, use_container_width=True, key="geo_map")

# # Simulated heatmap for market concentration
# st.markdown("#### Market Concentration Heatmap (Simulated)")
# fig_heat = go.Figure(data=go.Heatmap(z=[[1, 0.5], [0.7, 0.2]], x=['North', 'South'], y=['East', 'West'], colorscale='Greens'))
# st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

# ------------------------------
# Sentiment Analysis (Static for Now)
# ------------------------------
st.subheader("🗣 Customer Sentiment")
st.markdown("""
This section summarizes what customers are saying about the company and its products, based on online reviews and product focus. The visuals below are simulated for demonstration purposes.
""")
col6, col7 = st.columns(2)
with col6:
    st.markdown(f"*Average Online Rating:* {company_data['Avg Online Rating']}")
    # st.markdown(f"*Overall Sentiment:* {company_data['Sentiment Analysis']}")
    st.markdown("<span style='font-size: 0.95em; color: #555;'>Word Cloud of Most Commonly Mentioned Health Issues and Products</span>", unsafe_allow_html=True)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wc_text = company_data['Health Issues Targeted'] + ' ' + company_data['Top 3 Products']
    wordcloud = WordCloud(width=300, height=200, background_color='white', colormap='Greens').generate(wc_text)
    plt.figure(figsize=(4,2))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    st.caption("Larger words indicate more frequent mentions in product descriptions and reviews.")

# ------------------------------
# Sentiment Summary Table
# ------------------------------
st.subheader("📊 Sentiment Summary Table")
sentiment_df = data[["Company Name", "Avg Rating", "Sentiment Highlights"]].copy()
sentiment_df = sentiment_df.rename(columns={"Company Name": "Company"})
st.dataframe(sentiment_df, hide_index=True)

# ------------------------------
# OmniActive Comparison
# ------------------------------
st.subheader("🆚 Comparison with OmniActive")
st.markdown(company_data['Compared to OmniActive'])

# ------------------------------
# OmniActive Comparison (Detailed & Visual)
# ------------------------------
st.subheader("🆚 Detailed Comparison with OmniActive")

# Prepare comparison data
comparison_cols = [
    "Company Name", "Type", "Annual Revenue", "Growth Trend", "Product Categories", "No. of Products", "Avg Online Rating", "Patents Filed/Granted", "Regulatory Approvals", "Compared to OmniActive"
]
comparison_df = data[comparison_cols].copy()

# Add OmniActive as a reference row (simulated values)
omniactive_row = {
    "Company Name": "OmniActive Health Technologies",
    "Type": "B2B (nutraceutical ingredients)",
    "Annual Revenue": "~₹600 Cr (2024 est.)",
    "Growth Trend": "+10% YoY (simulated)",
    "Product Categories": "Lutein, Zeaxanthin, Curcumin, Plant Extracts",
    "No. of Products": "20+ ingredients",
    "Avg Online Rating": "N/A (B2B)",
    "Patents Filed/Granted": "30+ global patents",
    "Regulatory Approvals": "US FDA, FSSAI, EU Novel Food, GRAS",
    "Compared to OmniActive": "-"
}
comparison_df = pd.concat([comparison_df, pd.DataFrame([omniactive_row])], ignore_index=True)

# Show table
st.dataframe(comparison_df.set_index("Company Name"))

# Visual: Revenue Comparison (bar chart)
revenues = comparison_df["Annual Revenue"].apply(parse_revenue)
fig_rev = px.bar(
    x=comparison_df["Company Name"],
    y=revenues,
    labels={"x": "Company", "y": "Revenue (Cr INR, approx)"},
    title="Annual Revenue Comparison",
    color=comparison_df["Company Name"],
    color_discrete_sequence=px.colors.sequential.Greens
)
fig_rev.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_rev, use_container_width=True, key="rev_bar")

# Visual: Product Count
product_counts = comparison_df["No. of Products"].apply(parse_products)
fig_prod = px.bar(
    x=comparison_df["Company Name"],
    y=product_counts,
    labels={"x": "Company", "y": "# Products (approx)"},
    title="Number of Products/Ingredients",
    color=comparison_df["Company Name"],
    color_discrete_sequence=px.colors.sequential.Greens
)
fig_prod.update_layout(xaxis_tickangle=-30)
# st.plotly_chart(fig_prod, use_container_width=True, key="prod_bar")

# Visual: Growth Trend (numeric, not categorical)
growth_vals = comparison_df["Growth Trend"].apply(parse_growth)
fig_growth = px.bar(
    x=comparison_df["Company Name"],
    y=growth_vals,
    labels={"x": "Company", "y": "Growth % (YoY)"},
    title="Growth Trend (YoY %)",
    color=comparison_df["Company Name"],
    color_discrete_sequence=px.colors.sequential.Greens
)
fig_growth.update_layout(xaxis_tickangle=-30)
# st.plotly_chart(fig_growth, use_container_width=True, key="growth_bar")

# Visual: Patents Filed/Granted (simulated)
def parse_patents(val):
    if pd.isna(val): return 0
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 0
patent_counts = comparison_df["Patents Filed/Granted"].apply(parse_patents)
fig_pat = px.bar(
    x=comparison_df["Company Name"],
    y=patent_counts,
    labels={"x": "Company", "y": "# Patents (approx)"},
    title="Patents Filed/Granted",
    color=comparison_df["Company Name"],
    color_discrete_sequence=px.colors.sequential.Greens
)
fig_pat.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_pat, use_container_width=True, key="pat_bar")

# Visual: Regulatory Approvals (count, simulated)
reg_counts = comparison_df["Regulatory Approvals"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)
fig_reg = px.bar(
    x=comparison_df["Company Name"],
    y=reg_counts,
    labels={"x": "Company", "y": "# Regulatory Approvals (simulated)"},
    title="Regulatory Approvals (Count, Simulated)",
    color=comparison_df["Company Name"],
    color_discrete_sequence=px.colors.sequential.Greens
)
fig_reg.update_layout(xaxis_tickangle=-30)
# st.plotly_chart(fig_reg, use_container_width=True, key="reg_bar")

# Improved Radar Chart: Key Parameters (normalized)
radar_params = ["Revenue", "Product Count", "Growth", "Patents", "Reg Approvals"]
radar_data = pd.DataFrame({
    "Company": comparison_df["Company Name"],
    "Revenue": revenues / (revenues.max() or 1),
    "Product Count": product_counts / (product_counts.max() or 1),
    "Growth": (growth_vals - growth_vals.min()) / ((growth_vals.max() - growth_vals.min()) or 1),
    "Patents": patent_counts / (patent_counts.max() or 1),
    "Reg Approvals": reg_counts / (reg_counts.max() or 1)
})
radar_selected = radar_data[radar_data["Company"] == st.session_state.company]
radar_omni = radar_data[radar_data["Company"] == "OmniActive Health Technologies"]
import plotly.graph_objects as go
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=radar_selected.iloc[0,1:], theta=radar_params, fill='toself', name=st.session_state.company))
fig_radar.add_trace(go.Scatterpolar(r=radar_omni.iloc[0,1:], theta=radar_params, fill='toself', name="OmniActive"))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title="Comparative Radar Analysis (Normalized)")
st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart_comp")

# Improved Geographical Presence Map
import plotly.express as px
country_map = {
    "India": "IND", "USA": "USA", "Europe": "FRA", "Asia": "CHN", "Oceania": "AUS", "Middle East": "ARE", "Russia": "RUS", "Africa": "ZAF", "SAARC": "BGD"
}
def extract_countries(text):
    found = []
    for k, v in country_map.items():
        if k.lower() in text.lower():
            found.append(v)
    return found or ["IND"]
geo_countries = extract_countries(company_data['Geographical Presence'])
fig_geo = px.choropleth(locations=geo_countries, locationmode="ISO-3", color=[1]*len(geo_countries),
                        color_continuous_scale=px.colors.sequential.Greens, title="Geographical Presence (by Country)")
st.plotly_chart(fig_geo, use_container_width=True, key="geo_choropleth")