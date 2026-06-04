import streamlit as st
st.write("Streamlit OK")
try:
    import plotly
    st.success(f"Plotly Installed: {plotly.__version__}")
except Exception as e:
    st.error(f"Plotly Installed: {e}")


import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title = "Customer Churn Dashboard",
    layout = "wide",
    initial_sidebar_state = "auto",
)

st.markdown("""

<h1 style='text-align: center;
font-size:42px;
margin-bottom:5px;
background: linear-gradient(
90deg,
#00F5FF,
#00D4FF,
#7B61FF
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
-moz-background-color:transparent;
background-clip:text;
color:transparent;
'>
Customer Churn Prediction Dashboard
</h1>

<p style='text-align: center;
font-size:18px;
color:#AAB8C2;'>
This dashboard predicts customer churn using a Random Forest Machine Learing Model
And identifies customers at high risk of leaving.
""",unsafe_allow_html=True)
st.markdown("---")

# Risk Data load
risk_data = pd.read_csv(
    "data/customer_risk.csv"
)

# Feature data load
feature_data = pd.read_csv(
    "data/feature_importance.csv"
)

# performance data load confusion matrix
performance_data = pd.read_csv(
    "data/performance.csv"
)

# Sidebar
st.sidebar.markdown("""# Filters""")

st.markdown("""
<style>
/* selected tags */
span[data-baseweb="tag"]{
    background-color:#7B61FF !important;
    color:white !important;
    border-radius: 8px !important;
    }

/* high risk */
span[data-baseweb="risk"]{
    font-weight:bold;
    }

</style>
""", unsafe_allow_html=True)

selected_risk = st.sidebar.multiselect(
    "Select Risk Level",
options=risk_data["Risk_Level"].unique(),
default=risk_data["Risk_Level"].unique()
)
if not selected_risk:
    st.info(
        "  No Risk Level selected. Please select at least one filter option from the sidebar to view customer insights"
    )
    st.stop()
filtered_risk_data = risk_data[
    risk_data["Risk_Level"].isin(selected_risk)
]

# data loading
data = pd.read_csv("data/customer_churn.csv")
with st.expander("Dataset Preview"):
    st.dataframe(
        data,
        use_container_width = True,
        height=400
    )
st.subheader("Dataset Information")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Rows</div>
        <div class="metric-value">{data.shape[0]}</div>
    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Columns</div>
        <div class="metric-value">{data.shape[1]}</div>
    </div>
    """, unsafe_allow_html=True)

# churn matrix
total_customers = len(data)
churn_customers = (
    data['Churn?'] == 'True.'
).sum()

retained_customers = (
    data['Churn?'] == 'False.'
).sum()

churn_rate = (
    churn_customers / total_customers
) * 100

st.subheader("Customer Statistics")
col1, col2, col3, col4 = st.columns(4, gap="medium")
cards = [
    ("🧑‍🤝‍🧑🏻‍Total Customers", total_customers),
    ("⚠️Churn Customers", churn_customers),
    ("✅Retained Customers", retained_customers),
    ("📉Churn Rate", f"{churn_rate:.2f}%")
]
for col, (title, value) in zip([col1,col2,col3,col4], cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# Accuracy Card
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4, gap="medium")
metrics = [
    ("🎯Accuracy",
     f"{performance_data.iloc[0]['Value']*100:.2f}%"),
    ("📊Precision",
     f"{performance_data.iloc[1]['Value']*100:.2f}%"),
    ("📈Recall",
     f"{performance_data.iloc[2]['Value']*100:.2f}%"),
    ("🏆F1-Score",
     f"{performance_data.iloc[3]['Value']*100:.2f}%")
]
for col, (title, value) in zip([col1,col2,col3,col4], metrics):
    with col:
        st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.block-container{
padding-top: 2rem;
}

.metric-card{
    background: linear-gradient(135deg, #1C2333, #273449);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin: 8px 4px;
}

.metric-card:hover{
    transform: translateY(-6px);
    box-shadow: 0px 10px 25px rgba(0, 212, 255, 0.35);
    border: 1px solid #00D4FF;
}

.metric-title{
    font-size: clamp(12px, 1vw, 16px);
    color: #A0AEC0;
    margin-bottom: 10px;
    word-wrap: break-word;
}

.metric-value{
    font-size: clamp(20px, 2vw, 32px);
    font-weight: bold;
    color: white;
}
/* sidebar background */
[data-testid="stSidebar"]{
background: linear-gradient(
180deg,
#111827,
#1E293B
);
}

/* sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
color: #00D4FF;
}

@media(max-width: 1024px){
.metric-card{
    margin-bottom: 12px;
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("---")


col1, col2 = st.columns(2)
with col1:
    # Churn Distribution data visuals
    st.subheader("Churn Distribution")
    churn_counts = data['Churn?'].value_counts()
    fig = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        hole=0.45,
        title="Customer Churn Distribution"
    )
    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        )
    )
    st.plotly_chart(
        fig,
        use_container_width = True
    )
with col2:
    # Risk Distribution
    st.markdown("## Customer Risk Distribution")
    risk_counts = filtered_risk_data["Risk_Level"].value_counts()
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        labels={
            "x":"Risk Level",
            "y":"Customers"
        },
        title="Risk Distribution"
    )
    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        )
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    # Confusion Matrix visuals
    st.subheader("Confusion Matrix")
    st.image(
        "data/confusion_matrix.png",
        use_container_width = True
    )
with col2:
    # feature importance visuals
    st.subheader("Feature Importance")
    fig = px.bar(
        feature_data.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        )
    )
    st.plotly_chart(
        fig,
        use_container_width = True
    )

st.markdown("---")

# Risk Level Summery
high_risk = (
    filtered_risk_data["Risk_Level"] == "High Risk"
).sum()

medium_risk = (
    filtered_risk_data["Risk_Level"] == "Medium Risk"
).sum()
low_risk = (
    filtered_risk_data["Risk_Level"] == "Low Risk"
).sum()
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("## 🚨 Risk Summery")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        st.metric(" 🔴 High", high_risk)
    with risk_col2:
        st.metric(" 🟡 Medium", medium_risk)
    with risk_col3:
        st.metric(" 🟢 Low", low_risk)
with col2:
    # Search Customer by phone number Feature
    st.markdown("## 🔍 Search Customer")
    phone_search = st.text_input(
        "Enter Phone Number",
    )
    if phone_search:
        filtered = risk_data[
            risk_data["Phone"]
            .astype(str)
            .str.contains(phone_search)
        ]
        if filtered.empty:
            st.warning("No customer found.")
        else:
            st.dataframe(
                filtered,
                use_container_width = True,
                height=min(
                    max(len(filtered) * 35, 150),350
                )
            )

# High Risk Customer Data Table
st.markdown("## ⚠️ Top Risky Customers")
st.dataframe(
    filtered_risk_data.head(20),
    use_container_width=True,
    height=500,
)

# Download button
st.markdown(" 📩 Download Report")
csv = filtered_risk_data.to_csv(index=False)

st.download_button(
    label = "Download Customer Risk Report",
    data=csv,
    file_name="customer_risk_report.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("""
    <div style="text-align: center;
    background: #1C2333;
    padding:20px;
    border-radius:12px;
    text-align: center;
    color:#A0AEC0;
    font-size:clamp(12px, 1vw, 14px);
    margin-top:20px;
    ">
    Build with Python, Streamlit, Scikit-Learn, Pandas & Machine Learning
    <br><br>
    ©️2026 Lavakush Gond | Customer Churn Prediction Dashboard
    </div>
""", unsafe_allow_html=True)