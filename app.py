import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_pipeline import load_data, train_regression, train_classification, run_clustering

# Page Config
st.set_page_config(page_title="Gujarat Real Estate ML", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; }
    .sub-header { color: #94a3b8; font-size: 0.95rem; margin-top: -5px; margin-bottom: 20px; }
    .metric-card { background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 15px 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15); backdrop-filter: blur(10px); }
    .metric-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin: 0; }
</style>
""", unsafe_allow_html=True)

# Load Data
df, X, y_reg, y_clf = load_data()
reg_results = train_regression(X, y_reg)
clf_model, clf_acc = train_classification(X, y_clf)
df_clustered = run_clustering(df)

# Top Header
st.markdown('<p class="main-header">🏢 Gujarat Property ML Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Quantitative Market Grouping & Price Forecasting</p>', unsafe_allow_html=True)

# Top Level KPIs (Fills the top empty space)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total Properties</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Market Price</div><div class="metric-value">₹{df["Price_INR"].mean():,.0f}</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Most Expensive City</div><div class="metric-value">{df.groupby("City")["Price_INR"].mean().idxmax()}</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Distance to Transit</div><div class="metric-value">{df["Distance_to_Transit_km"].mean():.1f} km</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Property Parameters")
    city = st.selectbox("City", ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"])
    sqft = st.slider("Square Feet", 500, 5000, 1500, step=100)
    beds = st.slider("Bedrooms", 1, 5, 3)
    age = st.slider("Property Age (Years)", 0, 30, 5)
    transit = st.slider("Distance to Transit (km)", 0.0, 15.0, 2.0, step=0.5)
    gated = st.radio("Gated Society?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    
    input_data = pd.DataFrame({"City": [city], "SqFt": [sqft], "Bedrooms": [beds], "Property_Age_Years": [age], "Distance_to_Transit_km": [transit], "Is_Gated_Society": [gated]})

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Price Estimator", "🎯 Investment Tier"])

with tab1:
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### 🗺️ 2D Cluster Map")
        fig_scatter = px.scatter(df_clustered, x="SqFt", y="Price_INR", color="City", size="Bedrooms", hover_data=["Property_Age_Years"], color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_scatter.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
        fig_scatter.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=30, b=0), legend_title_text='Location', hovermode="x unified")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_t2:
        st.markdown("#### 🏘️ Price Distribution by City")
        fig_box = px.box(df, x="City", y="Price_INR", color="City", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_box.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    col_e1, col_e2 = st.columns([1, 2])
    
    with col_e1:
        st.markdown("#### Forecast Models (MAE)")
        res_table = [{"Model": k, "Error Margin": v["MAE (₹)"]} for k, v in reg_results.items()]
        st.dataframe(pd.DataFrame(res_table), hide_index=True, use_container_width=True)
        
        city_avg = df[df["City"] == city]["Price_INR"].mean()
        st.info(f"**Market Context:** The average property price in {city} is ₹{city_avg:,.0f}.")
        
    with col_e2:
        best_model = reg_results["XGBoost"]["Pipeline"]
        prediction = best_model.predict(input_data)[0]
        
        st.markdown("#### XGBoost Valuation")
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 40px;">
            <h3 style="margin:0; color:#94a3b8; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px;">Estimated Market Value</h3>
            <h1 style="margin:10px 0 0 0; color:#10b981; font-size: 4rem;">₹{prediction:,.0f}</h1>
            <p style="color: #64748b; margin-top: 10px;">Based on {sqft} sqft, {beds} BHK in {city}</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        tier_pred = clf_model.predict(input_data)[0]
        tier_color = "#10b981" if tier_pred == "Affordable" else "#38bdf8" if tier_pred == "Mid-Range" else "#ef4444"
        
        st.markdown("#### Predictive Classification")
        st.markdown(f"""
        <div class="metric-card" style="border-color: {tier_color}40; text-align: center; padding: 30px;">
            <h3 style="margin:0; color:#94a3b8;">Predicted Investment Tier</h3>
            <h2 style="margin:10px 0 0 0; color:{tier_color}; font-size: 3rem;">{tier_pred}</h2>
            <p style="margin-top: 15px; color: #64748b; font-size: 1rem;">Model Accuracy: {clf_acc:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_c2:
        st.markdown("#### Model Confidence (Probabilities)")
        # Extract probabilities for the pie/bar chart
        probs = clf_model.predict_proba(input_data)[0]
        classes = clf_model.classes_
        
        fig_prob = go.Figure(go.Bar(
            x=probs, y=classes, orientation='h',
            marker_color=["#10b981", "#ef4444", "#38bdf8"] 
        ))
        fig_prob.update_layout(
            template="plotly_dark", height=250, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Probability", xaxis=dict(tickformat=".0%")
        )
        st.plotly_chart(fig_prob, use_container_width=True)