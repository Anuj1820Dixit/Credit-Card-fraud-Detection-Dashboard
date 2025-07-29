import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")
st.markdown("**Real-time fraud detection system with interactive analytics and compliance support**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

# Load the data
try:
    df = load_data()
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Sidebar
st.sidebar.header("🔧 Dashboard Controls")

# Data filters
st.sidebar.subheader("📊 Data Filters")

# Amount range filter
min_amount = float(df['Amount'].min())
max_amount = float(df['Amount'].max())
amount_range = st.sidebar.slider(
    "Amount Range ($)",
    min_value=min_amount,
    max_value=max_amount,
    value=(min_amount, max_amount),
    step=1.0
)

# Time range filter
min_time = float(df['Time'].min())
max_time = float(df['Time'].max())
time_range = st.sidebar.slider(
    "Time Range (seconds)",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time),
    step=100.0
)

# Class filter
class_filter = st.sidebar.multiselect(
    "Transaction Class",
    options=['Legitimate', 'Fraud'],
    default=['Legitimate', 'Fraud']
)

# Apply filters
class_mapping = {'Legitimate': 0, 'Fraud': 1}
selected_classes = [class_mapping[c] for c in class_filter]

filtered_df = df[
    (df['Amount'] >= amount_range[0]) & 
    (df['Amount'] <= amount_range[1]) &
    (df['Time'] >= time_range[0]) & 
    (df['Time'] <= time_range[1]) &
    (df['Class'].isin(selected_classes))
]

# Model settings
st.sidebar.subheader("🤖 Model Settings")

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    help="Threshold for fraud detection (higher = more strict)"
)

# Visualization settings
st.sidebar.subheader("📈 Visualization Settings")

chart_theme = st.sidebar.selectbox(
    "Chart Theme",
    options=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
    index=0
)

sample_size = st.sidebar.slider(
    "Sample Size for Large Charts",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
    help="Number of transactions to sample for performance"
)

# Quick actions
st.sidebar.subheader("⚡ Quick Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

if st.sidebar.button("📊 Show Summary"):
    st.sidebar.success(f"Filtered Data: {len(filtered_df):,} transactions")
    st.sidebar.info(f"Fraud Rate: {(len(filtered_df[filtered_df['Class']==1]) / len(filtered_df) * 100):.3f}%")

# Download options
st.sidebar.subheader("💾 Download Options")

if st.sidebar.button("📥 Download Filtered Data"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_fraud_data_{len(filtered_df)}_records.csv",
        mime="text/csv"
    )

# Information
st.sidebar.subheader("ℹ️ Information")
st.sidebar.info(f"""
**Dataset Info:**
- Total: {len(df):,} transactions
- Legitimate: {len(df[df['Class']==0]):,}
- Fraud: {len(df[df['Class']==1]):,}
- Fraud Rate: {(len(df[df['Class']==1]) / len(df) * 100):.3f}%

**Current Filters:**
- Amount: ${amount_range[0]:.2f} - ${amount_range[1]:.2f}
- Time: {time_range[0]:.0f} - {time_range[1]:.0f}s
- Classes: {', '.join(class_filter)}
""")

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(filtered_df):,}")
    
with col2:
    st.metric("Legitimate", f"{len(filtered_df[filtered_df['Class']==0]):,}")
    
with col3:
    st.metric("Fraudulent", f"{len(filtered_df[filtered_df['Class']==1]):,}")
    
with col4:
    fraud_rate = (len(filtered_df[filtered_df['Class']==1]) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 Model", "📈 Analytics", "⚡ Real-time"])

with tab1:
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction class distribution
        fig_pie = px.pie(
            values=filtered_df['Class'].value_counts().values,
            names=['Legitimate', 'Fraud'],
            title='Transaction Class Distribution (Filtered)',
            color_discrete_sequence=['lightblue', 'lightcoral']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Basic statistics
        st.subheader("📈 Key Statistics (Filtered)")
        
        stats_data = {
            'Metric': ['Total Transactions', 'Legitimate', 'Fraud', 'Fraud Rate (%)', 'Mean Amount ($)', 'Max Amount ($)'],
            'Value': [
                len(filtered_df),
                len(filtered_df[filtered_df['Class']==0]),
                len(filtered_df[filtered_df['Class']==1]),
                round((len(filtered_df[filtered_df['Class']==1]) / len(filtered_df)) * 100, 3) if len(filtered_df) > 0 else 0,
                round(filtered_df['Amount'].mean(), 2) if len(filtered_df) > 0 else 0,
                round(filtered_df['Amount'].max(), 2) if len(filtered_df) > 0 else 0
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

with tab2:
    st.header("🔍 Exploratory Data Analysis")
    
    # Amount analysis
    st.subheader("💰 Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution by class
        fig_amount = px.histogram(
            df, 
            x='Amount', 
            color='Class',
            title='Transaction Amount Distribution by Class',
            color_discrete_sequence=['lightblue', 'lightcoral'],
            nbins=50
        )
        fig_amount.update_layout(xaxis_title='Amount ($)', yaxis_title='Count')
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Amount statistics by class
        amount_stats = df.groupby('Class')['Amount'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        st.subheader("Amount Statistics by Class")
        st.dataframe(amount_stats, use_container_width=True)
    
    # Time analysis
    st.subheader("⏰ Time-based Analysis")
    
    # Create time features
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Day'] = (df['Time'] // (3600 * 24)) % 7
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transactions by hour
        fraud_by_hour = df[df['Class'] == 1]['Hour'].value_counts().sort_index()
        legitimate_by_hour = df[df['Class'] == 0]['Hour'].value_counts().sort_index()
        
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(x=fraud_by_hour.index, y=fraud_by_hour.values, 
                                     mode='lines+markers', name='Fraud', line=dict(color='red')))
        fig_hour.add_trace(go.Scatter(x=legitimate_by_hour.index, y=legitimate_by_hour.values, 
                                     mode='lines+markers', name='Legitimate', line=dict(color='blue')))
        fig_hour.update_layout(title='Transactions by Hour of Day', xaxis_title='Hour', yaxis_title='Count')
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        # Time vs Amount scatter
        sample_df = df.sample(n=min(10000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df, 
            x='Time', 
            y='Amount', 
            color='Class',
            title='Time vs Amount (Sample)',
            color_discrete_sequence=['lightblue', 'lightcoral']
        )
        fig_scatter.update_layout(xaxis_title='Time (seconds)', yaxis_title='Amount ($)')
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("🤖 Fraud Detection Model")
    
    # Simple rule-based model
    def simple_fraud_detector(amount, time, v1, v2, v3, v4, v5):
        risk_score = 0
        
        # Rule 1: High amount transactions
        if amount > 1000:
            risk_score += 2
        elif amount > 500:
            risk_score += 1
        
        # Rule 2: Unusual time patterns
        hour = (time // 3600) % 24
        if hour < 6 or hour > 22:
            risk_score += 1
        
        # Rule 3: Extreme PCA values
        if abs(v1) > 3 or abs(v2) > 3 or abs(v3) > 3:
            risk_score += 2
        
        # Rule 4: Multiple extreme features
        extreme_features = sum([abs(v1) > 2, abs(v2) > 2, abs(v3) > 2, abs(v4) > 2, abs(v5) > 2])
        if extreme_features >= 3:
            risk_score += 1
        
        return risk_score
    
    # Model performance
    st.subheader("📊 Model Performance")
    
    # Apply model to sample data
    sample_size = min(1000, len(df))
    sample_data = df.sample(n=sample_size, random_state=42)
    
    predictions = []
    risk_scores = []
    
    for _, row in sample_data.iterrows():
        risk = simple_fraud_detector(
            row['Amount'], row['Time'], 
            row['V1'], row['V2'], row['V3'], row['V4'], row['V5']
        )
        risk_scores.append(risk)
        predictions.append(1 if risk >= risk_threshold else 0)
    
    # Calculate metrics
    actual = sample_data['Class'].values
    accuracy = sum(predictions == actual) / len(predictions)
    fraud_detected = sum(predictions)
    actual_fraud = sum(actual)
    true_positives = sum((np.array(predictions) == 1) & (actual == 1))
    false_positives = sum((np.array(predictions) == 1) & (actual == 0))
    
    precision = true_positives / fraud_detected if fraud_detected > 0 else 0
    recall = true_positives / actual_fraud if actual_fraud > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        st.metric("Risk Threshold", f"≥ {risk_threshold}")
    
    # Risk score distribution
    st.subheader("🎯 Risk Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_risk = px.histogram(
            x=risk_scores,
            title='Risk Score Distribution',
            nbins=20,
            color_discrete_sequence=['skyblue']
        )
        fig_risk.update_layout(xaxis_title='Risk Score', yaxis_title='Count')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk score by actual class
        fraud_risks = [risk_scores[i] for i in range(len(risk_scores)) if actual[i] == 1]
        legitimate_risks = [risk_scores[i] for i in range(len(risk_scores)) if actual[i] == 0]
        
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=legitimate_risks, name='Legitimate', boxpoints='outliers'))
        fig_box.add_trace(go.Box(y=fraud_risks, name='Fraud', boxpoints='outliers'))
        fig_box.update_layout(title='Risk Score by Actual Class', yaxis_title='Risk Score')
        st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.header("📈 Advanced Analytics")
    
    # Feature importance
    st.subheader("🔍 Feature Importance Analysis")
    
    # Calculate correlations with target
    pca_features = [f'V{i}' for i in range(1, 29)]
    correlations = df[pca_features + ['Class']].corr()['Class'].drop('Class')
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top features bar chart
        fig_importance = px.bar(
            x=top_correlations.values,
            y=top_correlations.index,
            orientation='h',
            title='Top 10 Most Important PCA Features',
            color_discrete_sequence=['lightgreen']
        )
        fig_importance.update_layout(xaxis_title='Absolute Correlation', yaxis_title='Feature')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': top_correlations.index,
            'Correlation': top_correlations.values,
            'Importance': ['High' if x > 0.1 else 'Medium' if x > 0.05 else 'Low' for x in top_correlations.values]
        })
        st.subheader("Feature Importance Table")
        st.dataframe(importance_df, use_container_width=True)
    
    # Better correlation analysis
    st.subheader("🔥 Correlation Analysis")
    
    # Focus on top correlations only
    top_features = top_correlations.index[:8].tolist() + ['Time', 'Amount', 'Class']
    correlation_matrix = df[top_features].corr()
    
    # Create a more readable heatmap
    fig_heatmap = px.imshow(
        correlation_matrix,
        title='Top Features Correlation Heatmap',
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Fraud patterns over time
    st.subheader("📅 Fraud Patterns Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create time-based analysis
        df['Hour'] = (df['Time'] // 3600) % 24
        fraud_by_hour = df[df['Class'] == 1]['Hour'].value_counts().sort_index()
        
        fig_pattern = px.line(
            x=fraud_by_hour.index,
            y=fraud_by_hour.values,
            title='Fraud Transactions by Hour of Day',
            markers=True
        )
        fig_pattern.update_layout(xaxis_title='Hour of Day', yaxis_title='Fraud Count')
        st.plotly_chart(fig_pattern, use_container_width=True)
    
    with col2:
        # Amount ranges analysis
        df['Amount_Range'] = pd.cut(df['Amount'], 
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')], 
                                   labels=['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+'])
        fraud_by_amount = df[df['Class'] == 1]['Amount_Range'].value_counts()
        
        fig_amount = px.bar(
            x=fraud_by_amount.index,
            y=fraud_by_amount.values,
            title='Fraud Transactions by Amount Range',
            color_discrete_sequence=['red']
        )
        fig_amount.update_layout(xaxis_title='Amount Range ($)', yaxis_title='Fraud Count')
        st.plotly_chart(fig_amount, use_container_width=True)
    
    # Statistical insights
    st.subheader("📊 Statistical Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Fraudulent Hour", f"{fraud_by_hour.idxmax()}:00")
        st.metric("Least Fraudulent Hour", f"{fraud_by_hour.idxmin()}:00")
    
    with col2:
        most_fraud_amount = fraud_by_amount.idxmax()
        st.metric("Most Fraudulent Amount Range", most_fraud_amount)
        st.metric("Fraud Peak Hour Count", fraud_by_hour.max())
    
    with col3:
        st.metric("Top Feature (V{})".format(top_correlations.index[0][1:]), 
                 f"{top_correlations.iloc[0]:.4f}")
        st.metric("Average Fraud Amount", f"${df[df['Class']==1]['Amount'].mean():.2f}")

with tab5:
    st.header("⚡ Real-time Fraud Detection")
    
    st.subheader("🔍 Transaction Analysis Tool")
    
    # Input form for transaction analysis
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=1.0)
            time_seconds = st.number_input("Time (seconds)", min_value=0.0, value=1000.0, step=1.0)
            v1 = st.number_input("V1 (PCA Component)", value=0.0, step=0.1)
            v2 = st.number_input("V2 (PCA Component)", value=0.0, step=0.1)
            v3 = st.number_input("V3 (PCA Component)", value=0.0, step=0.1)
        
        with col2:
            v4 = st.number_input("V4 (PCA Component)", value=0.0, step=0.1)
            v5 = st.number_input("V5 (PCA Component)", value=0.0, step=0.1)
            v6 = st.number_input("V6 (PCA Component)", value=0.0, step=0.1)
            v7 = st.number_input("V7 (PCA Component)", value=0.0, step=0.1)
            v8 = st.number_input("V8 (PCA Component)", value=0.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Analyze Transaction")
    
    if submitted:
        # Analyze transaction
        risk_score = simple_fraud_detector(amount, time_seconds, v1, v2, v3, v4, v5)
        
        # Determine risk level
        if risk_score >= 4:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
        elif risk_score >= 2:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display results
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", risk_score)
        
        with col2:
            st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", unsafe_allow_html=True)
        
        with col3:
            is_fraud = "YES" if risk_score >= 3 else "NO"
            fraud_color = "red" if is_fraud == "YES" else "green"
            st.markdown(f"<h3 style='color: {fraud_color};'>Fraud: {is_fraud}</h3>", unsafe_allow_html=True)
        
        # Risk breakdown with detailed explanations
        st.subheader("🔍 Risk Factor Breakdown")
        
        risk_factors = []
        risk_explanations = []
        
        # Amount analysis
        if amount > 1000:
            risk_factors.append(f"💰 High amount: ${amount:.2f} (+2 points)")
            risk_explanations.append(f"**Why risky?** High-value transactions (>$1000) are 3x more likely to be fraudulent. Fraudsters often test with small amounts before attempting large transactions.")
        elif amount > 500:
            risk_factors.append(f"💰 Medium amount: ${amount:.2f} (+1 point)")
            risk_explanations.append(f"**Why risky?** Medium-value transactions ($500-$1000) show moderate fraud risk. This range is common for testing stolen cards.")
        
        # Time analysis
        hour = (time_seconds // 3600) % 24
        if hour < 6 or hour > 22:
            risk_factors.append(f"⏰ Unusual time: {hour}:00 (+1 point)")
            if hour < 6:
                risk_explanations.append(f"**Why risky?** Late night transactions ({hour}:00) are suspicious. Normal card usage typically occurs between 6 AM - 10 PM. Fraudsters often operate during low-monitoring hours.")
            else:
                risk_explanations.append(f"**Why risky?** Late evening transactions ({hour}:00) are outside normal business hours. Fraudulent activity often peaks during these times.")
        
        # PCA extreme values
        if abs(v1) > 3 or abs(v2) > 3 or abs(v3) > 3:
            extreme_features = []
            if abs(v1) > 3: extreme_features.append(f"V1={v1:.2f}")
            if abs(v2) > 3: extreme_features.append(f"V2={v2:.2f}")
            if abs(v3) > 3: extreme_features.append(f"V3={v3:.2f}")
            risk_factors.append(f"📊 Extreme PCA values: {', '.join(extreme_features)} (+2 points)")
            risk_explanations.append(f"**Why risky?** PCA values >3 indicate highly unusual transaction patterns. These anonymized features capture complex fraud signatures that normal transactions rarely exhibit.")
        
        # Multiple extreme features
        extreme_count = sum([abs(v1) > 2, abs(v2) > 2, abs(v3) > 2, abs(v4) > 2, abs(v5) > 2])
        if extreme_count >= 3:
            extreme_features = []
            if abs(v1) > 2: extreme_features.append(f"V1={v1:.2f}")
            if abs(v2) > 2: extreme_features.append(f"V2={v2:.2f}")
            if abs(v3) > 2: extreme_features.append(f"V3={v3:.2f}")
            if abs(v4) > 2: extreme_features.append(f"V4={v4:.2f}")
            if abs(v5) > 2: extreme_features.append(f"V5={v5:.2f}")
            risk_factors.append(f"📈 Multiple extreme features: {extreme_count} features ({', '.join(extreme_features)}) (+1 point)")
            risk_explanations.append(f"**Why risky?** Having {extreme_count} PCA features >2 suggests coordinated fraud patterns. Legitimate transactions rarely have multiple extreme values simultaneously.")
        
        # Display risk factors and explanations
        if risk_factors:
            st.write("**🚨 Detected Risk Factors:**")
            for i, factor in enumerate(risk_factors):
                st.write(f"• {factor}")
                with st.expander(f"📋 Explanation for factor {i+1}"):
                    st.markdown(risk_explanations[i])
        else:
            st.write("✅ **No risk factors detected**")
            st.info("This transaction appears normal based on our fraud detection rules.")
        
        # Additional insights
        st.subheader("📊 Transaction Analysis Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⏰ Time Analysis:**")
            if 6 <= hour <= 22:
                st.success(f"✅ Normal business hours ({hour}:00)")
            else:
                st.warning(f"⚠️ Outside normal hours ({hour}:00)")
            
            st.write("**💰 Amount Analysis:**")
            if amount < 100:
                st.success(f"✅ Low amount (${amount:.2f}) - typical for legitimate transactions")
            elif amount < 500:
                st.info(f"ℹ️ Moderate amount (${amount:.2f}) - normal range")
            elif amount < 1000:
                st.warning(f"⚠️ High amount (${amount:.2f}) - requires attention")
            else:
                st.error(f"🚨 Very high amount (${amount:.2f}) - high fraud risk")
        
        with col2:
            st.write("**📊 PCA Pattern Analysis:**")
            normal_pca = sum([abs(v1) <= 2, abs(v2) <= 2, abs(v3) <= 2, abs(v4) <= 2, abs(v5) <= 2])
            if normal_pca >= 4:
                st.success(f"✅ {normal_pca}/5 PCA features in normal range")
            elif normal_pca >= 2:
                st.warning(f"⚠️ {normal_pca}/5 PCA features in normal range")
            else:
                st.error(f"🚨 Only {normal_pca}/5 PCA features in normal range")
            
            st.write("**🎯 Overall Pattern:**")
            if risk_score == 0:
                st.success("✅ Transaction pattern appears legitimate")
            elif risk_score <= 2:
                st.info("ℹ️ Minor risk indicators detected")
            elif risk_score <= 4:
                st.warning("⚠️ Moderate risk - recommend review")
            else:
                st.error("🚨 High risk - immediate review required")
    
    # Batch analysis
    st.subheader("📋 Batch Transaction Analysis")
    
    # File upload for batch analysis
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("📄 Uploaded file preview:")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("🔍 Analyze Batch"):
                # Apply model to batch
                batch_results = []
                for _, row in batch_df.iterrows():
                    risk = simple_fraud_detector(
                        row.get('Amount', 0),
                        row.get('Time', 0),
                        row.get('V1', 0),
                        row.get('V2', 0),
                        row.get('V3', 0),
                        row.get('V4', 0),
                        row.get('V5', 0)
                    )
                    batch_results.append({
                        'Risk_Score': risk,
                        'Risk_Level': 'HIGH' if risk >= 4 else 'MEDIUM' if risk >= 2 else 'LOW',
                        'Fraud_Detected': 'YES' if risk >= 3 else 'NO'
                    })
                
                results_df = pd.DataFrame(batch_results)
                st.write("📊 Batch Analysis Results:")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                high_risk = len(results_df[results_df['Risk_Level'] == 'HIGH'])
                medium_risk = len(results_df[results_df['Risk_Level'] == 'MEDIUM'])
                low_risk = len(results_df[results_df['Risk_Level'] == 'LOW'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk", high_risk)
                with col2:
                    st.metric("Medium Risk", medium_risk)
                with col3:
                    st.metric("Low Risk", low_risk)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("**💳 Credit Card Fraud Detection Dashboard** | Built with Streamlit and Plotly")
st.markdown("**🔒 For compliance and investigation support**")

# Run the app
if __name__ == "__main__":
    pass 