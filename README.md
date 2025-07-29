# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive fraud detection system that combines advanced data analytics, machine learning techniques, and interactive visualizations to identify suspicious financial transactions and support compliance investigations.

## 🎯 Project Overview

This project analyzes **284,807 credit card transactions** to develop a robust fraud detection system with real-time monitoring capabilities. The system achieves **99.83% legitimate** and **0.17% fraudulent** transaction classification through sophisticated pattern recognition and risk assessment.

## 📊 Key Insights

- **Fraud Amount Patterns**: Fraudulent transactions average $122.21 vs $88.29 for legitimate transactions
- **Temporal Clustering**: Peak fraud activity at 2:00 AM and 11:00 AM
- **Feature Importance**: PCA components V17, V14, and V12 show strongest fraud correlations
- **Class Imbalance**: Severe imbalance requiring specialized handling techniques

## 🚀 Features

### 🔍 **Interactive Dashboard**
- **Real-time transaction analysis** with configurable risk thresholds
- **Multi-tab interface** for comprehensive fraud investigation
- **Interactive visualizations** using Plotly and Streamlit
- **Data filtering and export** capabilities

### 🤖 **Fraud Detection Model**
- **Rule-based risk scoring** system with interpretable results
- **Multi-factor analysis** combining amount, time, and PCA features
- **Configurable thresholds** (1-6 points) for different sensitivity levels
- **Real-time processing** suitable for production environments

### 📈 **Advanced Analytics**
- **Exploratory Data Analysis (EDA)** with interactive visualizations
- **Feature importance ranking** and correlation analysis
- **Temporal pattern recognition** for fraud clustering
- **Statistical insights** and trend analysis

## 🖼️ Dashboard Screenshots

### Overview Dashboard
![Overview Dashboard](CD-1.png)
*Main dashboard showing transaction statistics, class distribution, and key metrics with interactive sidebar controls*

### Fraud Patterns Analysis
![Fraud Patterns](CD-2.png)
*Fraud patterns over time showing peak hours (2:00 AM, 11:00 AM) and amount range analysis with statistical insights*

### Real-time Detection Interface
![Real-time Detection](CD-3.png)
*Interactive transaction analysis tool with risk factor breakdown and detailed explanations for fraud detection*

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Custom rule-based algorithms
- **Deployment**: Web-based application

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
# Place creditcard.csv in the project directory
# Dataset should contain: Time, V1-V28 (PCA features), Amount, Class
```

4. **Run the dashboard**
```bash
streamlit run fraud_dashboard.py
```

5. **Access the application**
```
Open your browser and go to: http://localhost:8501
```

## 📁 Project Structure

```
credit-card-fraud-detection/
├── fraud_dashboard.py              # Main Streamlit dashboard
├── cd.ipynb                        # Jupyter notebook with analysis
├── creditcard.csv                  # Dataset (not included in repo)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Fraud_Detection_Insights_Report.md  # Comprehensive analysis report
└── images/                         # Dashboard screenshots
    ├── overview.png
    ├── patterns.png
    └── realtime.png
```

## 🎮 Usage Guide

### Dashboard Navigation

1. **Overview Tab**: Dataset statistics and transaction distribution
2. **EDA Tab**: Interactive exploratory data analysis
3. **Model Tab**: Fraud detection model performance and settings
4. **Analytics Tab**: Feature importance and correlation analysis
5. **Real-time Tab**: Individual transaction analysis and batch processing

### Sidebar Controls

- **Data Filters**: Adjust amount range, time range, and transaction classes
- **Model Settings**: Configure risk thresholds and sensitivity
- **Visualization Settings**: Choose chart themes and sample sizes
- **Quick Actions**: Refresh data and download filtered datasets

### Real-time Analysis

1. **Individual Transaction**: Enter transaction parameters and analyze risk
2. **Batch Processing**: Upload CSV files for bulk analysis
3. **Risk Factor Breakdown**: Detailed explanations for each risk indicator

## 📊 Model Performance

### Risk Scoring Logic
- **High Amount (>$1000)**: +2 points
- **Medium Amount ($500-$1000)**: +1 point
- **Unusual Time (outside 6 AM - 10 PM)**: +1 point
- **Extreme PCA Values (>3)**: +2 points
- **Multiple Extreme Features (≥3 features >2)**: +1 point

### Performance Metrics
- **Accuracy**: Configurable based on threshold setting
- **Precision**: Optimized for fraud detection
- **Recall**: Captures majority of fraudulent patterns
- **Risk Threshold**: ≥3 points (configurable)

## 🔍 Key Findings

### Fraud Patterns
1. **Amount Distribution**: Fraudulent transactions show higher mean amounts
2. **Temporal Clustering**: Distinct time windows with peak fraud activity
3. **Feature Interactions**: Complex PCA feature combinations indicate fraud
4. **Risk Indicators**: Multiple factors contribute to fraud probability

### Business Impact
- **Proactive Detection**: Real-time monitoring reduces financial losses
- **Operational Efficiency**: Automated screening reduces manual workload
- **Compliance Support**: Audit trails and detailed explanations
- **Scalable Architecture**: Adaptable for different business requirements

## 📈 Dashboard Features

### Interactive Visualizations
- **Pie Charts**: Transaction class distribution
- **Histograms**: Amount and time distributions
- **Scatter Plots**: Feature correlations and patterns
- **Line Charts**: Temporal fraud patterns
- **Heatmaps**: Feature correlation matrices
- **Box Plots**: Risk score distributions

### Real-time Capabilities
- **Live Transaction Analysis**: Instant risk assessment
- **Configurable Thresholds**: Adjustable sensitivity levels
- **Batch Processing**: Bulk transaction analysis
- **Export Functionality**: Download results and reports

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you have any questions or need support:
- Open an issue on GitHub
- Check the [comprehensive report](Fraud_Detection_Insights_Report.md) for detailed analysis
- Review the dashboard documentation

## 🙏 Acknowledgments

- **Dataset**: Credit card fraud detection dataset[kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Libraries**: Streamlit, Plotly, Pandas, NumPy
- **Community**: Open source contributors and fraud detection researchers

## 📊 Dataset Information

The dataset contains credit card transactions made by European cardholders in September 2013. It includes:
- **284,807 transactions** (492 fraudulent, 284,315 legitimate)
- **31 features** (28 PCA components + Time, Amount, Class)
- **No missing values** - clean dataset structure
- **Anonymized data** for privacy protection

## 🔒 Privacy and Security

- **No PII exposure** in analysis or reporting
- **PCA features** provide anonymized transaction characteristics
- **Secure data processing** with appropriate access controls
- **Compliance with data protection regulations**

---

**Built with ❤️ for fraud detection and prevention**

*For compliance and investigation support* 
