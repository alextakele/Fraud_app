<<<<<<< HEAD
# Fraud_app
This is our Deployed App
=======
# AI Fraud Detection Analytics Dashboard

A robust Streamlit-based web application for detecting and analyzing fraudulent transactions using machine learning.

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-green)

## Features

### Real-time Fraud Detection
- **Single Transaction Analysis**: Enter transaction details and get instant fraud probability
- **Risk Scoring**: Color-coded risk levels (Low/Medium/High)
- **SHAP Explanations**: Understand why the model made its prediction

### Batch Processing
- **CSV Upload**: Process entire datasets for fraud analysis
- **Filtering**: Filter results by fraud status and probability
- **Export**: Download analysis results as CSV

### Analytics Dashboard
- **Model Evaluation**: View accuracy, precision, recall, F1 scores
- **Confusion Matrix**: Visual representation of model performance
- **Feature Importance**: Understand which features drive predictions
- **Exploratory Data Analysis**: Comprehensive data exploration tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Fraud_app

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deployment

#### Streamlit Cloud
1. Push your code to a GitHub repository
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. The app will auto-deploy

#### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Project Structure

```
Fraud_app/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── models/
    ├── catboost_fraud_model.cbm   # Trained CatBoost model
    ├── catboost_fraud_model_feature_order.pkl
    └── scaler.pkl                 # Feature scaler
```

## Configuration

### Streamlit Settings (`.streamlit/config.toml`)

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200  # Maximum upload size in MB

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| STREAMLIT_SERVER_PORT | Port for Streamlit | 8501 |
| STREAMLIT_SERVER_ADDRESS | Server address | 0.0.0.0 |

## Usage Guide

### 1. Load Models
- The app will automatically load pre-trained fraud detection models

### 2. Upload Data
- Upload a CSV file with transaction data
- Required column: `Is_Fraud` (1=Fraud, 0=Legitimate)
- Recommended columns: `Transaction_Amount`, `Transaction_Date`, `Account_Balance`, etc.

### 3. Data Processing
- Preprocess data with automatic cleaning and feature engineering
- SMOTE for handling class imbalance

### 4. Run Analysis
- **Real-time Detection**: Single transaction analysis
- **Batch Processing**: Analyze entire datasets

## API Reference

### FraudDetectionPipeline

```python
pipeline = FraudDetectionPipeline()

# Load pre-trained models
pipeline.load_models()

# Load data from CSV
pipeline.load_data(uploaded_file)

# Preprocess data
pipeline.run_preprocessing()

# Run inference on single transaction
fraud_status, fraud_prob, shap_summary, rule_result, triggered_rules = pipeline.run_inference(input_data)
```

### ValidationModule

```python
# Validate uploaded data
is_valid, message = ValidationModule.validate_data_upload(df)

# Validate preprocessing results
is_valid, message = ValidationModule.validate_preprocessing(df)
```

## Dependencies

- **streamlit** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **catboost** - Gradient boosting model
- **scikit-learn** - Machine learning utilities
- **imbalanced-learn** - SMOTE for class imbalance
- **shap** - Model explainability
- **plotly** - Interactive visualizations
- **matplotlib** - Static plots
- **seaborn** - Statistical plots

See `requirements.txt` for full list.

## Model Performance

The CatBoost model achieves:
- **Accuracy**: ~95%+
- **ROC-AUC**: ~98%+
- **F1-Score**: ~94%+

Performance may vary based on data quality and distribution.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - feel free to use this for your projects.

## Support

For issues and questions:
- Open a GitHub issue
- Check the documentation
- Review Streamlit's troubleshooting guide

---

Built with  using Streamlit and CatBoost
>>>>>>> 2b0d0a1 (Add models directory)
