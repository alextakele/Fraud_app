import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from catboost import CatBoostClassifier
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, f1_score, precision_score, recall_score, 
                            roc_curve, auc, precision_recall_curve)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
import io
from scipy import stats
import sys

# -------------------- CONFIGURATION --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(SCRIPT_DIR, "visualizations")

for directory in [RESULTS_DIR, VISUALIZATIONS_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        pass

# -------------------- ENHANCED LOGGING --------------------
class StreamlitLogger:
    """Enhanced logger for Streamlit cloud environments."""
    
    def __init__(self, name: str = "fraud_detection"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        self.log_buffer = io.StringIO()
        handler = logging.StreamHandler(self.log_buffer)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def get_logs(self) -> str:
        return self.log_buffer.getvalue()

logger = StreamlitLogger()

# -------------------- CACHED MODEL LOADING --------------------
@st.cache_resource(show_spinner=False)
def load_cached_model() -> Optional[CatBoostClassifier]:
    """Load model with caching for better performance."""
    try:
        model_path = os.path.join(MODELS_DIR, "catboost_fraud_model.cbm")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_cached_feature_order() -> Optional[List[str]]:
    """Load feature order with caching."""
    try:
        feature_path = os.path.join(MODELS_DIR, "catboost_fraud_model_feature_order.pkl")
        if not os.path.exists(feature_path):
            logger.error(f"Feature order file not found: {feature_path}")
            return None
        
        with open(feature_path, 'rb') as f:
            feature_order = pickle.load(f)
        logger.info(f"Feature order loaded: {len(feature_order)} features")
        return feature_order
    except Exception as e:
        logger.error(f"Error loading feature order: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_cached_scaler() -> Optional[StandardScaler]:
    """Load scaler with caching."""
    try:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler file not found: {scaler_path}")
            return None
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return None

# Professional Color Scheme
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#2ca02c',
    'accent': '#ff7f0e',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'success': '#98df8a',
    'info': '#aec7e8',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

# Enhanced Bootstrap-inspired CSS
st.markdown(f"""
<style>
    /* Bootstrap-inspired Base Styles */
    :root {{
        --primary: {COLOR_SCHEME['primary']};
        --secondary: {COLOR_SCHEME['secondary']};
        --accent: {COLOR_SCHEME['accent']};
        --danger: {COLOR_SCHEME['danger']};
        --warning: {COLOR_SCHEME['warning']};
        --success: {COLOR_SCHEME['success']};
        --info: {COLOR_SCHEME['info']};
        --dark: {COLOR_SCHEME['dark']};
        --light: {COLOR_SCHEME['light']};
    }}
    
    .main-header {{
        font-size: 2.8rem;
        color: var(--dark);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }}
    
    /* Enhanced Button System */
    .btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        gap: 0.5rem;
        position: relative;
        overflow: hidden;
        margin: 0.25rem;
    }}
    
    .btn:before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .btn:hover:before {{
        left: 100%;
    }}
    
    .btn-primary {{
        background: linear-gradient(135deg, var(--primary), var(--info));
        color: white;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
    }}
    
    .btn-primary:hover {{
        background: linear-gradient(135deg, var(--secondary), var(--primary));
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.4);
    }}
    
    .btn-success {{
        background: linear-gradient(135deg, var(--success), var(--secondary));
        color: white;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }}
    
    .btn-success:hover {{
        background: linear-gradient(135deg, var(--secondary), var(--success));
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.4);
    }}
    
    .btn-warning {{
        background: linear-gradient(135deg, var(--warning), var(--accent));
        color: black;
        box-shadow: 0 4px 15px rgba(255, 187, 120, 0.3);
    }}
    
    .btn-warning:hover {{
        background: linear-gradient(135deg, var(--accent), var(--warning));
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 187, 120, 0.4);
    }}
    
    .btn-danger {{
        background: linear-gradient(135deg, var(--danger), #c0392b);
        color: white;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }}
    
    .btn-danger:hover {{
        background: linear-gradient(135deg, #c0392b, var(--danger));
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);
    }}
    
    .btn-outline {{
        background: transparent;
        color: var(--primary);
        border: 2px solid var(--primary);
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.1);
    }}
    
    .btn-outline:hover {{
        background: var(--primary);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.3);
    }}
    
    .btn:disabled {{
        background: #bdc3c7 !important;
        color: #7f8c8d !important;
        border-color: #bdc3c7 !important;
        cursor: not-allowed;
        transform: none !important;
        box-shadow: none !important;
    }}
    
    .btn-sm {{
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
    }}
    
    .btn-lg {{
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }}
    
    .btn-block {{
        width: 100%;
        display: flex;
    }}
    
    /* Enhanced Card System */
    .card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }}
    
    .card-header {{
        background: linear-gradient(135deg, var(--primary)15, var(--info)15);
        padding: 1rem 1.5rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        border-radius: 12px 12px 0 0;
        border-bottom: 2px solid var(--primary)30;
    }}
    
    .card-title {{
        margin: 0;
        color: var(--dark);
        font-size: 1.25rem;
        font-weight: 600;
    }}
    
    /* Enhanced Alert System */
    .alert {{
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        background: white;
    }}
    
    .alert-success {{
        background: var(--success)15;
        border-left-color: var(--success);
        color: var(--success);
    }}
    
    .alert-warning {{
        background: var(--warning)15;
        border-left-color: var(--warning);
        color: var(--warning);
    }}
    
    .alert-danger {{
        background: var(--danger)15;
        border-left-color: var(--danger);
        color: var(--danger);
    }}
    
    .alert-info {{
        background: var(--info)15;
        border-left-color: var(--info);
        color: var(--info);
    }}
    
    /* Enhanced Status System */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }}
    
    .status-pill-success {{
        background: var(--success)20;
        color: var(--success);
        border: 1px solid var(--success)40;
    }}
    
    .status-pill-warning {{
        background: var(--warning)20;
        color: var(--warning);
        border: 1px solid var(--warning)40;
    }}
    
    .status-pill-danger {{
        background: var(--danger)20;
        color: var(--danger);
        border: 1px solid var(--danger)40;
    }}
    
    .status-pill-info {{
        background: var(--info)20;
        color: var(--info);
        border: 1px solid var(--info)40;
    }}
    
    /* Enhanced Sidebar */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, var(--dark) 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 0px 15px 15px 0px;
        box-shadow: 3px 0 15px rgba(0,0,0,0.2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }}
    
    .nav-button {{
        width: 100%;
        margin: 6px 0;
        padding: 14px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, var(--primary), var(--info));
        color: white;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: start;
        gap: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-decoration: none;
    }}
    
    .nav-button:hover {{
        background: linear-gradient(135deg, var(--secondary), var(--primary));
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}
    
    .nav-button:disabled {{
        background: #7f8c8d;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }}
    
    .nav-button.active {{
        background: linear-gradient(135deg, var(--accent), var(--warning));
        box-shadow: 0 0 20px rgba(255, 127, 14, 0.5);
    }}
    
    .sidebar-section-header {{
        font-size: 1.2rem;
        color: var(--light);
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .sidebar-logo {{
        display: block;
        margin: 0 auto 20px auto;
        width: 90px;
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(31, 119, 180, 0.5);
        border: 3px solid var(--primary);
    }}
    
    /* Enhanced Progress System */
    .progress-bar {{
        background-color: #34495e;
        border-radius: 8px;
        height: 15px;
        margin: 12px 0;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }}
    
    .progress-bar-fill {{
        background: linear-gradient(90deg, var(--success), var(--secondary));
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
    }}
    
    /* Enhanced Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, var(--light), #ffffff);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid var(--primary);
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    
    /* Enhanced Form Controls */
    .form-group {{
        margin-bottom: 1.5rem;
    }}
    
    .form-label {{
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: var(--dark);
    }}
    
    .form-control {{
        width: 100%;
        padding: 0.75rem 1rem;
        border: 2px solid #e9ecef;
        border-radius: 0.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }}
    
    .form-control:focus {{
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primary)20;
    }}
    
    /* Enhanced Section Divider */
    .section-divider {{
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        margin: 25px 0;
        border-radius: 2px;
    }}
    
    /* Enhanced Pagination */
    .pagination-controls {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        margin: 20px 0;
        padding: 15px;
        background: var(--light);
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Quick Action Cards */
    .quick-action-card {{
        background: linear-gradient(135deg, var(--info)20, var(--primary)20);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--info)40;
        margin: 10px 0;
        transition: all 0.3s ease;
    }}
    
    .quick-action-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        border-color: var(--primary);
    }}
    
    /* Help Tooltip */
    .help-tooltip {{
        font-size: 0.92rem;
        color: #bdc3c7;
        margin-top: 1.2rem;
        padding: 15px;
        background: rgba(52, 73, 94, 0.8);
        border-radius: 10px;
        border-left: 4px solid var(--accent);
        line-height: 1.5;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------- CONFIGURATION --------------------
class Config:
    """Configuration class for directory paths and settings."""
    RESULTS_DIR = 'results'
    VISUALIZATIONS_DIR = 'visualizations'
    SHAP_PLOTS_DIR = 'shap_plots'
    FEATURE_IMPORTANCE_DIR = 'feature_importance'
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories for outputs."""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.VISUALIZATIONS_DIR, exist_ok=True)
        os.makedirs(cls.SHAP_PLOTS_DIR, exist_ok=True)
        os.makedirs(cls.FEATURE_IMPORTANCE_DIR, exist_ok=True)

# -------------------- ENHANCED BUTTON COMPONENTS --------------------
class EnhancedButtons:
    """Enhanced button components with Bootstrap-inspired styling"""
    
    @staticmethod
    def primary_button(text, key=None, disabled=False, icon=None, size="md", full_width=False):
        """Create a primary action button"""
        if full_width:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled,
                use_container_width=True
            )
        else:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled
            )
    
    @staticmethod
    def success_button(text, key=None, disabled=False, icon=None, size="md", full_width=False):
        """Create a success action button"""
        if full_width:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled,
                use_container_width=True
            )
        else:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled
            )
    
    @staticmethod
    def warning_button(text, key=None, disabled=False, icon=None, size="md", full_width=False):
        """Create a warning action button"""
        if full_width:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled,
                use_container_width=True
            )
        else:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled
            )
    
    @staticmethod
    def outline_button(text, key=None, disabled=False, icon=None, size="md", full_width=False):
        """Create an outline button"""
        if full_width:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled,
                use_container_width=True
            )
        else:
            return st.button(
                f"{icon + ' ' if icon else ''}{text}",
                key=key,
                disabled=disabled
            )

# -------------------- ENHANCED NAVIGATION STATE --------------------
class NavigationState:
    """Enhanced navigation state with smart button management"""
    
    MODULE_DEPENDENCIES = {
        "dashboard": [],
        "data_loading": [],
        "preprocessing": ["data_loading"],
        "eda": ["data_loading"],
        "evaluation": ["data_loading", "preprocessing"],
        "shap": ["data_loading", "preprocessing", "evaluation"],
        "feature_importance": ["data_loading", "preprocessing"],
        "prediction": ["data_loading", "preprocessing"],
        "inference": ["data_loading", "preprocessing"]
    }
    
    @staticmethod
    def get_module_status(pipeline, module_id):
        """Get status and appropriate button style for a module"""
        dependencies = NavigationState.MODULE_DEPENDENCIES.get(module_id, [])
        
        for dep in dependencies:
            if dep == "data_loading" and not pipeline.data_loaded:
                return "disabled", "Load data first", "outline"
            elif dep == "preprocessing" and not pipeline.preprocessing_done:
                return "disabled", "Run preprocessing first", "outline"
            elif dep == "evaluation" and not pipeline.models_loaded:
                return "disabled", "Load models first", "outline"
        
        return "enabled", "Ready", "primary"

# -------------------- VALIDATION MODULE --------------------
class ValidationModule:
    """Module for data and process validations with enhanced error handling."""
    
    @staticmethod
    def validate_data_upload(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate uploaded data for required columns and structure."""
        if df is None:
            return False, "No data provided"
        
        if len(df) == 0:
            return False, "Dataset is empty"
        
        required_columns = ['Is_Fraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        if len(df) < 100:
            return False, "Dataset too small (minimum 100 records required)"
        
        # Check for reasonable fraud distribution
        if 'Is_Fraud' in df.columns:
            fraud_rate = df['Is_Fraud'].mean()
            if fraud_rate < 0.001 or fraud_rate > 0.5:
                st.warning(f"⚠️ Unusual fraud rate detected: {fraud_rate:.2%}. Model performance may be affected.")
        
        # Check for all numeric target
        if 'Is_Fraud' in df.columns:
            unique_values = df['Is_Fraud'].unique()
            if not all(v in [0, 1] for v in unique_values):
                st.warning("⚠️ Target column should contain only 0 and 1 values")
        
        return True, "Data validation passed"
    
    @staticmethod
    def validate_preprocessing(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data after preprocessing."""
        if df is None:
            return False, "No data to validate"
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            return False, f"Data still contains {missing_values} missing values after preprocessing"
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if np.any(np.isinf(df[numeric_cols])):
                return False, "Data contains infinite values"
        
        # Check for empty dataframe
        if len(df) == 0:
            return False, "Dataframe is empty after preprocessing"
        
        return True, "Preprocessing validation passed"
    
    @staticmethod
    def validate_model_requirements(pipeline) -> Tuple[bool, str]:
        """Validate that all model requirements are met."""
        if not pipeline.models_loaded:
            return False, "Models not loaded"
        
        if not pipeline.data_loaded:
            return False, "Data not loaded"
        
        if not pipeline.preprocessing_done:
            return False, "Preprocessing not completed"
        
        # Additional model-specific checks
        if pipeline.model is None:
            return False, "Model object is None"
        
        if pipeline.feature_order is None:
            return False, "Feature order not set"
        
        return True, "All model requirements met"
    
    @staticmethod
    def validate_inference_input(input_data: Dict) -> Tuple[bool, str]:
        """Validate input data for inference."""
        if input_data is None:
            return False, "No input data provided"
        
        if not isinstance(input_data, dict):
            return False, "Input data must be a dictionary"
        
        required_fields = ['Transaction_Amount']
        missing_fields = [f for f in required_fields if f not in input_data or input_data[f] is None]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Validate numeric fields
        numeric_fields = ['Transaction_Amount', 'Account_Balance', 'Transaction_Hour', 
                         'Transaction_Day', 'Transaction_Month']
        for field in numeric_fields:
            if field in input_data and input_data[field] is not None:
                try:
                    float(input_data[field])
                except (ValueError, TypeError):
                    return False, f"Field '{field}' must be numeric"
        
        return True, "Input validation passed"

# -------------------- ENHANCED EDA MODULE --------------------
class EnhancedEDAModule:
    """Enhanced Exploratory Data Analysis module."""
    
    @staticmethod
    def run_comprehensive_eda(df, target_col='Is_Fraud'):
        """Run comprehensive EDA with multiple visualizations."""
        if df is None:
            return
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.header("📊 Enhanced Exploratory Data Analysis")
        
        # Basic Information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df), help="Total number of transactions in the dataset")
        with col2:
            st.metric("Features", len(df.columns), help="Total number of features/columns")
        with col3:
            fraud_count = df[target_col].sum() if target_col in df.columns else 0
            st.metric("Fraud Cases", fraud_count, help="Number of fraudulent transactions")
        with col4:
            fraud_rate = (fraud_count / len(df)) * 100 if target_col in df.columns else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%", help="Percentage of fraudulent transactions")
        
        # Data Quality Assessment
        st.subheader("🔍 Data Quality Assessment")
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            missing_total = df.isnull().sum().sum()
            st.metric("Total Missing Values", missing_total)
        
        with quality_col2:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)
        
        with quality_col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
        
        # Enhanced Target Analysis
        if target_col in df.columns:
            EnhancedEDAModule._analyze_target_variable(df, target_col)
        
        # Feature Distribution Analysis
        EnhancedEDAModule._analyze_feature_distributions(df, target_col)
        
        # Correlation Analysis
        EnhancedEDAModule._analyze_correlations(df, target_col)
        
        # Temporal Analysis (if date columns exist)
        EnhancedEDAModule._analyze_temporal_patterns(df, target_col)
        
        # Advanced Statistical Analysis
        EnhancedEDAModule._advanced_statistical_analysis(df, target_col)
    
    @staticmethod
    def _analyze_target_variable(df, target_col):
        """Analyze the target variable distribution."""
        st.subheader("🎯 Target Variable Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            fraud_counts = df[target_col].value_counts()
            fig_pie = px.pie(
                values=fraud_counts.values, 
                names=['Non-Fraud', 'Fraud'],
                title="Fraud Distribution",
                color_discrete_sequence=[COLOR_SCHEME['success'], COLOR_SCHEME['danger']]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=fraud_counts.index, 
                y=fraud_counts.values,
                title="Fraud Cases Count",
                color=fraud_counts.index.astype(str),
                color_discrete_map={'0': COLOR_SCHEME['success'], '1': COLOR_SCHEME['danger']}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col3:
            # Target variable statistics
            target_stats = pd.DataFrame({
                'Statistic': ['Total', 'Fraud Count', 'Fraud Rate', 'Non-Fraud Count'],
                'Value': [
                    len(df),
                    fraud_counts.get(1, 0),
                    f"{(fraud_counts.get(1, 0) / len(df)) * 100:.2f}%",
                    fraud_counts.get(0, 0)
                ]
            })
            st.dataframe(target_stats, use_container_width=True)
    
    @staticmethod
    def _analyze_feature_distributions(df, target_col):
        """Analyze distributions of numerical features."""
        st.subheader("📈 Feature Distributions")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if numerical_cols:
            # Select features to visualize
            selected_features = st.multiselect(
                "Select features to visualize:",
                numerical_cols,
                default=numerical_cols[:4] if len(numerical_cols) >= 4 else numerical_cols
            )
            
            if selected_features:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    feature_to_plot = st.selectbox("Select feature for distribution:", selected_features)
                    if feature_to_plot:
                        fig = px.histogram(
                            df, 
                            x=feature_to_plot,
                            color=target_col if target_col in df.columns else None,
                            title=f"Distribution of {feature_to_plot}",
                            marginal="box",
                            color_discrete_map={0: COLOR_SCHEME['primary'], 1: COLOR_SCHEME['danger']}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plots for selected features
                    fig = go.Figure()
                    for feature in selected_features[:4]:  # Limit to 4 features
                        fig.add_trace(go.Box(
                            y=df[feature],
                            name=feature,
                            marker_color=COLOR_SCHEME['accent']
                        ))
                    fig.update_layout(title="Box Plots of Selected Features")
                    st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _analyze_correlations(df, target_col):
        """Analyze feature correlations."""
        st.subheader("🔗 Correlation Analysis")
        
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation matrix
                corr_matrix = numerical_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation with target
                if target_col in numerical_df.columns:
                    target_correlations = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
                    fig = px.bar(
                        x=target_correlations.values,
                        y=target_correlations.index,
                        orientation='h',
                        title=f"Feature Correlations with {target_col}",
                        color=target_correlations.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _analyze_temporal_patterns(df, target_col):
        """Analyze temporal patterns if date columns exist."""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns and target_col in df.columns:
            st.subheader("⏰ Temporal Pattern Analysis")
            
            for date_col in date_columns[:2]:  # Analyze first 2 date columns
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    df_temp = df_temp.dropna(subset=[date_col])
                    
                    # Extract time components
                    df_temp[f'{date_col}_hour'] = df_temp[date_col].dt.hour
                    df_temp[f'{date_col}_day'] = df_temp[date_col].dt.day
                    df_temp[f'{date_col}_month'] = df_temp[date_col].dt.month
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Fraud by hour
                        fraud_by_hour = df_temp.groupby(f'{date_col}_hour')[target_col].mean()
                        fig = px.line(
                            x=fraud_by_hour.index,
                            y=fraud_by_hour.values,
                            title=f"Fraud Rate by Hour ({date_col})",
                            labels={'x': 'Hour of Day', 'y': 'Fraud Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Fraud by day of month
                        fraud_by_day = df_temp.groupby(f'{date_col}_day')[target_col].mean()
                        fig = px.line(
                            x=fraud_by_day.index,
                            y=fraud_by_day.values,
                            title=f"Fraud Rate by Day ({date_col})",
                            labels={'x': 'Day of Month', 'y': 'Fraud Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"Could not analyze temporal patterns for {date_col}: {e}")
    
    @staticmethod
    def _advanced_statistical_analysis(df, target_col):
        """Perform advanced statistical analysis."""
        st.subheader("📊 Advanced Statistical Analysis")
        
        if target_col in df.columns:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            
            if numerical_cols:
                # T-test for feature significance
                st.write("**Statistical Significance (T-test between Fraud/Non-Fraud):**")
                
                results = []
                for feature in numerical_cols[:10]:  # Limit to first 10 features
                    fraud_data = df[df[target_col] == 1][feature]
                    non_fraud_data = df[df[target_col] == 0][feature]
                    
                    if len(fraud_data) > 1 and len(non_fraud_data) > 1:
                        t_stat, p_value = stats.ttest_ind(fraud_data, non_fraud_data, equal_var=False)
                        results.append({
                            'Feature': feature,
                            'T-Statistic': t_stat,
                            'P-Value': p_value,
                            'Significant': p_value < 0.05
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

# -------------------- INFERENCE MODULE --------------------
class InferenceModule:
    """Module for running inference and generating results."""
    
    @staticmethod
    def run_inference_mode(pipeline, df):
        """Run inference and generate result tables using pipeline's preprocessing."""
        logger.info("Running INFERENCE mode")
        
        # Validate data first
        is_valid, message = ValidationModule.validate_data_upload(df)
        if not is_valid:
            st.error(f"❌ Data validation failed: {message}")
            return None
        
        # Preprocess data using pipeline's method
        X_train, X_test, y_train, y_test, test_indices, original_indices = pipeline.preprocess_data_exact_jupyter(df)
        
        # Generate the result table
        result_table = InferenceModule.generate_original_result_table(pipeline, X_test, df, test_indices, original_indices)
        
        # Save results
        result_table.to_csv(f'{Config.RESULTS_DIR}/final_predictions.csv', index=False)
        
        # Show fraud distribution
        fraud_counts = result_table['Fraud_Status'].value_counts()
        logger.info(f"\nFRAUD STATUS DISTRIBUTION:\n{fraud_counts}")
        
        logger.info("Inference completed successfully")
        return result_table
    
    @staticmethod
    def generate_original_result_table(pipeline, X_test, df_original, test_indices, original_indices):
        """Generate the exact result table from the Jupyter notebook."""
        
        # Align features with model
        X_test_aligned = pipeline.align_features_with_model(X_test)
        if X_test_aligned is None:
            raise ValueError("Feature alignment failed")
        
        # Get model predictions
        fraud_probs = pipeline.model.predict_proba(X_test_aligned)[:, 1] * 100
        fraud_status = ['Fraud' if p > 75 else 'Warning' if p > 50 else 'No' for p in fraud_probs]
        
        # Get rule-based results
        threshold = pipeline.compute_transaction_threshold(df_original)
        # Use original indices to align with df_original
        df_test_subset = df_original.loc[original_indices].reset_index(drop=True)
        triggered_rules = df_test_subset.apply(lambda row: pipeline.identify_triggered_rules(row, threshold), axis=1)
        rule_results = ['Flag' if any(rule != "No Rule Triggered" for rule in rules) else 'No Flag' for rules in triggered_rules]
        
        # Get SHAP explanations
        shap_details = pipeline.get_shap_details(X_test_aligned)
        shap_summaries = [pipeline.get_shap_summary(d) for d in shap_details]
        
        # Prepare raw data for output
        raw_data = X_test.reset_index(drop=True).to_dict('records')
        
        # Verify lengths
        n_samples = len(X_test)
        lengths = {
            'Transaction_ID': len(df_test_subset['Transaction_ID'].values if 'Transaction_ID' in df_test_subset.columns else range(n_samples)),
            'Raw_Transaction_Data': len(raw_data),
            'Triggered_Rule': len(triggered_rules),
            'Rule_Based_Result': len(rule_results),
            'Fraud_Probability': len(fraud_probs),
            'SHAP_Value_Summary': len(shap_summaries),
            'SHAP_Details': len(shap_details),
            'Fraud_Status': len(fraud_status)
        }
        if len(set(lengths.values())) > 1:
            logger.error(f"Mismatched array lengths: {lengths}")
            raise ValueError(f"Mismatched array lengths: {lengths}")
        
        # Create the result table
        result_df = pd.DataFrame({
            'Transaction_ID': df_test_subset['Transaction_ID'].values if 'Transaction_ID' in df_test_subset.columns else range(n_samples),
            'Raw_Transaction_Data': raw_data,
            'Triggered_Rule': triggered_rules,
            'Rule_Based_Result': rule_results,
            'Fraud_Probability (%)': fraud_probs,
            'SHAP_Value_Summary': shap_summaries,
            'SHAP_Details': shap_details,
            'Fraud_Status': fraud_status
        })
        
        return result_df

# -------------------- ENHANCED FRAUD DETECTION PIPELINE --------------------
class FraudDetectionPipeline:
    """Complete fraud detection pipeline with enhanced robustness."""
    
    def __init__(self):
        self.model = None
        self.feature_order = None
        self.scaler = None
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.models_loaded = False
        self.data_loaded = False
        self.preprocessing_done = False
        self.data_uploaded = False
        self._model_load_error = None
    
    def load_models(self) -> bool:
        """Load pre-trained models with improved error handling."""
        try:
            # Use cached model loading
            self.model = load_cached_model()
            if self.model is None:
                self._model_load_error = "Failed to load CatBoost model"
                return False
            
            # Load feature order
            self.feature_order = load_cached_feature_order()
            if self.feature_order is None:
                self._model_load_error = "Failed to load feature order"
                return False
            
            # Load scaler (optional, may not exist)
            self.scaler = load_cached_scaler()
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            self._model_load_error = str(e)
            logger.error(f"Error loading models: {e}")
            return False
    
    def check_model_status(self) -> Dict[str, Any]:
        """Check the status of model loading."""
        status = {
            "models_loaded": self.models_loaded,
            "model_loaded": self.model is not None,
            "feature_order_loaded": self.feature_order is not None,
            "scaler_loaded": self.scaler is not None,
            "error": self._model_load_error
        }
        return status
    
    def load_data(self, uploaded_file=None, sample_size=None) -> bool:
        """Load data from uploaded file with validation, sampling, and memory optimization."""
        try:
            if uploaded_file is None:
                st.warning("⚠️ Please upload a file")
                return False
            
            # Validate file type
            if not uploaded_file.name.endswith('.csv'):
                st.error("❌ Please upload a CSV file")
                return False
            
            # Show file info
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # For large files, offer sampling option
            if file_size > 100:  # If file is larger than 100MB
                st.warning("⚠️ Large file detected. For faster processing, consider sampling.")
                sample_option = st.radio(
                    "Processing option:",
                    ["Use full dataset", "Sample data for faster processing"],
                    key="sample_option"
                )
            else:
                sample_option = "Use full dataset"
            
            # Read the file with progress
            with st.spinner("Reading CSV file..."):
                # Use chunking for large files
                if file_size > 100:
                    # Read in chunks and concatenate
                    chunks = []
                    total_rows = 0
                    progress_bar = st.progress(0)
                    
                    for chunk in pd.read_csv(uploaded_file, encoding='utf-8', 
                                           on_bad_lines='skip', chunksize=10000):
                        chunks.append(chunk)
                        total_rows += len(chunk)
                        progress_bar.progress(min(total_rows / 500000, 1.0))
                    
                    self.df = pd.concat(chunks, ignore_index=True)
                    progress_bar.empty()
                else:
                    self.df = pd.read_csv(uploaded_file, encoding='utf-8', 
                                         on_bad_lines='skip')
            
            # Apply sampling if requested
            if sample_option == "Sample data for faster processing":
                sample_percentage = st.session_state.get('sample_percentage', 50)
                if sample_percentage < 100:
                    original_size = len(self.df)
                    sample_fraction = sample_percentage / 100
                    self.df = self.df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
                    st.info(f"✅ Sampled {len(self.df)} records ({sample_percentage}%) from original {original_size} records")
            
            # Validate data
            is_valid, message = ValidationModule.validate_data_upload(self.df)
            if not is_valid:
                st.error(f"❌ {message}")
                return False
            
            # Memory optimization
            self._optimize_memory()
            
            st.success(f"✅ Successfully loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
            
            # Data quality metrics
            missing_values = self.df.isnull().sum().sum()
            if missing_values > 0:
                st.warning(f"⚠️ Data contains {missing_values:,} missing values")
            
            duplicate_rows = self.df.duplicated().sum()
            if duplicate_rows > 0:
                st.warning(f"⚠️ Data contains {duplicate_rows:,} duplicate rows")
            
            self.data_loaded = True
            self.data_uploaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            logger.error(f"Data loading error: {e}")
            return False
    
    def _optimize_memory(self):
        """Optimize memory usage by downcasting data types."""
        try:
            for col in self.df.columns:
                col_type = self.df[col].dtype
                
                if col_type == 'float64':
                    self.df[col] = pd.to_numeric(self.df[col], downcast='float')
                elif col_type == 'int64':
                    self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
            
            logger.info(f"Memory optimization completed. Reduced memory usage.")
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def display_data_info(self):
        """Display comprehensive data information"""
        if self.df is None:
            st.warning("⚠️ No data loaded. Please load data first.")
            return
        
        st.subheader("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(self.df))
        with col2:
            st.metric("Total Columns", len(self.df.columns))
        with col3:
            st.metric("Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            missing_values = self.df.isnull().sum().sum()
            st.metric("Total Missing Values", missing_values)
        
        with st.expander("🔍 Detailed Data Analysis", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Data Types", "Missing Values", "Statistical Summary"])
            with tab1:
                st.dataframe(self.df.head(10), use_container_width=True)
            with tab2:
                dtype_df = pd.DataFrame({
                    'Column': self.df.columns,
                    'Data Type': self.df.dtypes,
                    'Null Count': self.df.isnull().sum(),
                    'Unique Values': [self.df[col].nunique() for col in self.df.columns]
                })
                st.dataframe(dtype_df, use_container_width=True)
            with tab3:
                missing_df = pd.DataFrame({
                    'Column': self.df.columns,
                    'Missing Values': self.df.isnull().sum(),
                    'Missing Percentage': (self.df.isnull().sum() / len(self.df)) * 100
                }).sort_values('Missing Percentage', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            with tab4:
                st.dataframe(self.df.describe(), use_container_width=True)

    def preprocess_data_exact_jupyter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
        """Exact replication of Jupyter notebook preprocessing with robust NaN handling"""
        df_processed = df.copy()
        df_processed = df_processed.reset_index()  # Preserve original indices
        
        # Log initial NaN status
        initial_nan = df_processed.isnull().sum()
        logger.info(f"Initial NaN values:\n{initial_nan[initial_nan > 0]}")
        
        # Remove single value columns
        single_value_columns = [col for col in df_processed.columns if df_processed[col].nunique() == 1]
        df_processed = df_processed.drop(columns=single_value_columns)
        logger.info(f"Removed single-value columns: {single_value_columns}")
        
        # Drop useless columns
        columns_to_drop = ['Customer_Contact', 'Customer_Email', 'Customer_Name', 
                         'Customer_ID', 'Transaction_Location', 'Bank_Branch', 'Merchant_ID']
        columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values for numerical columns
        numerical_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns
        if numerical_columns.size > 0:
            num_imputer = SimpleImputer(strategy='median')
            df_processed[numerical_columns] = num_imputer.fit_transform(df_processed[numerical_columns])
            logger.info(f"Imputed missing values in numerical columns: {numerical_columns.tolist()}")
        
        # Handle missing values for categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Is_Fraud':
                if df_processed[col].isnull().any():
                    mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                    df_processed[col] = df_processed[col].fillna(mode_val)
                    logger.info(f"Imputed missing values in {col} with mode: {mode_val}")
        
        # Datetime processing
        if 'Transaction_Date' in df_processed.columns:
            df_processed['Transaction_Date'] = pd.to_datetime(df_processed['Transaction_Date'], errors='coerce')
            df_processed['Transaction_Day'] = df_processed['Transaction_Date'].dt.day
            df_processed['Transaction_Month'] = df_processed['Transaction_Date'].dt.month
            df_processed['Transaction_Year'] = df_processed['Transaction_Date'].dt.year
            for col in ['Transaction_Day', 'Transaction_Month', 'Transaction_Year']:
                if col in df_processed.columns and df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    logger.info(f"Imputed NaNs in {col} with median")
        
        if 'Transaction_Time' in df_processed.columns:
            df_processed['Transaction_Time'] = pd.to_datetime(df_processed['Transaction_Time'], errors='coerce')
            df_processed['Transaction_Hour'] = df_processed['Transaction_Time'].dt.hour
            df_processed['Transaction_Minute'] = df_processed['Transaction_Time'].dt.minute
            df_processed['Transaction_Second'] = df_processed['Transaction_Time'].dt.second
            for col in ['Transaction_Hour', 'Transaction_Minute', 'Transaction_Second']:
                if col in df_processed.columns and df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    logger.info(f"Imputed NaNs in {col} with median")
        
        df_processed = df_processed.drop(columns=['Transaction_Date', 'Transaction_Time'], errors='ignore')
        
        # Remove single value columns again
        single_value_cols = [col for col in df_processed.columns if df_processed[col].nunique() == 1]
        df_processed = df_processed.drop(columns=single_value_cols, errors='ignore')
        logger.info(f"Removed single-value columns after datetime processing: {single_value_cols}")
        
        # Label encoding for categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Is_Fraud':
                df_processed[col] = df_processed[col].fillna('Unknown')
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                logger.info(f"Label encoded column: {col}")
        
        # Final NaN check before splitting
        if df_processed.isnull().sum().sum() > 0:
            nan_columns = df_processed.columns[df_processed.isnull().any()].tolist()
            st.error(f"❌ NaN values remain in columns: {nan_columns}")
            logger.error(f"NaN values remain in columns: {nan_columns}")
            raise ValueError(f"NaN values remain in columns: {nan_columns}")
        
        # Split data
        if 'Is_Fraud' not in df_processed.columns:
            raise ValueError("Target column 'Is_Fraud' not found")
        
        X = df_processed.drop(['Is_Fraud', 'index'], axis=1, errors='ignore')
        y = df_processed['Is_Fraud']
        original_indices = df_processed['index']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        original_test_indices = original_indices[X_test.index]
        
        # Verify no NaNs before SMOTE
        if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
            raise ValueError("NaN values detected in X_train or X_test before SMOTE")
        
        # SMOTE resampling
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info("SMOTE resampling completed")
        
        # Create a DataFrame to track indices
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
        # Since SMOTE creates synthetic samples, we can't directly map back to original indices
        # We'll use a placeholder for resampled indices and track original indices separately
        resampled_indices = pd.Series(range(len(X_train_res)))
        
        # Remove Gender and Age
        X_train_res = X_train_res.drop(columns=['Gender', 'Age'], errors='ignore')
        X_test = X_test.drop(columns=['Gender', 'Age'], errors='ignore')
        logger.info("Dropped Gender and Age columns")
        
        # Final train-test split on resampled data
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_res, y_train_res,
            test_size=0.2,
            random_state=42,
            stratify=y_train_res
        )
        test_indices = X_test.index
        # Map test indices back to original indices if possible
        # Since SMOTE creates synthetic samples, we use original_test_indices for df_original alignment
        original_indices_mapped = [original_test_indices.iloc[i % len(original_test_indices)] for i in range(len(X_test))]
        
        # Scale numerical features
        numeric_features = [
            'Transaction_Amount', 'Account_Balance',
            'Transaction_Day', 'Transaction_Hour',
            'Transaction_Minute', 'Transaction_Second'
        ]
        numeric_features = [col for col in numeric_features if col in X_train.columns]
        
        if numeric_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_train[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
                with open(os.path.join('models', 'scaler.pkl'), 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                X_train[numeric_features] = self.scaler.transform(X_train[numeric_features])
            X_test[numeric_features] = self.scaler.transform(X_test[numeric_features])
            logger.info(f"Scaled numerical features: {numeric_features}")
        
        # Save feature order
        self.feature_order = X_train.columns.tolist()
        with open(os.path.join('models', 'catboost_fraud_model_feature_order.pkl'), 'wb') as f:
            pickle.dump(self.feature_order, f)
        logger.info(f"Saved feature order: {self.feature_order}")
        
        logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test, test_indices, original_indices_mapped
    
    def align_features_with_model(self, X):
        """Align features with the model's expected feature order"""
        if self.feature_order is None:
            logger.warning("Feature order not loaded, using input columns as feature order")
            return X
        
        X_aligned = pd.DataFrame(index=X.index)
        for feature in self.feature_order:
            if feature in X.columns:
                X_aligned[feature] = X[feature]
            else:
                X_aligned[feature] = 0
        X_aligned = X_aligned[self.feature_order]
        return X_aligned
    
    def run_preprocessing(self):
        """Run the EXACT preprocessing pipeline from Jupyter notebook with validation"""
        if self.df is None:
            st.warning("⚠️ Please load data first.")
            return None
        
        # Validate data before preprocessing
        is_valid, message = ValidationModule.validate_data_upload(self.df)
        if not is_valid:
            st.error(f"❌ Cannot proceed with preprocessing: {message}")
            return None
        
        st.subheader("🔄 Data Preprocessing Pipeline")
        
        with st.spinner("Running EXACT Jupyter preprocessing..."):
            try:
                X_train, X_test, y_train, y_test, _, _ = self.preprocess_data_exact_jupyter(self.df)
                
                # Validate preprocessing results
                is_valid, message = ValidationModule.validate_preprocessing(pd.concat([X_train, X_test]))
                if not is_valid:
                    st.error(f"❌ Preprocessing validation failed: {message}")
                    return None
                
                self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
                self.preprocessing_done = True
                
                st.success("🎉 Data preprocessing completed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    st.metric("Fraud Rate", f"{(y_train.sum()/len(y_train))*100:.2f}%")
                
                return X_train, X_test, y_train, y_test
                
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")
                return None
    
    def run_eda(self):
        """Comprehensive Exploratory Data Analysis using enhanced module"""
        if self.df is None:
            st.warning("⚠️ Please load data first.")
            return
        
        EnhancedEDAModule.run_comprehensive_eda(self.df)
    
    def run_model_evaluation(self):
        """Comprehensive model evaluation with enhanced metrics"""
        if not self.models_loaded:
            st.warning("⚠️ Please load models first.")
            return
        if not self.preprocessing_done:
            st.warning("⚠️ Please run preprocessing first.")
            return
        
        # Validate model requirements
        is_valid, message = ValidationModule.validate_model_requirements(self)
        if not is_valid:
            st.error(f"❌ {message}")
            return
        
        st.header("📈 Enhanced Model Evaluation")
        
        with st.spinner("Evaluating model performance..."):
            try:
                X_test_aligned = self.align_features_with_model(self.X_test)
                if X_test_aligned is None:
                    return
                
                y_pred = self.model.predict(X_test_aligned)
                y_prob = self.model.predict_proba(X_test_aligned)[:, 1]
                
                # Calculate enhanced metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_prob)
                f1 = f1_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                cm = confusion_matrix(self.y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Additional metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_accuracy = (recall + specificity) / 2
                
                st.subheader("🎯 Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("ROC AUC", f"{roc_auc:.4f}")
                col3.metric("F1 Score", f"{f1:.4f}")
                col4.metric("Precision", f"{precision:.4f}")
                col5.metric("Recall", f"{recall:.4f}")
                
                col6, col7, col8, col9, col10 = st.columns(5)
                col6.metric("Specificity", f"{specificity:.4f}")
                col7.metric("Balanced Accuracy", f"{balanced_accuracy:.4f}")
                col8.metric("False Positives", fp)
                col9.metric("False Negatives", fn)
                col10.metric("True Positives", tp)
                
                # Enhanced Confusion Matrix
                st.subheader("📊 Enhanced Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Non-Fraud', 'Fraud'],
                           yticklabels=['Non-Fraud', 'Fraud'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # ROC and Precision-Recall Curves
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 ROC Curve")
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                               name=f'ROC curve (AUC = {roc_auc:.4f})',
                                               line=dict(color=COLOR_SCHEME['primary'], width=3)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                               name='Random classifier',
                                               line=dict(color=COLOR_SCHEME['danger'], width=2, dash='dash')))
                    fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                                        xaxis_title='False Positive Rate',
                                        yaxis_title='True Positive Rate')
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Precision-Recall Curve")
                    precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_prob)
                    pr_auc = auc(recall_curve, precision_curve)
                    
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', 
                                              name=f'PR curve (AUC = {pr_auc:.4f})',
                                              line=dict(color=COLOR_SCHEME['secondary'], width=3)))
                    fig_pr.update_layout(title='Precision-Recall Curve',
                                       xaxis_title='Recall',
                                       yaxis_title='Precision')
                    st.plotly_chart(fig_pr, use_container_width=True)
                
                # Classification Report
                st.subheader("📋 Detailed Classification Report")
                report = classification_report(self.y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Model evaluation failed: {e}")
    
    def run_shap_analysis(self):
        """SHAP analysis for model interpretability"""
        if not self.models_loaded:
            st.warning("⚠️ Please load models first.")
            return
        if not self.preprocessing_done:
            st.warning("⚠️ Please run preprocessing first.")
            return
        
        st.header("🔍 SHAP Analysis")
        
        with st.spinner("Generating SHAP explanations..."):
            try:
                X_test_aligned = self.align_features_with_model(self.X_test)
                if X_test_aligned is None:
                    return
                
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test_aligned)
                
                st.subheader("🌍 Global Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_aligned, plot_type="bar", show=False)
                plt.title('CatBoost - Global Feature Importance', fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("🔍 SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_aligned, show=False)
                plt.title('CatBoost - SHAP Summary Plot', fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ SHAP analysis failed: {e}")
    
    def run_feature_importance(self):
        """Feature importance analysis"""
        if not self.models_loaded:
            st.warning("⚠️ Please load models first.")
            return
        
        st.header("🎯 Feature Importance Analysis")
        
        try:
            importance = self.model.get_feature_importance()
            feature_names = self.feature_order if self.feature_order else [f"Feature_{i}" for i in range(len(importance))]
            
            min_length = min(len(importance), len(feature_names))
            importance = importance[:min_length]
            feature_names = feature_names[:min_length]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            top_features = importance_df.tail(15)
            fig = px.bar(top_features, x='Importance', y='Feature', 
                        title='Top 15 Feature Importance',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View All Features"):
                st.dataframe(importance_df.sort_values('Importance', ascending=False), 
                           use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Feature importance analysis failed: {e}")
    
    def compute_transaction_threshold(self, df: pd.DataFrame) -> float:
        """Compute threshold for high transaction amount rule"""
        if 'Transaction_Amount' not in df.columns:
            return 0
        q1 = df['Transaction_Amount'].quantile(0.25)
        q3 = df['Transaction_Amount'].quantile(0.75)
        iqr = q3 - q1
        iqr_thresh = q3 + 1.5 * iqr
        p99_thresh = df['Transaction_Amount'].quantile(0.99)
        return max(iqr_thresh, p99_thresh)
    
    def identify_triggered_rules(self, row: pd.Series, threshold: float) -> List[str]:
        """Identify which rules are triggered for a transaction"""
        rules_triggered = []
        
        if row.get('Transaction_Amount', 0) > threshold:
            rules_triggered.append("High Amount")
        
        if row.get('Transaction_Hour', 12) < 8 or row.get('Transaction_Hour', 12) > 20:
            rules_triggered.append("Off Hours")
        
        if row.get('Transaction_Amount', 0) > row.get('Account_Balance', 0):
            rules_triggered.append("Overdraft")
        
        desc = str(row.get('Transaction_Description', '')).lower()
        suspicious_keywords = ['urgent', 'refund', 'donation', 'unknown', 'verify']
        if any(kw in desc for kw in suspicious_keywords):
            rules_triggered.append("Suspicious Description")
        
        if row.get('Account_Type', '') == 'savings' and row.get('Transaction_Amount', 0) > 20000:
            rules_triggered.append("High Savings Transfer")
        
        return rules_triggered if rules_triggered else ["No Rule Triggered"]
    
    def get_shap_details(self, X):
        """Get SHAP feature importance details for each prediction"""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            top_n = 3
            details = []
            for i in range(X.shape[0]):
                shap_row = shap_values[i]
                top_indices = np.argsort(np.abs(shap_row))[-top_n:][::-1]
                detail = {X.columns[j]: float(X.iloc[i, j]) for j in top_indices}
                details.append(detail)
            return details
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return [{} for _ in range(X.shape[0])]
    
    def get_shap_summary(self, shap_detail: Dict) -> str:
        """Convert SHAP details to a readable summary"""
        return ', '.join([f"{k}: {v:.2f}" for k, v in shap_detail.items()])
    
    def run_inference(self, input_data=None):
        """Run inference on new data with feature alignment"""
        if not self.models_loaded:
            st.warning("⚠️ Please load models first.")
            return None, None, None, None, None
        
        try:
            input_df = pd.DataFrame([input_data])
            
            # Apply label encoding to categorical columns
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns:
                    valid_categories = encoder.classes_
                    mode_value = valid_categories[0]
                    input_df[col] = input_df[col].astype(str).apply(
                        lambda x: x if x in valid_categories else mode_value
                    )
                    input_df[col] = encoder.transform(input_df[col])
                    logger.info(f"Applied label encoding to {col}")
            
            # Handle missing values
            numerical_columns = input_df.select_dtypes(include=['float64', 'int64']).columns
            if numerical_columns.size > 0:
                num_imputer = SimpleImputer(strategy='median')
                input_df[numerical_columns] = num_imputer.fit_transform(input_df[numerical_columns])
            
            # Scale numerical features
            numeric_features = [
                'Transaction_Amount', 'Account_Balance',
                'Transaction_Day', 'Transaction_Hour',
                'Transaction_Minute', 'Transaction_Second'
            ]
            numeric_features = [col for col in numeric_features if col in input_df.columns]
            if numeric_features and self.scaler:
                input_df[numeric_features] = self.scaler.transform(input_df[numeric_features])
                logger.info(f"Scaled numerical features for inference: {numeric_features}")
            
            # Align features
            input_df_aligned = self.align_features_with_model(input_df)
            if input_df_aligned is None:
                return None, None, None, None, None
            
            # Get predictions
            fraud_prob = self.model.predict_proba(input_df_aligned)[:, 1][0] * 100
            fraud_status = 'Fraud' if fraud_prob > 75 else 'Warning' if fraud_prob > 50 else 'No'
            
            # Get SHAP details
            shap_details = self.get_shap_details(input_df_aligned)
            shap_summary = self.get_shap_summary(shap_details[0])
            
            # Get rule-based results
            threshold = self.compute_transaction_threshold(self.df) if self.df is not None else 0
            triggered_rules = self.identify_triggered_rules(input_data, threshold)
            rule_result = 'Flag' if any(rule != "No Rule Triggered" for rule in triggered_rules) else 'No Flag'
            
            return fraud_status, fraud_prob, shap_summary, rule_result, triggered_rules
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            logger.error(f"Prediction failed: {e}")
            return None, None, None, None, None

# -------------------- ENHANCED RENDERING FUNCTIONS --------------------
def render_enhanced_dashboard(pipeline):
    """Render the main dashboard view with enhanced Bootstrap styling"""
    #st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection Analytics Dashboard</h1>', 
    #            unsafe_allow_html=True)
    
    # Enhanced Quick Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1f77b4; margin: 0;'>🛡️</h3>
            <h4 style='margin: 5px 0;'>AI Protection</h4>
            <p style='margin: 0; color: #7f8c8d;'>Advanced CatBoost Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2ca02c; margin: 0;'>📊</h3>
            <h4 style='margin: 5px 0;'>Real-time Analysis</h4>
            <p style='margin: 0; color: #7f8c8d;'>Instant fraud detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #ff7f0e; margin: 0;'>🔍</h3>
            <h4 style='margin: 5px 0;'>SHAP Explanations</h4>
            <p style='margin: 0; color: #7f8c8d;'>Model interpretability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #d62728; margin: 0;'>📈</h3>
            <h4 style='margin: 5px 0;'>Batch Processing</h4>
            <p style='margin: 0; color: #7f8c8d;'>Full dataset analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Quick Start Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Quick Start Guide")
        
        steps = [
            ("1. Load AI Models", "Load pre-trained fraud detection models", pipeline.models_loaded),
            ("2. Upload Data", "Upload your transaction data CSV file", pipeline.data_loaded),
            ("3. Process Data", "Clean and prepare data for analysis", pipeline.preprocessing_done),
            ("4. Run Analysis", "Choose between real-time or batch analysis", pipeline.models_loaded and pipeline.data_loaded)
        ]
        
        for step, description, completed in steps:
            icon = "✅" if completed else "⏳"
            color = COLOR_SCHEME['success'] if completed else COLOR_SCHEME['warning']
            st.markdown(f"""
            <div class='card' style='border-left: 4px solid {color};'>
                <div style='display: flex; align-items: center;'>
                    <span style='font-size: 1.5rem; margin-right: 15px;'>{icon}</span>
                    <div>
                        <strong style='color: {color};'>{step}</strong>
                        <p style='margin: 0; color: #7f8c8d;'>{description}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        # Enhanced Model Status Card
        st.markdown("""
        <div class='card'>
            <h4>🤖 AI Models</h4>
            <p>Load machine learning models for fraud detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not pipeline.models_loaded:
            if st.button(
                "🔄 Load Models Now", 
                key="dashboard_load_models",
                use_container_width=True
            ):
                with st.spinner("Loading AI models..."):
                    if pipeline.load_models():
                        st.rerun()
        else:
            st.markdown('<div class="alert alert-success">✅ Models Loaded</div>', unsafe_allow_html=True)
        
        # Enhanced Data Upload Card
        st.markdown("""
        <div class='card'>
            <h4>📁 Data Upload</h4>
            <p>Upload transaction data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="dashboard_uploader", label_visibility="collapsed")
        
        if uploaded_file is not None and not pipeline.data_loaded:
            if st.button(
                "📊 Process Data", 
                key="dashboard_process",
                use_container_width=True
            ):
                with st.spinner("Validating and loading data..."):
                    if pipeline.load_data(uploaded_file=uploaded_file):
                        st.rerun()
        
        # Enhanced Analysis Options Card
        st.markdown("""
        <div class='card'>
            <h4>🔍 Analysis Tools</h4>
            <p>Choose your analysis method</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "🚀 Real-time", 
                key="dashboard_realtime",
                use_container_width=True,
                disabled=not pipeline.models_loaded
            ):
                st.session_state.current_module = "prediction"
                st.rerun()
        
        with col_b:
            if st.button(
                "📊 Batch", 
                key="dashboard_batch",
                use_container_width=True,
                disabled=not (pipeline.models_loaded and pipeline.preprocessing_done)
            ):
                st.session_state.current_module = "inference"
                st.rerun()
    
    # Enhanced System Status Section
    st.markdown("---")
    st.subheader("📊 System Status")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        status = "Ready" if pipeline.models_loaded else "Not Loaded"
        st.metric("AI Models", status)
    
    with status_col2:
        status = "Loaded" if pipeline.data_loaded else "No Data"
        st.metric("Transaction Data", status)
    
    with status_col3:
        status = "Processed" if pipeline.preprocessing_done else "Pending"
        st.metric("Data Processing", status)
    
    with status_col4:
        overall_status = "Operational" if pipeline.models_loaded and pipeline.data_loaded else "Setup Required"
        st.metric("System Status", overall_status)

def render_enhanced_data_loading(pipeline):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="alert alert-info">
            <strong>💡 Upload your transaction data CSV file for analysis</strong><br>
            • Ensure your data contains an 'Is_Fraud' column for supervised learning<br>
            • Supported formats: CSV with UTF-8 encoding<br>
            • Minimum dataset size: 100 records for reliable analysis<br>
            • Large files (>50MB) can be sampled for faster processing
        </div>
        """, unsafe_allow_html=True)
        
        # Professional file uploader
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file", 
            type="csv", 
            key="enhanced_data_upload",
            help="Upload your transaction data in CSV format"
        )
        
        if uploaded_file is not None:
            # Show file info and processing options
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            st.markdown(f"""
            <div class="card">
                <h4>📁 File Information</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size:.2f} MB</p>
            </div>
            """, unsafe_allow_html=True)
            
            if file_size > 50:
                st.markdown("""
                <div class="alert alert-warning">
                    ⚠️ Large file detected. Consider sampling for faster processing.
                </div>
                """, unsafe_allow_html=True)
                sample_option = st.radio(
                    "Processing option:",
                    ["Use full dataset", "Sample data for faster processing"],
                    key="data_loading_sample_option"
                )
                
                if sample_option == "Sample data for faster processing":
                    sample_percentage = st.slider(
                        "Sample percentage:", 
                        min_value=10, 
                        max_value=100, 
                        value=50,
                        key="data_loading_sample_percentage"
                    )
                    st.markdown(f"""
                    <div class="alert alert-info">
                        Will use {sample_percentage}% of the data
                    </div>
                    """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "🚀 Validate & Load Data", 
                    key="validate_load_data",
                    use_container_width=True
                ):
                    with st.spinner("Validating and loading data..."):
                        if pipeline.load_data(uploaded_file=uploaded_file):
                            st.markdown("""
                            <div class="alert alert-success">
                                ✅ Data loaded successfully!
                            </div>
                            """, unsafe_allow_html=True)
                            pipeline.display_data_info()
            
            with col2:
                if st.button(
                    "🔄 Clear Data", 
                    key="clear_data",
                    use_container_width=True,
                    type="secondary"
                ):
                    pipeline.df = None
                    pipeline.data_loaded = False
                    st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h4>📋 Data Requirements</h4>
            </div>
            <p><strong>Required Columns:</strong></p>
            <ul>
                <li><code>Is_Fraud</code>: Target variable (1=Fraud, 0=Legitimate)</li>
            </ul>
            <p><strong>Recommended Columns:</strong></p>
            <ul>
                <li><code>Transaction_Amount</code></li>
                <li><code>Transaction_Date/Time</code></li>
                <li><code>Account_Balance</code></li>
                <li><code>Transaction_Type</code></li>
                <li><code>Customer_Demographics</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_preprocessing(pipeline):
    if not pipeline.data_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ <strong>Please load data first</strong><br>
            • Use the Data Loading module to upload and validate your transaction data<br>
            • Ensure data meets the required format before preprocessing
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="alert alert-info">
        🔧 <strong>Advanced Data Preprocessing Pipeline</strong><br>
        • Automated data cleaning and transformation<br>
        • Handling of missing values and outliers<br>
        • Feature engineering and encoding<br>
        • Data splitting and balancing using SMOTE
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button(
            "🚀 Run Advanced Preprocessing", 
            key="run_preprocessing",
            use_container_width=True
        ):
            with st.spinner("Running advanced preprocessing pipeline..."):
                result = pipeline.run_preprocessing()
                if result is not None:
                    st.markdown("""
                    <div class="alert alert-success">
                        ✅ Advanced preprocessing completed successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show preprocessing summary
                    X_train, X_test, y_train, y_test = result
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training Samples", f"{len(X_train):,}")
                    with col2:
                        st.metric("Test Samples", f"{len(X_test):,}")
                    with col3:
                        st.metric("Features", X_train.shape[1])
                    with col4:
                        fraud_rate = (y_train.sum()/len(y_train))*100
                        st.metric("Balanced Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col2:
        if st.button(
            "📊 View Processed Data",
            key="view_processed",
            use_container_width=True,
            disabled=not pipeline.preprocessing_done
        ):
            if pipeline.X_train is not None:
                st.subheader("Processed Training Data")
                st.dataframe(pipeline.X_train.head(), use_container_width=True)

def render_enhanced_eda(pipeline):
    if not pipeline.data_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please load data first using the Data Loading module.
        </div>
        """, unsafe_allow_html=True)
        return
    
    pipeline.run_eda()

def render_enhanced_evaluation(pipeline):
    if not pipeline.models_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please load AI models first using the Quick Actions in sidebar.
        </div>
        """, unsafe_allow_html=True)
        return
    if not pipeline.preprocessing_done:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please run preprocessing first to prepare data for evaluation.
        </div>
        """, unsafe_allow_html=True)
        return
    
    pipeline.run_model_evaluation()

def render_enhanced_shap(pipeline):
    if not pipeline.models_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please load AI models first.
        </div>
        """, unsafe_allow_html=True)
        return
    if not pipeline.preprocessing_done:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please run preprocessing first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    pipeline.run_shap_analysis()

def render_enhanced_feature_importance(pipeline):
    if not pipeline.models_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please load AI models first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    pipeline.run_feature_importance()

def render_enhanced_prediction(pipeline):
    """Render the prediction interface with sidebar inputs and main page outputs."""
    if not pipeline.models_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            Please load AI models first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("### Real-time Fraud Detection")
    
    # Create two columns for layout
    col_sidebar, col_main = st.columns([1, 2])
    
    with col_sidebar:
        st.markdown("#### Transaction Details")
        
        with st.form("enhanced_prediction_form"):
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
            time_of_day = st.slider("Hour", 0, 23, 12)
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=1000.0, step=100.0)
            transaction_day = st.slider("Day", 1, 31, 15)
            transaction_month = st.slider("Month", 1, 12, 6)
            transaction_year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
            transaction_type = st.selectbox("Type", ["POS", "Online", "ATM", "Transfer", "Payment"])
            account_type = st.selectbox("Account", ["savings", "checking", "credit", "business"])
            description = st.text_input("Description", value="Purchase")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Analyze Risk", use_container_width=True)
            with col2:
                clear_btn = st.form_submit_button("Clear", use_container_width=True, type="secondary")
            
            if clear_btn:
                st.rerun()
    
    with col_main:
        st.markdown("#### Fraud Analysis Results")
        
        if submitted:
            input_data = {
                'Transaction_Amount': amount,
                'Transaction_Hour': time_of_day,
                'Account_Balance': balance,
                'Transaction_Day': transaction_day,
                'Transaction_Month': transaction_month,
                'Transaction_Year': transaction_year,
                'Transaction_Type': transaction_type,
                'Account_Type': account_type,
                'Transaction_Description': description,
                'Transaction_Minute': 0,
                'Transaction_Second': 0
            }
            
            fraud_status, fraud_prob, shap_summary, rule_result, triggered_rules = pipeline.run_inference(input_data=input_data)
            
            if fraud_status is not None:
                # Risk Level Alert
                if fraud_status == 'Fraud':
                    risk_color = COLOR_SCHEME['danger']
                    risk_message = "HIGH RISK - FRAUD DETECTED"
                    alert_class = "alert-danger"
                elif fraud_status == 'Warning':
                    risk_color = COLOR_SCHEME['warning']
                    risk_message = "MEDIUM RISK - SUSPICIOUS"
                    alert_class = "alert-warning"
                else:
                    risk_color = COLOR_SCHEME['success']
                    risk_message = "LOW RISK - LEGITIMATE"
                    alert_class = "alert-success"
                
                st.markdown(f"""
                <div class="{alert_class}" style="font-size: 1.1rem; font-weight: bold;">
                    {risk_message}
                </div>
                """, unsafe_allow_html=True)
                
                # Summary Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Probability", f"{fraud_prob:.1f}%")
                m2.metric("Risk Level", fraud_status)
                m3.metric("Rule-Based", rule_result)
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size": 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 50], 'color': COLOR_SCHEME['success']},
                            {'range': [50, 75], 'color': COLOR_SCHEME['warning']},
                            {'range': [75, 100], 'color': COLOR_SCHEME['danger']}],
                    }
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Actionable Suggestions
                st.markdown("**Recommended Actions:**")
                if fraud_status == 'Fraud':
                    st.markdown("""
                    - Flag transaction for immediate review
                    - Contact customer to verify
                    - Consider temporary block
                    """)
                elif fraud_status == 'Warning':
                    st.markdown("""
                    - Monitor closely
                    - Request additional verification
                    """)
                else:
                    st.markdown("""
                    - Transaction appears normal
                    - No immediate action required
                    """)
                
                # Visual Insights
                with st.expander("Visual Insights", expanded=True):
                    tab1, tab2 = st.tabs(["SHAP Explanation", "Feature Details"])
                    with tab1:
                        st.write(f"**Top Factors:** {shap_summary}")
                    with tab2:
                        rules_df = pd.DataFrame({
                            'Rule': triggered_rules,
                            'Status': ['Triggered' if r != "No Rule Triggered" else 'Not Triggered' for r in triggered_rules]
                        })
                        st.dataframe(rules_df, use_container_width=True)
        else:
            st.info("Enter transaction details and click 'Analyze Risk' to get started.")

def render_enhanced_inference(pipeline):
    if not pipeline.models_loaded:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ <strong>Please load AI models first.</strong>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="alert alert-info">
        🔎 <strong>Batch Transaction Analysis</strong><br>
        • Run fraud detection on entire dataset<br>
        • Advanced pagination for large datasets<br>
        • Filter by fraud status and probability<br>
        • Export comprehensive results
    </div>
    """, unsafe_allow_html=True)
    
    # Single Transaction Analysis Card
    with st.expander("🚀 Quick Single Transaction Check", expanded=False):
        with st.form("quick_single_form"):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("💰 Amount ($)", min_value=0.0, value=100.0, key="quick_amount")
                time_of_day = st.slider("⏰ Hour", 0, 23, 12, key="quick_hour")
                balance = st.number_input("💳 Balance ($)", min_value=0.0, value=1000.0, key="quick_balance")
            with col2:
                transaction_type = st.selectbox("💸 Type", ["POS", "Online", "ATM", "Transfer"], key="quick_type")
                account_type = st.selectbox("🏦 Account Type", ["savings", "checking", "credit"], key="quick_account")
                description = st.text_input("📝 Description", value="Purchase", key="quick_desc")
            
            submitted = st.form_submit_button(
                "🔍 Analyze Single Transaction",
                use_container_width=True
            )
            
            if submitted:
                input_data = {
                    'Transaction_Amount': amount,
                    'Transaction_Hour': time_of_day,
                    'Account_Balance': balance,
                    'Transaction_Day': 15,
                    'Transaction_Month': 6,
                    'Transaction_Year': 2024,
                    'Transaction_Type': transaction_type,
                    'Account_Type': account_type,
                    'Transaction_Description': description,
                    'Transaction_Minute': 0,
                    'Transaction_Second': 0
                }
                
                fraud_status, fraud_prob, shap_summary, rule_result, triggered_rules = pipeline.run_inference(input_data=input_data)
                
                if fraud_status is not None:
                    st.markdown("""
                    <div class="alert alert-success">
                        ✅ Analysis completed!
                    </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fraud Probability", f"{fraud_prob:.1f}%")
                    with col2:
                        st.metric("Risk Level", fraud_status)
                    with col3:
                        st.metric("Rule Result", rule_result)
    
    # Full Batch Analysis
    st.markdown("---")
    st.markdown("### 📊 Full Dataset Batch Analysis")
    
    if not pipeline.preprocessing_done:
        st.markdown("""
        <div class="alert alert-warning">
            ⚠️ Please run preprocessing first to enable full dataset batch analysis.
        </div>
        <div class="alert alert-info">
            💡 Go to <strong>Data Processing Pipeline</strong> in the sidebar to preprocess your data.
        </div>
        """, unsafe_allow_html=True)
    else:
        analysis_col1, analysis_col2 = st.columns([3, 1])
        
        with analysis_col2:
            st.markdown("### ⚙️ Analysis Settings")
            
            # Sampling options for large datasets
            if pipeline.df is not None and len(pipeline.df) > 10000:
                st.markdown("""
                <div class="alert alert-warning">
                    Large dataset detected. Consider sampling for faster analysis.
                </div>
                """, unsafe_allow_html=True)
                use_sampling = st.checkbox("Use sampling for faster processing", value=True)
                if use_sampling:
                    sample_size = st.slider("Sample size", 1000, min(10000, len(pipeline.df)), 5000)
                    st.markdown(f"""
                    <div class="alert alert-info">
                        Will analyze {sample_size} samples
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button(
                "🚀 Run Full Batch Analysis",
                key="full_batch_analysis",
                use_container_width=True
            ):
                with st.spinner("Running comprehensive batch analysis..."):
                    try:
                        result_table = InferenceModule.run_inference_mode(pipeline, pipeline.df)
                        st.session_state.batch_results = result_table
                        st.session_state.current_page = 1
                        st.markdown("""
                        <div class="alert alert-success">
                            ✅ Batch analysis completed successfully!
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="alert alert-danger">
                            ❌ Full batch analysis failed: {e}
                        </div>
                        """, unsafe_allow_html=True)
                        logger.error(f"Batch analysis error: {e}")
        
        with analysis_col1:
            if 'batch_results' in st.session_state and st.session_state.batch_results is not None:
                result_table = st.session_state.batch_results
                
                st.markdown("""
                <div class="alert alert-success">
                    ✅ Full batch analysis completed successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Summary Statistics
                st.subheader("📋 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_transactions = len(result_table)
                    st.metric("Total Transactions", f"{total_transactions:,}")
                with col2:
                    fraud_count = (result_table['Fraud_Status'] == 'Fraud').sum()
                    st.metric("Fraud Detected", fraud_count)
                with col3:
                    warning_count = (result_table['Fraud_Status'] == 'Warning').sum()
                    st.metric("Warnings", warning_count)
                with col4:
                    legit_count = (result_table['Fraud_Status'] == 'No').sum()
                    st.metric("Legitimate", legit_count)
                
                # Enhanced Results Table with Pagination
                st.subheader("📄 Detailed Results")
                
                # Filter Options
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                with filter_col1:
                    fraud_status_filter = st.multiselect(
                        "Filter by Fraud Status:",
                        options=['No', 'Warning', 'Fraud'],
                        default=['No', 'Warning', 'Fraud'],
                        key="fraud_status_filter"
                    )
                
                with filter_col2:
                    probability_range = st.slider(
                        "Fraud Probability Range:",
                        0, 100, (0, 100),
                        key="probability_range"
                    )
                
                with filter_col3:
                    items_per_page = st.selectbox(
                        "Rows per page:",
                        [100, 250, 500, 1000],
                        index=1,
                        key="items_per_page"
                    )
                
                # Apply filters
                filtered_table = result_table[
                    (result_table['Fraud_Status'].isin(fraud_status_filter)) &
                    (result_table['Fraud_Probability (%)'] >= probability_range[0]) &
                    (result_table['Fraud_Probability (%)'] <= probability_range[1])
                ]
                
                st.metric("Filtered Transactions", len(filtered_table))
                
                # Display filtered results
                if len(filtered_table) > 0:
                    st.dataframe(filtered_table, use_container_width=True, height=400)
                else:
                    st.markdown("""
                    <div class="alert alert-info">
                        No transactions match the current filters.
                    </div>
                    """, unsafe_allow_html=True)

# -------------------- ENHANCED MAIN APPLICATION --------------------
def main():
    st.set_page_config(
        page_title="AI Fraud Detection Analytics Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    st.markdown('<h1 class="main-header">AI Fraud Detection Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state with proper defaults
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = FraudDetectionPipeline()
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "dashboard"
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    pipeline = st.session_state.pipeline
    
    # Enhanced Sidebar with Bootstrap-inspired styling
    with st.sidebar:
        st.markdown('<img src="https://via.placeholder.com/80/1f77b4/ffffff?text=FD" class="sidebar-logo" alt="Fraud Detection Logo">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">Navigation Menu</div>', unsafe_allow_html=True)
        
        # Main Navigation with enhanced buttons
        nav_options = [
            ("dashboard", "🏠", "Dashboard", "Overview and quick actions"),
            ("data_loading", "📥", "Data Management", "Upload and validate transaction data"),
            ("preprocessing", "🔄", "Data Processing", "Clean and prepare data for analysis"),
            ("eda", "📊", "Data Analysis", "Explore data patterns and insights"),
            ("evaluation", "📈", "Model Evaluation", "Evaluate AI model performance"),
            ("shap", "🔍", "Model Insights", "Explain model predictions with SHAP"),
            ("feature_importance", "🎯", "Feature Analysis", "View feature importance"),
            ("prediction", "🚀", "Real-time Detection", "Make instant fraud predictions"),
            ("inference", "🔎", "Batch Processing", "Run analysis on entire dataset")
        ]
        
        for module_id, icon, name, description in nav_options:
            status, message, button_type = NavigationState.get_module_status(pipeline, module_id)
            is_active = st.session_state.current_module == module_id
            
            if status == "enabled":
                if st.button(f"{icon} {name}", key=f"nav_{module_id}", use_container_width=True):
                    st.session_state.current_module = module_id
                    st.rerun()
            else:
                st.button(f"{icon} {name} ⚠️", key=f"nav_{module_id}", disabled=True, use_container_width=True,
                         help=message)
        
        st.markdown("---")
        
        # Enhanced System Status
        st.markdown('<div class="sidebar-section-header">System Status</div>', unsafe_allow_html=True)
        
        # Status pills
        col1, col2 = st.columns(2)
        with col1:
            status_class = "status-pill-success" if pipeline.data_loaded else "status-pill-danger"
            st.markdown(f'<div class="status-pill {status_class}">📊 Data</div>', unsafe_allow_html=True)
            
            status_class = "status-pill-success" if pipeline.preprocessing_done else "status-pill-warning"
            st.markdown(f'<div class="status-pill {status_class}">🔄 Processed</div>', unsafe_allow_html=True)
        
        with col2:
            status_class = "status-pill-success" if pipeline.models_loaded else "status-pill-danger"
            st.markdown(f'<div class="status-pill {status_class}">🤖 Models</div>', unsafe_allow_html=True)
            
            status_class = "status-pill-success" if pipeline.data_loaded and pipeline.models_loaded else "status-pill-info"
            st.markdown(f'<div class="status-pill {status_class}">✅ Ready</div>', unsafe_allow_html=True)
        
        # Enhanced Progress Bar
        progress_items = [pipeline.data_loaded, pipeline.models_loaded, pipeline.preprocessing_done]
        progress = sum(progress_items) / len(progress_items) * 100
        st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {progress}%"></div>
        </div>
        <small>Pipeline Progress: {progress:.0f}%</small>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Quick Actions
        st.markdown('<div class="sidebar-section-header">Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if not pipeline.models_loaded:
                if st.button(
                    "🔄 Load Models", 
                    key="quick_load_models",
                    use_container_width=True
                ):
                    with st.spinner("Loading AI models..."):
                        if pipeline.load_models():
                            st.rerun()
        
        with col2:
            uploaded_file = st.file_uploader("📁 Upload Data", type="csv", key="sidebar_uploader",
                                           help="Upload transaction data CSV", label_visibility="collapsed")
            
            if uploaded_file is not None and not pipeline.data_loaded:
                if st.button(
                    "📊 Process Data", 
                    key="quick_process",
                    use_container_width=True
                ):
                    with st.spinner("Validating and loading data..."):
                        if pipeline.load_data(uploaded_file=uploaded_file):
                            st.session_state.data_uploaded = True
                            st.rerun()
        
        # Enhanced Quick Links
        st.markdown("---")
        st.markdown('<div class="sidebar-section-header">Quick Links</div>', unsafe_allow_html=True)
        
        if st.button(
            "🔍 Single Transaction Check", 
            key="quick_single",
            use_container_width=True,
            disabled=not pipeline.models_loaded
        ):
            st.session_state.current_module = "prediction"
            st.rerun()
            
        if st.button(
            "📊 Full Dataset Analysis", 
            key="quick_batch",
            use_container_width=True,
            disabled=not (pipeline.models_loaded and pipeline.preprocessing_done)
        ):
            st.session_state.current_module = "inference"
            st.rerun()
        
        # Enhanced Help Section
        st.markdown("---")
        st.markdown("""
        <div class="help-tooltip">
            <strong>Need Help?</strong><br>
            Contact support for assistance.
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Footer
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #bdc3c7; font-size: 0.8rem;">
            <p style="font-weight: bold; color: {COLOR_SCHEME['primary']};">Developed by</p>
            <p>Smart AI Team</p>
            <p>2025 Fraud Analytics Inc.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    current_module = st.session_state.current_module
    
    # Dashboard View
    if current_module == "dashboard":
        render_enhanced_dashboard(pipeline)
    else:
        module_headers = {
            "data_loading": "📥 Data Management",
            "preprocessing": "🔄 Data Processing Pipeline", 
            "eda": "📊 Exploratory Data Analysis",
            "evaluation": "📈 Model Performance Evaluation",
            "shap": "🔍 Model Interpretation & SHAP Analysis",
            "feature_importance": "🎯 Feature Importance Analysis",
            "prediction": "🚀 Real-time Fraud Detection",
            "inference": "🔎 Batch Transaction Analysis"
        }
        
        if current_module in module_headers:
            st.markdown(f'<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader(f"{module_headers[current_module]}")
            
            # Render appropriate module
            if current_module == "data_loading":
                render_enhanced_data_loading(pipeline)
            elif current_module == "preprocessing":
                render_enhanced_preprocessing(pipeline)
            elif current_module == "eda":
                render_enhanced_eda(pipeline)
            elif current_module == "evaluation":
                render_enhanced_evaluation(pipeline)
            elif current_module == "shap":
                render_enhanced_shap(pipeline)
            elif current_module == "feature_importance":
                render_enhanced_feature_importance(pipeline)
            elif current_module == "prediction":
                render_enhanced_prediction(pipeline)
            elif current_module == "inference":
                render_enhanced_inference(pipeline)

if __name__ == "__main__":
    main()
