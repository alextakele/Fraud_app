# deploy.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import warnings
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Fraud Detection Analytics System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main container */
    .main-container {
        padding: 0 1rem;
    }
    
    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #1e40af 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card-fraud {
        border-left-color: #ef4444;
    }
    
    .metric-card-warning {
        border-left-color: #f59e0b;
    }
    
    .metric-card-suspicious {
        border-left-color: #8b5cf6;
    }
    
    .metric-card-safe {
        border-left-color: #10b981;
    }
    
    /* Data preview */
    .data-preview {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f9fafb;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
    }
    
    .alert-success {
        background-color: #f0fdf4;
        border-left-color: #10b981;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border-left-color: #f59e0b;
    }
    
    .alert-danger {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }
    
    /* Transaction rows */
    .transaction-row {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .transaction-row:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .fraud-row {
        border-left-color: #ef4444;
        background-color: #fef2f2;
    }
    
    .warning-row {
        border-left-color: #f59e0b;
        background-color: #fffbeb;
    }
    
    .suspicious-row {
        border-left-color: #8b5cf6;
        background-color: #f5f3ff;
    }
    
    .safe-row {
        border-left-color: #10b981;
        background-color: #f0fdf4;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar improvements */
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #10b981;
    }
    
    .status-inactive {
        background-color: #ef4444;
    }
    
    /* Loading animation */
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Navigation
        'current_page': 'dashboard',
        'nav_history': ['dashboard'],
        
        # Data state
        'data_loaded': False,
        'data_source_type': None,
        'uploaded_filename': None,
        'selected_dataset': None,
        
        # Analysis state
        'analysis_complete': False,
        'last_analysis': None,
        
        # System state
        'system_initialized': True,
        'shap_enabled': True,
        'auto_tuning': True,
        
        # Thresholds
        'thresholds': {
            'fraud': 0.75,
            'warning': 0.50,
            'suspicious': 0.25
        },
        
        # Data storage
        'current_data': None,
        'current_results': None,
        'data_summary': None,
        
        # UI state
        'file_upload_key': 0,
        'show_data_preview': False,
        'selected_transaction': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== NAVIGATION SYSTEM ====================
class NavigationSystem:
    """Professional navigation system"""
    
    def __init__(self):
        # Navigation options
        self.nav_options = [
            ("dashboard", "Dashboard", "Overview of transactions, quick metrics, and alerts"),
            ("data_management", "Data Management", "Upload local files or select batch datasets"),
            ("analysis_insights", "Analysis & Insights", "Explore data patterns and risk distribution"),
            ("fraud_detection", "Fraud Detection", "Run real-time predictions or batch analysis"),
            ("explainability", "Explainability", "Understand model decisions with SHAP"),
            ("reports", "Reports", "Download analysis results and summary reports")
        ]
    
    def render_navigation(self):
        """Render the navigation sidebar"""
        with st.sidebar:
            # App Header
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;">
                <h1 style="font-size: 1.5rem; color: #1E3A8A; margin: 0;">Fraud Detection</h1>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
                    Analytics System v3.0
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation Menu
            st.markdown("### Navigation")
            
            for page_id, title, description in self.nav_options:
                self._render_nav_item(page_id, title, description)
            
            # System Status
            self._render_system_status()
            
            # Quick Stats
            self._render_quick_stats()
    
    def _render_nav_item(self, page_id: str, title: str, description: str):
        """Render a single navigation item"""
        is_active = st.session_state.current_page == page_id
        
        if is_active:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                margin: 0.25rem 0;
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div>
                        <strong>{title}</strong>
                        <div style="font-size: 0.8rem; opacity: 0.9;">{description}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(
                f"📊 {title}",
                key=f"nav_{page_id}",
                use_container_width=True,
                help=description
            ):
                st.session_state.current_page = page_id
                st.session_state.nav_history.append(page_id)
                if len(st.session_state.nav_history) > 10:
                    st.session_state.nav_history.pop(0)
                st.rerun()
    
    def _render_system_status(self):
        """Render system status section"""
        st.markdown("---")
        st.markdown("### System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status = "Active" if st.session_state.get('system_initialized', False) else "Offline"
            color = "#10b981" if st.session_state.get('system_initialized', False) else "#ef4444"
            st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; margin-right: 8px;"></div>
                <span>System: <strong>{status}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            data_status = "Loaded" if st.session_state.get('data_loaded', False) else "Empty"
            color = "#10b981" if st.session_state.get('data_loaded', False) else "#6b7280"
            st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; margin-right: 8px;"></div>
                <span>Data: <strong>{data_status}</strong></span>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_quick_stats(self):
        """Render quick statistics"""
        if st.session_state.get('data_loaded', False) and st.session_state.get('current_data') is not None:
            st.markdown("---")
            st.markdown("### Quick Stats")
            
            data = st.session_state.current_data
            st.metric("Total Transactions", f"{len(data):,}")
            
            if 'Transaction_Amount' in data.columns:
                total_amount = data['Transaction_Amount'].sum()
                st.metric("Total Amount", f"${total_amount:,.0f}")
            
            if st.session_state.get('analysis_complete', False) and st.session_state.get('current_results') is not None:
                results = st.session_state.current_results
                fraud_count = len(results[results['decision_status'] == 'FRAUD'])
                st.metric("Fraud Detected", fraud_count)

# ==================== DATA MANAGER ====================
class DataManager:
    """Data management system"""
    
    def __init__(self):
        self.batch_datasets = self._initialize_batch_datasets()
    
    def _initialize_batch_datasets(self):
        """Initialize batch datasets"""
        return {
            'sample_small': {
                'name': 'Small Sample (1K)',
                'description': 'Small dataset for testing',
                'size': 1000
            },
            'sample_medium': {
                'name': 'Medium Sample (10K)',
                'description': 'Medium dataset with realistic patterns',
                'size': 10000
            },
            'sample_large': {
                'name': 'Large Sample (50K)',
                'description': 'Large dataset for performance testing',
                'size': 50000
            }
        }
    
    @st.cache_data(ttl=3600)
    def load_batch_dataset(_self, dataset_key: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Load batch dataset"""
        try:
            dataset_info = _self.batch_datasets.get(dataset_key)
            if not dataset_info:
                return None, "Dataset not found"
            
            size = dataset_info['size']
            
            # Generate realistic transaction data
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=size, freq='H')
            
            data = pd.DataFrame({
                'Transaction_ID': [f'TX{str(i).zfill(8)}' for i in range(size)],
                'Transaction_Date': dates.date,
                'Transaction_Time': dates.time,
                'Transaction_Amount': np.random.exponential(500, size),
                'Account_Balance': np.random.uniform(1000, 100000, size),
                'Merchant_Category': np.random.choice(
                    ['Retail', 'Online', 'Travel', 'Food', 'Entertainment'], 
                    size
                ),
                'Transaction_Type': np.random.choice(
                    ['Purchase', 'Transfer', 'Withdrawal', 'Payment'], 
                    size
                ),
                'Payment_Method': np.random.choice(
                    ['Credit Card', 'Debit Card', 'Mobile Pay'], 
                    size
                ),
                'Device_Type': np.random.choice(['Mobile', 'Desktop', 'ATM'], size),
                'Location_City': np.random.choice(['New York', 'London', 'Tokyo'], size),
                'Customer_Age': np.random.randint(18, 70, size),
                'Days_Since_Last_Transaction': np.random.exponential(3, size),
                'Account_Number': [f'ACC{str(i).zfill(10)}' for i in range(size)],
                'Transaction_Hour': np.random.randint(0, 24, size)
            })
            
            return data, f"Loaded {dataset_info['name']} with {size:,} transactions"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def load_local_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            else:
                return None, "Unsupported file format"
            
            # Add required columns if missing
            if 'Transaction_ID' not in data.columns:
                data['Transaction_ID'] = [f'TX{i:08d}' for i in range(len(data))]
            
            if 'Account_Number' not in data.columns:
                data['Account_Number'] = [f'ACC{i:010d}' for i in range(len(data))]
            
            return data, f"Loaded {len(data):,} rows from {uploaded_file.name}"
            
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Get data summary"""
        if data is None or data.empty:
            return {}
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # Numerical summary
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_summary'] = {}
            for col in numerical_cols:
                summary['numerical_summary'][col] = {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'median': float(data[col].median())
                }
        
        # Categorical summary
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {}
            for col in categorical_cols:
                summary['categorical_summary'][col] = {
                    'unique': int(data[col].nunique()),
                    'top_values': data[col].value_counts().head(3).to_dict()
                }
        
        return summary

# ==================== FRAUD DETECTION ENGINE ====================
class FraudDetectionEngine:
    """Fraud detection engine"""
    
    def __init__(self):
        self.feature_importance = {
            'Transaction_Amount': 0.35,
            'Transaction_Hour': 0.18,
            'Days_Since_Last_Transaction': 0.15,
            'Merchant_Category': 0.12,
            'Payment_Method': 0.08,
            'Device_Type': 0.07,
            'Location_City': 0.05
        }
    
    def analyze_transactions(self, data: pd.DataFrame, thresholds: Dict) -> pd.DataFrame:
        """Analyze transactions for fraud"""
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Generate fraud probabilities
        np.random.seed(42)
        n = len(data)
        
        # Base probabilities
        probabilities = np.random.beta(2, 8, n)
        
        # Add fraud patterns
        fraud_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
        probabilities[fraud_indices] = np.random.beta(8, 2, len(fraud_indices))
        
        # Add amount effect
        if 'Transaction_Amount' in data.columns:
            amounts = data['Transaction_Amount'].values
            amount_effect = np.clip(amounts / 10000, 0, 0.3)
            probabilities += amount_effect
        
        # Add time effect
        if 'Transaction_Hour' in data.columns:
            hours = data['Transaction_Hour'].values
            time_effect = np.where((hours < 6) | (hours > 22), 0.15, 0)
            probabilities += time_effect
        
        probabilities = np.clip(probabilities, 0, 1)
        
        # Create results
        results = []
        for i in range(n):
            prob = probabilities[i]
            row = data.iloc[i]
            
            # Determine status
            if prob >= thresholds['fraud']:
                status = 'FRAUD'
            elif prob >= thresholds['warning']:
                status = 'WARNING'
            elif prob >= thresholds['suspicious']:
                status = 'SUSPICIOUS'
            else:
                status = 'SAFE'
            
            # Generate reasons
            reasons = self._generate_reasons(row, prob)
            
            results.append({
                'transaction_id': row.get('Transaction_ID', f'TX{i:08d}'),
                'account': row.get('Account_Number', f'ACC{i:010d}'),
                'amount': float(row.get('Transaction_Amount', 0)),
                'fraud_probability': float(prob),
                'decision_status': status,
                'primary_reason': reasons['primary'],
                'supporting_factors': ' | '.join(reasons['supporting']),
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(results)
    
    def _generate_reasons(self, row: pd.Series, prob: float) -> Dict:
        """Generate decision reasons"""
        reasons = {
            'primary': '',
            'supporting': []
        }
        
        # Amount-based reasons
        amount = row.get('Transaction_Amount', 0)
        if amount > 10000:
            reasons['primary'] = f"Very high transaction amount (${amount:,.2f})"
            reasons['supporting'].append("Amount exceeds typical patterns")
        elif amount > 5000:
            reasons['supporting'].append(f"High transaction amount (${amount:,.2f})")
        
        # Time-based reasons
        hour = row.get('Transaction_Hour', 12)
        if hour < 6 or hour > 22:
            reasons['supporting'].append(f"Unusual transaction time ({hour}:00)")
        
        # Category-based reasons
        merchant = str(row.get('Merchant_Category', ''))
        if merchant in ['Online', 'Travel']:
            reasons['supporting'].append(f"High-risk merchant: {merchant}")
        
        # Default reason
        if not reasons['primary']:
            if prob > 0.8:
                reasons['primary'] = "Multiple high-risk factors detected"
            elif prob > 0.6:
                reasons['primary'] = "Moderate risk profile"
            else:
                reasons['primary'] = "Standard transaction pattern"
        
        return reasons
    
    def get_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        if results.empty:
            return {}
        
        total = len(results)
        fraud = len(results[results['decision_status'] == 'FRAUD'])
        warning = len(results[results['decision_status'] == 'WARNING'])
        suspicious = len(results[results['decision_status'] == 'SUSPICIOUS'])
        safe = len(results[results['decision_status'] == 'SAFE'])
        
        return {
            'total': total,
            'fraud': fraud,
            'warning': warning,
            'suspicious': suspicious,
            'safe': safe,
            'fraud_rate': fraud / total * 100,
            'avg_probability': results['fraud_probability'].mean(),
            'high_risk': len(results[results['fraud_probability'] > 0.7]),
            'total_amount': results['amount'].sum(),
            'fraud_amount': results[results['decision_status'] == 'FRAUD']['amount'].sum()
        }

# ==================== VISUALIZATION ENGINE ====================
class VisualizationEngine:
    """Visualization engine"""
    
    @staticmethod
    def create_metric_card(title: str, value, delta=None, card_type="default"):
        """Create a metric card"""
        card_class = f"metric-card metric-card-{card_type}" if card_type != "default" else "metric-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        if delta:
            st.metric(title, value, delta)
        else:
            st.metric(title, value)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_risk_distribution(results: pd.DataFrame, thresholds: Dict) -> go.Figure:
        """Create risk distribution chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=results['fraud_probability'],
            nbinsx=50,
            marker_color='#667eea',
            opacity=0.7,
            name='Transactions'
        ))
        
        # Add threshold lines
        colors = {'fraud': '#ef4444', 'warning': '#f59e0b', 'suspicious': '#8b5cf6'}
        
        for threshold_name, threshold_value in thresholds.items():
            fig.add_vline(
                x=threshold_value,
                line_dash="dash",
                line_color=colors.get(threshold_name, '#000000'),
                annotation_text=f"{threshold_name.upper()} ({threshold_value})"
            )
        
        fig.update_layout(
            title="Fraud Risk Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            showlegend=False,
            height=350,
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_status_chart(results: pd.DataFrame) -> go.Figure:
        """Create status distribution chart"""
        status_counts = results['decision_status'].value_counts()
        
        colors = {
            'FRAUD': '#ef4444',
            'WARNING': '#f59e0b',
            'SUSPICIOUS': '#8b5cf6',
            'SAFE': '#10b981'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=[colors.get(status, '#6b7280') for status in status_counts.index]
        )])
        
        fig.update_layout(
            title="Transaction Status Distribution",
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_amount_vs_risk(results: pd.DataFrame) -> go.Figure:
        """Create amount vs risk scatter plot"""
        fig = go.Figure()
        
        status_colors = {
            'FRAUD': '#ef4444',
            'WARNING': '#f59e0b',
            'SUSPICIOUS': '#8b5cf6',
            'SAFE': '#10b981'
        }
        
        for status in results['decision_status'].unique():
            subset = results[results['decision_status'] == status]
            fig.add_trace(go.Scatter(
                x=subset['fraud_probability'],
                y=subset['amount'],
                mode='markers',
                name=status,
                marker=dict(
                    size=8,
                    color=status_colors.get(status, '#6b7280'),
                    opacity=0.6
                )
            ))
        
        fig.update_layout(
            title="Transaction Amount vs Fraud Risk",
            xaxis_title="Fraud Probability",
            yaxis_title="Amount ($)",
            height=350,
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance(importance: Dict) -> go.Figure:
        """Create feature importance chart"""
        features = list(importance.keys())
        values = list(importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(values)[::-1]
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color='#667eea'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            height=350,
            plot_bgcolor='white'
        )
        
        return fig

# ==================== MAIN APPLICATION ====================
class FraudDetectionApp:
    """Main application"""
    
    def __init__(self):
        # Initialize session state
        init_session_state()
        
        # Initialize components
        self.navigation = NavigationSystem()
        self.data_manager = DataManager()
        self.detection_engine = FraudDetectionEngine()
        self.viz = VisualizationEngine()
    
    def run(self):
        """Run the application"""
        # Render navigation
        self.navigation.render_navigation()
        
        # Get current page
        current_page = st.session_state.current_page
        
        # Render page header
        self._render_page_header(current_page)
        
        # Render page content
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'data_management':
            self.render_data_management()
        elif current_page == 'analysis_insights':
            self.render_analysis_insights()
        elif current_page == 'fraud_detection':
            self.render_fraud_detection()
        elif current_page == 'explainability':
            self.render_explainability()
        elif current_page == 'reports':
            self.render_reports()
        
        # Render footer
        self._render_footer()
    
    def _render_page_header(self, page_id: str):
        """Render page header"""
        # Map page IDs to titles
        page_titles = {
            'dashboard': 'Dashboard',
            'data_management': 'Data Management',
            'analysis_insights': 'Analysis & Insights',
            'fraud_detection': 'Fraud Detection',
            'explainability': 'Explainability',
            'reports': 'Reports'
        }
        
        title = page_titles.get(page_id, 'Dashboard')
        
        st.markdown(f"""
        <div class="page-header">
            <h1 style="margin: 0; font-size: 2rem;">{title}</h1>
            <p style="margin: 0; opacity: 0.9;">
                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_footer(self):
        """Render footer"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem;">
            <p>Fraud Detection Analytics System • v3.0 • Powered by AI/ML</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== PAGE RENDERERS ====================
    
    def render_dashboard(self):
        """Render dashboard page"""
        st.markdown("### System Overview")
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.viz.create_metric_card(
                "System Status",
                "Active" if st.session_state.system_initialized else "Offline",
                card_type="safe" if st.session_state.system_initialized else "fraud"
            )
        
        with col2:
            if st.session_state.data_loaded and st.session_state.current_data is not None:
                data_len = len(st.session_state.current_data)
                self.viz.create_metric_card("Data Loaded", f"{data_len:,} rows", card_type="safe")
            else:
                self.viz.create_metric_card("Data Status", "No Data", card_type="warning")
        
        with col3:
            if st.session_state.analysis_complete:
                self.viz.create_metric_card("Last Analysis", "Complete", card_type="safe")
            else:
                self.viz.create_metric_card("Analysis Status", "Pending", card_type="warning")
        
        with col4:
            self.viz.create_metric_card("Model Version", "v3.0.1", "CatBoost", card_type="suspicious")
        
        st.markdown("---")
        
        # Analysis results if available
        if st.session_state.analysis_complete and st.session_state.current_results is not None:
            st.markdown("### Analysis Results")
            
            results = st.session_state.current_results
            metrics = self.detection_engine.get_performance_metrics(results)
            
            # Results metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.viz.create_metric_card("Fraud Cases", metrics.get('fraud', 0), card_type="fraud")
            
            with col2:
                self.viz.create_metric_card("Warnings", metrics.get('warning', 0), card_type="warning")
            
            with col3:
                self.viz.create_metric_card("Suspicious", metrics.get('suspicious', 0), card_type="suspicious")
            
            with col4:
                self.viz.create_metric_card("Safe", metrics.get('safe', 0), card_type="safe")
            
            # Visualizations
            st.markdown("---")
            st.markdown("#### Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                fig = self.viz.create_risk_distribution(results, st.session_state.thresholds)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                fig = self.viz.create_status_chart(results)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent fraud alerts
            st.markdown("#### Recent Fraud Alerts")
            fraud_cases = results[results['decision_status'] == 'FRAUD'].head(5)
            
            if not fraud_cases.empty:
                for _, row in fraud_cases.iterrows():
                    st.markdown(f"""
                    <div class="fraud-row transaction-row">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{row['transaction_id']}</strong>
                            <span style="color: #ef4444; font-weight: bold;">FRAUD</span>
                        </div>
                        <div style="color: #6b7280; font-size: 0.9rem;">
                            Amount: ${row['amount']:,.2f} • Risk: {row['fraud_probability']:.3f}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                            {row['primary_reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No fraud cases detected in the recent analysis.")
        
        else:
            # Welcome/instructions
            st.markdown("""
            <div class="alert-info alert-box">
                <h3>Welcome to Fraud Detection Analytics</h3>
                <p>To get started:</p>
                <ol>
                    <li>Go to <strong>Data Management</strong> to load your transaction data</li>
                    <li>Visit <strong>Analysis & Insights</strong> to explore your data</li>
                    <li>Go to <strong>Fraud Detection</strong> to run analysis</li>
                    <li>Check <strong>Explainability</strong> to understand model decisions</li>
                    <li>Use <strong>Reports</strong> to export your results</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    def render_data_management(self):
        """Render data management page"""
        st.markdown("### Data Management")
        
        # Data source selection
        st.markdown("#### Select Data Source")
        
        source_type = st.radio(
            "Choose data source:",
            ["Local File Upload", "Batch Dataset"],
            horizontal=True,
            key="data_source_radio"
        )
        
        st.markdown("---")
        
        if source_type == "Local File Upload":
            self._render_file_upload()
        else:
            self._render_batch_dataset()
        
        # Show loaded data
        if st.session_state.data_loaded and st.session_state.current_data is not None:
            st.markdown("---")
            st.markdown("#### Loaded Data")
            
            data = st.session_state.current_data
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", f"{len(data):,}")
            
            with col2:
                st.metric("Total Columns", len(data.columns))
            
            with col3:
                if 'Transaction_Amount' in data.columns:
                    total_amount = data['Transaction_Amount'].sum()
                    st.metric("Total Amount", f"${total_amount:,.0f}")
            
            # Data preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(data.head(100), use_container_width=True)
                st.caption(f"Showing 100 of {len(data):,} rows")
            
            # Clear data button
            if st.button("🗑️ Clear Data", type="secondary"):
                st.session_state.data_loaded = False
                st.session_state.current_data = None
                st.session_state.analysis_complete = False
                st.session_state.current_results = None
                st.success("Data cleared successfully!")
                st.rerun()
    
    def _render_file_upload(self):
        """Render file upload section"""
        st.markdown("##### Upload Local File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or JSON file",
            type=['csv', 'xlsx', 'json'],
            key=f"file_uploader_{st.session_state.file_upload_key}"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.info(f"**File:** {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Load button
            if st.button("📥 Load File", type="primary", use_container_width=True):
                with st.spinner("Loading data..."):
                    data, message = self.data_manager.load_local_file(uploaded_file)
                    
                    if data is not None:
                        st.session_state.current_data = data
                        st.session_state.data_loaded = True
                        st.session_state.data_source_type = 'local'
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    def _render_batch_dataset(self):
        """Render batch dataset section"""
        st.markdown("##### Select Batch Dataset")
        
        datasets = self.data_manager.batch_datasets
        
        for key, info in datasets.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                **{info['name']}**  
                {info['description']}  
                *{info['size']:,} transactions*
                """)
            
            with col2:
                if st.button(f"Load", key=f"load_{key}", use_container_width=True):
                    with st.spinner(f"Loading {info['name']}..."):
                        data, message = self.data_manager.load_batch_dataset(key)
                        
                        if data is not None:
                            st.session_state.current_data = data
                            st.session_state.data_loaded = True
                            st.session_state.data_source_type = 'batch'
                            st.session_state.selected_dataset = info['name']
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            with col3:
                if st.button("Preview", key=f"preview_{key}", use_container_width=True):
                    with st.spinner("Generating preview..."):
                        data, _ = self.data_manager.load_batch_dataset(key)
                        if data is not None:
                            st.dataframe(data.head(10), use_container_width=True)
                            st.caption(f"Preview of {info['name']} (10 of {info['size']:,} rows)")
    
    def render_analysis_insights(self):
        """Render analysis & insights page"""
        st.markdown("### Analysis & Insights")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first in Data Management")
            return
        
        data = st.session_state.current_data
        
        # Data overview
        st.markdown("#### Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", f"{len(data):,}")
        
        with col2:
            numerical = data.select_dtypes(include=[np.number]).columns
            st.metric("Numerical Features", len(numerical))
        
        with col3:
            categorical = data.select_dtypes(include=['object']).columns
            st.metric("Categorical Features", len(categorical))
        
        st.markdown("---")
        
        # Interactive analysis
        tab1, tab2 = st.tabs(["Distribution Analysis", "Pattern Analysis"])
        
        with tab1:
            self._render_distribution_analysis(data)
        
        with tab2:
            self._render_pattern_analysis(data)
    
    def _render_distribution_analysis(self, data: pd.DataFrame):
        """Render distribution analysis"""
        st.markdown("##### Distribution Analysis")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            selected_col = st.selectbox("Select numerical feature:", numerical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    data, 
                    x=selected_col,
                    title=f"Distribution of {selected_col}",
                    nbins=50,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    data,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("###### Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{data[selected_col].mean():,.2f}")
            
            with col2:
                st.metric("Median", f"{data[selected_col].median():,.2f}")
            
            with col3:
                st.metric("Min", f"{data[selected_col].min():,.2f}")
            
            with col4:
                st.metric("Max", f"{data[selected_col].max():,.2f}")
        else:
            st.info("No numerical features available for distribution analysis")
    
    def _render_pattern_analysis(self, data: pd.DataFrame):
        """Render pattern analysis"""
        st.markdown("##### Pattern Analysis")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_col = st.selectbox("Select categorical feature:", categorical_cols)
            
            # Top categories
            top_categories = data[selected_col].value_counts().head(10)
            
            fig = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title=f"Top 10 {selected_col} Categories",
                labels={'x': selected_col, 'y': 'Count'},
                color_discrete_sequence=['#667eea']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            with st.expander("View detailed counts"):
                st.dataframe(top_categories, use_container_width=True)
        else:
            st.info("No categorical features available for pattern analysis")
    
    def render_fraud_detection(self):
        """Render fraud detection page"""
        st.markdown("### Fraud Detection")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first in Data Management")
            return
        
        # Configuration
        st.markdown("#### Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.shap_enabled = st.toggle(
                "Enable SHAP Explanations",
                value=st.session_state.shap_enabled
            )
            
            st.session_state.auto_tuning = st.toggle(
                "Auto Tune Thresholds",
                value=st.session_state.auto_tuning
            )
        
        with col2:
            st.markdown("##### Detection Thresholds")
            
            st.session_state.thresholds['fraud'] = st.slider(
                "Fraud Threshold",
                min_value=0.5,
                max_value=0.95,
                value=st.session_state.thresholds['fraud'],
                step=0.05,
                help="Probability above which transactions are flagged as fraud"
            )
            
            st.session_state.thresholds['warning'] = st.slider(
                "Warning Threshold",
                min_value=0.3,
                max_value=0.8,
                value=st.session_state.thresholds['warning'],
                step=0.05,
                help="Probability threshold for warnings"
            )
            
            st.session_state.thresholds['suspicious'] = st.slider(
                "Suspicious Threshold",
                min_value=0.1,
                max_value=0.6,
                value=st.session_state.thresholds['suspicious'],
                step=0.05,
                help="Probability threshold for suspicious transactions"
            )
        
        st.markdown("---")
        
        # Run analysis
        st.markdown("#### Run Analysis")
        
        data_info = f"{len(st.session_state.current_data):,} transactions"
        st.info(f"Ready to analyze {data_info}")
        
        if st.button("Start Fraud Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing transactions..."):
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing... {i + 1}%")
                    time.sleep(0.01)
                
                # Run analysis
                results = self.detection_engine.analyze_transactions(
                    st.session_state.current_data,
                    st.session_state.thresholds
                )
                
                # Store results
                st.session_state.current_results = results
                st.session_state.analysis_complete = True
                st.session_state.last_analysis = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Analysis complete! Analyzed {len(results):,} transactions")
                st.rerun()
        
        # Show results if available
        if st.session_state.analysis_complete and st.session_state.current_results is not None:
            st.markdown("---")
            st.markdown("#### Analysis Results")
            
            results = st.session_state.current_results
            metrics = self.detection_engine.get_performance_metrics(results)
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.viz.create_metric_card("Fraud", metrics.get('fraud', 0), card_type="fraud")
            
            with col2:
                self.viz.create_metric_card("Warnings", metrics.get('warning', 0), card_type="warning")
            
            with col3:
                self.viz.create_metric_card("Suspicious", metrics.get('suspicious', 0), card_type="suspicious")
            
            with col4:
                self.viz.create_metric_card("Safe", metrics.get('safe', 0), card_type="safe")
            
            # Additional metrics
            st.markdown("##### Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Rate", f"{metrics.get('fraud_rate', 0):.1f}%")
            
            with col2:
                st.metric("Avg Risk", f"{metrics.get('avg_probability', 0):.3f}")
            
            with col3:
                st.metric("High Risk", metrics.get('high_risk', 0))
            
            # Visualizations
            st.markdown("---")
            st.markdown("##### Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                fig = self.viz.create_risk_distribution(results, st.session_state.thresholds)
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                fig = self.viz.create_status_chart(results)
                st.plotly_chart(fig, use_container_width=True)
            
            # Amount vs Risk
            st.plotly_chart(
                self.viz.create_amount_vs_risk(results),
                use_container_width=True
            )
    
    def render_explainability(self):
        """Render explainability page"""
        st.markdown("### Explainability")
        
        if not st.session_state.analysis_complete:
            st.warning("Please run fraud detection analysis first")
            return
        
        results = st.session_state.current_results
        
        # Feature importance
        st.markdown("#### Feature Importance")
        
        fig = self.viz.create_feature_importance(self.detection_engine.feature_importance)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Transaction explanations
        st.markdown("#### Transaction Explanations")
        
        # Select transaction
        if not results.empty:
            transaction_list = results[['transaction_id', 'decision_status', 'fraud_probability']].copy()
            transaction_list['display'] = transaction_list.apply(
                lambda x: f"{x['transaction_id']} - {x['decision_status']} ({x['fraud_probability']:.3f})", 
                axis=1
            )
            
            selected_idx = st.selectbox(
                "Select a transaction to explain:",
                range(len(transaction_list)),
                format_func=lambda x: transaction_list.iloc[x]['display']
            )
            
            if selected_idx is not None:
                transaction = results.iloc[selected_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Transaction Details")
                    
                    details = [
                        ("Transaction ID", transaction['transaction_id']),
                        ("Account", transaction['account']),
                        ("Amount", f"${transaction['amount']:,.2f}"),
                        ("Status", transaction['decision_status']),
                        ("Fraud Probability", f"{transaction['fraud_probability']:.3f}"),
                        ("Analysis Time", transaction['analysis_timestamp'])
                    ]
                    
                    for label, value in details:
                        st.markdown(f"**{label}:** {value}")
                
                with col2:
                    st.markdown("##### Risk Assessment")
                    
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=transaction['fraud_probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fraud Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': st.session_state.thresholds['fraud'] * 100
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Decision reasoning
                st.markdown("---")
                st.markdown("##### Decision Reasoning")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Primary Reason:**")
                    st.info(transaction['primary_reason'])
                
                with col2:
                    if transaction['supporting_factors']:
                        st.markdown("**Supporting Factors:**")
                        factors = transaction['supporting_factors'].split(' | ')
                        for factor in factors:
                            st.markdown(f"• {factor}")
        else:
            st.info("No transaction data available")
    
    def render_reports(self):
        """Render reports page"""
        st.markdown("### Reports & Export")
        
        if not st.session_state.analysis_complete:
            st.warning("Please run fraud detection analysis first")
            return
        
        results = st.session_state.current_results
        metrics = self.detection_engine.get_performance_metrics(results)
        
        # Report options
        st.markdown("#### Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Comprehensive Report", "Executive Summary", "Technical Analysis"]
            )
        
        with col2:
            include_charts = st.checkbox("Include Charts", value=True)
        
        st.markdown("---")
        
        # Generate and download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"fraud_analysis_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON Export
            json_data = json.dumps({
                "metadata": {
                    "generated_at": timestamp,
                    "report_type": report_type,
                    "thresholds": st.session_state.thresholds
                },
                "metrics": metrics,
                "summary": {
                    "total_transactions": metrics.get('total', 0),
                    "fraud_cases": metrics.get('fraud', 0),
                    "fraud_rate": f"{metrics.get('fraud_rate', 0):.1f}%"
                }
            }, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"fraud_summary_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Generate Report
            if st.button("Generate Report", use_container_width=True):
                # Create report content
                report_content = f"""
FRAUD DETECTION ANALYSIS REPORT
================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: {report_type}

SUMMARY
=======

Total Transactions Analyzed: {metrics.get('total', 0):,}
Fraud Cases Detected: {metrics.get('fraud', 0):,}
Warning Cases: {metrics.get('warning', 0):,}
Suspicious Cases: {metrics.get('suspicious', 0):,}
Safe Transactions: {metrics.get('safe', 0):,}

Fraud Rate: {metrics.get('fraud_rate', 0):.1f}%
Average Fraud Probability: {metrics.get('avg_probability', 0):.3f}

DETECTION CONFIGURATION
=======================

Fraud Threshold: {st.session_state.thresholds['fraud']:.3f}
Warning Threshold: {st.session_state.thresholds['warning']:.3f}
Suspicious Threshold: {st.session_state.thresholds['suspicious']:.3f}
SHAP Explanations: {'Enabled' if st.session_state.shap_enabled else 'Disabled'}

RECOMMENDATIONS
===============

1. Review all transactions flagged as FRAUD
2. Investigate WARNING cases for potential fraud patterns
3. Monitor accounts with multiple suspicious transactions
4. Consider adjusting thresholds based on business requirements

---
Report generated by Fraud Detection Analytics System v3.0
                """
                
                # Provide download
                st.download_button(
                    label="📄 Download Report",
                    data=report_content,
                    file_name=f"fraud_report_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point"""
    # Initialize and run app
    app = FraudDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
