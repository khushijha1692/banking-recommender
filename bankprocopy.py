import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import pytesseract
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

# Set up pytesseract path (adjust this for your system if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Store original data before preprocessing
    original_df = df.copy()
    
    if 'Bank Name' in df.columns:
        bank_names = df['Bank Name'].reset_index(drop=True)
        df.drop(columns=['Bank Name'], inplace=True)
    else:
        bank_names = None  
    
    # Store original column ranges before normalization
    original_ranges = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        original_ranges[col] = {'min': df[col].min(), 'max': df[col].max(), 'mean': df[col].mean()}
    
    # Continue with preprocessing
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(0, inplace=True)
    
    # Get categorical columns before creating dummies
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    df = pd.get_dummies(df, drop_first=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, bank_names, numeric_cols, original_ranges, categorical_cols, original_df

# Document scanning and processing functions
def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def extract_text_from_image(image):
    """Extract text from preprocessed image using OCR"""
    processed_img = preprocess_image(image)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(processed_img)
    return text

def extract_document_info(text):
    """Extract relevant banking information from OCR text"""
    # Initialize dictionary to store extracted data
    extracted_data = {}
    
    # Extract loan amount if present
    if "loan amount" in text.lower() or "amount" in text.lower():
        # Simple regex-like approach to find amounts
        lines = text.split('\n')
        for line in lines:
            if "loan amount" in line.lower() or "amount" in line.lower():
                # Try to find numbers in this line
                import re
                amounts = re.findall(r'[\d,]+', line)
                if amounts:
                    # Take the largest number as the loan amount
                    extracted_data['Loan Amount'] = float(amounts[-1].replace(',', ''))
    
    # Extract loan type if present
    loan_types = {
        "home": "Home Loan",
        "housing": "Home Loan",
        "car": "Car Loan",
        "auto": "Car Loan",
        "vehicle": "Car Loan",
        "personal": "Personal Loan",
        "business": "Business Loan",
        "enterprise": "Business Loan",
        "credit card": "Credit Card",
        "savings": "Savings Account",
        "checking": "Checking Account"
    }
    
    for keyword, loan_type in loan_types.items():
        if keyword.lower() in text.lower():
            extracted_data['Product Type'] = loan_type
            break
    
    # Extract interest rate if present
    if "interest" in text.lower() or "rate" in text.lower() or "apr" in text.lower():
        lines = text.split('\n')
        for line in lines:
            if "interest" in line.lower() or "rate" in line.lower() or "apr" in line.lower():
                import re
                rates = re.findall(r'\d+\.\d+%|\d+%', line)
                if rates:
                    # Extract the percentage and convert to float
                    rate_str = rates[0].replace('%', '')
                    extracted_data['Interest Rate'] = float(rate_str)
    
    # Extract credit card specific features
    if "cashback" in text.lower() or "cash back" in text.lower():
        lines = text.split('\n')
        for line in lines:
            if "cashback" in line.lower() or "cash back" in line.lower():
                import re
                cashback_rates = re.findall(r'\d+\.\d+%|\d+%', line)
                if cashback_rates:
                    rate_str = cashback_rates[0].replace('%', '')
                    extracted_data['Cashback Rate'] = float(rate_str)
    
    # Extract annual fee
    if "annual fee" in text.lower() or "yearly fee" in text.lower():
        lines = text.split('\n')
        for line in lines:
            if "annual fee" in line.lower() or "yearly fee" in line.lower():
                import re
                fees = re.findall(r'[\d,]+', line)
                if fees:
                    extracted_data['Annual Fee'] = float(fees[0].replace(',', ''))
    
    # Extract hidden fees or charges
    if "fee" in text.lower() or "charge" in text.lower():
        lines = text.split('\n')
        hidden_fees = []
        for line in lines:
            if "fee" in line.lower() or "charge" in line.lower():
                if not any(skip in line.lower() for skip in ["annual fee", "processing fee", "application fee"]):
                    hidden_fees.append(line.strip())
        if hidden_fees:
            extracted_data['Hidden Fees'] = hidden_fees
    
    return extracted_data

# Build MLP model
def build_mlp_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train MLP model
def train_mlp_model(df, feature_cols):
    X = df[feature_cols]
    y = df['Product Type'] if 'Product Type' in df.columns else np.random.rand(len(df))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_mlp_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    return model, feature_cols

# Normalize user input to match model's expected format
def normalize_user_input(user_input, df, original_ranges):
    normalized_input = {}
    
    for col, value in user_input.items():
        if col in original_ranges:
            # Normalize the value using min-max scaling
            min_val = original_ranges[col]['min']
            max_val = original_ranges[col]['max']
            if max_val > min_val:  # Avoid division by zero
                normalized_input[col] = (value - min_val) / (max_val - min_val)
            else:
                normalized_input[col] = 0.5  # Default mid-point if range is zero
        else:
            # For categorical or already normalized values
            normalized_input[col] = value
            
    return normalized_input

# Get bank features for display
def get_bank_features(bank_name, original_df):
    """Get original features for a bank to display to users"""
    if bank_name in original_df['Bank Name'].values:
        bank_data = original_df[original_df['Bank Name'] == bank_name].iloc[0].to_dict()
        # Remove bank name from dict
        if 'Bank Name' in bank_data:
            del bank_data['Bank Name']
        return bank_data
    return {}

# Recommend banks using MLP and return original bank data
def recommend_banks(user_preferences, df, bank_names, model, feature_cols, original_df, top_n=5):
    # Create a dataframe from user preferences for only the available feature columns
    available_features = [col for col in feature_cols if col in user_preferences]
    user_df = pd.DataFrame([{col: user_preferences.get(col, df[col].mean()) for col in feature_cols}])
    
    # Generate recommendations based on similarity scores
    similarity = cosine_similarity(user_df, df[feature_cols])
    top_indices = similarity.argsort()[0][-top_n:][::-1]
    recommended_banks = bank_names.iloc[top_indices].tolist()
    
    # Get similarity scores for recommendations
    scores = [similarity[0][i] for i in top_indices]
    
    # Get original bank data for display
    bank_details = []
    for bank in recommended_banks:
        details = get_bank_features(bank, original_df)
        bank_details.append(details)
    
    return recommended_banks, scores, bank_details

# Evaluate model
def evaluate_model(df, bank_names, model, feature_cols, original_df):
    train_df, test_df, train_banks, test_banks = train_test_split(df, bank_names, test_size=0.2, random_state=42)
    train_banks = train_banks.reset_index(drop=True)
    test_banks = test_banks.reset_index(drop=True)
    
    predictions = []
    for i, row in test_df.iterrows():
        user_pref = row.to_dict()
        rec, _, _ = recommend_banks(user_pref, train_df, train_banks, model, feature_cols, original_df, top_n=1)
        predictions.append(rec[0] if rec else None)
    
    accuracy = np.mean([1 if pred == actual else 0 for pred, actual in zip(predictions, test_banks) if pred is not None])
    return accuracy

# Main Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Smart Bank Recommender",
        page_icon="🏦",
        layout="wide"
    )
    
    # Custom CSS to improve appearance with better contrast
    st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        background-color: #e6e6e6;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5;
        color: white;
    }
    h1 {
        color: #0d47a1;
        font-size: 2.5rem;
        font-weight: 700;
    }
    h2, h3 {
        color: #1565c0;
        font-weight: 600;
    }
    p {
        color: #333333;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1e88e5;
    }
    .bank-name {
        color: #0d47a1;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .match-score {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .feature-label {
        color: #555555;
        font-weight: 600;
    }
    .feature-value {
        color: #1565c0;
        font-weight: 500;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    .product-type-button {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 2px solid #1565c0;
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s;
    }
    .product-type-button:hover {
        background-color: #bbdefb;
        cursor: pointer;
    }
    .product-type-button.selected {
        background-color: #1565c0;
        color: white;
    }
    .info-box {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    .section-header {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 15px 0 10px 0;
        color: #0d47a1;
        font-weight: 600;
    }
    .benefit-tag {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .fee-tag {
        display: inline-block;
        background-color: #ffebee;
        color: #c62828;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .expandable-section {
        cursor: pointer;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🏦")
    with col2:
        st.title("Find Your Perfect Bank")
        st.markdown("### Personalized bank and financial product recommendations based on your needs")
    
    # Load data
    filepath = "indian_banks_data2.csv"  # Updated to use relative path
    
    if not os.path.exists(filepath):
        st.error(f"Could not find the dataset at {filepath}. Please check the file path.")
        st.info("For demonstration purposes, we'll continue with sample data.")
        # Create enhanced sample data for demonstration that includes credit card and account info
        sample_df = pd.DataFrame({
            'Bank Name': ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Punjab National Bank', 'Bank of Baroda', 
                         'Yes Bank', 'Kotak Mahindra', 'IndusInd Bank', 'Union Bank'],
            'Product Type': ['Home Loan', 'Credit Card', 'Savings Account', 'Personal Loan', 'Credit Card', 'Business Loan',
                           'Credit Card', 'Savings Account', 'Home Loan', 'Credit Card'],
            'Interest Rate': [8.0, 26.5, 3.5, 9.5, 24.5, 8.0, 28.0, 4.0, 7.5, 25.0],
            'Loan Term (Years)': [5, None, None, 15, None, 5, None, None, 7, None],
            'Processing Fee (%)': [1.0, None, None, 1.0, None, 1.0, None, None, 0.75, None],
            'Loan Amount Min': [100000, None, None, 300000, None, 100000, None, None, 500000, None],
            'Loan Amount Max': [5000000, None, None, 8000000, None, 5000000, None, None, 10000000, None],
            'Approval Time (Days)': [7, 3, 1, 4, 2, 6, 2, 1, 5, 3],
            'Customer Rating': [4.2, 4.5, 4.0, 4.0, 3.8, 4.1, 3.7, 4.7, 4.3, 3.5],
            'Annual Fee': [None, 499, None, None, 999, None, 1499, None, None, 749],
            'Cashback Rate (%)': [None, 2.5, None, None, 1.5, None, 3.0, None, None, 1.0],
            'Reward Points': [None, 'Yes', None, None, 'Yes', None, 'Yes', None, None, 'Yes'],
            'Airport Lounge Access': [None, 'Yes', None, None, 'No', None, 'Yes', None, None, 'No'],
            'Minimum Balance (₹)': [None, None, 5000, None, None, None, None, 10000, None, None],
            'Savings Interest (%)': [None, None, 3.5, None, None, None, None, 4.0, None, None],
            'ATM Fee (₹)': [None, None, 20, None, None, None, None, 0, None, None],
            'Net Banking Fee': [None, None, 'Free', None, None, None, None, 'Free', None, None],
            'Hidden Fees': [None, 'Late Payment, Foreign Transaction', 'Account Closure', None, 
                           'Card Replacement, Cash Advance', None, 'Balance Enquiry, Statement Request', 
                           'SMS Alerts', None, 'Over-limit, Cash Withdrawal']
        })
        sample_df.to_csv(filepath, index=False)
    
    df, bank_names, feature_cols, original_ranges, categorical_cols, original_df = load_and_preprocess_data(filepath)
    mlp_model, feature_cols = train_mlp_model(df, feature_cols)
    model_accuracy = evaluate_model(df, bank_names, mlp_model, feature_cols, original_df)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Find Your Match", "💳 Credit Cards", "🏦 Bank Accounts", "📄 Document Scanner"])
    
    with tab1:
        # Use columns for better layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="section-header">Your Banking Preferences</div>', unsafe_allow_html=True)
            
            # Container for preferences with better styling
            with st.container():
                user_input = {}
                
                # Product Type Selection with clear buttons
                st.markdown("#### What type of banking product are you looking for?")
                product_options = ['Home Loan', 'Car Loan', 'Personal Loan', 'Business Loan', 'Credit Card', 'Savings Account']
                
                # Initialize session state for selected product
                if 'selected_product' not in st.session_state:
                    st.session_state.selected_product = None
                
                # First row of product type buttons
                product_cols = st.columns(2)
                if product_cols[0].button("🏠 Home Loan", use_container_width=True, 
                                    key="home_loan", 
                                    help="For buying, renovating, or constructing a house"):
                    st.session_state.selected_product = "Home Loan"
                    
                if product_cols[1].button("🚗 Car Loan", use_container_width=True,
                                     key="car_loan",
                                     help="For purchasing a new or used vehicle"):
                    st.session_state.selected_product = "Car Loan"
                
                # Second row of product type buttons
                product_cols = st.columns(2)
                if product_cols[0].button("👤 Personal Loan", use_container_width=True,
                                     key="personal_loan",
                                     help="For personal expenses, education, medical, etc."):
                    st.session_state.selected_product = "Personal Loan"
                    
                if product_cols[1].button("💼 Business Loan", use_container_width=True,
                                     key="business_loan",
                                     help="For starting or expanding a business"):
                    st.session_state.selected_product = "Business Loan"
                
                # Third row of product type buttons
                product_cols = st.columns(2)
                if product_cols[0].button("💳 Credit Card", use_container_width=True,
                                     key="credit_card",
                                     help="Find credit cards with the best benefits and lowest fees"):
                    st.session_state.selected_product = "Credit Card"
                    
                if product_cols[1].button("💰 Savings Account", use_container_width=True,
                                     key="savings_account",
                                     help="For saving money with interest"):
                    st.session_state.selected_product = "Savings Account"
                
                # Show selected product with clear visual feedback
                if st.session_state.selected_product:
                    st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; 
                                margin-top: 10px; border-left: 5px solid #43a047;">
                        <p style="color: #2e7d32; font-weight: 600; margin: 0;">
                            ✅ Selected: {st.session_state.selected_product}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'Product Type' in df.columns:
                        user_input['Product Type'] = product_options.index(st.session_state.selected_product) / len(product_options)
                else:
                    st.markdown("""
                    <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; 
                                margin-top: 10px; border-left: 5px solid #ff9800;">
                        <p style="color: #e65100; font-weight: 600; margin: 0;">
                            Please select a product type to continue
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display form fields based on selected product type
                if st.session_state.selected_product:
                    st.markdown('<div class="section-header">Financial Details</div>', unsafe_allow_html=True)
                    
                    # Show different form fields based on product type
                    if st.session_state.selected_product in ['Home Loan', 'Car Loan', 'Personal Loan', 'Business Loan']:
                        # Interest Rate (actual percentage)
                        min_rate = 5.0  # Example min interest rate
                        max_rate = 15.0  # Example max interest rate
                        if 'Interest Rate' in original_ranges:
                            min_rate = original_ranges['Interest Rate']['min']
                            max_rate = original_ranges['Interest Rate']['max']
                        
                        interest_rate = st.slider(
                            "Interest Rate (%)", 
                            min_value=float(min_rate), 
                            max_value=float(max_rate), 
                            value=float((min_rate + max_rate) / 2),
                            step=0.1,
                            format="%.1f%%"
                        )
                        user_input['Interest Rate'] = interest_rate
                        
                        # Loan Amount with better formatting
                        if 'Loan Amount Min' in original_df.columns:
                            min_amount = original_df['Loan Amount Min'].min()
                            max_amount = original_df['Loan Amount Max'].max()
                            
                            # Format with lakhs for better user understanding
                            loan_amount = st.slider(
                                "Loan Amount (₹)", 
                                min_value=int(min_amount), 
                                max_value=int(max_amount), 
                                value=int((min_amount + max_amount) / 2),
                                step=50000,
                                format="%d"
                            )
                            # Show as lakhs for Indian users
                            st.markdown(f"<p style='text-align: right; color: #1976d2;'>= ₹{loan_amount/100000:.2f} Lakhs</p>", unsafe_allow_html=True)
                            user_input['Loan Amount Min'] = loan_amount
                            user_input['Loan Amount Max'] = loan_amount
                        
                        # Loan Term in years (not normalized)
                        if 'Loan Term (Years)' in original_df.columns:
                            min_term = int(original_df['Loan Term (Years)'].min())
                            max_term = int(original_df['Loan Term (Years)'].max())
                            loan_term = st.slider(
                                "Loan Term", 
                                min_value=min_term, 
                                max_value=max_term, 
                                value=int((min_term + max_term) / 2),
                                format="%d years"
                            )
                            user_input['Loan Term (Years)'] = loan_term
                        
                        # Processing Fee as percentage
                        if 'Processing Fee (%)' in original_df.columns:
                            min_fee = original_df['Processing Fee (%)'].min()
                            max_fee = original_df['Processing Fee (%)'].max()
                            fee = st.slider(
                                "Processing Fee", 
                                min_value=float(min_fee), 
                                max_value=float(max_fee), 
                                value=float((min_fee + max_fee) / 2),
                                step=0.1,
                                format="%.1f%%"
                            )
                            user_input['Processing Fee (%)'] = fee
                    
                    elif st.session_state.selected_product == "Credit Card":
                        # Credit card specific preferences
                        # Annual Fee
                        if 'Annual Fee' in original_df.columns:
                            annual_fee_options = ["No Annual Fee", "Up to ₹500", "₹500-₹1500", "₹1500+"]
                            annual_fee = st.selectbox("Annual Fee Preference", annual_fee_options)
                            
                            # Map selection to numerical values
                            fee_mapping = {
                                "No Annual Fee": 0,
                                "Up to ₹500": 250,
                                "₹500-₹1500": 1000,
                                "₹1500+": 2000
                            }
                            user_input['Annual Fee'] = fee_mapping[annual_fee]
                        
                        # Cashback preference
                        if 'Cashback Rate (%)' in original_df.columns:
                            max_cashback = original_df['Cashback Rate (%)'].max()
                            cashback = st.slider(
                                "Minimum Cashback Rate", 
                                min_value=0.0, 
                                max_value=float(max_cashback) if not pd.isna(max_cashback) else 5.0, 
                                value=1.0,
                                step=0.5,
                                format="%.1f%%"
                            )
                            user_input['Cashback Rate (%)'] = cashback
                        
                        # Reward points
                        user_input['Reward Points'] = st.checkbox("I want reward points", value=True)
                        
                        # Airport lounge access
                        user_input['Airport Lounge Access'] = st.checkbox("Airport lounge access", value=False)
                        
                        # Interest rate for credit cards (APR)
                        if 'Interest Rate' in original_df.columns:
                            # Filter for only credit card rates
                            credit_card_df = original_df[original_df['Product Type'] == 'Credit Card']
                            if not credit_card_df.empty:
                                min_rate = credit_card_df['Interest Rate'].min()
                                max_rate = credit_card_df['Interest Rate'].max()
                            else:
                                min_rate = 18.0  # Typical credit card minimum APR
                                max_rate = 36.0  # Typical credit card maximum APR
                            
                            interest_rate = st.slider(
                                "Maximum APR (%)", 
                                min_value=float(min_rate), 
                                max_value=float(max_rate), 
                                value=float(min_rate + 2),  # Default to lower APR
                                step=0.5,
                                format="%.1f%%"
                            )
                            user_input['Interest Rate'] = interest_rate
                    
                    elif st.session_state.selected_product == "Savings Account":
                        # Savings account specific preferences
                        # Minimum balance
                        if 'Minimum Balance (₹)' in original_df.columns:
                            min_balance_options = ["Zero Balance", "Up to ₹1000", "₹1000-₹5000", "₹5000+"]
                            min_balance = st.selectbox("Minimum Balance Requirement", min_balance_options)
                            
                            # Map selection to numerical values
                            balance_mapping = {
                                "Zero Balance": 0,
                                "Up to ₹1000": 500,
                                "₹1000-₹5000": 2500,
                                "₹5000+": 7500
                            }
                            user_input['Minimum Balance (₹)'] = balance_mapping[min_balance]
                        
                      # Savings interest rate preference
                        if 'Savings Interest (%)' in original_df.columns:
                            savings_interest_rates = original_df['Savings Interest (%)'].dropna()
                            if not savings_interest_rates.empty:
                                min_rate = savings_interest_rates.min()
                                max_rate = savings_interest_rates.max()
                            else:
                                min_rate = 2.0  # Default minimum savings interest rate
                                max_rate = 6.0  # Default maximum savings interest rate
                            
                            interest_rate = st.slider(
                                "Minimum Savings Interest Rate (%)", 
                                min_value=float(min_rate), 
                                max_value=float(max_rate), 
                                value=float(min_rate),
                                step=0.1,
                                format="%.1f%%"
                            )
                            user_input['Savings Interest (%)'] = interest_rate
                        
                        # ATM Fee preference
                        if 'ATM Fee (₹)' in original_df.columns:
                            atm_fee_options = ["No ATM Fees", "Up to ₹10 per transaction", "Up to ₹20 per transaction", "Any fee structure"]
                            atm_fee = st.selectbox("ATM Fee Preference", atm_fee_options)
                            
                            # Map selection to numerical values
                            fee_mapping = {
                                "No ATM Fees": 0,
                                "Up to ₹10 per transaction": 10,
                                "Up to ₹20 per transaction": 20,
                                "Any fee structure": 50
                            }
                            user_input['ATM Fee (₹)'] = fee_mapping[atm_fee]
                        
                        # Net banking fee preference
                        user_input['Net Banking Fee'] = st.checkbox("Free Net Banking", value=True)
                    
                    # Common preferences for all product types
                    st.markdown('<div class="section-header">Other Preferences</div>', unsafe_allow_html=True)
                    
                    # Customer Rating
                    if 'Customer Rating' in original_df.columns:
                        min_rating = original_df['Customer Rating'].min()
                        max_rating = original_df['Customer Rating'].max()
                        rating = st.slider(
                            "Minimum Customer Rating", 
                            min_value=float(min_rating), 
                            max_value=float(max_rating), 
                            value=3.5,
                            step=0.1
                        )
                        user_input['Customer Rating'] = rating
                    
                    # Approval Time (Days) with a more user-friendly selector
                    if 'Approval Time (Days)' in original_df.columns:
                        approval_time_options = [
                            "Same day (1 day)", 
                            "Quick (2-3 days)", 
                            "Standard (4-7 days)", 
                            "Any duration"
                        ]
                        approval_preference = st.selectbox(
                            "Approval Time Preference", 
                            approval_time_options,
                            index=1  # Default to "Quick"
                        )
                        
                        # Map selection to numerical value
                        time_mapping = {
                            "Same day (1 day)": 1,
                            "Quick (2-3 days)": 3,
                            "Standard (4-7 days)": 7,
                            "Any duration": 10
                        }
                        user_input['Approval Time (Days)'] = time_mapping[approval_preference]
                
                # Submit button for recommendations
                if st.session_state.selected_product and st.button("🔍 Find My Best Match", key="find_match", 
                                                           use_container_width=True,
                                                           help="Submit your preferences to find the best matches"):
                    # Store user input in session state for recommendations
                    st.session_state.user_preferences = user_input
                    st.session_state.recommendations_made = True
        
        with col2:
            if hasattr(st.session_state, 'recommendations_made') and st.session_state.recommendations_made:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #2e7d32; margin: 0 0 10px 0;">Top Recommended Banks for {st.session_state.selected_product}</h3>
                    <p>Based on your preferences, we've found these top matches for you.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get recommendations
                recs, scores, details = recommend_banks(
                    st.session_state.user_preferences, 
                    df, 
                    bank_names, 
                    mlp_model, 
                    feature_cols, 
                    original_df,
                    top_n=3
                )
                
                # Display recommendations in styled cards
                for i, (bank, score, detail) in enumerate(zip(recs, scores, details)):
                    score_percentage = int(score * 100)
                    
                    # Start the card
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="bank-name">{i+1}. {bank}</div>
                        <div class="match-score">Match Score: {score_percentage}%</div>
                        <div class="feature-grid">
                    """, unsafe_allow_html=True)
                    
                    # Display relevant features based on product type
                    display_features = []
                    if st.session_state.selected_product in ['Home Loan', 'Car Loan', 'Personal Loan', 'Business Loan']:
                        loan_features = [
                            ('Interest Rate', 'Interest Rate (%)', lambda x: f"{x:.2f}%"),
                            ('Loan Term (Years)', 'Term', lambda x: f"{x} years"),
                            ('Processing Fee (%)', 'Processing Fee', lambda x: f"{x:.1f}%"),
                            ('Approval Time (Days)', 'Approval Time', lambda x: f"{x} days"),
                        ]
                        display_features.extend(loan_features)
                        
                    elif st.session_state.selected_product == 'Credit Card':
                        card_features = [
                            ('Interest Rate', 'APR', lambda x: f"{x:.2f}%"),
                            ('Annual Fee', 'Annual Fee', lambda x: f"₹{x}" if x > 0 else "No Fee"),
                            ('Cashback Rate (%)', 'Cashback', lambda x: f"{x:.1f}%" if x > 0 else "No Cashback"),
                            ('Reward Points', 'Rewards', lambda x: "Yes" if x == 'Yes' else "No"),
                            ('Airport Lounge Access', 'Airport Lounge', lambda x: x),
                        ]
                        display_features.extend(card_features)
                        
                    elif st.session_state.selected_product == 'Savings Account':
                        account_features = [
                            ('Savings Interest (%)', 'Interest Rate', lambda x: f"{x:.2f}%"),
                            ('Minimum Balance (₹)', 'Min. Balance', lambda x: f"₹{x}" if x > 0 else "Zero Balance"),
                            ('ATM Fee (₹)', 'ATM Fee', lambda x: f"₹{x} per txn" if x > 0 else "Free"),
                            ('Net Banking Fee', 'Net Banking', lambda x: x),
                        ]
                        display_features.extend(account_features)
                    
                    # Always display customer rating
                    display_features.append(('Customer Rating', 'Rating', lambda x: f"{x:.1f}★"))
                    
                    # Render the features in the grid
                    for key, label, formatter in display_features:
                        if key in detail and not pd.isna(detail[key]):
                            formatted_value = formatter(detail[key])
                            st.markdown(f"""
                            <div>
                                <span class="feature-label">{label}:</span>
                                <span class="feature-value">{formatted_value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Close the feature grid
                    st.markdown("""</div>""", unsafe_allow_html=True)
                    
                    # Display benefits and fees sections
                    benefits = []
                    fees = []
                    
                    # Add product-specific benefits
                    if st.session_state.selected_product == 'Credit Card':
                        if 'Cashback Rate (%)' in detail and detail['Cashback Rate (%)'] > 2:
                            benefits.append(f"High Cashback Rate ({detail['Cashback Rate (%)']}%)")
                        if 'Reward Points' in detail and detail['Reward Points'] == 'Yes':
                            benefits.append("Reward Points")
                        if 'Airport Lounge Access' in detail and detail['Airport Lounge Access'] == 'Yes':
                            benefits.append("Airport Lounge Access")
                        if 'Annual Fee' in detail and detail['Annual Fee'] == 0:
                            benefits.append("No Annual Fee")
                    
                    elif st.session_state.selected_product in ['Home Loan', 'Car Loan', 'Personal Loan', 'Business Loan']:
                        if 'Interest Rate' in detail and detail['Interest Rate'] < 8:
                            benefits.append(f"Low Interest Rate ({detail['Interest Rate']}%)")
                        if 'Processing Fee (%)' in detail and detail['Processing Fee (%)'] < 1:
                            benefits.append(f"Low Processing Fee ({detail['Processing Fee (%)']}%)")
                        if 'Approval Time (Days)' in detail and detail['Approval Time (Days)'] <= 3:
                            benefits.append(f"Fast Approval ({detail['Approval Time (Days)']} days)")
                    
                    elif st.session_state.selected_product == 'Savings Account':
                        if 'Savings Interest (%)' in detail and detail['Savings Interest (%)'] > 3.5:
                            benefits.append(f"High Interest Rate ({detail['Savings Interest (%)']}%)")
                        if 'Minimum Balance (₹)' in detail and detail['Minimum Balance (₹)'] == 0:
                            benefits.append("Zero Balance Account")
                        if 'ATM Fee (₹)' in detail and detail['ATM Fee (₹)'] == 0:
                            benefits.append("Free ATM Transactions")
                        if 'Net Banking Fee' in detail and detail['Net Banking Fee'] == 'Free':
                            benefits.append("Free Net Banking")
                    
                    # Extract fees
                    if 'Hidden Fees' in detail and not pd.isna(detail['Hidden Fees']):
                        if isinstance(detail['Hidden Fees'], str):
                            fees_list = detail['Hidden Fees'].split(', ')
                            fees.extend(fees_list)
                    
                    # Display benefits if any
                    if benefits:
                        st.markdown("""<div style="margin-top: 15px;"><strong>Key Benefits:</strong></div>""", unsafe_allow_html=True)
                        benefits_html = ""
                        for benefit in benefits:
                            benefits_html += f'<span class="benefit-tag">{benefit}</span> '
                        st.markdown(f"""<div style="margin-top: 5px;">{benefits_html}</div>""", unsafe_allow_html=True)
                    
                    # Display fees if any
                    if fees:
                        st.markdown("""<div style="margin-top: 15px;"><strong>Additional Fees:</strong></div>""", unsafe_allow_html=True)
                        fees_html = ""
                        for fee in fees:
                            fees_html += f'<span class="fee-tag">{fee}</span> '
                        st.markdown(f"""<div style="margin-top: 5px;">{fees_html}</div>""", unsafe_allow_html=True)
                    
                    # Close the card
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                # Add apply now button for each bank (won't actually do anything in this demo)
                col1, col2, col3 = st.columns(3)
                for i, col in enumerate([col1, col2, col3]):
                    if i < len(recs):
                        if col.button(f"Apply Now: {recs[i]}", key=f"apply_{i}", use_container_width=True):
                            st.success(f"Application process initiated for {recs[i]}! This is a demo, so no actual application will be processed.")
                
                # Disclaimer
                st.markdown("""
                <div style="margin-top: 30px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 0.8rem;">
                    <p style="margin: 0; color: #666666;">
                        <strong>Disclaimer:</strong> The recommendations provided are based on your preferences and our algorithm's analysis.
                        Actual loan terms, interest rates, and other features may vary. Always read the full terms and conditions before applying.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Welcome message with tips
                st.markdown("""
                <div style="height: 100%; display: flex; flex-direction: column; justify-content: center; text-align: center; padding: 20px;">
                    <div style="font-size: 64px; margin-bottom: 20px;">👈</div>
                    <h3>Welcome to Smart Bank Recommender</h3>
                    <p>Select your product type and preferences to get personalized bank recommendations.</p>
                    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: left;">
                        <h4 style="margin-top: 0;">Tips for finding your perfect match:</h4>
                        <ul>
                            <li>Be specific about your product needs (Home Loan, Credit Card, etc.)</li>
                            <li>Set realistic interest rate and loan amount expectations</li>
                            <li>Consider fees and additional features that matter most to you</li>
                            <li>Check customer ratings for service quality insights</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## Credit Card Comparison")
        
        st.markdown("""
        <div class="info-box">
            Compare different credit cards to find the best match for your lifestyle and spending habits.
        </div>
        """, unsafe_allow_html=True)
        
        # Filter for credit cards
        credit_cards_df = original_df[original_df['Product Type'] == 'Credit Card'] if 'Product Type' in original_df.columns else pd.DataFrame()
        
        if credit_cards_df.empty:
            st.warning("No credit card data available. Using sample data for demonstration.")
            # Generate sample credit card data
            credit_cards_df = pd.DataFrame({
                'Bank Name': ['HDFC Bank', 'ICICI Bank', 'SBI Card', 'Axis Bank', 'Kotak Mahindra'],
                'Card Name': ['Regalia', 'Rubyx', 'SimplySAVE', 'Neo', 'Royale'],
                'Annual Fee': [2500, 3000, 499, 1500, 2000],
                'Interest Rate': [35.88, 36.0, 39.0, 40.0, 38.0],
                'Cashback Rate (%)': [1.5, 2.0, 5.0, 1.0, 2.5],
                'Reward Points': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
                'Fuel Surcharge Waiver': ['Yes', 'Yes', 'No', 'Yes', 'No'],
                'Airport Lounge Access': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
                'Welcome Benefits': ['10000 points', '5000 points', '₹500 cashback', '8000 points', '₹1000 voucher'],
                'Foreign Transaction Fee (%)': [3.5, 3.0, 3.5, 3.25, 3.5],
                'Joining Fee': [1000, 2000, 0, 750, 1500],
                'Card Type': ['Premium', 'Premium', 'Basic', 'Standard', 'Premium'],
                'Customer Rating': [4.5, 4.3, 4.0, 4.2, 4.1]
            })
        
        # Create credit card filter options
        st.markdown('<div class="section-header">Filter Credit Cards</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Annual fee filter
            fee_options = ["All", "No Annual Fee", "Up to ₹1000", "₹1000-₹2500", "₹2500+"]
            fee_filter = st.selectbox("Annual Fee", fee_options)
            
            # Rewards filter
            reward_filter = st.multiselect(
                "Features", 
                ["Cashback", "Reward Points", "Airport Lounge", "Fuel Surcharge Waiver"],
                default=["Cashback", "Reward Points"]
            )
        
        with col2:
            # Card type filter
            card_types = ["All"] + list(credit_cards_df['Card Type'].unique()) if 'Card Type' in credit_cards_df.columns else ["All", "Basic", "Standard", "Premium"]
            card_type_filter = st.selectbox("Card Type", card_types)
            
            # Sort by options
            sort_by = st.selectbox(
                "Sort By", 
                ["Customer Rating", "Annual Fee (Low to High)", "Cashback Rate (High to Low)"]
            )
        
        # Filter the dataframe based on user selections
        filtered_cards = credit_cards_df.copy()
        
        # Apply annual fee filter
        if fee_filter != "All":
            if fee_filter == "No Annual Fee":
                filtered_cards = filtered_cards[filtered_cards['Annual Fee'] == 0]
            elif fee_filter == "Up to ₹1000":
                filtered_cards = filtered_cards[filtered_cards['Annual Fee'] <= 1000]
            elif fee_filter == "₹1000-₹2500":
                filtered_cards = filtered_cards[(filtered_cards['Annual Fee'] > 1000) & (filtered_cards['Annual Fee'] <= 2500)]
            else:  # ₹2500+
                filtered_cards = filtered_cards[filtered_cards['Annual Fee'] > 2500]
        
        # Apply rewards filter
        if "Cashback" in reward_filter:
            filtered_cards = filtered_cards[filtered_cards['Cashback Rate (%)'] > 0]
        if "Reward Points" in reward_filter:
            filtered_cards = filtered_cards[filtered_cards['Reward Points'] == 'Yes']
        if "Airport Lounge" in reward_filter:
            filtered_cards = filtered_cards[filtered_cards['Airport Lounge Access'] == 'Yes']
        if "Fuel Surcharge Waiver" in reward_filter:
            filtered_cards = filtered_cards[filtered_cards['Fuel Surcharge Waiver'] == 'Yes']
        
        # Apply card type filter
        if card_type_filter != "All":
            filtered_cards = filtered_cards[filtered_cards['Card Type'] == card_type_filter]
        
        # Apply sorting
        if sort_by == "Customer Rating":
            filtered_cards = filtered_cards.sort_values('Customer Rating', ascending=False)
        elif sort_by == "Annual Fee (Low to High)":
            filtered_cards = filtered_cards.sort_values('Annual Fee', ascending=True)
        elif sort_by == "Cashback Rate (High to Low)":
            filtered_cards = filtered_cards.sort_values('Cashback Rate (%)', ascending=False)
        
        # Display the filtered credit cards in a more attractive format
        if not filtered_cards.empty:
            st.markdown('<div class="section-header">Matching Credit Cards</div>', unsafe_allow_html=True)
            
            for i, (_, card) in enumerate(filtered_cards.iterrows()):
                # Create a styled credit card display
                card_html = f"""
                <div style="background: linear-gradient(135deg, #1a237e, #283593, #303f9f); color: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0; color: white;">{card['Bank Name']} {card['Card Name']}</h3>
                            <div style="background-color: rgba(255,255,255,0.2); display: inline-block; padding: 3px 10px; border-radius: 10px; margin-top: 5px;">
                                {card['Card Type']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{'₹' + str(int(card['Annual Fee'])) if card['Annual Fee'] > 0 else 'No Fee'}</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Annual Fee</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Cashback</div>
                            <div>{card['Cashback Rate (%)']}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">APR</div>
                            <div>{card['Interest Rate']}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Rewards</div>
                            <div>{card['Reward Points']}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Rating</div>
                            <div>{card['Customer Rating']}★</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Features:</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                """
                
                # Add feature badges
                features = []
                if card['Airport Lounge Access'] == 'Yes':
                    features.append("Airport Lounge")
                if card['Fuel Surcharge Waiver'] == 'Yes':
                    features.append("Fuel Surcharge Waiver")
                if 'Foreign Transaction Fee (%)' in card and card['Foreign Transaction Fee (%)'] < 3.0:
                    features.append("Low Forex Fee")
                if 'Welcome Benefits' in card and not pd.isna(card['Welcome Benefits']):
                    features.append("Welcome Gift")
                
                for feature in features:
                    card_html += f'<div style="background-color: rgba(255,255,255,0.15); padding: 3px 10px; border-radius: 10px; font-size: 0.8rem;">{feature}</div>'
                
                # Close all open divs
                card_html += """
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Apply button for each card
                if st.button(f"Apply for {card['Bank Name']} {card['Card Name']}", key=f"apply_card_{i}"):
                    st.success(f"Application process initiated for {card['Bank Name']} {card['Card Name']}! This is a demo, so no actual application will be processed.")
        else:
            st.warning("No credit cards match your filter criteria. Try adjusting your filters.")
    
    with tab3:
        st.markdown("## Bank Account Comparison")
        
        st.markdown("""
        <div class="info-box">
            Compare savings and current accounts from different banks to find the best fit for your needs.
        </div>
        """, unsafe_allow_html=True)
        
        # Filter for bank accounts
        accounts_df = original_df[original_df['Product Type'] == 'Savings Account'] if 'Product Type' in original_df.columns else pd.DataFrame()
        
        if accounts_df.empty:
            st.warning("No bank account data available. Using sample data for demonstration.")
            # Generate sample bank account data
            accounts_df = pd.DataFrame({
                'Bank Name': ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Kotak Mahindra', 'Bank of Baroda'],
                'Account Type': ['Regular Savings', 'Premium Savings', 'Senior Citizen', 'Regular Savings', 'Digital Savings', 'Rural Savings'],
                'Minimum Balance (₹)': [1000, 10000, 0, 5000, 0, 2000],
                'Savings Interest (%)': [2.70, 3.00, 3.50, 3.10, 4.00, 2.90],
                'ATM Fee (₹)': [20, 0, 10, 15, 5, 0],
                'Debit Card Annual Fee': [150, 0, 100, 0, 199, 0],
                'Net Banking Fee': ['Free', 'Free', 'Free', 'Free', 'Free', 'Free'],
                'Mobile Banking': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
                'Free Transactions': [5, 10, 8, 7, 999, 6],
                'Digital Access': ['Basic', 'Advanced', 'Basic', 'Advanced', 'Advanced', 'Basic'],
                'Branches': [22000, 5000, 5300, 4500, 1600, 9500],
                'ATM Network': [58000, 13000, 14000, 11000, 2500, 13500],
                'IMPS/RTGS/NEFT Fee': ['Free', 'Free', 'Paid', 'Free', 'Free', 'Paid'],
                'Customer Rating': [3.8, 4.5, 4.2, 4.0, 4.7, 3.7]
            })
        
        # Create account filter options
        st.markdown('<div class="section-header">Filter Bank Accounts</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Minimum balance filter
            balance_options = ["All", "Zero Balance", "Up to ₹1000", "₹1000-₹5000", "₹5000+"]
            balance_filter = st.selectbox("Minimum Balance", balance_options)
            
            # Interest rate filter
            min_interest = float(accounts_df['Savings Interest (%)'].min()) if not accounts_df.empty else 2.5
            max_interest = float(accounts_df['Savings Interest (%)'].max()) if not accounts_df.empty else 4.5
            interest_filter = st.slider(
                "Minimum Interest Rate", 
                min_value=min_interest, 
                max_value=max_interest,
                value=min_interest,
                step=0.1,
                format="%.1f%%"
            )
        
        with col2:
            # Transaction filter
            transaction_options = ["All", "Free Digital Transactions", "5+ Free ATM Transactions", "No ATM Fees"]
            transaction_filter = st.selectbox("Transaction Benefits", transaction_options)
            
            # Sort by options
            sort_by = st.selectbox(
                "Sort By", 
                ["Interest Rate (High to Low)", "Minimum Balance (Low to High)", "Customer Rating"]
            )
        
        # Filter the dataframe based on user selections
        filtered_accounts = accounts_df.copy()
        
        # Apply minimum balance filter
        if balance_filter != "All":
            if balance_filter == "Zero Balance":
                filtered_accounts = filtered_accounts[filtered_accounts['Minimum Balance (₹)'] == 0]
            elif balance_filter == "Up to ₹1000":
                filtered_accounts = filtered_accounts[filtered_accounts['Minimum Balance (₹)'] <= 1000]
            elif balance_filter == "₹1000-₹5000":
                filtered_accounts = filtered_accounts[(filtered_accounts['Minimum Balance (₹)'] > 1000) & (filtered_accounts['Minimum Balance (₹)'] <= 5000)]
            else:  # ₹5000+
                filtered_accounts = filtered_accounts[filtered_accounts['Minimum Balance (₹)'] > 5000]
        
        # Apply interest rate filter
        filtered_accounts = filtered_accounts[filtered_accounts['Savings Interest (%)'] >= interest_filter]
        
        # Apply transaction filter
        if transaction_filter != "All":
            if transaction_filter == "Free Digital Transactions":
                filtered_accounts = filtered_accounts[
                    (filtered_accounts['Net Banking Fee'] == 'Free') | 
                    (filtered_accounts['IMPS/RTGS/NEFT Fee'] == 'Free')
                ]
            elif transaction_filter == "5+ Free ATM Transactions":
                filtered_accounts = filtered_accounts[filtered_accounts['Free Transactions'] >= 5]
            elif transaction_filter == "No ATM Fees":
                filtered_accounts = filtered_accounts[filtered_accounts['ATM Fee (₹)'] == 0]
        
        # Apply sorting
        if sort_by == "Interest Rate (High to Low)":
            filtered_accounts = filtered_accounts.sort_values('Savings Interest (%)', ascending=False)
        elif sort_by == "Minimum Balance (Low to High)":
            filtered_accounts = filtered_accounts.sort_values('Minimum Balance (₹)', ascending=True)
        elif sort_by == "Customer Rating":
            filtered_accounts = filtered_accounts.sort_values('Customer Rating', ascending=False)
        
        # Display the filtered accounts in a more attractive format
        if not filtered_accounts.empty:
            st.markdown('<div class="section-header">Matching Bank Accounts</div>', unsafe_allow_html=True)
            
            for i, (_, account) in enumerate(filtered_accounts.iterrows()):
                # Create a styled account display
                account_html = f"""
                <div style="background: linear-gradient(135deg, #00695c, #00897b, #009688); color: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space
                    <div style="background: linear-gradient(135deg, #00695c, #00897b, #009688); color: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0; color: white;">{account['Bank Name']} {account['Account Type']}</h3>
                            <div style="background-color: rgba(255,255,255,0.2); display: inline-block; padding: 3px 10px; border-radius: 10px; margin-top: 5px;">
                                {account['Digital Access']} Account
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{account['Savings Interest (%)']}%</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Interest Rate</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Min. Balance</div>
                            <div>{'₹' + str(int(account['Minimum Balance (₹)'])) if account['Minimum Balance (₹)'] > 0 else 'Zero Balance'}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">ATM Fee</div>
                            <div>{'₹' + str(account['ATM Fee (₹)']) if account['ATM Fee (₹)'] > 0 else 'Free'}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Free Transactions</div>
                            <div>{account['Free Transactions']}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Rating</div>
                            <div>{account['Customer Rating']}★</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Features:</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                """
                
                # Add feature badges
                features = []
                if account['Net Banking Fee'] == 'Free':
                    features.append("Free Net Banking")
                if account['Mobile Banking'] == 'Yes':
                    features.append("Mobile Banking")
                if account['IMPS/RTGS/NEFT Fee'] == 'Free':
                    features.append("Free IMPS/NEFT")
                if account['Debit Card Annual Fee'] == 0:
                    features.append("Free Debit Card")
                
                for feature in features:
                    account_html += f'<div style="background-color: rgba(255,255,255,0.15); padding: 3px 10px; border-radius: 10px; font-size: 0.8rem;">{feature}</div>'
                
                # Add bank network info
                account_html += f"""
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; font-size: 0.8rem; opacity: 0.8;">
                        <span>{account['Branches']} Branches</span> • <span>{account['ATM Network']} ATMs</span>
                    </div>
                </div>
                """
                
                st.markdown(account_html, unsafe_allow_html=True)
                
                # Apply button for each account
                if st.button(f"Open Account with {account['Bank Name']}", key=f"open_account_{i}"):
                    st.success(f"Account opening process initiated for {account['Bank Name']} {account['Account Type']}! This is a demo, so no actual application will be processed.")
        else:
            st.warning("No bank accounts match your filter criteria. Try adjusting your filters.")
    
    with tab4:
        st.markdown("## Loan Comparison")
        
        st.markdown("""
        <div class="info-box">
            Compare loan offers from different banks to find the best interest rates and terms.
        </div>
        """, unsafe_allow_html=True)
        
        # Create loan type selector
        loan_types = ["Home Loan", "Car Loan", "Personal Loan", "Business Loan"]
        selected_loan_type = st.selectbox("Select Loan Type", loan_types)
        
        # Filter for loan data
        loans_df = original_df[original_df['Product Type'] == selected_loan_type] if 'Product Type' in original_df.columns else pd.DataFrame()
        
        if loans_df.empty:
            st.warning(f"No {selected_loan_type} data available. Using sample data for demonstration.")
            # Generate sample loan data based on selected type
            if selected_loan_type == "Home Loan":
                loans_df = pd.DataFrame({
                    'Bank Name': ['SBI', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'LIC Housing'],
                    'Interest Rate': [6.95, 7.00, 7.10, 7.20, 6.90],
                    'Processing Fee (%)': [0.35, 0.50, 0.50, 0.40, 0.30],
                    'Loan Term (Years)': [20, 20, 20, 20, 20],
                    'Max Loan Amount (Lakhs)': [90, 100, 90, 85, 75],
                    'Min Income Required (Lakhs)': [3.0, 5.0, 4.0, 4.5, 3.5],
                    'Approval Time (Days)': [7, 5, 6, 4, 8],
                    'Pre-Payment Penalty': ['No', 'Yes', 'Yes', 'No', 'No'],
                    'Age Eligibility': ['21-65', '21-70', '21-65', '21-70', '21-60'],
                    'Loan to Value (%)': [80, 85, 80, 80, 75],
                    'Down Payment (%)': [20, 15, 20, 20, 25],
                    'Customer Rating': [4.3, 4.5, 4.2, 4.0, 4.1]
                })
            elif selected_loan_type == "Car Loan":
                loans_df = pd.DataFrame({
                    'Bank Name': ['HDFC Bank', 'ICICI Bank', 'Axis Bank', 'SBI', 'Kotak'],
                    'Interest Rate': [7.45, 7.30, 7.70, 7.20, 7.80],
                    'Processing Fee (%)': [1.0, 1.0, 0.75, 0.5, 1.0],
                    'Loan Term (Years)': [7, 7, 7, 7, 7],
                    'Max Loan Amount (Lakhs)': [100, 85, 80, 75, 80],
                    'Min Income Required (Lakhs)': [2.5, 3.0, 2.7, 2.4, 3.0],
                    'Approval Time (Days)': [2, 3, 2, 4, 3],
                    'Pre-Payment Penalty': ['Yes', 'Yes', 'No', 'No', 'Yes'],
                    'Age Eligibility': ['21-65', '21-65', '21-65', '21-70', '21-65'],
                    'Loan to Value (%)': [90, 85, 85, 90, 80],
                    'Down Payment (%)': [10, 15, 15, 10, 20],
                    'Customer Rating': [4.0, 4.2, 3.9, 4.3, 3.8]
                })
            elif selected_loan_type == "Personal Loan":
                loans_df = pd.DataFrame({
                    'Bank Name': ['HDFC Bank', 'ICICI Bank', 'Axis Bank', 'SBI', 'Bajaj Finserv'],
                    'Interest Rate': [10.25, 10.75, 11.25, 9.85, 11.50],
                    'Processing Fee (%)': [2.5, 2.0, 2.0, 1.0, 3.0],
                    'Loan Term (Years)': [5, 5, 5, 5, 5],
                    'Max Loan Amount (Lakhs)': [40, 40, 35, 20, 50],
                    'Min Income Required (Lakhs)': [3.0, 2.5, 3.0, 2.5, 2.0],
                    'Approval Time (Days)': [2, 3, 2, 5, 1],
                    'Pre-Payment Penalty': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
                    'Minimum CIBIL Score': [700, 725, 720, 700, 685],
                    'Debt-Income Ratio': [0.5, 0.55, 0.5, 0.45, 0.6],
                    'Customer Rating': [4.1, 4.3, 4.0, 4.2, 3.9]
                })
            else:  # Business Loan
                loans_df = pd.DataFrame({
                    'Bank Name': ['SBI', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Bank of Baroda'],
                    'Interest Rate': [9.5, 10.25, 10.75, 11.00, 9.25],
                    'Processing Fee (%)': [1.5, 2.0, 2.0, 1.8, 1.0],
                    'Loan Term (Years)': [10, 10, 8, 10, 10],
                    'Max Loan Amount (Lakhs)': [200, 150, 100, 150, 100],
                    'Min Business Age (Years)': [3, 2, 3, 2, 5],
                    'Approval Time (Days)': [10, 7, 8, 7, 14],
                    'Collateral Required': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
                    'Min Annual Turnover (Lakhs)': [50, 75, 60, 50, 40],
                    'Debt-Income Ratio': [0.6, 0.65, 0.6, 0.65, 0.55],
                    'Customer Rating': [4.0, 4.2, 4.1, 3.9, 3.8]
                })
        
        # Loan calculator section
        st.markdown('<div class="section-header">Loan Calculator</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loan amount
            max_amount = loans_df['Max Loan Amount (Lakhs)'].max() * 100000 if 'Max Loan Amount (Lakhs)' in loans_df.columns else 10000000
            loan_amount = st.number_input(
                "Loan Amount (₹)", 
                min_value=100000, 
                max_value=int(max_amount),
                value=1000000,
                step=100000,
                format="%d"
            )
            
            # Interest rate range
            min_rate = loans_df['Interest Rate'].min() if 'Interest Rate' in loans_df.columns else 7.0
            max_rate = loans_df['Interest Rate'].max() if 'Interest Rate' in loans_df.columns else 12.0
            interest_rate = st.slider(
                "Maximum Interest Rate (%)", 
                min_value=float(min_rate), 
                max_value=float(max_rate),
                value=float(max_rate),
                step=0.1,
                format="%.2f%%"
            )
        
        with col2:
            # Loan tenure
            max_tenure = loans_df['Loan Term (Years)'].max() if 'Loan Term (Years)' in loans_df.columns else 30
            loan_tenure = st.slider(
                "Loan Tenure (Years)", 
                min_value=1, 
                max_value=int(max_tenure),
                value=min(20, int(max_tenure)),
                step=1
            )
            
            # Loan filters
            include_prepayment_penalty = st.checkbox("Include loans with prepayment penalty", value=True)
        
        # Calculate EMI
        def calculate_emi(p, r, t):
            r = r / (12 * 100)  # Monthly rate
            t = t * 12  # Total months
            return p * r * (1 + r) ** t / ((1 + r) ** t - 1)
        
        # Filter loans based on user preferences
        filtered_loans = loans_df[loans_df['Interest Rate'] <= interest_rate].copy()
        if not include_prepayment_penalty and 'Pre-Payment Penalty' in filtered_loans.columns:
            filtered_loans = filtered_loans[filtered_loans['Pre-Payment Penalty'] == 'No']
        
        # Calculate EMI for each loan
        if not filtered_loans.empty:
            filtered_loans['EMI'] = filtered_loans['Interest Rate'].apply(
                lambda x: calculate_emi(loan_amount, x, loan_tenure)
            )
            
            # Sort by EMI
            filtered_loans = filtered_loans.sort_values('EMI')
            
            # Display the filtered loans
            st.markdown('<div class="section-header">Recommended Loans</div>', unsafe_allow_html=True)
            
            for i, (_, loan) in enumerate(filtered_loans.iterrows()):
                # Calculate total interest payable
                total_payment = loan['EMI'] * loan_tenure * 12
                total_interest = total_payment - loan_amount
                
                # Create a styled loan display
                loan_html = f"""
                <div style="background: linear-gradient(135deg, #212121, #424242, #616161); color: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0; color: white;">{loan['Bank Name']} {selected_loan_type}</h3>
                            <div style="background-color: rgba(255,255,255,0.2); display: inline-block; padding: 3px 10px; border-radius: 10px; margin-top: 5px;">
                                {loan['Approval Time (Days)']} Day Approval
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem; font-weight: bold;">{loan['Interest Rate']}%</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Interest Rate</div>
                        </div>
                    </div>
                    
                    <div style="background-color: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 0.8rem; opacity: 0.8;">Monthly EMI</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">₹{int(loan['EMI']):,}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.8rem; opacity: 0.8;">Total Interest</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">₹{int(total_interest):,}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Processing Fee</div>
                            <div>{loan['Processing Fee (%)']}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Max Loan</div>
                            <div>₹{int(loan['Max Loan Amount (Lakhs)'] * 100000):,}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Pre-Payment Penalty</div>
                            <div>{loan['Pre-Payment Penalty']}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">Rating</div>
                            <div>{loan['Customer Rating']}★</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 5px;">Eligibility:</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                """
                
                # Add eligibility criteria
                eligibility = []
                if 'Min Income Required (Lakhs)' in loan:
                    eligibility.append(f"Min Income: ₹{loan['Min Income Required (Lakhs)']} Lakhs")
                if 'Age Eligibility' in loan:
                    eligibility.append(f"Age: {loan['Age Eligibility']}")
                if 'Minimum CIBIL Score' in loan:
                    eligibility.append(f"CIBIL: {int(loan['Minimum CIBIL Score'])}")
                if 'Min Business Age (Years)' in loan:
                    eligibility.append(f"Business Age: {int(loan['Min Business Age (Years)'])} Years")
                if 'Min Annual Turnover (Lakhs)' in loan:
                    eligibility.append(f"Turnover: ₹{int(loan['Min Annual Turnover (Lakhs)'])} Lakhs")
                
                for criteria in eligibility:
                    loan_html += f'<div style="background-color: rgba(255,255,255,0.15); padding: 3px 10px; border-radius: 10px; font-size: 0.8rem;">{criteria}</div>'
                
                # Close all divs
                loan_html += """
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(loan_html, unsafe_allow_html=True)
                
                # Apply button for each loan
                if st.button(f"Apply for {loan['Bank Name']} {selected_loan_type}", key=f"apply_loan_{i}"):
                    st.success(f"Application process initiated for {loan['Bank Name']} {selected_loan_type}! This is a demo, so no actual application will be processed.")
            
            # Add a little more information if needed
            if 'Debt-Income Ratio' in filtered_loans.columns:
                avg_ratio = filtered_loans['Debt-Income Ratio'].mean()
                st.info(f"Note: Most banks require a debt-to-income ratio below {avg_ratio:.2f} for this type of loan. Make sure your total monthly debt payments don't exceed {int(avg_ratio*100)}% of your monthly income.")
        else:
            st.warning("No loans match your criteria. Try adjusting your preferences.")

# Add custom CSS for styling
st.markdown("""
<style>
.section-header {
    background-color: #f0f2f6;
    padding: 8px 15px;
    border-radius: 5px;
    font-weight: bold;
    margin: 20px 0 15px 0;
    color: #1e3a8a;
}

.recommendation-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 5px solid #2e7d32;
}

.bank-name {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 5px;
    color: #1e3a8a;
}

.match-score {
    color: #2e7d32;
    font-weight: bold;
    margin-bottom: 15px;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 15px;
}

.feature-label {
    font-weight: bold;
    color: #555;
}

.feature-value {
    color: #000;
}

.benefit-tag {
    display: inline-block;
    background-color: #e8f5e9;
    color: #2e7d32;
    padding: 3px 10px;
    border-radius: 15px;
    margin-right: 5px;
    margin-bottom: 5px;
    font-size: 0.8rem;
}

.fee-tag {
    display: inline-block;
    background-color: #ffebee;
    color: #c62828;
    padding: 3px 10px;
    border-radius: 15px;
    margin-right: 5px;
    margin-bottom: 5px;
    font-size: 0.8rem;
}

.info-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: #0d47a1;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)