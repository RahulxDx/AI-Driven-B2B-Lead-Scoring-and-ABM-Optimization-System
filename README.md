# AI-Driven B2B Lead Scoring and Account-Based Marketing (ABM) Optimization System

A complete machine learning system for scoring B2B leads and optimizing account-based marketing strategies using XGBoost classification.

## 🎯 Features

- **Synthetic Dataset Generation**: Creates realistic B2B lead data with 2000 samples
- **Multiple ML Models**: Compares Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost
- **Stratified Cross-Validation**: Ensures robust model evaluation
- **Production-Ready Web App**: Streamlit interface for easy lead scoring
- **Automatic Lead Ranking**: Intelligent ranking system identifies best leads
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Importance Visualization**: Understand what drives lead quality

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python b2b_lead_scoring.py
```

This will:
- Generate 2000 synthetic B2B leads
- Train and compare 5 ML models
- Select the best model (based on F1-score)
- Save the model as `best_model.pkl`
- Generate feature importance visualization

### 3. Run the Web Application

```bash
streamlit run app.py
```

Upload a CSV file with lead data to get predictions.

## 📊 Data Format

Your CSV file should contain these columns:

**Categorical:**
- `Company_Size`: Small, Medium, or Large
- `Industry`: Tech, Finance, Healthcare, Manufacturing, Retail, etc.
- `Lead_Source`: Organic, Paid Ads, Referral, Social Media, Email Campaign
- `Region`: North America, Europe, Asia, Others

**Numerical:**
- `Annual_Revenue`: Company annual revenue (numeric)
- `Website_Visits`: Number of website visits (integer)
- `Email_Opens`: Number of email opens (integer)
- `Ad_Clicks`: Number of ad clicks (integer)
- `Engagement_Score`: Engagement score (0-100)
- `Previous_Purchases`: Number of previous purchases (integer)

## 🎯 Lead Scoring

The system predicts three lead quality categories:
- **High**: Premium quality leads - prioritize immediately
- **Medium**: Good quality leads - follow up soon
- **Low**: Lower quality leads - follow up later

### How to Identify Best Leads

1. **Primary**: `Predicted_Lead_Score` = High (best)
2. **Secondary**: `Probability_High` percentage (higher is better)
3. **Tie-breakers**: `Confidence`, `Annual_Revenue`, `Engagement_Score`

The app automatically ranks leads using a composite scoring system.

## 📁 Project Structure

```
B2B Project/
├── b2b_lead_scoring.py      # Model training script
├── app.py                    # Streamlit web application
├── best_model.pkl           # Trained XGBoost model (generated)
├── demo_leads.csv           # Sample dataset for testing
├── feature_importance.png   # Feature importance visualization (generated)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Model Training Details

- **Train-Test Split**: 80/20 stratified split
- **Cross-Validation**: 5-fold stratified K-Fold
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best Model Selection**: Based on weighted F1-score
- **Preprocessing**: One-hot encoding + StandardScaler normalization

## 📈 Output

The training script generates:
- Model performance comparison table
- Best model saved as `best_model.pkl`
- Feature importance plot (`feature_importance.png`)
- Detailed metrics for each model

## 🌐 Web Application Features

- **Upload CSV**: Simple drag-and-drop interface
- **Automatic Preprocessing**: Handles missing values and encoding
- **Real-time Predictions**: Instant lead scoring
- **Top Leads Dashboard**: Automatically identifies and ranks best leads
- **Interactive Visualizations**: Bar charts and pie charts
- **Filter & Sort**: Flexible data exploration
- **CSV Export**: Download results with predictions

## 🎨 Interface Highlights

- Clean, professional design
- Color-coded lead scores (Green=High, Yellow=Medium, Red=Low)
- Responsive layout
- Real-time metrics and insights

## 📝 Example Usage

1. Train model: `python b2b_lead_scoring.py`
2. Start app: `streamlit run app.py`
3. Upload `demo_leads.csv` to test
4. View predictions and download results

## 🔍 Model Performance

The system compares 5 models and automatically selects the best performer:
- Logistic Regression
- Random Forest
- XGBoost (typically best)
- LightGBM
- CatBoost

## 📞 Support

For issues or questions, refer to the code comments or adjust parameters in the scripts.

## 📄 License

This project is provided as-is for educational and commercial use.

---

**Built with**: Python, Scikit-learn, XGBoost, LightGBM, CatBoost, Streamlit

