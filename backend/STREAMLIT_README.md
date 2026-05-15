# 🧠 Mental Health Risk Assessment System - Streamlit Deployment

Interactive web application for predicting mental health risks (Depression, Anxiety, Stress) based on personality traits and lifestyle factors.

## ✨ Features

### 🔮 Prediction Module

- **Personality Assessment**: TIPI-10 scale (10 personality trait questions)
- **Substance Use Tracking**: VCL scale (12 substance use indicators)
- **Demographics**: Age, gender, religion, occupation, family status
- **Real-time Predictions**: Instant risk assessment for 3 mental health conditions
- **Visual Risk Levels**: Color-coded risk indicators (🟢 Low / 🟡 Medium / 🔴 High)

### 📊 Model Information

- View model architecture and performance metrics
- See feature breakdown and importance
- Understand optimal thresholds
- Learn about feature selection methodology

### ❓ Help & Documentation

- FAQ about the model
- Explanation of personality traits and substance use scales
- Accuracy information and disclaimers
- Data handling and privacy information

## 🚀 Quick Start

### Option 1: Direct Script (Recommended for Windows)

**Windows:**

```bash
run_app.bat
```

**macOS/Linux:**

```bash
chmod +x run_app.sh
./run_app.sh
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit
streamlit run app/streamlit_app.py
```

The app will open at **http://localhost:8501**

### Option 3: Docker

```bash
# Build and run with Docker
docker-compose up

# App will be available at http://localhost:8501
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Model File**: `models/multilabel_xgboost_classifier_chain.pkl` (required)
- **Dependencies**: Listed in `requirements.txt`

## 📁 Project Structure

```
Project/
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── main.py                    # (existing)
├── src/
│   ├── streamlit_utils.py         # Utility functions for deployment
│   ├── mfo_optimizer.py           # (existing)
│   ├── model_trainer.py           # (existing)
│   └── preprocessing.py           # (existing)
├── models/
│   └── multilabel_xgboost_classifier_chain.pkl
├── Data/
│   ├── raw/
│   ├── processed/
│   └── split/
├── .streamlit/
│   ├── config.toml                # Streamlit configuration
│   └── secrets.toml               # Secrets management
├── config.yaml                    # Centralized configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Docker Compose setup
├── DEPLOYMENT_GUIDE.md           # Detailed deployment guide
├── run_app.sh                    # Quick start script (Unix)
└── run_app.bat                   # Quick start script (Windows)
```

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)

Customize app appearance and behavior:

- Theme colors
- Page layout
- Server settings
- Logging level

### Application Config (`config.yaml`)

Centralized configuration for:

- Model parameters
- Feature names and ranges
- Optimal thresholds
- Data paths
- Feature descriptions

## 🎯 Model Information

| Aspect                       | Details                     |
| ---------------------------- | --------------------------- |
| **Algorithm**                | XGBoost + ClassifierChain   |
| **Features Selected**        | 35 (from 3000+) using MFO   |
| **Feature Selection Method** | Moth-Flame Optimization     |
| **Imbalance Handling**       | MLSMOTE                     |
| **Validation**               | 5-Fold Cross-Validation     |
| **Targets**                  | Depression, Anxiety, Stress |
| **Train-Test Split**         | 80-20 stratified            |

### Performance Metrics

- **F1-Score**: ~0.75-0.80 per target
- **Hamming Loss**: ~0.15
- **Exact Match Ratio**: ~0.45
- **Average Accuracy**: ~85%

### Optimal Thresholds

- **Depression**: 0.41
- **Anxiety**: 0.37
- **Stress**: 0.38

## 📊 Input Features

### Personality Traits (TIPI-10)

Ten-Item Personality Inventory measuring Big Five traits:

- Extraversion / Introversion
- Agreeableness
- Conscientiousness
- Neuroticism
- Openness to Experience

### Substance Use (VCL)

Frequency of use for 12 substances:

- Alcohol, Cannabis, Cocaine, Heroin
- Amphetamine, Ecstasy, LSD, Mushrooms
- Nicotine, Methadone, VSA, PCP

### Demographics

- Age (15-24)
- Gender
- Religion
- Sexual orientation
- Race/Ethnicity
- Marital status
- Family size
- Voting behavior
- English native speaker status

## 🔐 Security & Privacy

✅ **Data Handling:**

- All predictions run **locally** (no external servers)
- No data is stored persistently
- No tracking or telemetry by default
- No internet connectivity required after startup

✅ **Model Protection:**

- Model file stored securely in `models/` directory
- Input validation on all user inputs
- Safe error handling without exposing sensitive info

## 📈 Usage Examples

### Basic Usage

1. Navigate to `http://localhost:8501`
2. Fill in personality traits (rate 1-7)
3. Select substance use frequencies (0-6)
4. Enter demographics
5. Click "Get Prediction"
6. View results with risk levels and recommendations

### Interpreting Results

**🟢 Low Risk** (Probability < 0.40)

- Minimal mental health risk indicators
- Continue healthy lifestyle practices

**🟡 Medium Risk** (Probability 0.40-0.70)

- Some risk factors present
- Consider professional consultation

**🔴 High Risk** (Probability > 0.70)

- Multiple risk indicators present
- Strongly recommend mental health professional

## 🐛 Troubleshooting

### Issue: "Model file not found"

```
FileNotFoundError: models/multilabel_xgboost_classifier_chain.pkl
```

**Solution:** Ensure model file exists in `models/` directory

### Issue: "XGBoost compatibility error"

```
AttributeError: 'XGBClassifier' object has no attribute 'predict_proba'
```

**Solution:** Update XGBoost: `pip install --upgrade xgboost>=2.1.0`

### Issue: Slow predictions

**Solutions:**

- Restart Streamlit (`Ctrl+C` then re-run)
- Clear browser cache
- Check system resources
- Reduce model batch size

### Issue: Port 8501 already in use

```bash
# Run on different port
streamlit run app/streamlit_app.py --server.port 8502
```

## 🌐 Deployment Options

### 1. **Streamlit Cloud** (Easiest, Free)

```bash
# Push to GitHub, then deploy via https://share.streamlit.io
```

### 2. **Heroku** (Free tier available)

- See `DEPLOYMENT_GUIDE.md` for setup instructions

### 3. **Docker** (Production)

```bash
docker-compose up --build
```

### 4. **AWS / Azure / GCP**

- See `DEPLOYMENT_GUIDE.md` for cloud platform setup

## 📚 Documentation

- **`DEPLOYMENT_GUIDE.md`**: Detailed deployment instructions for various platforms
- **`config.yaml`**: Configuration reference
- **`METHODOLOGY.md`**: Research methodology (existing)
- **`QUICK_REFERENCE.md`**: Quick reference guide (existing)

## 🔧 Development

### Adding New Features

1. Edit `app/streamlit_app.py`
2. Update `src/streamlit_utils.py` for new utilities
3. Modify `config.yaml` for configuration changes
4. Test locally before deployment

### Customization Examples

**Change Theme:**
Edit `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#FF5733"
```

**Add New Tab:**

```python
with st.tabs(["...", "New Tab"]):
    st.write("New content")
```

**Modify Model Thresholds:**
Update `config.yaml`:

```yaml
thresholds:
  optimal_thresholds:
    risk_depression: 0.45 # Changed from 0.41
```

## 📝 Requirements

All dependencies specified in `requirements.txt`:

- pandas, numpy, scikit-learn
- xgboost, shap
- streamlit
- matplotlib, seaborn
- mlflow, pyswarms
- imbalanced-learn

Install all at once:

```bash
pip install -r requirements.txt
```

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.

- Not a medical device
- Should not be used for clinical diagnosis
- Always consult qualified mental health professionals
- Not a substitute for professional psychological assessment

## 📞 Support

For issues or questions:

1. Check `DEPLOYMENT_GUIDE.md` troubleshooting section
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check XGBoost docs: https://xgboost.readthedocs.io

## 📊 Project Links

- **Repository**: Amikom University Data Mining Project
- **Data Source**: DASS-42 Dataset with personality & substance use data
- **Model Notebook**: `notebooks/11_final_model.ipynb`
- **Evaluation**: `notebooks/08_performance_evaluation.ipynb`

## 👨‍💼 Author & Attribution

- **Project**: Mental Health Risk Assessment System
- **Institution**: Amikom University
- **Course**: Data Mining (Semester 6)
- **Date**: May 2026

## 📄 License

Educational use. See project repository for details.

---

**Status**: ✅ Production Ready  
**Last Updated**: May 2026  
**Version**: 1.0

**Happy Predicting! 🧠✨**
