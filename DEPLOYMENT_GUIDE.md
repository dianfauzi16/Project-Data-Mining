# 🚀 Streamlit Deployment Guide

## Quick Start

### Prerequisites

- Python 3.8+
- Trained model in `models/multilabel_xgboost_classifier_chain.pkl`

### Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Verify model exists:**

```bash
# Make sure the model file exists
ls models/multilabel_xgboost_classifier_chain.pkl
```

### Running the Application

#### Local Development

```bash
# Navigate to project directory
cd "d:\Amikom\Semester 6\Project Data Mining\Project"

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will be available at: **http://localhost:8501**

#### With Custom Config

```bash
# Run with custom port
streamlit run app/streamlit_app.py --server.port 8502

# Run with logger disabled
streamlit run app/streamlit_app.py --logger.level=error

# Full screen mode
streamlit run app/streamlit_app.py --client.showErrorDetails=false
```

---

## 📊 Features

### 🔮 Prediction Tab

- Input personality traits (TIPI-10)
- Input substance use information (VCL)
- Input demographic information
- Get predictions for Depression, Anxiety, Stress risk
- See visual risk levels with color coding

### 📊 Model Info Tab

- View model specifications
- See selected features breakdown
- View optimal thresholds
- Understand feature selection method

### ❓ Help Tab

- FAQ about the model
- Accuracy information
- Disclaimers and recommendations
- Data handling information

---

## 🌐 Deployment Options

### Option 1: Local Machine (Development)

```bash
streamlit run app/streamlit_app.py
```

### Option 2: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Deploy directly from repository
4. App gets public URL

### Option 3: Heroku Deployment

**1. Create `Procfile`:**

```
web: streamlit run app/streamlit_app.py --server.port $PORT --server.address "0.0.0.0"
```

**2. Create `setup.sh`:**

```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.json

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**3. Deploy:**

```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Option 4: Docker Deployment

**1. Create `Dockerfile`:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Build and run:**

```bash
docker build -t mental-health-app .
docker run -p 8501:8501 mental-health-app
```

### Option 5: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.10 mental-health-app

# Create environment
eb create mental-health-env

# Deploy
eb deploy
```

---

## 🔧 Configuration

### Streamlit Config (`config.toml`)

Located at `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[server]
port = 8501
headless = true
runOnSave = false
```

### Application Config (`config.yaml`)

Centralized configuration for:

- Model parameters
- Feature names and ranges
- Thresholds
- Data paths
- Feature descriptions

---

## 📈 Performance Tips

### 1. Optimize Model Loading

```python
@st.cache_resource
def load_model():
    with open("models/multilabel_xgboost_classifier_chain.pkl", "rb") as f:
        return pickle.load(f)
```

### 2. Optimize Predictions

- Batch process multiple predictions
- Use caching for repeated inputs
- Minimize data transformations

### 3. UI Optimization

- Use columns for parallel layout
- Minimize re-renders with `@st.cache_data`
- Use `session_state` for user inputs

### 4. Memory Management

- Clear large objects after use
- Use generators for large datasets
- Monitor with `st.write(st.session_state)`

---

## 🐛 Troubleshooting

### Issue: Model Not Found

```
FileNotFoundError: models/multilabel_xgboost_classifier_chain.pkl
```

**Solution:**

- Verify model file exists: `ls models/`
- Check file path in code
- Ensure running from project root directory

### Issue: XGBoost Version Mismatch

```
AttributeError: 'XGBClassifier' object has no attribute 'predict_proba'
```

**Solution:**

- Update XGBoost: `pip install --upgrade xgboost`
- Model uses custom `chain_predict_proba()` function
- Ensure XGBoost >= 2.1.0

### Issue: Slow Predictions

**Solutions:**

- Reduce input size
- Increase `n_jobs=-1` in model
- Use GPU acceleration if available
- Cache predictions for common inputs

### Issue: Out of Memory

**Solutions:**

- Limit Streamlit cache size: `@st.cache_data(max_entries=10)`
- Reduce batch size for predictions
- Monitor memory usage

---

## 📊 Monitoring & Logging

### View Logs

```bash
# Streamlit logs
streamlit run app/streamlit_app.py --logger.level=debug

# Custom logs
tail -f logs/app.log
```

### Track Predictions

```python
# Add to streamlit_app.py
import logging

logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log predictions
logging.info(f"Prediction: {predictions}, Probabilities: {probabilities}")
```

---

## 🔐 Security Considerations

1. **Data Privacy**
   - Predictions run locally (not sent to server)
   - No data storage by default
   - Add HTTPS for production

2. **Model Protection**
   - Don't expose model paths in error messages
   - Use environment variables for sensitive config
   - Implement API authentication if needed

3. **Input Validation**
   - Validate all user inputs
   - Check data types and ranges
   - Use `DataPreprocessor.validate_input()`

---

## 📚 Additional Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [Streamlit Cloud](https://share.streamlit.io)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-Learn ClassifierChain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)

---

## 📝 Support & Contributing

For issues or improvements:

1. Check existing GitHub issues
2. Create detailed bug reports
3. Submit pull requests with improvements
4. Follow project code style

---

**Last Updated:** May 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
