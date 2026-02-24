# 🚀 Streamlit Deployment Guide

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Streamlit Cloud

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

### 3. Your app will be live at:
```
https://YOUR_USERNAME-YOUR_REPO.streamlit.app
```

## Alternative Deployments

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## Environment Variables (Optional)
No environment variables required for basic deployment.

---
**Ready for production deployment!**
