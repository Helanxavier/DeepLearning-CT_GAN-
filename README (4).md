# CTGAN Synthetic Data Generator 🎲

This project is a **Streamlit web app** that demonstrates how to generate synthetic tabular data using a trained **CTGAN model**.  

It allows users to:
- Load a pre-trained CTGAN model
- Generate synthetic samples
- Visualize data distributions
- Compare real vs. synthetic datasets

---

## 🚀 How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/ctgan-streamlit.git
cd ctgan-streamlit
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser at:
👉 http://localhost:8501

---

## 🌐 Deployment on Streamlit Cloud
1. Push this repo to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in.  
3. Click **New App** → select this repo and branch → choose `app.py`.  
4. Click **Deploy**.  
5. Your app will be live at a public URL like:
   ```
   https://<your-app-name>.streamlitapp.com
   ```

---

## 📂 Project Structure
```
.
├── app.py                # Main Streamlit app
├── ctgan_model.pkl       # Pre-trained CTGAN model (optional if hosted externally)
├── adult.csv             # Example dataset
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Files to ignore in Git
```

---

## ⚠️ Notes
- If your `ctgan_model.pkl` is **>100MB**, don’t push directly to GitHub.  
  Instead, host it externally (e.g., Google Drive, Hugging Face Hub) and modify `app.py` to download it at runtime.  
- To keep secrets (like URLs or API keys), use **Streamlit Cloud → App → Settings → Secrets**.  

---

## 📜 License
MIT License – feel free to use and modify.
