# Financial Fraud Intelligence Engine

AI-powered credit card fraud detection and transaction risk analysis built with **Python**, **Scikit-learn**, and **Streamlit**.

This internship-ready project trains a **Random Forest** classifier on the Kaggle Credit Card Fraud Detection dataset and serves predictions through a premium fintech-style analytics dashboard.

---

## Project Description

The Financial Fraud Intelligence Engine helps users:

- Explore fraud vs genuine transaction patterns
- Score a single transaction with risk insights
- Run batch predictions on uploaded CSV files
- View KPIs and distribution charts on an interactive dashboard

The machine learning model uses anonymized PCA features (**V1–V28**) plus **Time** and **Amount** to classify transactions as **Genuine** or **Fraud**, returning fraud probability and a risk score out of 100.

---

## Features

- **Home** — Hero landing page, feature cards, KPI overview, and project author details
- **Fraud Detection (Demo)** — Load Random / Genuine / Fraud samples from the dataset; edit Amount and Hour; predict with hidden V1–V28
- **Fraud Detection (Upload CSV)** — Validate columns, score every row, download results
- **Batch Analysis** — Dedicated batch CSV scoring workflow
- **Dashboard** — Total / fraud / genuine KPIs plus amount, class, probability, and risk charts
- **About** — Project overview, model, dataset, and technology stack
- **Risk Scoring** — Fraud probability (%) and risk score (/100) with confidence and recommended actions
- **Recent Predictions** — Session history of the last 10 demo predictions

---

## Technologies Used

| Technology | Role |
|---|---|
| Python | Core language |
| Pandas | Data loading and processing |
| NumPy | Numerical operations |
| Scikit-learn | Random Forest training and evaluation |
| Matplotlib | Charts and visual analytics |
| Streamlit | Interactive web application |
| Joblib | Model serialization (`fraud_model.pkl`) |

---

## Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Target:** `Class` (`0` = Genuine, `1` = Fraud)
- **Features:** `Time`, `V1`–`V28`, `Amount`
- **Split:** 80% train / 20% test (`random_state=42`, stratified)
- **Artifacts:** Saved locally as `fraud_model.pkl` after training
- **Outputs:** Prediction label, fraud probability, risk score

Training and prediction logic live in `model.py`. Helper utilities (summary stats, risk score, charts) live in `utils.py`.

---

## Dataset Information

This project uses the public **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** dataset from Kaggle (by the Machine Learning Group of ULB).

| Property | Detail |
|---|---|
| File name | `creditcard.csv` |
| Rows | ~284,807 transactions |
| Features | `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class` |
| Class balance | Highly imbalanced (fraud is rare) |

> **Important:** `creditcard.csv` is **not** included in this repository because of its size.

### Download and place the dataset

1. Open the Kaggle dataset page:  
   [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Sign in to Kaggle and click **Download**.
3. Extract the archive to get `creditcard.csv`.
4. Create a `data` folder in the **project root** (if it does not already exist).
5. Place the file so the final path is:

```text
Financial-Fraud-Engine/
└── data/
    └── creditcard.csv
```

The application and training scripts load the dataset from `data/creditcard.csv`.

---

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Financial-Fraud-Engine.git
cd Financial-Fraud-Engine
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `creditcard.csv` from Kaggle and place it at:

```text
data/creditcard.csv
```

### 5. Train the model (required once)

```bash
python model.py
```

This creates `fraud_model.pkl` in the project root.

---

## How to Run the Project

After installing dependencies, adding the dataset, and training the model:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

### Quick checklist

1. `pip install -r requirements.txt`
2. Place `data/creditcard.csv`
3. `python model.py`
4. `streamlit run app.py`

---

## Deploying on Vercel

This project includes Vercel configuration so root `app.py` is **not** treated as a FastAPI/Flask app (which caused the missing `app` / `application` / `handler` export error).

### Files used for Vercel

| File | Purpose |
|---|---|
| `vercel.json` | Routes deployment through `api/index.py` instead of scanning Streamlit `app.py` as a WSGI/ASGI app |
| `api/index.py` | Vercel Python entrypoint (`handler`) that launches Streamlit from `app.py` |
| `Dockerfile.vercel` | **Recommended** container image that runs `streamlit run app.py` on `$PORT` |
| `runtime.txt` | Pins Python 3.12 for the Python runtime |
| `pyproject.toml` | Sets `tool.vercel.entrypoint = "api.index:handler"` so Vercel ignores root `app.py` as the framework entry |
| `.vercelignore` | Keeps large/local files out of the upload |

### Recommended: container deploy (full Streamlit UI)

Vercel can host Streamlit through **`Dockerfile.vercel`** (container Functions). The container listens on `$PORT` (default `80`) and starts the existing Streamlit app.

1. Push this repository to GitHub.
2. Import the project in [Vercel](https://vercel.com).
3. Deploy (Vercel detects `Dockerfile.vercel`).
4. Ensure `data/creditcard.csv` and `fraud_model.pkl` are available in the deployment environment (upload via build pipeline, private asset store, or bake into the image carefully — the CSV is large and is gitignored by default).

### About the previous error

> Found app.py but it does not export a top-level "app", "application", or "handler"

Vercel’s Python runtime looked at root `app.py` and expected a FastAPI/Flask export. Streamlit apps do not use that pattern. This repo keeps Streamlit `app.py` unchanged and moves the Vercel entrypoint to `api/index.py` + optional `Dockerfile.vercel`.

---

## Project Structure

```text
Financial-Fraud-Engine/
├── app.py                 # Streamlit UI (unchanged ML/UI logic)
├── model.py               # Train / save / load / predict (Random Forest)
├── utils.py               # Data helpers, summary stats, chart helpers, risk score
├── api/
│   └── index.py           # Vercel Python entrypoint (launches Streamlit)
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for Vercel
├── vercel.json            # Vercel routing / function config
├── Dockerfile.vercel      # Container image for full Streamlit on Vercel
├── pyproject.toml         # Points Vercel entrypoint to api.index:handler
├── .vercelignore          # Files excluded from Vercel uploads
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── .gitignore             # Ignores dataset, venv, caches, model artifacts
├── data/
│   └── .gitkeep           # Placeholder; add creditcard.csv here (not committed)
├── screenshots/
│   └── .gitkeep           # Optional UI screenshots for documentation
└── fraud_model.pkl        # Generated locally after training (not committed)
```

---

## Screenshots

Add screenshots to the `screenshots/` folder and replace the placeholders below.

### Home

![Home Page](screenshots/home.png)

### Fraud Detection

![Fraud Detection](screenshots/fraud_detection.png)

### Batch Analysis

![Batch Analysis](screenshots/batch_analysis.png)

### Dashboard

![Dashboard](screenshots/dashboard.png)

---

## Future Enhancements

- Compare additional models (Logistic Regression, XGBoost, LightGBM)
- Improve recall on rare fraud cases with resampling or calibrated thresholds
- Add SHAP / feature-importance explainability views
- Support larger batch uploads with progress streaming
- Deploy the Streamlit app to Streamlit Community Cloud
- Add user authentication and prediction audit logs for production demos

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Srivarshini U R**  
Email: [srivarshini.ur2025@vitstudent.ac.in](mailto:srivarshini.ur2025@vitstudent.ac.in)  

Internship / academic project — Financial Fraud Intelligence Engine.
