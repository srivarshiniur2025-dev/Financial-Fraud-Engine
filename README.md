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

### Why you saw `404: NOT_FOUND`

After switching away from the serverless `/api` stub, `vercel.json` was:

```json
{ "framework": null }
```

That configuration:

1. Disabled framework auto-detection
2. Removed all rewrites to `/api`
3. Did **not** explicitly wire traffic to `Dockerfile.vercel`

If Vercel did not build/attach the container (common on plans or settings where container Functions are unavailable or not detected), the deployment has **no routes and no static output** → Vercel returns **`404: NOT_FOUND`**.

### Current fix

`vercel.json` now declares an explicit container service and rewrites every path to it:

```json
{
  "framework": null,
  "services": {
    "streamlit": {
      "root": ".",
      "entrypoint": "Dockerfile.vercel"
    }
  },
  "rewrites": [
    { "source": "/(.*)", "destination": { "service": "streamlit" } }
  ]
}
```

`Dockerfile.vercel` still runs:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Platform limitation (important)

| Approach | Result |
|---|---|
| Vercel Serverless (`api/index.py`) | Cannot host Streamlit UI (no persistent WebSockets / long-running server) |
| Vercel Container (`Dockerfile.vercel`) | Can host Streamlit **if** container Functions are enabled for your Vercel plan/project |
| Streamlit Community Cloud / Render / Railway | Best fit for Streamlit with **zero app code changes** |

Streamlit is a long-running Tornado/ASGI-style app with WebSockets. Vercel’s default model is short-lived serverless request handlers. Containers are the only Vercel path that can work, and they may require Fluid compute / a plan that supports custom container images.

### Minimum architectural change if containers are unavailable

Do **not** rewrite the ML or Streamlit UI. Instead host the same repo on:

1. [Streamlit Community Cloud](https://streamlit.io/cloud) (connect GitHub → set `app.py`)
2. Or Render / Railway (Docker or `streamlit run app.py`)

That is the smallest change: **platform only**, same `app.py` / `model.py` / `utils.py`.

### Deploy checklist on Vercel

1. Push latest `vercel.json` + `Dockerfile.vercel`
2. Redeploy
3. In build logs, confirm **Docker image build** (not only Python serverless)
4. Provide `data/creditcard.csv` and `fraud_model.pkl` in the runtime (gitignored by default)

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
