# 🫀 RedPulse — Heart Risk Analyzer

## 📌 Overview

RedPulse is an AI-powered heart disease risk prediction system built using Machine Learning and Streamlit. It analyzes patient health parameters and predicts the likelihood of heart disease, helping in early detection and preventive care.

---

## 🚀 Features

* 🔍 Heart disease risk prediction using ML model (KNN)
* 📊 Probability-based risk analysis
* 📈 Interactive visualizations (Gauge & Bar Charts)
* 📋 Detailed patient report
* 📥 Downloadable CSV report
* 💡 Personalized health tips
* 🎨 Modern dark-themed UI with custom styling

---

## 🧠 Machine Learning Model

* Algorithm: K-Nearest Neighbors (KNN)
* Preprocessing: StandardScaler
* Input Features:

  * Age
  * Sex
  * Chest Pain Type
  * Resting Blood Pressure
  * Cholesterol
  * Fasting Blood Sugar
  * Resting ECG
  * Max Heart Rate
  * Exercise-induced Angina
  * Oldpeak
  * ST Slope

---

## 🏗️ Project Structure

```
RedPulse/
│
├── app.py
├── model/
│   ├── KNN_heart.pkl
│   ├── scaler.pkl
│   └── columns.pkl
├── assets/
│   └── style.css
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/redpulse.git
cd redpulse
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. User enters patient health data in the sidebar
2. Data is preprocessed and scaled
3. ML model predicts heart disease risk
4. Results are displayed with probability and charts
5. User can download the report

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical advice.

---

## 🔮 Future Scope

* Add more advanced ML models (Random Forest, XGBoost)
* Improve accuracy with larger datasets
* Add real-time patient monitoring
* Deploy as a web application
* Integrate with hospital systems

---

## 👨‍💻 Author

Devansh Tiwari

---

## ⭐ Acknowledgment

Special thanks to faculty and mentors for their guidance and support during the development of this project.
