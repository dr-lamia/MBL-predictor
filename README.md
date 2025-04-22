
 🦷 AI-Powered Marginal Bone Loss (MBL) Predictor – Enhanced

This interactive Streamlit app predicts marginal bone loss (MBL) around implant-supported crowns based on clinical and radiographic features. It now supports:

- ✅ Individual prediction with SHAP explanations
- ✅ Batch prediction via CSV upload
- ✅ SHAP summary plots
- ✅ Custom branding (icon, title, layout)

---

🔍 Features

- Predict MBL using:
  - Crown type (PEKK or LD)
  - Gender
  - Implant site (Central or Lateral)
  - Baseline HU (CBCT bone density)
  - ΔHU (change in bone density)

- Visual explanation of prediction via SHAP waterfall and summary plots
- Upload a batch `.csv` file to predict MBL for multiple patients at once
- Download prediction results as `.csv`

---

📂 Example Input File

```csv
crown_type,gender,implant_site,baseline_HU,delta_HU
PEKK,Female,Central,1179,-335
LD,Male,Lateral,695,-126
```

---

## 🚀 Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run ai_mbl_predictor_enhanced.py
```

---

## 💡 Citation

This tool is based on the manuscript:  
**“Artificial Intelligence for Predicting Marginal Bone Loss and Esthetic Outcomes in Anterior Implant-Supported Crowns: A Pilot Study”**

---

## 📬 Contact

Developed by:lamiaa Elfadaly  
Email: dr.l.fadaly@gmail.com 
Institution: MSA university
```

