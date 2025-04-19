# 💧 Water Sensor Prediction using Machine Learning

A machine learning project designed to predict the status of water sensors based on key environmental features. This solution helps in automating the monitoring process and ensuring timely alerts for potential water quality issues.

---

## 📌 Project Overview
This project uses classification algorithms to predict whether the status of a water sensor is safe or not. It includes:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation

---

## 📂 Dataset Description
The dataset consists of environmental attributes that influence water quality such as:
- **Temperature**
- **Conductivity**
- **pH**
- **Turbidity**
- **Sensor Status** (Target)

The target variable (`Sensor Status`) is used to predict the health or functionality of the water sensor.

---

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/water-sensor-prediction.git
cd water-sensor-prediction
```
2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
Launch the Jupyter Notebook to run the full pipeline:
```bash
jupyter notebook Water_Sensor_Prediction.ipynb
```

Steps included:
- Data preprocessing
- EDA visualization
- Training classification models (e.g., Logistic Regression, Random Forest)
- Model evaluation using accuracy and confusion matrix

---

## 🧐 Modeling Techniques
Machine Learning models used:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Metrics:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

---

## 📊 Results
The models were tested on a hold-out test set and yielded the following results:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 92.5%    | 93%       | 91%    | 92%      |
| Decision Tree        | 91.0%    | 90%       | 89%    | 89%      |
| Random Forest        | **96.3%**| **97%**   | **95%**| **96%**  |
| SVM                  | 93.2%    | 94%       | 92%    | 93%      |

- The **Random Forest Classifier** showed the best performance across all metrics.
- It proved to be highly effective in identifying unsafe sensor readings and maintained consistent performance on validation.

---

## 🧰 Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

Add them to `requirements.txt` for quick setup.

---

## 📝 License
This project is licensed under the **MIT License**.

.

