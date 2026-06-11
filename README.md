#  AI-Powered Fashion Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Neural%20Networks-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Optimized-green.svg)]()

##  Project Overview
The **Fashion Recommendation System** is an AI-driven tool designed to eliminate the guesswork in online shopping. By analyzing physical body measurements, the system predicts the most accurate **Size**, **Fit**, and **Body Shape** for a user. This reduces return rates for e-commerce brands and improves customer satisfaction by providing a personalized sizing experience.

###  Key Features
- **Automated Size Prediction:** Accurately predicts sizes (M, L, XL, etc.) based on biometric data.
- **Body Shape Classification:** Categorizes users into shapes (Apple, Pear, Hourglass, Rectangle).
- **Fit Analysis:** Determines if a "Slim" or "Regular" fit is most suitable.
- **Hyperparameter Optimized:** The model has been tuned via Grid Search for maximum precision.

---

##  Technical Stack
- **Language:** Python 3.x
- **Libraries:** 
  - `Pandas` & `NumPy` (Data Manipulation)
  - `Scikit-Learn` (Preprocessing & Evaluation)
  - `TensorFlow/Keras` (Neural Network Architecture)
  - `Matplotlib/Seaborn` (Confusion Matrix Visualization)

---

##  Model Optimization Report
*(This section was specifically developed to validate the impact of hyperparameter tuning on model intelligence)*

To improve the predictive performance of the fashion recommendation model, I implemented a systematic hyperparameter tuning process using a Grid Search approach, testing 32 different architectural configurations. The primary focus was on optimizing the learning rate to **0.001** to ensure stable convergence and adjusting the hidden layer capacity to a **128 $\rightarrow$ 64** neuron structure. This allowed the model to better capture the complex biological relationships between body measurements and size categories while preventing overfitting through the introduction of a **0.2 dropout rate**.

The results of this tuning are evident in the final metrics, where the Size prediction accuracy increased from **93% to 97%**. As seen in the confusion matrices below, there is a significant reduction in misclassifications between 'Large' and 'XL' sizes, indicating that the model has developed a much finer boundary for borderline cases. While the 'Rectangle' body shape remains a challenge due to limited data samples, the overall system now demonstrates a superior balance between computational efficiency and predictive precision.

---

## 🖼️ Visual Comparison (Confusion Matrices)

### 1. Size Prediction
| Before Tuning | After Tuning |
|---|---|
| ![Before](plots/size_before.png) | ![After](plots/size_after.png) |

### 2. Fit Prediction
| Before Tuning | After Tuning |
|---|---|
| ![Before](plots/fit_before.png) | ![After](plots/fit_after.png) |

### 3. Body Shape Prediction
| Before Tuning | After Tuning |
|---|---|
| ![Before](plots/shape_before.png) | ![After](plots/shape_after.png) |

---

##  Installation & Execution

### Project Layout
```text
fashion_rec_system/
├── data/               # Raw datasets (caesar.csv)
├── src/                # Core logic (model, preprocessing, tuner, evaluation)
├── plots/              # Visual results (Confusion Matrices)
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

### Running the Project
1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fashion_rec_system.git
   cd fashion_rec_system
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the System:**
   ```bash
   python main.py
   ```

---

## 🔮 Future Scope
- **Dataset Expansion:** Collecting more samples for the "Rectangle" body shape to eliminate the remaining 2% error.
- **API Integration:** Wrapping the model in a FastAPI/Flask wrapper for mobile app integration.
- **GUI Development:** Creating a user-friendly frontend where users can input their measurements manually.
