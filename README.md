<div align="center">

[1]: https://github.com/Akshat1661  
[2]: https://www.linkedin.com/in/akshat-desai-10bba1235/

<a href="https://github.com/Akshat1661">
    <img src="https://github.com/user-attachments/assets/9bfce800-9161-48e7-9a3b-f6937614c8e6" alt="Github" width="50" height="50">
</a>  
<a href="https://www.linkedin.com/in/akshat-desai-10bba1235/">
    <img src="https://github.com/user-attachments/assets/b2aa4b06-5336-46e8-9bef-f4ad296b2ade" alt="LinkedIn" width="50" height="50">
</a>

</div>

# <div align="center">Machine Learning in Healthcare</div>

<div align="center">
<img src="https://github.com/your-repo-path/output/your-image.gif?raw=true">
</div>

## Table of Contents
1. [Overview](#overview)
2. [Models Used](#models-used)
3. [Dataset](#dataset)
4. [Implementation](#implementation)
    - [Libraries Used](#libraries-used)
    - [Data Preprocessing](#data-preprocessing)
    - [Models](#models)
5. [Web Deployment](#web-deployment)
6. [Example Predictions](#example-predictions)
7. [How to Run](#how-to-run)
8. [Contact](#contact)

## Overview
In this project, we apply **Machine Learning** models to healthcare data to predict outcomes for multiple infectious diseases such as kidney disease, liver infection, and heart disease. We use a variety of **classification models** to enhance the predictive accuracy for each disease category. By leveraging advanced algorithms and data preprocessing techniques, we aim to provide timely and accurate predictions that can assist healthcare professionals in making informed decisions. Additionally, this project emphasizes the importance of accessibility in healthcare, allowing patients and providers to access predictive insights through an intuitive web interface. Ultimately, our goal is to improve patient outcomes and streamline the healthcare process using innovative technology.

## Models Used
- Random Forest (Best average accuracy: ~90%)
- Support Vector Classifier (SVC)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost

The project involves **data preprocessing**, **model training**, and **deployment** using Flask, with prediction pages specific to heart, kidney, and liver disease based on their unique input features.

## Dataset
You can find the dataset in the downloaded zip file, which contains three Excel files, one for each disease.

## Implementation
The following **Python** libraries were utilized:
- `NumPy`
- `pandas`
- `scikit-learn`
- `Flask`
- `Matplotlib`
- `XGBoost`

### Data Preprocessing
- Data is cleaned and standardized across multiple notebooks for consistency.
- Feature engineering and scaling techniques are applied to improve model performance.

### Models
- **Random Forest** was the most effective model with ~90% accuracy across multiple diseases.
- **Other classifiers** such as **SVC**, **Logistic Regression**, and **Naive Bayes** were also evaluated to compare performance metrics.

## Web Deployment
- **Flask** was used for web deployment.
- Separate pages are available for predicting heart, kidney, and liver diseases.
- Each page takes input features specific to the disease and returns predictions.

<div align="center">
<img src="https://github.com/user-attachments/assets/d917e1c2-0a04-4253-8676-1ce264a3c808">
</div>

## Example Predictions
#### Heart Disease:
<a href="https://github.com/your-repo-path/output/heart-prediction.png?raw=true">
    <img src="https://github.com/user-attachments/assets/d8f67b18-674e-4c5c-888d-65037390e8d9" alt="Heart Disease Example" width="750" height="400">
</a>

#### Kidney Disease:
<a href="https://github.com/your-repo-path/output/kidney-prediction.png?raw=true">
    <img src="https://github.com/user-attachments/assets/a8d27d66-ab51-4b66-bbf6-fc8fcfc65756" alt="Kidney Disease Example" width="750" height="400">
</a>

#### Liver Disease:
<a href="https://github.com/your-repo-path/output/liver-prediction.png?raw=true">
    <img src="https://github.com/user-attachments/assets/25ffcd51-8b4a-4eba-90a3-2ccd19d79886" alt="Liver Disease Example" width="750" height="400">
</a>

## Libraries Used
- **Python 3.x**: Core language for implementing machine learning models and Flask.
- **Flask**: For web app deployment and routing.
- **scikit-learn**: Used for various classification models like Random Forest, SVC, Logistic Regression, etc.
- **XGBoost**: For implementing the XGBoost classifier.
- **Matplotlib**: To visualize data and model performance.
- **NumPy**: For numerical operations and handling arrays.
- **pandas**: For data manipulation and preprocessing.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-path.git
2. Navigate to the project directory:
   ```bash
   cd your-repo-path
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Flask app:
   ```bash
   python app.py

## Contact
For any questions or feedback, feel free to contact me at akshat.desai.754@gmail.com
