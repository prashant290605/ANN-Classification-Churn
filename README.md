# Customer Churn Prediction and Salary Regression

This repository contains two machine learning applications built with Streamlit:

1. **Customer Churn Prediction**: A classification model to predict whether a bank customer will leave the bank or not.
2. **Salary Regression**: A regression model to predict employee salaries based on various features.

## Project Structure

- `app.py`: Streamlit application for customer churn prediction
- `streamlit_regression.py`: Streamlit application for salary prediction
- `experiment.ipynb`: Jupyter notebook with exploratory data analysis and model development for churn prediction
- `salary_regression.ipynb`: Jupyter notebook with exploratory data analysis and model development for salary prediction
- `model.h5`: Trained neural network model for churn prediction
- `regression_model.h5`: Trained neural network model for salary prediction
- `scaler.pkl`: Feature scaler for churn prediction model
- `label_encoder_gender.pkl`: Label encoder for gender feature
- `onehot_encoder_geo.pkl`: One-hot encoder for geography feature
- `requirements.txt`: List of dependencies required to run the applications

## Features

### Customer Churn Prediction
- Interactive UI for inputting customer data
- Real-time prediction of churn probability
- Visual representation of prediction results
- Handles categorical features with appropriate encoding

### Salary Regression
- User-friendly interface for entering employee details
- Accurate prediction of expected salary
- Proper handling of categorical and numerical features
- Visualization of prediction results

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: For building and training neural network models
- **Streamlit**: For creating interactive web applications
- **Pandas**: For data manipulation and analysis
- **Scikit-learn**: For data preprocessing and model evaluation
- **Matplotlib/Seaborn**: For data visualization

## How to Run

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the churn prediction app:
   ```
   streamlit run app.py
   ```
4. Run the salary prediction app:
   ```
   streamlit run streamlit_regression.py
   ```

## Model Details

Both applications use artificial neural networks (ANNs) built with TensorFlow/Keras:

- The churn prediction model is a classification ANN with multiple hidden layers
- The salary regression model is a regression ANN optimized for numerical prediction

## Future Improvements

- Add more visualization options
- Implement model explainability features
- Add option to retrain models with new data
- Enhance UI with more interactive elements 