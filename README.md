# Mobile Price Classification

This project uses machine learning to predict the price range of mobile devices based on various features like battery power, RAM, screen size, and more. The model is developed using AWS Sagemaker, along with Python libraries such as Pandas and Scikit-learn for data preprocessing and training.

## Project Overview

In this project, we build a machine learning model to classify mobile devices into price categories based on their features. The dataset contains information about various specifications of mobile phones and their respective price ranges.

### Key Features:
- **Battery Power**
- **RAM**
- **Mobile Weight**
- **Screen Size**
- **Touch Screen**
- **Camera Features**
- **4G/3G connectivity**
- **WiFi availability**

The price range is divided into the following categories:
- `0`: Low range
- `1`: Medium range
- `2`: High range

## Technologies Used

- **AWS Sagemaker**: Used for training and deploying the model on the cloud.
- **Python**: Programming language used for data analysis, preprocessing, and model training.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning models and evaluation.
- **Boto3**: AWS SDK for interacting with S3 and other AWS services.

## Dataset

The dataset used contains 2000 samples of mobile phones with 21 features such as battery power, screen size, and RAM. The target variable is `price_range`, which is used to classify the mobile phones into one of the three price categories.

### Sample Data Columns:
- `battery_power`: The mobile phone's battery capacity.
- `blue`: Indicates if the phone supports Bluetooth.
- `clock_speed`: Processor clock speed.
- `dual_sim`: Indicates if the phone supports dual SIM cards.
- `ram`: The RAM capacity of the mobile phone.
- `price_range`: The price range (target variable).

## Steps

1. **Data Loading and Exploration**: We load the dataset and explore the initial rows to understand the structure.
2. **Data Preprocessing**: The data is cleaned and prepared for model training. Features and target variables are separated.
3. **Model Training**: We split the dataset into training and testing sets, then use machine learning models like Logistic Regression or Random Forest to predict the price range.
4. **Model Evaluation**: The model's performance is evaluated based on accuracy and other metrics.

## Installation

To run this project, you'll need Python 3.x along with the following dependencies:

```bash
pip install sagemaker boto3 pandas scikit-learn matplotlib
