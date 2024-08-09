
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import pandas as pd
import os
import argparse
import numpy as np

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__=='__main__':
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    #arguments for RFC model
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    #arguments for sagemaker
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train_file', type=str, default='trainV1.csv')
    parser.add_argument('--test_file', type=str, default='testV1.csv')

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")
    print()

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = train_df.columns.tolist()
    label = features.pop(-1)

    print("Building training and testing datasets")
    print()

    x_train = train_df[features]
    x_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print("column order: ")
    print(features)
    print()

    print("Label column: ")
    print(label)
    print()

    print("Data Shape:")
    print()
    print("---TRAINING DATA 80%---")
    print(x_train.shape)
    print(y_train.shape)
    print()
    print("---TEST DATA 20%---")
    print(x_test.shape)
    print(y_test.shape)

    print("Training Random Forest Classifier")
    print()

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(x_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print("Model saved at: {}", model_path)

    y_pred_test = model.predict(x_test)
    tes_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print()
    print("---METRICS RESULTA FOR TEST DATA---")   
    print()
    print("Total rows are: ", x_test.shape[0])
    print("Accuracy: ", tes_acc)
    print("Testing Report: ")
    print(test_rep)







    
