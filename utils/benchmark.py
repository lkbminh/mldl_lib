import time
import numpy
from utils.data_operation import RMSE, R2, ConfusionMatrix

class Benchmarker:
    def __init__(self):
        self.results = {}

    def reges_eval(self, name, model, X_train, X_test, y_train, y_test):
        start_train_time = time.time()
        model.fit(X_train, y_train)
        end_train_time = time.time()

        start_pred_time = time.time()
        y_pred = model.predict(X_test)
        end_pred_time = time.time()

        rmse_score = RMSE(y_test, y_pred) 
        r2_score = R2(y_test, y_pred)

        self.results[name] = {
            "Train_time" : end_train_time - start_train_time,
            "Prediction_time": end_pred_time - start_pred_time,
            "RMSE": rmse_score,
            "R2": r2_score
        }

    def class_eval(self, name, model, X_train, X_test, y_train, y_test):
        start_train_time = time.time()
        model.fit(X_train, y_train)
        end_train_time = time.time()

        start_pred_time = time.time()
        y_pred = model.predict(X_test)
        end_pred_time = time.time()

        accuracy, precision, recall, f1_score = ConfusionMatrix(y_test, y_pred) 

        self.results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }