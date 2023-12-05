#this module is responsible for handling trained models, train them and use them to predict
#last update 5/12/2023

import os
import joblib
from sklearn.ensemble import RandomForestRegressor

class ModelLibrary:

    def __init__(self):
        print("ModelLibrary: new object of model library created")
        self.model = None
    
    def get_path(self, name):
        return name + ".joblib"

    def load_model(self, name): 
        if os.path.exists(self.get_path(name)):
            print(f"ModelLibrary: model named {name} found")
            self.model = joblib.load(self.get_path(name))
            return self.model
        else:
            print("ModelLibrary: Model file not found")
            return None

    def save_model(self, name):
        joblib.dump(self.model, self.get_path(name))
        print("ModelLibrary: Model saved.")

    def train_model(self, data, labels):
        self.model.fit(data, labels)
        print("ModelLibrary: Model trained.")

    def predict(self, data):
        print("ModelLibrary: Predicting....")
        return self.model.predict(data)