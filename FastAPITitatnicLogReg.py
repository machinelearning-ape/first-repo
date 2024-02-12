from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np

app = FastAPI(debug=True)

class ModelWrapper:
    def __init__(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

    def predict(self, input_data):
        input_array = np.array([input_data])
        prediction = self.model.predict(input_array)
        return prediction[0]

# Instantiate the model wrapper
model_wrapper = ModelWrapper(r"C:\Users\godof\OneDrive\Desktop\Studying\Mentor\Kaggle\Titanic\model.pkl")

@app.get("/")
def home():
    return {"text": "Kaggle Titanic Logistic Regression"}

@app.get("/predict")
def predict(Pclass: int, Age: int, SibSp: int, Parch: int, Fare: int, Cabin: int, sex_encoded: int, embarked_encoded: int):
    input_data = [Pclass, Age, SibSp, Parch, Fare, Cabin, sex_encoded, embarked_encoded]
    prediction_result = model_wrapper.predict(input_data)

    if prediction_result == 0:
        return 'This passenger has not survived'
    elif prediction_result == 1:
        return 'This passenger has survived'
    else:
        return 'Invalid input'

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# Instantiate the model wrapper
model_wrapper = ModelWrapper(r"C:\Users\godof\OneDrive\Desktop\Studying\Mentor\Kaggle\Titanic\model.pkl")

@app.get("/")
def home():
    return {"text": "Kaggle Titanic Logistic Regression"}

@app.get("/predict")
def predict(Pclass: int, Age: int, SibSp: int, Parch: int, Fare: int, Cabin: int, sex_encoded: int, embarked_encoded: int):
    input_data = [Pclass, Age, SibSp, Parch, Fare, Cabin, sex_encoded, embarked_encoded]
    prediction_result = model_wrapper.predict(input_data)

    if prediction_result == 0:
        return 'This passenger has not survived'
    elif prediction_result == 1:
        return 'This passenger has survived'
    else:
        return 'Invalid input'

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)




