import pickle

with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

print("✅ Model loaded successfully!")
