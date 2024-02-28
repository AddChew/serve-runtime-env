import numpy as np
import xgboost as xgb

X = np.loadtxt("datasets/x.txt")
y = np.loadtxt("datasets/y.txt")

dataset = xgb.DMatrix(X, y)
params = {'objective': 'multi:softmax', 'num_class': 3}
model = xgb.train(params, dataset, 5)
model.save_model("models/xgboost_model.json")