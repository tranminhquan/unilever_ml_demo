# import libraries
import pandas as pd
from uregres.utils.split_data import split_train_test
from uregres.processing.preprocess import impute
from uregres.processing.features_engineering import scale
from uregres.models.cls_models import buid_logistic_regression
from uregres.utils.training import train_model
from uregres.utils.predicting import predict
from uregres.utils.evaluating import evaluate_accuracy

# load data
df = pd.read_csv('data/breast_cancer.csv')

# split data
(X_train, y_train), (X_test, y_test) = split_train_test(df)

# impute
X_train = impute(X_train)
X_test = impute(X_test)

# scale
X_train, X_test = scale(X_train, X_test)

# build logistic model
model = buid_logistic_regression()

# train model
model = train_model(model, X_train, y_train)

# predict
preds = predict(model, X_test)

# evaluate
accuracy = evaluate_accuracy(preds, y_test.to_numpy())
print('Accuracy: ', accuracy)

