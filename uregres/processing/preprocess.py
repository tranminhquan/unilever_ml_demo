
from sklearn.impute import SimpleImputer

def impute(data):
  imputer = SimpleImputer(strategy="median")
  imputer.fit(data)
  data = imputer.transform(data)
  
  return data