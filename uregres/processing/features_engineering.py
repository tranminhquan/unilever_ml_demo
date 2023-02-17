from sklearn.preprocessing import StandardScaler

def scale(X_train, X_test):
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  
  return X_train, X_test