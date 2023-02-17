import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df: pd.DataFrame) -> pd.DataFrame:
  df.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
  
  # Define your (X,y)
  X = df.drop('diagnosis',axis=1)
  y = df['diagnosis']
  
  # Split
  X_train, X_test, y_train, y_test = train_test_split(
                             X, y, 
                             test_size=0.15, 
                             random_state=42, 
                             shuffle=True,
                             stratify=y)
  
  return (X_train, y_train), (X_test, y_test)