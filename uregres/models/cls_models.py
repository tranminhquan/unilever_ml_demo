from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def buid_logistic_regression(random_state=42):
  model = LogisticRegression(random_state=random_state)
  
  return model


def build_decision_tree(random_state=42):
  model = DecisionTreeClassifier(random_state=random_state)
  
  return model


def build_random_forest(random_state=42, criterion='entropy'):
  
  model = RandomForestClassifier(random_state=random_state, criterion='entropy')
  
  return model
