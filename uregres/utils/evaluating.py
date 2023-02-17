from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def evaluate_accuracy(y_pred, labels):
  ac = accuracy_score(labels, y_pred)
  
  return ac