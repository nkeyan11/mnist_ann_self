from training import training
import tensorflow as tf
from dataset import split_dataset

def predictions():
    
    # X_test, y_test,X_valid,y_valid,X_train,y_train=split_dataset()
    training.model_clf.evaluate(split_dataset.X_test, split_dataset.y_test)
    