from model_compile import compilation
import tensorflow as tf
from dataset import split_dataset

def training():
    model_clf=compilation()
    X_test, y_test,X_valid,y_valid,X_train,y_train=split_dataset()
    EPOCHS = 3
    VALIDATION = (X_valid, y_valid)
    model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)
    model_clf.evaluate(X_test, y_test)
    model_clf.save("model.h5")
