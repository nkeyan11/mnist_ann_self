from dataset import split_dataset
from model_compile import compilation
from training import training
from predict import predictions
from save_model import saving

def mainn():

    split_dataset()
    compilation()
    training()
    # predictions()
    # saving()

if __name__ == '__main__':
    mainn()
   

