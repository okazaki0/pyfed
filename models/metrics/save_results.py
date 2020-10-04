import sys
import os
import glob
import pickle
import datetime
import pandas as pd    


#Save function
def saveres(args,accuracy_scores,loss_scores,training_loss):
    """ Save the training results in csv file
    args:
        args: the argument
        accuracy_scores: test accuracy scores
        loss_scores: test loss scores
        training_loss: training loss
    """
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    date_time = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

    # ----------------Testing Loss ---------------#
    title1 = "Testing_loss_" + args.dataset
    df = pd.DataFrame(loss_scores)  
    df.to_csv(results_dir + title1 + "_" + date_time + ".csv", index=False)
    # ----------------Testing Accuracy ---------------#
    title2 = "Testing_accuracy_" + args.dataset
    df = pd.DataFrame(accuracy_scores)  
    df.to_csv(results_dir + title2 + "_"+ date_time + ".csv", index=False)
    # ----------------Training Loss  ---------------#
    title3 = "Training_loss_" + args.dataset
    df = pd.DataFrame(training_loss)  
    df.to_csv(results_dir + title3 + "_"+ date_time + ".csv", index=False)

