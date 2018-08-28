import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import ast

def dump_model(trainer, dir_name):
    os.mkdir(dir_name)
    cls = trainer.model
    torch.save(cls.state_dict(), os.path.join(dir_name, "model_state.tc"))
    torch.save(trainer.optimizer, os.path.join(dir_name, "optimizer.tc"))
    torch.save((trainer.train_metrics, trainer.validation_metrics), os.path.join(dir_name, "metrics.tc"))
    

def load_model(trainer, dir_name):
    trainer.model.load_state_dict(torch.load(os.path.join(dir_name, "model_state.tc")))
    trainer.optimizer.load_state_dict(torch.load(os.path.join(dir_name, "optimizer.tc")).state_dict())
    (trainer.train_metrics, trainer.validation_metrics) = torch.load(os.path.join(dir_name, "metrics.tc"))

def load_plain_model(model, dir_name):
    model.load_state_dict(torch.load(os.path.join(dir_name, "model_state.tc")))
    return model
   
    
import seaborn as sns
import matplotlib.pyplot as plt
def build_confusion_matrix(predicted_probs, true):
    n_labels = predicted_probs.shape[1]
    result = np.zeros(shape=(n_labels, n_labels))
    
    pred = predicted_probs.argmax(axis=1)

    
    accuracy = np.count_nonzero(pred == true.ravel())/len(true)
    print("Accuracy = ", accuracy)
    
    for pred_cls in range(n_labels):
        for true_cls in range(n_labels):
            result[true_cls, pred_cls] = np.count_nonzero(true[pred == pred_cls] == true_cls)
    norm = result.sum(axis=1)
    norm = np.maximum(norm, 1)
    return result

def plot_confusion_matrix(confusion_matrix):
    fig = plt.figure( figsize=(20, 20))
    plt.xlabel("True classes")
    plt.ylabel("Predicted classes")
    sns.heatmap(confusion_matrix, annot=True, vmin=0.0, cmap="YlGnBu")
    
    
def read_all_gcj(path = "../CodeStylometry/Corpus/temp/codejamfolder/py"):
    result = {}
    for handle in os.listdir(path):
        handle_path = os.path.join(path, handle)
        result_for_handle = defaultdict(str)
#         for contest in os.listdir(handle_path):
#             contest_path = os.path.join(handle_path, contest)
        for solution in os.listdir(handle_path):
                solution_path = os.path.join(handle_path, solution)
                with open(solution_path, "r") as f:
                    try:
                        result_for_handle[solution] = f.read()
                    except Exception as e:
                        print(solution_path)
                        print(e)
                    
        result[handle] = result_for_handle
        
    return result

def fails(func):
    try:
        func()
        return False
    except Exception as e:
#         print(e)
        return True


def filter_by_count(data, min_count, max_count):
    result = {}
    for handle, result_for_handle in data.items():
        current = {}
        for problem, solution in result_for_handle.items():
            if not fails(lambda: ast.parse(solution)):
                current[problem.split(".")[0]] = solution


        if len(current) >= min_count and len(current) <= max_count:
            result[handle] = current
    
    return result

def filter_ast_size(df, mn=0, mx=500):
    result = {}
    for handle, result_for_handle in df.items():
        current_result = {}
        for problem, submission in result_for_handle.items():
            try:
                parsed = ast.parse(submission)
                length = len(list(ast.walk(parsed)))
                print(length)
                if length >= mn and length <= mx:
                    current_result[problem] = submission
            except:
                pass
            
        result[handle] = current_result
    
    return result

def filter_people(df, people):
    result = {}
    for handle, result_for_handle in df.items():
        if handle in people:
            result[handle] = result_for_handle
    
    return result