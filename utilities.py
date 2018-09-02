import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import ast
import pickle as pkl

def dump_batcher(batcher, dir_name, override=False):
    os.makedirs(dir_name, exist_ok=override)
    with open(os.path.join(dir_name, "batcher_state.pkl"), "wb") as f:
        d = {
            "classes":batcher.classes,
            "x_train":batcher.x_train,
            "y_train":batcher.y_train,
            "x_test":batcher.x_test,
            "y_test":batcher.y_test
        }
        
        if hasattr(batcher, "raw_x_train"):
            d["raw_x_train"] = batcher.raw_x_train
            d["raw_x_test"] = batcher.raw_x_test
        
        pkl.dump(d, f)
        
def load_batcher(batcher, dir_name):
    with open(os.path.join(dir_name, "batcher_state.pkl"), "rb") as f:
        d = pkl.load(f)
        if hasattr(batcher, "raw_x_train"):
            batcher.raw_x_train = d["raw_x_train"]
            batcher.raw_x_test = d["raw_x_test"]
         
        batcher.classes = d["classes"]
        batcher.x_train = d["x_train"]
        batcher.y_train = d["y_train"]
        batcher.x_test = d["x_test"]
        batcher.x_test = d["x_test"]
        batcher.y_test = d["y_test"]
        
        
def dump_all(trainer, model, batcher, dir_name, override=False):
    dump_model(trainer, dir_name, override)
    dump_batcher(batcher, dir_name, override)
    
def load_all(trainer, model, batcher, dir_name):
    load_model(trainer, dir_name)
    load_batcher(batcher, dir_name)
    

def dump_model(trainer, dir_name, override=False):
    os.makedirs(dir_name, exist_ok=override)
    cls = trainer.model
    torch.save(cls.state_dict(), os.path.join(dir_name, "model_state.tc"))
    torch.save(trainer.optimizer, os.path.join(dir_name, "optimizer.tc"))
    torch.save((trainer.train_metrics, trainer.validation_metrics), os.path.join(dir_name, "metrics.tc"))
    torch.save((trainer.current_epoch, trainer.all_params), os.path.join(dir_name, "params_epochs.tc"))
    
    cls_path = os.path.join(dir_name, "components")
    os.makedirs(cls_path, exist_ok=override)
    cls.save(cls_path)
    

def load_model(trainer, dir_name):
    trainer.model.load_state_dict(torch.load(os.path.join(dir_name, "model_state.tc")))
    trainer.optimizer.load_state_dict(torch.load(os.path.join(dir_name, "optimizer.tc")).state_dict())
    (trainer.train_metrics, trainer.validation_metrics) = torch.load(os.path.join(dir_name, "metrics.tc"))
    (trainer.current_epoch, trainer.all_params) = torch.load(os.path.join(dir_name, "params_epochs.tc"))

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

def print_confusion_matrix(model, x, y):
    model.eval()
#     x, y = batch_sampler.x_train, batch_sampler.y_train
#     pred = torch.nn.functional.softmax(cls.forward(x), dim=1)
    pred = model.predict_proba_batches(x)
    plot_confusion_matrix(build_confusion_matrix(pred.detach().numpy(), y.reshape(-1, 1)))


from sklearn.decomposition import PCA
import matplotlib.cm as cm


def plot_pca(batch_sampler, model):
    def plot(x, y, alpha=1.0):
#         plt.figure(figsize=(20, 20))
        plt.scatter(x[:, 0], x[:, 1], color=list(map(colors.get, y)), alpha=alpha)
        plt.show()
    n_classes = batch_sampler.get_n_classes()
    colors = {i:c for i, c in enumerate(cm.rainbow(np.linspace(0, 1, n_classes)))}
    pca = PCA(n_components=2)
    x_transformed = model.transform_batch(batch_sampler.x_train)
    x_test_transformed = model.transform_batch(batch_sampler.x_test)
    pca.fit(x_transformed.detach())
    
    plt.figure(figsize=(20, 20))
    plt.title("train_set")
    plot(pca.transform(x_transformed.detach()), batch_sampler.y_train, alpha=1.0)
    plt.figure(figsize=(20, 20))
    plt.title("test_set")
    plot(pca.transform(x_test_transformed.detach()), batch_sampler.y_test, alpha=1.0)
    plt.show()
    
    
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

def read_all(path = "../cf/Solutions"):
    result = {}
    for handle in os.listdir(path):
        handle_path = os.path.join(path, handle)
        result_for_handle = defaultdict(list)
        for contest in os.listdir(handle_path):
            contest_path = os.path.join(handle_path, contest)
            for solution in os.listdir(contest_path):
                solution_path = os.path.join(contest_path, solution)
                with open(solution_path, "r") as f:
                    result_for_handle[contest + ":" + solution] = f.read()
                    
        result[handle] = result_for_handle
        
    return result

def read_all_anytask(path = "../anytask"):
    result = defaultdict(dict)
    for term in os.listdir(path):
        term_path = os.path.join(path, term)
        for problem in os.listdir(term_path):
            problem_path = os.path.join(term_path, problem)
            for student in os.listdir(problem_path):
                student_path = os.path.join(problem_path, student)
                if not os.path.isdir(student_path):
                    continue
                for file in os.listdir(student_path):
                    if not file.endswith(".py"):
                        continue
                    file_path = os.path.join(student_path, file)
                    src = open(file_path, "r").read()
                    
                    problem_name = ":".join([term, problem, file])
                    result[student][problem_name] = src
                    
                    
       
        
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