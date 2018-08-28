import tqdm
import time
import numpy as np
import torch
#import datetime
#from launch_utils import logger
from collections import defaultdict
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from IPython import display
import ast

import os
from collections import defaultdict

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

class Batcher:
    def __init__(self, data, batch_size):
        np.random.seed(42)

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))

        self.class_encoder = {handle:id for id, handle in enumerate(self.classes)}


        self.y = []
        self.x = []
        for handle, submissions in data.items():
            for submission, src in submissions.items():
                self.y.append(self.class_encoder[handle])
                self.x.append(src)

        self.y = np.array(self.y)
        self.x = np.array(self.x)

        order = np.random.permutation(np.arange(len(self.x)))

        self.y = self.y[order]
        self.x = self.x[order]

    def get_n_classes(self):
        return len(self.classes)


class NameBatcher:
    def __init__(self, data, batch_size, train_names, test_names):
        np.random.seed(42)

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))

        self.class_encoder = {handle:id for id, handle in enumerate(self.classes)}


        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []

        for handle, submissions in data.items():
            for submission, src in submissions.items():
                if submission in train_names:
                    self.y_train.append(self.class_encoder[handle])
                    self.x_train.append(src)
                elif submission in test_names:
                    self.y_test.append(self.class_encoder[handle])
                    self.x_test.append(src)

        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        order = np.random.permutation(np.arange(len(self.x_train)))

        self.y_train = self.y_train[order]
        self.x_train = self.x_train[order]

        self.y_test = np.array(self.y_test)
        self.x_test = np.array(self.x_test)

    def get_n_classes(self):
        return len(self.classes)



    def train(self):
        for pos in range(0, len(self.x_train), self.batch_size):
            yield self.x_train[pos:pos + self.batch_size], torch.from_numpy(np.array(self.y_train[pos:pos + self.batch_size], dtype=np.int64))

    def test(self):
        for pos in range(0, len(self.x_test), self.batch_size):
            yield self.x_test[pos:pos + self.batch_size], torch.from_numpy(np.array(self.y_test[pos:pos + self.batch_size], dtype=np.int64))


def split(d, train_frac, seed):
    np.random.seed(seed)
    train_data = {}
    test_data = {}
    for handle, result_for_handle in sorted(d.items()):
        current_train = {}
        current_test = {}
        n_items = len(result_for_handle)
        train_size = int(train_frac * n_items)
        test_size = n_items - train_size
        result_for_handle = sorted(list(result_for_handle.items()))
        
        result_for_handle = np.random.permutation(result_for_handle)
        
        for i in range(train_size):
            item = result_for_handle[i]
            current_train[item[0]] = item[1]
        
        for i in range(train_size, n_items, 1):
            item = result_for_handle[i]
            current_test[item[0]] = item[1]
            
        train_data[handle] = current_train
        test_data[handle] = current_test
    
    
    return train_data, test_data
        
        
        
        
        

class StratifiedBatcher:
    def __init__(self, data, batch_size, train_frac, seed=42):

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))
        train, test = split(data, train_frac, seed)

        self.train_data = train
        self.test_data = test

        self.class_encoder = {handle:id for id, handle in enumerate(self.classes)}


        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []

        for handle, submissions in sorted(train.items()):
            for submission, src in sorted(submissions.items()):
                    self.y_train.append(self.class_encoder[handle])
                    self.x_train.append(src)


        for handle, submissions in sorted(test.items()):
            for submission, src in sorted(submissions.items()):
                    self.y_test.append(self.class_encoder[handle])
                    self.x_test.append(src)


        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        order = np.random.permutation(np.arange(len(self.x_train)))
        self.y_train = self.y_train[order]
        self.x_train = self.x_train[order]

        self.y_test = np.array(self.y_test)
        self.x_test = np.array(self.x_test)


    def get_n_classes(self):
        return len(self.classes)

    def train(self):
        for pos in range(0, len(self.x_train), self.batch_size):
            yield self.x_train[pos:pos + self.batch_size], torch.from_numpy(np.array(self.y_train[pos:pos + self.batch_size], dtype=np.int64))

    def test(self):
        for pos in range(0, len(self.x_test), self.batch_size):
            yield self.x_test[pos:pos + self.batch_size], torch.from_numpy(np.array(self.y_test[pos:pos + self.batch_size], dtype=np.int64))


class StratifiedBatcherPreprocessed(StratifiedBatcher):
    def __init__(self, data, batch_size, train_frac, seed=42):

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))
        train, test = split(data, train_frac, seed)

        self.train_data = train
        self.test_data = test

        self.class_encoder = {handle:id for id, handle in enumerate(self.classes)}


        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []

        for handle, submissions in sorted(train.items()):
            for submission, src in sorted(submissions.items()):
                    self.y_train.append(self.class_encoder[handle])
                    self.x_train.append(ast.parse(src))


        for handle, submissions in test.items():
            for submission, src in submissions.items():
                    self.y_test.append(self.class_encoder[handle])
                    self.x_test.append(ast.parse(src))


        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        order = np.random.permutation(np.arange(len(self.x_train)))
        self.y_train = self.y_train[order]
        self.x_train = self.x_train[order]

        self.y_test = np.array(self.y_test)
        self.x_test = np.array(self.x_test)

def transform_for_problems(d):
    result = defaultdict(list)
    for handle, result_for_handle in d.items():
        for problem, src in result_for_handle.items():
            result[problem].append(src)
            
    return result


class StratifiedBatcherPreprocessedRandom(StratifiedBatcher):
    def __init__(self, data, batch_size, train_frac, seed=42):

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))
        train, test = split(data, train_frac, seed)

        self.train_data = train
        self.test_data = test

        self.class_encoder = {handle:id for id, handle in enumerate(self.classes)}


        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []

        for handle, submissions in sorted(train.items()):
            for submission, src in sorted(submissions.items()):
                    self.y_train.append(self.class_encoder[handle])
                    self.x_train.append(ast.parse(src))


        for handle, submissions in test.items():
            for submission, src in submissions.items():
                    self.y_test.append(self.class_encoder[handle])
                    self.x_test.append(ast.parse(src))


        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        order = np.random.permutation(np.arange(len(self.x_train)))
        self.y_train = self.y_train[order]
        self.x_train = self.x_train[order]

        self.y_test = np.array(self.y_test)
        self.x_test = np.array(self.x_test)

def transform_for_problems(d):
    result = defaultdict(list)
    for handle, result_for_handle in d.items():
        for problem, src in result_for_handle.items():
            result[problem].append(src)
            
    return result
            
            
def split_problems(d, train_frac, seed):
    np.random.seed(seed)
    train_data = {}
    test_data = {}
    for problem, solutions in d.items():
        n_items = len(solutions)
        train_size = int(train_frac * n_items)
        solutions = np.random.permutation(solutions)
        train_data[problem] = solutions[:train_size]
        test_data[problem] = solutions[train_size:]
    
    
    return train_data, test_data
            

class StratifiedProblemBatcher(StratifiedBatcher):
    def __init__(self, data, batch_size, train_frac, seed=42):

        self.batch_size = batch_size
        self.data = data
        self.classes = list(sorted(data.keys()))
        train, test = split_problems(data, train_frac, seed)

        self.train_data = train
        self.test_data = test

        self.class_encoder = {problem:id for id, problem in enumerate(self.classes)}


        self.y_train = []
        self.x_train = []
        self.y_test = []
        self.x_test = []
        
        for problem, solutions in train.items():
            for src in solutions:
                self.y_train.append(self.class_encoder[problem])
                self.x_train.append(src)
                
                
        for problem, solutions in test.items():
            for src in solutions:
                self.y_test.append(self.class_encoder[problem])
                self.x_test.append(src)



        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        order = np.random.permutation(np.arange(len(self.x_train)))
        self.y_train = self.y_train[order]
        self.x_train = self.x_train[order]

        self.y_test = np.array(self.y_test)
        self.x_test = np.array(self.x_test)



class Trainer:
    def __init__(self, model, loss_object, optimizer):

        torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
                
        self.train_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.all_params = {}
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

                
    def restore(self, path):
        if self.global_iterations > 0:
            raise ValueError(
                "Cannot restore variables to an already trained model! (Trained for {} global iterations)"\
                    .format(self.global_iterations))
            
        self.model.load_state_dict(torch.load(path))


    
    def log_global_iteration(self):
        pass


    def validate(self):
        pass



    
    def train(self, batch_sampler, params):
        self.model.train()

        grads_1 = []
        grads_2 = []
        grads_embeddings = []
        for epoch_id in range(params['n_epochs']):
            self.model.train()
            start_time = time.time()
            for x, y in batch_sampler.train():
                # print(y)
                self.optimizer.zero_grad()
                prediction = self.model(x)
                # print(prediction)
                # print(y)
                loss = self.loss_object(prediction, y)
                regularized_loss = loss + self.model.regularizer() #+ 0.1 * torch.norm(prediction, p = 1)
                regularized_loss.backward()
                # print(regularized_loss)
                
                grads_1.append(self.model.ast_encoder.subtree_network.weight_ih_l0.grad.norm())
                grads_2.append(self.model.ast_encoder.subtree_network.weight_hh_l0.grad.norm())
                grads_embeddings.append(self.model.ast_encoder.name_embedding_layer.weight.grad.norm())
                print(grads_embeddings[-1])

#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.train_metrics['loss'].append(loss.detach().numpy())
                self.train_metrics['regularizer'].append((regularized_loss - loss).detach().numpy())

                display.clear_output(wait=True)
                plt.figure(figsize=(15, 10))
                plt.grid()
                plt.plot(self.train_metrics['loss'], color='red', label='train')
                plt.plot(self.validation_metrics['validation_iterations'], self.validation_metrics['loss'], color='blue', label='val')
                plt.legend()
                plt.show()

                plt.figure(figsize=(15, 10))
                plt.grid()
                plt.plot(self.validation_metrics['validation_iterations'], self.validation_metrics['loss'], color='blue', label='val')
                plt.legend()
                plt.show()

                plt.figure(figsize=(15, 10))
                plt.grid()
                plt.plot(self.train_metrics['regularizer'], color='red', label='train')
                plt.legend()
                plt.show()

                plt.figure(figsize=(15, 10))
                plt.grid()
                plt.plot(self.train_metrics['accuracy'], color='red', label='train')
                plt.plot(self.validation_metrics['accuracy'], color='blue', label='val')
                plt.legend()
                plt.show()

                plt.figure(figsize=(15, 10))
                plt.grid()
                plt.plot(grads_1)
                plt.plot(grads_2)
                plt.plot(grads_embeddings, label='embeddings')
                plt.legend()
                plt.show()
                # print(y)
                # print(prediction)

            print("Epoch took ", time.time() - start_time)
            #print(self.train_metrics['loss'][-1])


            if epoch_id % 2 == 0:
                self.model.eval()
                z = 0
                n_items = 0
                for x, y in batch_sampler.train():
                    prediction = self.model(x)
                    print(prediction)
                    _, prediction = prediction.max(dim=1)
                    z += np.count_nonzero(prediction == y)
                    n_items += len(prediction)
                    print(prediction)

                self.train_metrics['accuracy'].append(z/n_items)
                print(self.train_metrics['accuracy'][-1])


                z = 0
                n_items = 0
                loss_acc = 0
                for x, y in batch_sampler.test():
                    prediction = self.model(x)
                    loss_acc += self.loss_object(prediction, y).detach().numpy() * len(x)
                    _, prediction = prediction.max(dim=1)
                    z += np.count_nonzero(prediction == y)
                    n_items += len(prediction)

                self.validation_metrics['validation_iterations'].append(len(self.train_metrics['loss']))
                self.validation_metrics['loss'].append(loss_acc/n_items)
                self.validation_metrics['accuracy'].append(z/n_items)
                print(self.validation_metrics['accuracy'][-1])

            # raise ValueError()
