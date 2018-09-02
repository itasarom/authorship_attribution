import numpy as np
import torch
import ast
import torch.multiprocessing as mp
import os
import json


# NODE_TYPES = ['AST', 'Add', 'And', 'Assert', 'Assign', 'AsyncFor', 'AsyncFunctionDef', 'AsyncWith', 'Attribute', 'AugAssign', 'AugLoad', 'AugStore', 'Await', 'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp', 'Break', 'Bytes', 'Call', 'ClassDef', 'Compare', 'Continue', 'Del', 'Delete', 'Dict', 'DictComp', 'Div', 'Ellipsis', 'Eq', 'ExceptHandler', 'Expr', 'Expression', 'ExtSlice', 'FloorDiv', 'For', 'FunctionDef', 'GeneratorExp', 'Global', 'Gt', 'GtE', 'If', 'IfExp', 'Import', 'ImportFrom', 'In', 'Index', 'Interactive', 'Invert', 'Is', 'IsNot', 'LShift', 'Lambda', 'List', 'ListComp', 'Load', 'Lt', 'LtE', 'MatMult', 'Mod', 'Module', 'Mult', 'Name', 'NameConstant', 'NodeTransformer', 'NodeVisitor', 'Nonlocal', 'Not', 'NotEq', 'NotIn', 'Num', 'Or', 'Param', 'Pass', 'Pow', 'PyCF_ONLY_AST', 'RShift', 'Raise', 'Return', 'Set', 'SetComp', 'Slice', 'Starred', 'Store', 'Str', 'Sub', 'Subscript', 'Suite', 'Try', 'Tuple', 'UAdd', 'USub', 'UnaryOp', 'While', 'With', 'Yield', 'YieldFrom', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'alias', 'arg', 'arguments', 'boolop', 'cmpop', 'comprehension', 'copy_location', 'dump', 'excepthandler', 'expr', 'expr_context', 'fix_missing_locations', 'get_docstring', 'increment_lineno', 'iter_child_nodes', 'iter_fields', 'keyword', 'literal_eval', 'mod', 'operator', 'parse', 'slice', 'stmt', 'unaryop', 'walk', 'withitem']

NODE_TYPES = ['AST', 'Add', 'And', 'AnnAssign', 'Assert', 'Assign', 'AsyncFor', 'AsyncFunctionDef', 'AsyncWith', 'Attribute', 'AugAssign', 'AugLoad', 'AugStore', 'Await', 'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp', 'Break', 'Bytes', 'Call', 'ClassDef', 'Compare', 'Constant', 'Continue', 'Del', 'Delete', 'Dict', 'DictComp', 'Div', 'Ellipsis', 'Eq', 'ExceptHandler', 'Expr', 'Expression', 'ExtSlice', 'FloorDiv', 'For', 'FormattedValue', 'FunctionDef', 'GeneratorExp', 'Global', 'Gt', 'GtE', 'If', 'IfExp', 'Import', 'ImportFrom', 'In', 'Index', 'Interactive', 'Invert', 'Is', 'IsNot', 'JoinedStr', 'LShift', 'Lambda', 'List', 'ListComp', 'Load', 'Lt', 'LtE', 'MatMult', 'Mod', 'Module', 'Mult', 'Name', 'NameConstant', 'NodeTransformer', 'NodeVisitor', 'Nonlocal', 'Not', 'NotEq', 'NotIn', 'Num', 'Or', 'Param', 'Pass', 'Pow', 'PyCF_ONLY_AST', 'RShift', 'Raise', 'Return', 'Set', 'SetComp', 'Slice', 'Starred', 'Store', 'Str', 'Sub', 'Subscript', 'Suite', 'Try', 'Tuple', 'UAdd', 'USub', 'UnaryOp', 'While', 'With', 'Yield', 'YieldFrom', '_NUM_TYPES', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'alias', 'arg', 'arguments', 'boolop', 'cmpop', 'comprehension', 'copy_location', 'dump', 'excepthandler', 'expr', 'expr_context', 'fix_missing_locations', 'get_docstring', 'increment_lineno', 'iter_child_nodes', 'iter_fields', 'keyword', 'literal_eval', 'mod', 'operator', 'parse', 'slice', 'stmt', 'unaryop', 'walk', 'withitem']

# def read_pretrained_vocabs(embeddings_path):
#         f = open(embeddings_path, "r")
#         embeddings  = []
#         words = []

#         n_words, embedding_dim = map(int, f.readline().strip().split())
#         bad_words = 0

#         word_set = set()
#         for line in f:
#             line = line.strip().split(" ")
#             word = line[0]
#             vec = np.array(list(map(float, line[1:]))).reshape(1, -1)

#             word_set.add(word)
#             words.append(word) 
#             embeddings.append(vec)
#             assert embedding_dim == vec.shape[1], (embedding_dim, vec.shape[1], vec)
                
        
#         # print(self.embeddings[0])
#         transformation = dict(zip(words, range(len(words))))
#         embeddings  = torch.from_numpy(np.vstack(embeddings))

#         return embedding_dim, transformation, embeddings


class EmbeddingVisitor(ast.NodeVisitor):
    def __init__(self, embedding_layer, subtree_network, params):
        super(EmbeddingVisitor, self).__init__()
        self.params = params
        self.embedding_dim = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer
        self.subtree_network = subtree_network

#         self.embedding_network = embedding_network
        self.mapping = {name : index for index, name in enumerate(NODE_TYPES)}

    def visit(self, node):
        return self.embed_node(node)

    def node_to_index(self, node):
        return self.mapping[node.__class__.__name__]

    def embed_node_proper(self, node):
        tmp = np.array([self.node_to_index(node)])
        # print(tmp)
        return self.embedding_layer(torch.from_numpy(tmp))
        # result = torch.zeros(1, self.embedding_dim, requires_grad=False)
        # result[0, self.node_to_index(node)] = 1
        # return result

    def embed_node(self, node):
        node_embedding = self.embed_node_proper(node)
        result = node_embedding
        # print("Embedding ", node)
        if len(list(ast.iter_child_nodes(node))):
            embeddings = [node_embedding]
            embeddings += self.embed_subtree(node)
            # result = self.embedding_network(torch.cat((node_embedding, subtree_embedding), -1))
            # c_0 = torch.zeros(1, self.embedding_dim)
            # h_0 = torch.zeros(1, self.embedding_dim)

            embeddings = torch.cat(embeddings, 0).unsqueeze(0)


            # print(len(embeddings))
            # for node_embedding in embeddings:
                # (c_0, h_0) = self.subtree_network(node_embedding, (c_0, h_0))

            lstm_result, _ = self.subtree_network(embeddings)

            # print(lstm_result[0])
            # print("Ended embedding", node)
            # print(lstm_result.size())
            result = lstm_result[0][-1]
        result = torch.nn.functional.dropout(result, self.params['dropout'], training=self.subtree_network.training)
            # return 

            # print(result)

        return result

    def embed_subtree(self, node):
        children_embeddings = []
        # c_0 = torch.zeros(1, self.embedding_dim)  
        # h_0 = torch.zeros(1, self.embedding_dim)
        # n_children = 0
        # print("Embedding the subtree of ", node)
        # print("\t\t\t", list(ast.iter_child_nodes(node)))
        for child in ast.iter_child_nodes(node):
            # print(child)
            child_embedding = self.embed_node(child).view(1, -1)
            children_embeddings.append(child_embedding)
            # (c_0, h_0) = self.subtree_network(child_embedding, (c_0, h_0))
            # h_0 += child_embedding
            # n_children += 1

        # h_0 /= n_children
        # indices = torch.from_numpy()
        # embeddings = self.embedding_layer(indices)
        
        # children_embeddings = torch.cat(children_embeddings, 0)[None, :, :]
        # print(children_embeddings)
        
        return children_embeddings


def init_ast_embeddings(n_nodes, params):
    embedding_layer = torch.nn.Embedding(n_nodes, params['embedding_dims'])
    torch.nn.init.xavier_normal_(embedding_layer.weight)
#     if not pre_init:
    return embedding_layer

#     embedding_dim, transformation, embeddings = read_pretrained_vocabs("../training_embeddings_tree/trained_embeddings_2.vec")

#     new_weight = torch.rand(n_nodes, embedding_dims)
#     for word, id in transformation.items():
#         if word not in NODE_TYPES:
#             continue 
#         else:
#             new_weight[NODE_TYPES.index(word)] = embeddings[id]


#     embedding_layer.weight.data = new_weight

#     return embedding_layer


class ASTEncoder(torch.nn.Module):
    def __init__(self, params):
        super(ASTEncoder, self).__init__()
        n_nodes = len(NODE_TYPES)
        
        params['NODE_TYPES'] = NODE_TYPES
        # embedding_dims = n_nodes
        self.embedding_dims = params['embedding_dims']
        self.params = params
        # self.subtree_network = torch.nn.LSTMCell(embedding_dims, embedding_dims)
        self.subtree_network = torch.nn.LSTM(self.embedding_dims, self.embedding_dims, num_layers=params['num_layers'], dropout=params['rnn_dropout'], batch_first=True)

        torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l0)
        torch.nn.init.constant_(self.subtree_network.bias_ih_l0, 0)
        torch.nn.init.constant_(self.subtree_network.bias_hh_l0, 1)

        # torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l1)
        # torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l1)
        # torch.nn.init.constant_(self.subtree_network.bias_ih_l1, 0)
        # torch.nn.init.constant_(self.subtree_network.bias_ih_l1, 1)

        # torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l2)
        # torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l2)
        # torch.nn.init.constant_(self.subtree_network.bias_ih_l2, 0)
        # torch.nn.init.constant_(self.subtree_network.bias_ih_l2, 1)
        #self.embedding_network = torch.nn.Sequential(
        #                        torch.nn.Linear(2 * embedding_dims, 256),
        #                        torch.nn.ReLU(),
        #                        torch.nn.Linear(256, embedding_dims)
        #                    )
        
        self.embedding_layer = init_ast_embeddings(n_nodes, self.params)
        
        # self.embedding_layer.weight.

        self.visitor = EmbeddingVisitor(self.embedding_layer, self.subtree_network, self.params)
        
    def save(self, path):
        torch.save(self.embedding_layer.state_dict(), os.path.join(path, "ast_embeddings.tc"))
        torch.save(self.subtree_network.state_dict(), os.path.join(path, "subtree_network.tc"))
        with open(os.path.join(path, "ast_encoder_params.json"), "w") as f:
            f.write(json.dumps(self.params, sort_keys=True, indent=4))
    
    def load(self, path):
        self.embedding_layer.load_state_dict(torch.load(os.path.join(path, "ast_embeddings.tc")))
        self.subtree_network.load_state_dict(torch.load(os.path.join(path, "subtree_network.tc")))
        with open(os.path.join(path, "ast_encoder_params.json")) as f:
            self.params = json.loads(f.read())
    
        
    def forward(self, node):
        return self.visitor.visit(node)


class Model(torch.nn.Module):
    def __init__(self, params):
        # print(self.__class__)
        super(Model, self).__init__()
        self.params = params
        n_classes = self.params['n_classes']
#         embedding_dims = self.params['embedding_dims']
        
        self.ast_encoder = ASTEncoder(params['encoder_params'])
        self.softmax_head = torch.nn.Sequential(torch.nn.Linear(self.ast_encoder.embedding_dims, n_classes))
        torch.nn.init.xavier_normal_(self.softmax_head[0].weight)


#         self.preprocessed = preprocessed
        # self.softmax_head = torch.nn.Sequential(
                        ## torch.nn.BatchNorm1d(self.ast_encoder.embedding_dims),
                        # torch.nn.Linear(self.ast_encoder.embedding_dims, 256),
                        # torch.nn.ReLU(),
                        # torch.nn.Linear(256, n_classes),
                        # torch.nn.Softmax(dim=1)
                    # )
                    
    def save(self, path):
        self.ast_encoder.save(path)
        torch.save(self.softmax_head.state_dict(), os.path.join(path, "softmax_head.tc"))
        with open(os.path.join(path, "model_params.json"), "w") as f:
            f.write(json.dumps(self.params, sort_keys=True, indent=4))
    
    def load(self, path):
        self.ast_encoder.load(path)
        self.softmax_head.load_state_dict(torch.load(os.path.join(path, "softmax_head.tc")))
        with open(os.path.join(path, "model_params.json")) as f:
            self.params = json.loads(f.read())

    def transform_batch(self, input):
        result = torch.zeros(len(input), self.ast_encoder.embedding_dims)

#         if self.preprocessed:
#             for id, root in enumerate(input):
#                 result[id] = self.ast_encoder(root)
#         else:
        for id, code in enumerate(input):
                result[id] = self.ast_encoder(ast.parse(code))

        return result
    
    """
    def transform_one(self, queue, process_id, batch):
        result = torch.zeros(len(batch), self.ast_encoder.embedding_dims)
        if self.preprocessed:
            for id, root in enumerate(batch):
                result[id] = self.ast_encoder(root)
        else:
            for id, code in enumerate(batch):
                result[id] = self.ast_encoder(ast.parse(code))

        queue.put(result)
        queue.close()
        queue.join_thread()

    def transform_batch(self, input):
        num_processes = 4
        self.share_memory()
        processes = []
        result_queue = mp.Queue()
        for rank in range(num_processes):
            p = mp.Process(target=self.transform_one, args=(result_queue, rank, input))
            p.start()
            processes.append(p)


        for p in processes:
            p.join()


        result = []

        while not result_queue.empty():
            result.append(result_queue.get())


        print(result)

        result = torch.cat(result, 0)

        return result
    """

    def regularizer(self):
        result = torch.sum(self.ast_encoder.subtree_network.weight_ih_l0 ** 2) + \
                torch.sum(self.ast_encoder.subtree_network.weight_hh_l0 ** 2) + \
                torch.sum(self.ast_encoder.subtree_network.bias_ih_l0 ** 2) + \
                torch.sum(self.ast_encoder.subtree_network.bias_hh_l0 ** 2) + \
                torch.sum(self.softmax_head[0].weight ** 2) + \
                torch.sum(self.ast_encoder.embedding_layer.weight ** 2)
                # result = torch.norm(self.ast_encoder.subtree_network.weight_ih_l0) ** 2 + \
                # torch.norm(self.ast_encoder.subtree_network.weight_hh_l0) ** 2 + 
                # torch.norm(self.ast_encoder.subtree_network.weight_ih_l1) + \
                # torch.norm(self.ast_encoder.subtree_network.weight_hh_l1) + \
                # torch.norm(self.ast_encoder.subtree_network.weight_ih_l2) + \
                # torch.norm(self.ast_encoder.subtree_network.weight_hh_l2)


#         return result
        return self.params['regularizer_alpha'] * result


    def forward(self, input):

        embeddings = self.transform_batch(input)

        # print(embeddings)

        result = self.softmax_head(embeddings)

        return result
    
    def get_representations(self, input):
        return self.transform_batch(input)
    
    def predict_proba(self, input):
        return torch.nn.functional.softmax(self.forward(input), dim=1)
    
    def predict(self, input):
        return self.predict_proba(input).max(dim=1)
    
    
    def predict_proba_batches(self, input, batch_size=64):
        result = []
        for pos in range(0, len(input), batch_size):
            result.append(self.predict_proba(input[pos:pos+batch_size]))
        
        result = torch.cat(result)
        
        return result
    
    def predict_batches(self, input, batch_size=64):
        result = self.predict_proba_batches(input).max(dim=1)
        
        return result
        



#==================================================
#
#
#
#
#
#
#==================================================
import pickle
class NameEmbeddingVisitor(EmbeddingVisitor):
    def __init__(self, embedding_layer, name_mapping,  name_embedding_layer, name_combiner, subtree_network, params):
        super(self.__class__, self).__init__(embedding_layer, subtree_network, params)
        
        self.OOV_ID = params['OOV_ID']
        self.nodes_with_names = {"Name", "Attribute"}
#         self.params['OOV_ID'] = self.OOV_ID
        self.params['nodes_with_names'] = list(self.nodes_with_names)

        
        self.name_mapping = name_mapping
        self.name_combiner = name_combiner
        self.name_embedding_layer = name_embedding_layer

        
        # self.embedding_dim = embedding_layer.embedding_dim
        # self.embedding_layer = embedding_layer
        # self.subtree_network = subtree_network

        # self.embedding_network = embedding_network
        # self.mapping = {name : index for index, name in enumerate(NODE_TYPES)}


    def node_to_index(self, node):
        return self.mapping[node.__class__.__name__]

    def embed_node_proper(self, node):
        node_type_id = torch.from_numpy(np.array([self.node_to_index(node)]))
        node_type_embedding = self.embedding_layer(node_type_id)
        result = node_type_embedding

        node_class = node.__class__.__name__
        if node_class in self.nodes_with_names:
            if node_class == "Name":
                name = node.id
            elif node_class == "Attribute":
                name = node.attr

            # print("Yeeeee!")

            node_name_id = torch.from_numpy(np.array([self.name_mapping.get(name, self.OOV_ID)]))
            # print(self.name_mapping.get(name, self.OOV_ID))
            node_name_embedding = self.name_embedding_layer(node_name_id).view(1, -1)
            node_type_embedding = node_type_embedding.view(1, -1)

            combiner_input = torch.cat([node_type_embedding, node_name_embedding], dim=1)

            result = self.name_combiner(combiner_input)


        return result 


class NameASTEncoder(ASTEncoder):
#     embedding_dims, name_embedding_dims, combiner_dims
    def __init__(self, params):
        super(NameASTEncoder, self).__init__(params)
        n_nodes = len(NODE_TYPES)
        
        embedding_dims = params['embedding_dims']
#         name_embedding_dims, combiner_dims
        # self.embedding_dims = embedding_dims
        # # self.subtree_network = torch.nn.LSTMCell(embedding_dims, embedding_dims)
        # self.subtree_network = torch.nn.LSTM(embedding_dims, embedding_dims, num_layers=1, dropout=0.0, batch_first=True)

        # torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l0)
        # torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l0)
        # torch.nn.init.constant_(self.subtree_network.bias_ih_l0, 0)
        # torch.nn.init.constant_(self.subtree_network.bias_hh_l0, 1)

        # self.embedding_layer = init_ast_embeddings(n_nodes, embedding_dims, False)

#         most_common_names_file = "most_common_names.pkl"
        most_common_names_file = self.params['most_common_names_file']
        with open(most_common_names_file, "rb") as f:
            names = pickle.load(f)

        self.name_mapping = {name : params["OOV_ID"] + index + 1 for index, (name, count) in enumerate(names)}
        self.name_mapping["<OOV>"] = params["OOV_ID"]
        self.params['name_mapping'] = self.name_mapping


        self.name_embedding_layer = init_ast_embeddings(len(self.name_mapping), params)
        self.name_combiner = torch.nn.Sequential(
                torch.nn.Linear(params['embedding_dims'] + params['name_embedding_dims'], params['combiner_dims']),
                torch.nn.ReLU(),
                torch.nn.Linear(params['combiner_dims'], params['embedding_dims'])
            )

        self.visitor = NameEmbeddingVisitor(self.embedding_layer, self.name_mapping, self.name_embedding_layer, self.name_combiner, self.subtree_network, params)
        
    def save(self, path):
        super(NameASTEncoder, self).save(path)
        torch.save(self.name_embedding_layer.state_dict(), os.path.join(path, "name_embeddings.tc"))
        torch.save(self.name_combiner.state_dict(), os.path.join(path, "name_combiner.tct"))
        
    def load(self, load):
        super(NameASTEncoder, self).load(path)
        self.name_embedding_layer.load_state_dict(torch.load(os.path.join(path, "name_embeddings.tc")))
        self.name_combiner.load_state_dict(torch.load(os.path.join(path, "name_combiner.tct")))


class NameModel(Model):
    def __init__(self, params):
        super(self.__class__, self).__init__(params)
        self.ast_encoder = NameASTEncoder(params['encoder_params'])
#         self.softmax_head = torch.nn.Sequential(torch.nn.Linear(self.ast_encoder.embedding_dims, n_classes))

    def regularizer(self):
#         print(super(Model
        result = super(NameModel, self).regularizer()
        current =  torch.sum(self.ast_encoder.name_combiner[0].weight ** 2) + \
                torch.sum(self.ast_encoder.name_combiner[2].weight ** 2) + \
                torch.sum(self.ast_encoder.name_embedding_layer.weight ** 2)
            
        result += self.params.get('regularizer_alpha_names', self.params['regularizer_alpha']) * current
        
        return result