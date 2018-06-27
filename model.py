import numpy as np
import torch
import ast


NODE_TYPES = ['AST', 'Add', 'And', 'Assert', 'Assign', 'AsyncFor', 'AsyncFunctionDef', 'AsyncWith', 'Attribute', 'AugAssign', 'AugLoad', 'AugStore', 'Await', 'BinOp', 'BitAnd', 'BitOr', 'BitXor', 'BoolOp', 'Break', 'Bytes', 'Call', 'ClassDef', 'Compare', 'Continue', 'Del', 'Delete', 'Dict', 'DictComp', 'Div', 'Ellipsis', 'Eq', 'ExceptHandler', 'Expr', 'Expression', 'ExtSlice', 'FloorDiv', 'For', 'FunctionDef', 'GeneratorExp', 'Global', 'Gt', 'GtE', 'If', 'IfExp', 'Import', 'ImportFrom', 'In', 'Index', 'Interactive', 'Invert', 'Is', 'IsNot', 'LShift', 'Lambda', 'List', 'ListComp', 'Load', 'Lt', 'LtE', 'MatMult', 'Mod', 'Module', 'Mult', 'Name', 'NameConstant', 'NodeTransformer', 'NodeVisitor', 'Nonlocal', 'Not', 'NotEq', 'NotIn', 'Num', 'Or', 'Param', 'Pass', 'Pow', 'PyCF_ONLY_AST', 'RShift', 'Raise', 'Return', 'Set', 'SetComp', 'Slice', 'Starred', 'Store', 'Str', 'Sub', 'Subscript', 'Suite', 'Try', 'Tuple', 'UAdd', 'USub', 'UnaryOp', 'While', 'With', 'Yield', 'YieldFrom', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'alias', 'arg', 'arguments', 'boolop', 'cmpop', 'comprehension', 'copy_location', 'dump', 'excepthandler', 'expr', 'expr_context', 'fix_missing_locations', 'get_docstring', 'increment_lineno', 'iter_child_nodes', 'iter_fields', 'keyword', 'literal_eval', 'mod', 'operator', 'parse', 'slice', 'stmt', 'unaryop', 'walk', 'withitem']

def read_pretrained_vocabs(embeddings_path):
        f = open(embeddings_path, "r")
        embeddings  = []
        words = []

        n_words, embedding_dim = map(int, f.readline().strip().split())
        bad_words = 0

        word_set = set()
        for line in f:
            line = line.strip().split(" ")
            word = line[0]
            vec = np.array(list(map(float, line[1:]))).reshape(1, -1)

            word_set.add(word)
            words.append(word) 
            embeddings.append(vec)
            assert embedding_dim == vec.shape[1], (embedding_dim, vec.shape[1], vec)
                
        
        # print(self.embeddings[0])
        transformation = dict(zip(words, range(len(words))))
        embeddings  = torch.from_numpy(np.vstack(embeddings))

        return embedding_dim, transformation, embeddings


class EmbeddingVisitor(ast.NodeVisitor):
    def __init__(self, embedding_layer, subtree_network, embedding_network):
        super(self.__class__, self).__init__()
        self.embedding_dim = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer
        self.subtree_network = subtree_network

        self.embedding_network = embedding_network
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

            return lstm_result[0][-1]

        return node_embedding

    def embed_subtree(self, node):
        children_embeddings = []
        # c_0 = torch.zeros(1, self.embedding_dim)  
        # h_0 = torch.zeros(1, self.embedding_dim)
        # n_children = 0

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


def init_ast_embeddings(n_nodes, embedding_dims, pre_init=False):
    embedding_layer = torch.nn.Embedding(n_nodes, embedding_dims)
    if not pre_init:
        return embedding_layer

    embedding_dim, transformation, embeddings = read_pretrained_vocabs("./pretrained_vectors.txt")

    new_weight = torch.rand(n_nodes, embedding_dims)
    for word, id in transformation.items():
        if word not in NODE_TYPES:
            continue 
        else:
            new_weight[NODE_TYPES.index(word)] = embeddings[id]


    embedding_layer.weight.data = new_weight

    return embedding_layer


class ASTEncoder(torch.nn.Module):
    def __init__(self, embedding_dims):
        super(self.__class__, self).__init__()
        n_nodes = len(NODE_TYPES)

        # embedding_dims = n_nodes
        self.embedding_dims = embedding_dims
        # self.subtree_network = torch.nn.LSTMCell(embedding_dims, embedding_dims)
        self.subtree_network = torch.nn.LSTM(embedding_dims, embedding_dims, num_layers=2, dropout=0.2, batch_first=True)

        torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l0)
        torch.nn.init.constant_(self.subtree_network.bias_ih_l0, 0)
        torch.nn.init.constant_(self.subtree_network.bias_ih_l0, 1)

        torch.nn.init.xavier_normal_(self.subtree_network.weight_ih_l1)
        torch.nn.init.xavier_normal_(self.subtree_network.weight_hh_l1)
        torch.nn.init.constant_(self.subtree_network.bias_ih_l1, 0)
        torch.nn.init.constant_(self.subtree_network.bias_ih_l1, 1)
        #self.embedding_network = torch.nn.Sequential(
        #                        torch.nn.Linear(2 * embedding_dims, 256),
        #                        torch.nn.ReLU(),
        #                        torch.nn.Linear(256, embedding_dims)
        #                    )
        
        self.embedding_layer = init_ast_embeddings(n_nodes, embedding_dims, False)
        
        # self.embedding_layer.weight.

        self.visitor = EmbeddingVisitor(self.embedding_layer, self.subtree_network, None)

    def forward(self, node):
        return self.visitor.visit(node)


class Model(torch.nn.Module):
    def __init__(self, n_classes, embedding_dims):
        super(self.__class__, self).__init__()
        self.ast_encoder = ASTEncoder(embedding_dims)
        self.softmax_head = torch.nn.Sequential(torch.nn.Linear(self.ast_encoder.embedding_dims, n_classes))
        # self.softmax_head = torch.nn.Sequential(
                        ## torch.nn.BatchNorm1d(self.ast_encoder.embedding_dims),
                        # torch.nn.Linear(self.ast_encoder.embedding_dims, 256),
                        # torch.nn.ReLU(),
                        # torch.nn.Linear(256, n_classes),
                        # torch.nn.Softmax(dim=1)
                    # )



    def transform_batch(self, input):
        result = torch.zeros(len(input), self.ast_encoder.embedding_dims)

        for id, code in enumerate(input):
            result[id] = self.ast_encoder(ast.parse(code))

        return result

    def forward(self, input):

        embeddings = self.transform_batch(input)

        # print(embeddings)

        result = self.softmax_head(embeddings)

        return result


