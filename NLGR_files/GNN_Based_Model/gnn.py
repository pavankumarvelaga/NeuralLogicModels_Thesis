import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch.nn.init import normal_ as normal_init
import numpy as np
from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks,device, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.device = device

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        elif graph_pooling == "logical":

            self.true_vector = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 0.1, size=self.emb_dim).astype(np.float32)),
            requires_grad=True) 

            # Layers of NOT network
            self.not_layer_1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
            self.not_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
#             self.not_layer_3 = torch.nn.Linear(self.emb_dim, self.emb_dim)
            #self.not_layer_4 = torch.nn.Linear(self.emb_dim, self.emb_dim)
            
            
            # Layers of OR Network 
#             self.and_layer_1 = torch.nn.Linear(2 * self.emb_dim, self.emb_dim)
#             self.and_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
            
            # Layers of AND network
            self.and_layer_1 = torch.nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.and_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
#             self.and_layer_3 = torch.nn.Linear(self.emb_dim, self.emb_dim)
#             self.and_layer_4 = torch.nn.Linear(self.emb_dim, self.emb_dim)

            self.dropout_layer_logic = torch.nn.Dropout(0.01)

#             self.logical_init_weights()
            
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)


#     def logical_init_weights(self):
#         """
#         It initializes all the weights of the neural architecture as reported in the paper.
#         """
#         # not
#         normal_init(self.not_layer_1.weight, mean=0.0, std=0.01)
#         normal_init(self.not_layer_1.bias, mean=0.0, std=0.01)
#         normal_init(self.not_layer_2.weight, mean=0.0, std=0.01)
#         normal_init(self.not_layer_2.bias, mean=0.0, std=0.01)     
# #         # or
# # #         normal_init(self.or_layer_1.weight, mean=0.0, std=0.01)
# # #         normal_init(self.or_layer_1.bias, mean=0.0, std=0.01)
# # #         normal_init(self.or_layer_2.weight, mean=0.0, std=0.01)
# # #         normal_init(self.or_layer_2.bias, mean=0.0, std=0.01)
# #         # and
#         normal_init(self.and_layer_1.weight, mean=0.0, std=0.01)
#         normal_init(self.and_layer_1.bias, mean=0.0, std=0.01)
#         normal_init(self.and_layer_2.weight, mean=0.0, std=0.01)
#         normal_init(self.and_layer_2.bias, mean=0.0, std=0.01)
# #         normal_init(self.and_layer_3.weight, mean=0.0, std=0.01)
# #         normal_init(self.and_layer_3.bias, mean=0.0, std=0.01)
# #         normal_init(self.and_layer_4.weight, mean=0.0, std=0.01)
# #         normal_init(self.and_layer_4.bias, mean=0.0, std=0.01)

    def logic_not(self, vector):

        # ReLU is the activation function selected in the paper
        vector = F.relu(self.not_layer_1(vector))
        if self.training:
            vector = self.dropout_layer_logic(vector)
        out = self.not_layer_2(vector)
#         if self.training:
#             out = self.dropout_layer_logic(out)
#         out = self.not_layer_3(out)
#         if self.training:
#             out = self.dropout_layer_logic(out)
#         out = self.not_layer_4(out)
        return out

    def logic_and(self, vector1, vector2, dim=1):
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer_logic(vector)
        out = self.and_layer_2(vector)
#         if self.training:
#             out = self.dropout_layer_logic(out)
#         out = self.and_layer_3(out)
#         if self.training:
#             out = self.dropout_layer_logic(out)
#         out = self.and_layer_4(out)
        return out
    
    def mse(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        return (vector1 - vector2) ** 2

    def dot_product(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        result = (vector1 * vector2).sum(dim=-1)
        vector1_pow = vector1.pow(2).sum(dim=-1).pow(self.sim_alpha)
        vector2_pow = vector2.pow(2).sum(dim=-1).pow(self.sim_alpha)
        result = result / torch.clamp(vector1_pow * vector2_pow, min=1e-8)
        return result

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * 10
        if sigmoid:
            return result.sigmoid()
        return result

    def padding_tensor(self,sequences):
        num = len(sequences)
        max_len = max([s.shape[0] for s in sequences])
        out_dims = (num, max_len, *sequences[0].shape[1:])
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        mask = sequences[0].data.new(*out_dims).fill_(0)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            out_tensor[i,length:] =  self.true_vector
            mask[i, :length] = 1
        return out_tensor
    
    def forward(self, batched_data,train):
        h_node = self.gnn_node(batched_data)
        constraints = []
        if self.graph_pooling != 'logical':
            h_graph = self.pool(h_node, batched_data.batch)
            prediction = self.graph_pred_linear(h_graph)
            
        else:


            size = int(batched_data.batch.max().item() + 1)
            batch = batched_data.batch
            graph = []
            j=0
            for i in range(size):
                output = []
                k = 0
                while j <h_node.shape[0] and batch[j]==i:
                    output.append(h_node[j])
                    j += 1
                    k += 1
                graph.append(torch.cat(output).reshape(k,h_node.shape[1]))


            h_node = self.padding_tensor(graph)
#            print(h_node.shape)
            output_vec = h_node[:,0,:]
            indxs = list(range(1,h_node.shape[1]))
            if train:
                for i in range(len(indxs)):
                    idx = np.random.choice(indxs,replace=False)
                    output_vec = self.logic_and(output_vec,h_node[:,idx,:])
                    constraints.append(output_vec)
                    indxs.remove(idx)
            else:
                for i in indxs:
                    idx = i
                    output_vec = self.logic_and(output_vec,h_node[:,idx,:])
                    constraints.append(output_vec)
                    
 #           print(constraints,len(constraints))
            if len(constraints) > 0:
                constraints = torch.cat(constraints, dim=1)
		
            #print(constraints,constraints.shape)
            constraints = constraints.view(-1, self.emb_dim)
#            print(constraints,constraints.shape)
            # constraints = torch.cat(constraints,dim=0)
            prediction = self.similarity(output_vec, self.true_vector).view([-1])
        return constraints,prediction

#             while len(h_node) > 1:
#                 op1 = h_node[-1]
#                 op2 = h_node[-2]
#                 h_node = h_node[:-2 or None]
#     #             print(op1)
                
#                 and_res = self.logic_and(op1,op2)
#     #             print(len(struct_list))
#                 h_node.append(and_res)
#                 constraints.append(and_res)
# #         print(constraints)
#             if len(constraints) > 0:
#                 constraints = torch.cat(constraints, dim=1)
#             constraints = constraints.view( constraints.size(1)//self.emb_dim, self.emb_dim)
#             prediction = self.similarity(h_node[0], self.true_vector).view([-1])
#         return constraints,prediction


if __name__ == '__main__':
    GNN(num_tasks = 2)

