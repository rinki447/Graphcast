import torch
import torch.nn as nn
from mlp import MLP

class message_pass(nn.Module):
    def __init__(self,mesh_nodes,mesh_edges):
      super(message_pass,self).__init__()
     
      self.adj_mat=torch.randint(high=2, size=(mesh_nodes, mesh_edges)).float()
      print("matrix for message passing in processor:Row(Node) * Column(Edge)=",self.adj_mat)
      print("message passing matrix of processor has shape=",self.adj_mat.shape)
     
    def forward(self,em_updated,vm_updated):
     
      edge_sum= torch.matmul(self.adj_mat,em_updated)
      vm_processor=torch.cat((vm_updated,edge_sum),dim=1)
      return vm_processor
     
     
class Mesh_GNN(nn.Module):
    def __init__(self,mesh_edge_features,mesh_node_features,mesh_hidden_dim,embed_feature_latent_dim):
      super(Mesh_GNN, self).__init__()
     
      # MLP for updating mesh edge features 
      self.em_MLP = MLP(embed_feature_latent_dim*3,mesh_hidden_dim)
        
      # MLP for updating mesh node features
      self.vm_MLP = MLP(embed_feature_latent_dim*2,mesh_hidden_dim)


    def forward(self,vm_updated,em_embedded,mesh_nodes,mesh_edges):
     
      # create a torch tensor containing infor about source and destination mesh node number for all mesh edges
        ## position of each column of tensor: index of mesh edge
        ## first row of the torch tensor: index of source mesh node
        ## second row of the torch tensor: index of destination mesh node
      mesh_edge_indices = torch.randint(high=mesh_nodes, size=(2, mesh_edges))


      #updates each of the mesh edges using information of the adjacent mesh nodes
      vm_source=vm_updated[mesh_edge_indices[0, :]]
      vm_dst=vm_updated[mesh_edge_indices[1, :]]

      em_processor=torch.cat((em_embedded,vm_source,vm_dst),dim=1)

      em_new=self.em_MLP(em_processor)

      #updates each of the mesh nodes, aggregating information from all of the edges arriving at that mesh node
      GNN_part1=message_pass(mesh_nodes,mesh_edges)
      vm_changed = GNN_part1(em_new, vm_updated)

      vm_new=self.vm_MLP(vm_changed)

      return vm_new,em_new