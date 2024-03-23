import torch
import torch.nn as nn
from lib.mlp import MLP

class Embedder(nn.Module):
    def __init__(self,grid_node_features,embed_feature_latent_dim,mesh_node_features,
    mesh_edge_features,grid2mesh_edge_features,mesh2grid_edge_features):
        super(Embedder, self).__init__()

        # Define MLPs for each feature type
        self.mlp_grid_node = MLP(grid_node_features, embed_feature_latent_dim)
        self.mlp_mesh_node = MLP(mesh_node_features, embed_feature_latent_dim)
        self.mlp_mesh_edge = MLP(mesh_edge_features, embed_feature_latent_dim)
        self.mlp_g2m_edge = MLP(grid2mesh_edge_features, embed_feature_latent_dim)
        self.mlp_m2g_edge = MLP(mesh2grid_edge_features, embed_feature_latent_dim)

    def forward(self, vg, vm, em, eg2m, em2g):
        # Embedding the features for all nodes and edges
        vg_embedded = self.mlp_grid_node(vg)
        vm_embedded = self.mlp_mesh_node(vm)
        em_embedded = self.mlp_mesh_edge(em)
        eg2m_embedded = self.mlp_g2m_edge(eg2m)
        em2g_embedded = self.mlp_m2g_edge(em2g)

        return vg_embedded, vm_embedded, em_embedded, eg2m_embedded, em2g_embedded
