import torch
import torch.nn as nn
from lib.mlp import MLP

# Define the Grid2MeshGNN module
class Grid2Mesh(nn.Module):
    def __init__(self, embed_feature_latent_dim, grid_hidden_dim, mesh_hidden_dim, edge_hidden_dim):
        super(Grid2Mesh, self).__init__()
        # MLP for updating grid-to-mesh edge features which takes concatenated features of a grid node, a mesh node, and the edge as input
        
        self.edge_update_mlp = MLP(embed_feature_latent_dim *3, edge_hidden_dim)

        # MLP for updating mesh node features which processes the aggregated edge features and the original mesh nodefeatures
        
        self.mesh_node_update_mlp = MLP(embed_feature_latent_dim+mesh_hidden_dim, mesh_hidden_dim)

        # MLP for updating grid node features (aggregation not added)
        self.grid_node_update_mlp = MLP(embed_feature_latent_dim, grid_hidden_dim)

    def forward(self, vg_embed, vm_embed, eg2m_embed, edge_indices):

        # Obtain features for source nodes (grid nodes) using the first row of edge_indices
        src_features = vg_embed[edge_indices[0, :]]

        # Obtain features for destination nodes (mesh nodes) using the second row of edge_indices
        dst_features = vm_embed[edge_indices[1, :]]

        # Concatenate source node features, destination node features, and edge features for each edge
        edge_features = torch.cat((src_features, dst_features, eg2m_embed), dim=1)


        # Update edge features using the defined MLP
        eg2m_embed_updated = self.edge_update_mlp(edge_features)


        ### ToDo: This part needs to be replaced by GNN libraries

        # Initialize a tensor to accumulate aggregated features for each mesh node
        vm_embed_aggregated = torch.zeros(vm_embed.shape[0],eg2m_embed_updated.shape[1])

        # Iterate over each edge to aggregate the updated edge features to the corresponding destination node
        for i, dst_idx in enumerate(edge_indices[1, :]):
            vm_embed_aggregated[dst_idx] += eg2m_embed_updated[i]

        # Update mesh node features by combining original features with aggregated edge features using MLP
        vm_embed_updated = self.mesh_node_update_mlp(torch.cat((vm_embed, vm_embed_aggregated), dim=1))

        # Update grid nodes
        # Directly update grid node features using the MLP defined
        vg_embed_updated = self.grid_node_update_mlp(vg_embed)

        ##### Apply residual connections to all nodes and edges
        # Addding the original features to the updated features for grid nodes, mesh nodes, and edges
        vg_embed_final = vg_embed + vg_embed_updated  # Final grid node features
        vm_embed_final = vm_embed + vm_embed_updated  # Final mesh node features
        eg2m_embed_final = eg2m_embed + eg2m_embed_updated  # Final edge features

        return vg_embed_final, vm_embed_final, eg2m_embed_final
