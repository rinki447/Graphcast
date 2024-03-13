# **Graphcast:**

import torch
import torch.nn as nn

### All values defined in the paper (Following section 3.3,  Pg 26 and beyond)
'''V(G):At 0.25° resolution, there is a total of 721 × 1440 = 1, 038, 240 grid nodes, each with (5 surface variables + 6 atmospheric variables
× 37 levels) × 2 steps + 5 forcings × 3 steps + 5 constant = 474 input features.'''
grid_nodes = 103  #8240
grid_node_features = 474


'''v(M): features associated with each mesh node include the cosine of the latitude, and the sine and cosine of the longitude. '''
mesh_nodes = 40  #962
mesh_node_features = 3

'''e(M):For each edge 𝑒(s →𝑣) connecting a sender mesh node 𝑣 to a receiver mesh node 𝑣, we build
edge features using the position on the unit sphere of the mesh nodes. This includes the
length of the edge, and the vector difference between the 3d positions of the sender node and the
receiver node computed in a local coordinate system of the receiver....
total of 327,660 mesh edges (See Table 4), each with 4 input features.'''
mesh_edges = 32  #7660
mesh_edge_features = 4

'''e(G->M): unidirectional edges that connect sender grid nodes to receiver mesh nodes.
 Features are built the same way as those for the mesh edges. This results on a total of 1,618,746 Grid2Mesh edges, each with 4 input features.'''
grid2mesh_edge = 161  #8746
grid2mesh_edge_features = 4


mesh2grid_edge = 311  #4720
mesh2grid_edge_features = 4


'''embedding dim is not specified particularly in the paper, but hidden and o/p dim are given as 512.
 So mathematically embedded_feature_latent dim has to be 512'''
grid_hidden_dim = 512
mesh_hidden_dim = 512
edge_hidden_dim = 512
embed_feature_latent_dim = 512

processor_num_layers = 16

""" As only inference steps are mimicked, we have not trained MLP weights. However, even for inference, we need the trained weights for the MLPs. So we initialized these weights :randomly."""

## Standard MLP for embedding
''' instead of “swish” activation function, used RELU for all MLPs '''
''' hidden layer and output layer has same dimension (512) for encoder and processor'''
''' weights of these MLP are randomly initialized and used for single pass'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):

        return self.layers(x)

class Embedder(nn.Module):
    def __init__(self):
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


"""# ***Encoder oparations:***"""

# Creating dummy features
vg = torch.randn(grid_nodes, grid_node_features)
vm = torch.randn(mesh_nodes, mesh_node_features)

# For mesh edges, assuming a simple source-target format for the edges, with random features
em = torch.randn(mesh_edges, mesh_edge_features)

# For grid to mesh and mesh to grid edges, initializing edge features
eg2m = torch.randn(grid2mesh_edge, grid2mesh_edge_features)
em2g = torch.randn(mesh2grid_edge, mesh2grid_edge_features)

#Encoder operations:

encoder = Embedder()

## One dummy forward pass through Encoder
vg_embedded, vm_embedded, em_embedded, eg2m_embedded, em2g_embedded = encoder(vg, vm, em, eg2m, em2g)

# Output shapes from Encoder
print("vg_embedded shape:", vg_embedded.shape)
print("vm_embedded shape:", vm_embedded.shape)
print("em_embedded shape:", em_embedded.shape)
print("eg2m_embedded shape:", eg2m_embedded.shape)
print("em2g_embedded shape:", em2g_embedded.shape)



# Assuming edge_indices is a tensor indicating the connections from grid nodes to mesh nodes
# For simplicity, randomly generated indices are used

## Since it is a bipartite graph from grid nodes to edge nodes, first column is the source node (grid) and the second column is the destination node(mesh)
edge_grid_source = torch.randint(0, grid_nodes, (1, grid2mesh_edge), dtype=torch.long)
edge_mesh_dst = torch.randint(0, mesh_nodes, (1, grid2mesh_edge), dtype=torch.long)

## Final concatenated bipartite graph
edge_indices_g2m = torch.cat((edge_grid_source, edge_mesh_dst), dim=0)


# Instantiate the Grid2MeshGNN model
grid2mesh_gnn = Grid2Mesh(embed_feature_latent_dim,grid_hidden_dim, mesh_hidden_dim, edge_hidden_dim)

# Use outputs from Encoder and perform forward pass through the GNN to Update features using the Grid2MeshGNN
vg_updated, vm_updated, eg2m_updated = grid2mesh_gnn(vg_embedded, vm_embedded, eg2m_embedded, edge_indices_g2m)

print(f"Updated Grid Node Features Shape: {vg_updated.shape}")
print(f"Updated Mesh Node Features Shape: {vm_updated.shape}")
print(f"Updated Grid2Mesh Edge Features Shape: {eg2m_updated.shape}")

"""# ***Processor:***

* Message passing should have 6 loops- starting from M^6 to M^0
* For each M^R: mesh node features are updated using neighbouring edge features.
* The list of neighbours for each M^R are different. For example, neighbour of a node in M^6 icosahedron mesh are:{1,2,3,4,5}, but neighbour of that node in M^5 icosahedron mesh are:{6,7,8,9,10}. Also, nodes:{6,7,8,9,10} are more distant than nodes:{1,2,3,4,5} from the concerned node.
* Creation of this neighbour list for each M^R icosahedron based mesh, requires information about (x,y,z) coordinates of those mesh nodes.These, (x,y,z) mesh node coordinate values should be calculated using M^R icosahedron mesh edge length and angle between edges of that icosahedron.

* **I have skipped these calculations for brevity of the code. Also, I consider that, I have just 1 mesh, so the following operations are performed once and not in loops. Information of neighbours of a node are not generated using geometry of icosahedron. Instead they are randomly generated**.
"""


#define the message passing using modified version of adjancency matrix
    ## the modified adjacency matrix's row indices are mesh_node indices and column indices are mesh_edge indices
    ## summed feature of neighbouring edges for all mesh_nodes=adj_matrix * mesh_edge feature matrix


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
     def __init__(self,mesh_edge_features,mesh_node_features,mesh_hidden_dim):
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
     
# the representations are updated with a residual connection
Mesh=Mesh_GNN(mesh_edge_features,mesh_node_features,mesh_hidden_dim)
vm_new,em_new=Mesh(vm_updated,em_embedded,mesh_nodes,mesh_edges)
vm_final=vm_updated+vm_new
em_final=em_embedded+em_new
print("After processor, mesh node shape =",vm_final.shape)
print("After processor, mesh edge shape =",em_final.shape)
