import torch
import torch.nn as nn

from lib.embed import Embedder
from lib.g2m import Grid2Mesh
from lib.processor import Mesh_GNN

# main.py
def main():
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

    """# ***Encoder oparations:***"""
    # Creating dummy featuresvg = torch.randn(grid_nodes, grid_node_features)vm = torch.randn(mesh_nodes, mesh_node_features)
    vg = torch.randn(grid_nodes, grid_node_features)
    vm = torch.randn(mesh_nodes, mesh_node_features)

    # For mesh edges, assuming a simple source-target format for the edges, with random featuresem = torch.randn(mesh_edges, mesh_edge_features)
    em = torch.randn(mesh_edges, mesh_edge_features)
    
    # For grid to mesh and mesh to grid edges, initializing edge featureseg2m = torch.randn(grid2mesh_edge, grid2mesh_edge_features)em2g = torch.randn(mesh2grid_edge, mesh2grid_edge_features)
    eg2m = torch.randn(grid2mesh_edge, grid2mesh_edge_features)
    em2g = torch.randn(mesh2grid_edge, mesh2grid_edge_features)

    #Encoder operations:
    encoder = Embedder(grid_node_features,embed_feature_latent_dim,mesh_node_features,
    mesh_edge_features,grid2mesh_edge_features,mesh2grid_edge_features)

    ## One dummy forward pass through Encodervg_embedded, vm_embedded, em_embedded, eg2m_embedded, em2g_embedded = encoder(vg, vm, em, eg2m, em2g)
    vg_embedded, vm_embedded, em_embedded, eg2m_embedded, em2g_embedded = encoder(vg, vm, em, eg2m, em2g)

    
    # Output shapes from Encoderprint("vg_embedded shape:", vg_embedded.shape)print("vm_embedded shape:", vm_embedded.shape)print("em_embedded shape:", em_embedded.shape)print("eg2m_embedded shape:", eg2m_embedded.shape)print("em2g_embedded shape:", em2g_embedded.shape)
    print("vg_embedded shape:", vg_embedded.shape)
    print("vm_embedded shape:", vm_embedded.shape)
    print("em_embedded shape:", em_embedded.shape)
    print("eg2m_embedded shape:", eg2m_embedded.shape)
    print("em2g_embedded shape:", em2g_embedded.shape)

    # Assuming edge_indices is a tensor indicating the connections from grid nodes to mesh nodes# For simplicity, randomly generated indices are used
    ## Since it is a bipartite graph from grid nodes to edge nodes, first column is the source node (grid) and the second column is the destination node(mesh)edge_grid_source = torch.randint(0, grid_nodes, (1, grid2mesh_edge), dtype=torch.long)edge_mesh_dst = torch.randint(0, mesh_nodes, (1, grid2mesh_edge), dtype=torch.long)
    edge_grid_source = torch.randint(0, grid_nodes, (1, grid2mesh_edge), dtype=torch.long)
    edge_mesh_dst = torch.randint(0, mesh_nodes, (1, grid2mesh_edge), dtype=torch.long)

    
    ## Final concatenated bipartite graphedge_indices_g2m = torch.cat((edge_grid_source, edge_mesh_dst), dim=0)
    edge_indices_g2m = torch.cat((edge_grid_source, edge_mesh_dst), dim=0)

    # Instantiate the Grid2MeshGNN modelgrid2mesh_gnn = Grid2Mesh(embed_feature_latent_dim,grid_hidden_dim, mesh_hidden_dim, edge_hidden_dim)
    grid2mesh_gnn = Grid2Mesh(embed_feature_latent_dim,grid_hidden_dim, mesh_hidden_dim, edge_hidden_dim)

    
    # Use outputs from Encoder and perform forward pass through the GNN to Update features using the Grid2MeshGNNvg_updated, vm_updated, eg2m_updated = grid2mesh_gnn(vg_embedded, vm_embedded, eg2m_embedded, edge_indices_g2m)
    vg_updated, vm_updated, eg2m_updated = grid2mesh_gnn(vg_embedded, vm_embedded, eg2m_embedded, edge_indices_g2m)
    print(f"Updated Grid Node Features Shape: {vg_updated.shape}")
    print(f"Updated Mesh Node Features Shape: {vm_updated.shape}")
    print(f"Updated Grid2Mesh Edge Features Shape: {eg2m_updated.shape}")
    print(f"Updated Grid Node Features Shape: {vg_updated.shape}")
    print(f"Updated Mesh Node Features Shape: {vm_updated.shape}")
    print(f"Updated Grid2Mesh Edge Features Shape: {eg2m_updated.shape}")
    
    # the representations are updated with a residual connection
    
    Mesh=Mesh_GNN(mesh_edge_features,mesh_node_features,mesh_hidden_dim,embed_feature_latent_dim)
    vm_new,em_new=Mesh(vm_updated,em_embedded,mesh_nodes,mesh_edges)
    vm_final=vm_updated+vm_new
    em_final=em_embedded+em_new
    print("After processor, mesh node shape =",vm_final.shape)
    print("After processor, mesh edge shape =",em_final.shape)


if __name__ == "__main__":
    main()
