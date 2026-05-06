import torch
import numpy as np
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import mesh_face_areas_normals
import time

def get_adjacency_matrix(vertices, faces, valid_len):
    """
    Args:
        vertices: [B, N, 3] tensor
        faces: [B, M, 3] tensor, padded with -1
        valid_len: [B] tensor
    Returns:
        adj_matrix: [B, N, N] tensor
    """
    B, N, _ = vertices.shape
    device = vertices.device
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, faces.size(1))  # [B, M]
    valid_faces_mask = (faces >= 0).all(dim=-1)  # [B, M]
    edges_0 = torch.stack([faces[..., 0], faces[..., 1]], dim=-1)  # [B, M, 2]
    edges_1 = torch.stack([faces[..., 1], faces[..., 2]], dim=-1)  # [B, M, 2]
    edges_2 = torch.stack([faces[..., 2], faces[..., 0]], dim=-1)  # [B, M, 2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1)  # [B, 3M, 2]
    batch_idx = batch_idx.unsqueeze(-1).expand(-1, -1, 2)  # [B, M, 2]
    batch_idx = torch.cat([batch_idx, batch_idx, batch_idx], dim=1)  # [B, 3M, 2]
    valid_mask = valid_faces_mask.unsqueeze(-1).expand(-1, -1, 2)  # [B, M, 2]
    valid_mask = torch.cat([valid_mask, valid_mask, valid_mask], dim=1)  # [B, 3M, 2]
    flat_edges = edges[valid_mask].view(-1, 2)  # [Valid_edges, 2]
    flat_batch = batch_idx[valid_mask].view(-1, 2)[:, 0]  # [Valid_edges]
    indices = torch.stack([
        flat_batch,
        flat_edges[:, 0],
        flat_edges[:, 1]
    ], dim=0)  # [3, Valid_edges]
    values = torch.ones(indices.size(1), device=device)
    adj = torch.sparse_coo_tensor(
        indices, 
        values, 
        (B, N, N),
        device=device
    ).to_dense()
    adj = (adj + adj.transpose(1, 2)) > 0
    mask = torch.arange(N, device=device)[None, :] < valid_len[:, None]  # [B, N]
    mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
    adj = adj & mask
    return adj

def mesh_preprocess(vertices, faces, max_length=4096):
    if vertices.ndim == 4:
        vertices, faces = vertices[0], faces[0]
    center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
    vertices = vertices - center
    v_max = vertices[0].abs().max()
    vertices = vertices / v_max
    valid_mask = torch.ones(vertices.shape[1], dtype=torch.bool)
    faces = torch.tensor(faces, dtype=torch.int64)
    faces_max_length = int(max_length * 2.5)
    vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], max_length-vertices.shape[1], 3)], dim=1)
    vertices_color = torch.cat([vertices_color, -1.0 * torch.ones(max_length-vertices_color.shape[0], 3)], dim=0)
    faces = torch.cat([faces, -1 * torch.ones(faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
    valid_mask = torch.cat([valid_mask, torch.zeros(max_length-valid_mask.shape[0], dtype=torch.bool)])[None]
    valid_length = valid_mask.sum(dim=-1)
    adj_matrix = get_adjacency_matrix(vertices[0][None], faces[None], valid_length)
    return vertices, vertices_color, faces, valid_mask, valid_length, adj_matrix

def merge_identical_vertices(vertices, faces):

    T, N, _ = vertices.shape
    first_frame = vertices[0]
    
    rounded_vertices = first_frame
    _, unique_indices, inverse_indices = np.unique(
        rounded_vertices.view([('', rounded_vertices.dtype)]*3),
        return_index=True,
        return_inverse=True
    )
    
    merged_vertices = vertices[:, unique_indices]
    merged_faces = inverse_indices[faces]
    if merged_faces.ndim == 3:
        merged_faces = merged_faces.squeeze(-1)
    
    max_valid_index = len(unique_indices) - 1
    valid_faces_mask = (merged_faces <= max_valid_index).all(axis=1)
    merged_faces = merged_faces[valid_faces_mask]
    
    valid_faces_mask = np.apply_along_axis(lambda x: len(np.unique(x)), 1, merged_faces) == 3
    merged_faces = merged_faces[valid_faces_mask]
    
    sorted_faces = np.sort(merged_faces, axis=1)
    _, unique_face_idx = np.unique(sorted_faces.view([('', sorted_faces.dtype)]*3),
                                 return_index=True)
    merged_faces = merged_faces[unique_face_idx]

    print("Before merging: verts: {}, faces: {}".format(vertices.shape[1], faces.shape[0]))
    print("After merging: verts: {}, faces: {}".format(merged_vertices.shape[1], merged_faces.shape[0]))

    assert merged_faces.max() < len(unique_indices), "Face indices out of bounds"
    assert merged_faces.min() >= 0, "Negative face indices found"
    
    return merged_vertices, merged_faces

def find_indices_in_merged(vertices_list, merged_vertices):
    indices_list = []
    for vertices in vertices_list:
        matches = (merged_vertices.unsqueeze(1) == vertices.unsqueeze(0))
        matches = matches.all(dim=2)  
        indices = matches.nonzero()[:, 0]
        indices = indices.reshape(vertices.shape[0])
        indices_list.append(indices)
    return indices_list

def merge_identical_vertices_with_indices(vertices_list, faces_list):
    """
    Args:
        vertices_list: list of torch.Tensor[(Ni,3)]
        faces_list: list of torch.Tensor[(Fi,3)]
    Returns:
        merged_vertices: torch.Tensor 
        merged_faces: torch.Tensor 
        indices_list: list of torch.Tensor 
    """
    all_vertices = torch.cat(vertices_list, dim=0)
    all_faces = torch.cat(faces_list, dim=0)
    vertices_tuple = torch.stack(all_vertices.unbind(dim=-1), dim=-1)
    unique_vertices, inverse_indices = torch.unique(
        vertices_tuple,
        dim=0,
        return_inverse=True,
        sorted=True
    )
    merged_faces = inverse_indices[all_faces]
    face_vertices_equal = (merged_faces[:,[0,0,1]] == merged_faces[:,[1,2,2]]).any(dim=1)
    valid_faces = ~face_vertices_equal
    merged_faces = merged_faces[valid_faces]
    sorted_faces, _ = torch.sort(merged_faces, dim=1)
    faces_tuple = torch.stack(sorted_faces.unbind(dim=-1), dim=-1)
    merged_faces = torch.unique(faces_tuple, dim=0, sorted=True)
    start_idx = 0
    indices_list = []
    for vertices in vertices_list:
        indices_list.append(inverse_indices[start_idx:start_idx + len(vertices)])
        start_idx += len(vertices)
    assert merged_faces.max() < len(unique_vertices), "Face indices out of bounds"
    assert merged_faces.min() >= 0, "Negative face indices found"
    return unique_vertices, merged_faces, indices_list

def get_edge_lengths_from_verts(verts, adj_matrix, valid_mask):
    
    B, T, N, _ = verts.shape
    device = verts.device
    adj_matrix = adj_matrix.bool()
    triu_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    sparse_adj = adj_matrix & triu_mask.unsqueeze(0)
    edge_indices = torch.where(sparse_adj)
    batch_indices, edge_starts, edge_ends = edge_indices
    is_start_valid = valid_mask.bool()[batch_indices, edge_starts]
    is_end_valid = valid_mask.bool()[batch_indices, edge_ends]
    valid_edge_filter = is_start_valid & is_end_valid
    batch_indices_for_edges = batch_indices[valid_edge_filter]
    edge_starts_final = edge_starts[valid_edge_filter]
    edge_ends_final = edge_ends[valid_edge_filter]
    num_total_valid_edges = len(batch_indices_for_edges)
    if num_total_valid_edges == 0:
        return (
            torch.empty(0, T, device=device), 
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device)
        )
    # (B, T, N, 3) -> (B, N, T, 3)
    verts_permuted = verts.permute(0, 2, 1, 3)
    start_v = verts_permuted[batch_indices_for_edges, edge_starts_final] # Shape: (E_valid, T, 3)
    end_v = verts_permuted[batch_indices_for_edges, edge_ends_final]   # Shape: (E_valid, T, 3)
    edge_lengths = torch.sqrt(torch.sum((start_v - end_v) ** 2, dim=-1) + 1e-12) # Shape: (E_valid, T)

    return edge_lengths, batch_indices_for_edges, edge_starts_final, edge_ends_final

@torch.no_grad()
def get_strain_limit_raw_stats(gt_edge_lengths, recon_edge_lengths, threshold):
    total_num_edges = torch.tensor(gt_edge_lengths.numel(), device=gt_edge_lengths.device, dtype=torch.long)
    if total_num_edges == 0:
        device = gt_edge_lengths.device
        return torch.tensor(0.0, device=device), torch.tensor(0, device=device, dtype=torch.long), torch.tensor(0, device=device, dtype=torch.long)

    epsilon = 1e-8
    strain_ratio = recon_edge_lengths / (gt_edge_lengths + epsilon)

    is_stretched = strain_ratio > threshold
    is_compressed = strain_ratio < (1.0 / threshold)
    abnormal_mask = is_stretched | is_compressed

    num_abnormal_edges = abnormal_mask.sum()

    if num_abnormal_edges == 0:
        total_error_on_abnormal = torch.tensor(0.0, device=gt_edge_lengths.device)
    else:
        strain_error_values = (strain_ratio - 1.0) ** 2
        total_error_on_abnormal = strain_error_values[abnormal_mask].sum()
    
    return total_error_on_abnormal, num_abnormal_edges, total_num_edges

@torch.no_grad()
def calc_n_hops(adj_matrix, num_hops=4, alpha_hops=0.5, density_threshold=0.05, mode="band", no_norm=True):
    """
    Calculate multi-scale adjacency bands and merge into a weighted adjacency matrix,
    supporting dynamic dense/sparse switching to save GPU memory.
    This version fixes bugs in the original code that only processed the first sample in the batch
    and cross-sample data contamination issues.
    
    Args:
        adj_matrix (torch.Tensor): Boolean or float adjacency matrix of shape (B, N, N).
        num_hops (int): Number of scales to compute (total hop count).
        alpha_hops (float): Distance decay factor.
        density_threshold (float): Density threshold, beyond which switches to dense computation.
        mode (str): Reserved parameter, current implementation is "band" mode.
        no_norm (bool): If True, skip row normalization.
        
    Returns:
        torch.Tensor: Weighted adjacency matrix (B, N, N).
    """
    if not isinstance(num_hops, int) or num_hops < 1:
        raise ValueError("num_hops must be an integer greater than or equal to 1.")

    device = adj_matrix.device
    batch_size, N, _ = adj_matrix.shape
    DENSITY_THRESHOLD = density_threshold
    
    # Initialize the final result matrix
    adj_matrix_float = torch.zeros(adj_matrix.shape, device=device, dtype=torch.float32)
    
    # Batch process 1-hop (scale=1)
    # This is the foundation of all calculations, weight is alpha_hops^0 = 1.0
    scale_1_matrix = adj_matrix.float()
    scale_1_matrix.diagonal(dim1=1, dim2=2).fill_(0)  # Ensure diagonal is 0
    adj_matrix_float += scale_1_matrix
    
    # If only 1-hop is needed, normalize and return directly
    if num_hops == 1:
        if no_norm:
            return adj_matrix_float
        else:
            row_sums = adj_matrix_float.sum(dim=-1, keepdim=True)
            return adj_matrix_float / (row_sums + 1e-12)
    
    # Calculate subsequent scales (2-hop to num_hops) sample by sample
    # Sample-wise loop is necessary because sparse/dense switching logic is based on individual graph density
    for b in range(batch_size):
        # C_prev: represents nodes reachable in (scale-1) hops or fewer
        # Initially, C_prev is the 1-hop adjacency matrix
        C_prev = scale_1_matrix[b]
        
        # Decide whether to create sparse representation based on initial density
        density = C_prev.count_nonzero().item() / (N * N)
        C_prev_sparse = C_prev.to_sparse().coalesce() if density < DENSITY_THRESHOLD else None
        
        # Calculate scale 2 to num_hops
        for scale_level in range(2, num_hops + 1):
            # Dynamically choose dense or sparse matrix multiplication
            if C_prev_sparse is not None:
                try:
                    # Try sparse multiplication C_prev * C_prev
                    mult_res_sparse = torch.sparse.mm(C_prev_sparse, C_prev_sparse).coalesce()
                    mult_res = mult_res_sparse.to_dense()
                except RuntimeError:
                    # If sparse computation fails (e.g., out of memory), fall back to dense computation
                    mult_res = torch.mm(C_prev, C_prev)
                    C_prev_sparse = None  # Mark to not use sparse computation for subsequent steps
            else:
                # Use dense computation
                mult_res = torch.mm(C_prev, C_prev)
            
            # C_curr: represents nodes reachable in scale hops or fewer
            # (C_prev U (C_prev * C_prev))
            C_curr = torch.logical_or(C_prev > 0, mult_res > 0).float()
            C_curr.diagonal().fill_(0) # Ensure diagonal is 0
            
            # band: precisely find node pairs with shortest path length of scale_level
            # C_curr - C_prev
            band = torch.logical_and(C_curr > 0, torch.logical_not(C_prev > 0)).float()
            band.diagonal().fill_(0) # Ensure diagonal is 0
            
            # --- This is the key fix point ---
            # Add the computed weighted band to the corresponding matrix slice of the current sample
            # No more if b == 0 and broadcast contamination issues
            weight = alpha_hops ** (scale_level - 1)
            adj_matrix_float[b] += weight * band
            
            # Update iteration variables
            C_prev = C_curr
            
            # Update sparse representation for next iteration
            new_density = C_prev.count_nonzero().item() / (N * N)
            if new_density < DENSITY_THRESHOLD:
                C_prev_sparse = C_prev.to_sparse().coalesce()
            else:
                C_prev_sparse = None
    
    # Finally ensure all sample diagonals are 0 again
    adj_matrix_float.diagonal(dim1=1, dim2=2).fill_(0)
    
    if no_norm:
        return adj_matrix_float
    else:
        # Batch row normalization
        row_sums = adj_matrix_float.sum(dim=-1, keepdim=True)
        # Prevent division by zero
        return adj_matrix_float / (row_sums + 1e-12)