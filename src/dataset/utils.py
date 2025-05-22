import numpy as np
import torch
def normalize_vertices(vertices):
    # vertices: (B, N, 3)
    center = vertices.mean(dim=1, keepdim=True)  # (B, 1, 3)
    vertices = vertices - center  # (B, N, 3)
    scale = torch.norm(vertices, dim=2).max(dim=1)[0].view(-1, 1, 1)  # (B, 1, 1)
    scale = torch.where(scale < 1e-8, torch.tensor(1.0, device=vertices.device), scale)
    vertices = vertices / scale
    return vertices

def move_root_to_origin(vertices, wrist_idx=0):
    wrist = vertices[:, wrist_idx:wrist_idx + 1, :]  # (B, 1, 3)
    vertices = vertices - wrist  # wristを原点に平行移動
    return vertices

def add_noise(input, noise_level=0.005):
    input += np.random.normal(0, noise_level, size=input.shape)
    return input

def rotate_vertices_xyz(vertices, max_angle=180):
    angles = np.radians(np.random.uniform(-max_angle, max_angle, size=3))
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]),  np.cos(angles[0])]])
    
    Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0,                  1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0,                  0,                 1]])
    
    R = Rz @ Ry @ Rx
    
    return vertices @ R.T

import torch

def rotation_matrix_to_axis_angle(R):
    # R: (..., 3, 3)
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    if isinstance(cos_theta, np.ndarray):
        cos_theta = torch.from_numpy(cos_theta)
    theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6))

    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1) / (2 * torch.sin(theta).unsqueeze(-1))

    axis_angle = axis * theta.unsqueeze(-1)
    return axis_angle.view(-1, 45)  # (B, 45)