import numpy as np
import torch

def mpjpe(pred, gt):
    pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt
    return np.mean(np.linalg.norm(pred - gt, axis=-1))

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """
    if isinstance(S1, np.ndarray):
        S1 = torch.from_numpy(S1)
    if isinstance(S2, np.ndarray):
        S2 = torch.from_numpy(S2)
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    # print("S1 NaN?", torch.isnan(S1).any())
    # print("S2 NaN?", torch.isnan(S2).any())

    # Step 2: S1_hat 出力確認
    S1_hat = compute_similarity_transform(S1, S2)
    # print("S1_hat NaN?", torch.isnan(S1_hat).any())

    # Step 3: 差分確認
    diff = S1_hat - S2
    # print("diff NaN?", torch.isnan(diff).any())

    # Step 4: 最終距離の中身確認
    dist = ((diff)**2).sum(dim=-1)
    # print("dist min/max:", dist.min(), dist.max())
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt(((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    # print(f"re: {re}")
    return re

import trimesh
import pyrender
import numpy as np
import cv2

def render_mesh(verts, faces, mesh_color=(0.5, 0.5, 1.0), save_path=None):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=mesh_color)
    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)

    if save_path:
        cv2.imwrite(save_path, color[..., ::-1])  # RGB→BGR
    return color