import smplx
import torch
def calc_mano_mesh(r_mano_layer,
                   l_mano_layer, 
                   mano_pose,
                   mano_betas, 
                   mano_trans, 
                   mano_side,
                   batch_size=64, 
                   device='cuda'):
    vertices_list = []
    for i in range(batch_size):
        if mano_side[i] == "right":
            mano_output = r_mano_layer(
                global_orient=torch.zeros(1, 3, device=device),
                hand_pose=mano_pose[i].reshape(1, 45).to(device),
                betas=mano_betas[i].reshape(1, 10).to(device),
                transl=mano_trans[i].reshape(1, 3).to(device)
            )
            
        else:
            mano_output = l_mano_layer(
                global_orient=torch.zeros(1, 3, device=device),
                hand_pose=mano_pose[i].reshape(1, 45).to(device),
                betas=mano_betas[i].reshape(1, 10).to(device),
                transl=mano_trans[i].reshape(1, 3).to(device)
            )
        vertices_list.append(mano_output.vertices)
    return torch.cat(vertices_list, dim=0).to(device)

