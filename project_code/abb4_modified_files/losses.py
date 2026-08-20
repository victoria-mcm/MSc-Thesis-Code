import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
import abb4.data.utils as du
from abb4.data import all_atom
from abb4.data import so3_utils
from openfold.utils.loss import compute_fape
from openfold.utils.superimposition import superimpose
from openfold.np.residue_constants import restype_name_to_atom14_names, restypes, restype_1to3, van_der_waals_radius
import ot as pot


#### constants
chain_term = 1000
reg_def = {'cdr1': np.arange(27, 38+1), 'cdr2': np.arange(56, 65+1), 'cdr3': np.arange(105, 117+1)}
reg_def['cdrs'] = np.concatenate([reg_def['cdr1'], reg_def['cdr2'], reg_def['cdr3']])
reg_def['fw'] = np.array([x for x in range(1, 128+1) if x not in reg_def['cdrs']])

residue_van_der_waals_radius = {
    res: [van_der_waals_radius[atom[0]] if atom != '' else 0
          for atom in restype_name_to_atom14_names[restype_1to3[res]]]
          for res in restypes
}
residue_van_der_waals_radius['U'] = [0]*14 # unknown residue

eps = 1e-8 # small constant to prevent 0 division


def superimpose_differentiable(reference, coords, mask):
    """
    Differentiable superimposition that preserves gradients.
    Uses Kabsch algorithm implemented in PyTorch.
    
    Args:
        reference: [*, N, 3] reference tensor
        coords: [*, N, 3] tensor to align
        mask: [*, N] mask tensor
    Returns:
        superimposed coords [*, N, 3], rmsd [*] tensor
    """
    # Handle batch dimensions
    batch_dims = reference.shape[:-2]
    n_atoms = reference.shape[-2]
    
    # Flatten batch dimensions for processing
    flat_reference = reference.reshape(-1, n_atoms, 3)
    flat_coords = coords.reshape(-1, n_atoms, 3)
    flat_mask = mask.reshape(-1, n_atoms)
    
    batch_size = flat_reference.shape[0]
    superimposed_list = []
    rmsd_list = []
    
    for i in range(batch_size):
        ref_i = flat_reference[i]  # [N, 3]
        coords_i = flat_coords[i]  # [N, 3]
        mask_i = flat_mask[i]      # [N]
        
        # Select only unmasked coordinates (like original implementation)
        mask_bool = mask_i > 0
        if not mask_bool.any():
            superimposed_list.append(coords_i)
            rmsd_list.append(torch.tensor(0.0, device=coords_i.device))
            continue
            
        ref_unmasked = ref_i[mask_bool]  # [n_unmasked, 3]
        coords_unmasked = coords_i[mask_bool]  # [n_unmasked, 3]
        
        # Center coordinates (only using unmasked coordinates)
        ref_mean = ref_unmasked.mean(dim=0, keepdim=True)  # [1, 3]
        coords_mean = coords_unmasked.mean(dim=0, keepdim=True)  # [1, 3]
        
        ref_centered = ref_unmasked - ref_mean  # [n_unmasked, 3]
        coords_centered = coords_unmasked - coords_mean  # [n_unmasked, 3]
        
        # Compute covariance matrix H = coords_centered^T @ ref_centered
        H = torch.mm(coords_centered.t(), ref_centered)  # [3, 3]
        
        # SVD
        U, S, Vt = torch.linalg.svd(H, full_matrices=False)
        
        # Compute rotation matrix
        R = torch.mm(Vt.t(), U.t())  # [3, 3]
        
        # Ensure proper rotation (det(R) = 1)
        det_R = torch.linalg.det(R)
        if det_R < 0:
            Vt_corrected = Vt.clone()
            Vt_corrected[-1, :] *= -1
            R = torch.mm(Vt_corrected.t(), U.t())
        
        # Apply transformation ONLY to unmasked coordinates (like original)
        coords_centered_unmasked = coords_unmasked - coords_mean  # [n_unmasked, 3]
        coords_rotated_unmasked = torch.mm(coords_centered_unmasked, R.t())  # [n_unmasked, 3]
        superimposed_unmasked = coords_rotated_unmasked + ref_mean  # [n_unmasked, 3]
        
        # Place rotated coordinates back into full-size tensor (masked positions stay as original)
        superimposed = torch.zeros_like(coords_i)  # [N, 3] - zeros for masked positions
        superimposed[mask_bool] = superimposed_unmasked  # Fill unmasked positions
        
        # Compute RMSD (only using unmasked coordinates)
        diff_unmasked = superimposed_unmasked - ref_unmasked
        rmsd = torch.sqrt(torch.mean(torch.sum(diff_unmasked ** 2, dim=1)))
        
        superimposed_list.append(superimposed)
        rmsd_list.append(rmsd)
    
    # Stack results and reshape to original batch dimensions
    superimposed_stacked = torch.stack(superimposed_list, dim=0)  # [batch_size, N, 3]
    rmsd_stacked = torch.stack(rmsd_list, dim=0)  # [batch_size]
    
    superimposed_reshaped = superimposed_stacked.reshape(batch_dims + (n_atoms, 3))
    rmsd_reshaped = rmsd_stacked.reshape(batch_dims)
    
    return superimposed_reshaped, rmsd_reshaped


def flow_normalisations(t_trans, t_rot, t_seq, clip, likelihood_weighting):
    trans_norm_scale = 1 - torch.min(
        t_trans[..., None], torch.tensor(clip)) # (B, 1, 1)
    rots_norm_scale = 1 - torch.min(
        t_rot[..., None], torch.tensor(clip)) # (B, 1, 1)
    if likelihood_weighting:
        seq_norm_scale = 1 - torch.min(t_seq, torch.tensor(clip)) # (B, 1)
    else:
        seq_norm_scale = 1.0

    return trans_norm_scale, rots_norm_scale, seq_norm_scale


def flow_losses(cfg, noisy_batch, model_output, num_batch, num_res, 
                loss_mask, loss_denom, norm_scales):
    # aatypes loss
    ce_loss = F.cross_entropy(
        model_output['pred_logits'].reshape(-1, 21),
        noisy_batch['aatypes_1'].flatten().long(),
        reduction='none',
    ).reshape(num_batch, num_res) / norm_scales[2]
    aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / loss_denom

    # translation VF loss
    trans_error = (noisy_batch['trans_1'] - model_output['pred_trans']) / norm_scales[0] * cfg.trans_scale
    trans_loss = torch.sum(
        trans_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / (loss_denom * 3)
    trans_loss = torch.clamp(trans_loss, max=2.5) # 5 in original code, but multiplied by trans_loss_weight_first

    # rotation VF loss
    gt_rot_vf = so3_utils.calc_rot_vf(
            noisy_batch['rotmats_t'], noisy_batch['rotmats_1'].type(torch.float32))
    pred_rots_vf = so3_utils.calc_rot_vf(noisy_batch['rotmats_t'], model_output['pred_rotmats'])
    rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scales[1]
    rots_vf_loss = torch.sum(
        rots_vf_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / (loss_denom * 3)

    return aatypes_loss, trans_loss, rots_vf_loss


def BB_fape(
    pred,
    batch,
    loss_mask,
    norm_scale,
    clamp_distance=10, # default in ABB2 aux BB fape
    ):
    '''Compute FAPE for an antibody.
    FAPE between CDRs and Fw is clapmed at 30, otherwise at 10.
    '''
    # unpack
    gt_trans_1, gt_rotmats_1 = batch['trans_1'], batch['rotmats_1']
    pred_trans_1, pred_rotmats_1 = pred['pred_trans'], pred['pred_rotmats']

    # here loss divided by normalisation not multiplied
    num_batch = gt_trans_1.shape[0]
    normalisation = (1 / norm_scale).view(num_batch, 1, 1)

    # get rigids and point
    gt_rigids = du.create_rigid(gt_rotmats_1, gt_trans_1)
    pred_rigids = du.create_rigid(pred_rotmats_1, pred_trans_1)
    gt_bb_points = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3].reshape(num_batch, -1, 3)
    pred_bb_points = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3].reshape(num_batch, -1, 3)
  
    positions_mask = loss_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(num_batch, -1)
    
    fape = compute_fape(
                pred_rigids,
                gt_rigids,
                loss_mask,
                pred_bb_points,
                gt_bb_points,
                positions_mask,
                normalisation,
                l1_clamp_distance=clamp_distance,
                )

    return fape


def mab_specific_all_atom_fape(
    pred,
    batch,
    loss_mask,
    length_scale,
    inter_weight, # weight for cdr-fw fape
    ):
    '''Compute FAPE for an antibody.
    FAPE between CDRs and Fw is clapmed at 30, otherwise at 10.
    '''
    num_batch = loss_mask.shape[0]

    # get all frames
    pred_rigids = du.create_rigid(pred['pred_rotmats'], pred['pred_trans'])
    all_frames_pred = all_atom.torsion_angles_to_frames(r=pred_rigids, alpha=pred['pred_torsions'], aatype=batch['aatypes_1'])
    all_frames_pred = all_frames_pred.from_tensor_4x4(all_frames_pred.to_tensor_4x4().reshape(num_batch, -1, 4, 4)) # [B, N, 8] -> [B, 8N]
    gt_rigids = du.create_rigid(batch['rotmats_1'], batch['trans_1'])
    all_frames_gt = all_atom.torsion_angles_to_frames(r=gt_rigids, alpha=batch['torsion_angles'], aatype=batch['aatypes_1'])
    all_frames_gt = all_frames_gt.from_tensor_4x4(all_frames_gt.to_tensor_4x4().reshape(num_batch, -1, 4, 4)) # [B, N, 8] -> [B, 8N]

    # get all atom positions
    all_atom_pred = all_atom.atom14_from_trans_rot_torsion(pred['pred_trans'], pred['pred_rotmats'], pred['pred_torsions'], batch['aatypes_1'])
    all_atom_pred = all_atom_pred.reshape(num_batch, -1, 3) # [B, N, 14, 3] -> [B, 14N, 3]
    all_atom_gt = batch['atom14_1']
    all_atom_gt = all_atom_gt.reshape(num_batch, -1, 3) # [B, N, 14, 3] -> [B, 14N, 3]

    # create mask for valid frames
    rigids_mask = torch.concat([batch['res_mask'][...,None], batch['torsion_mask']], dim=-1)

    # here loss divided by normalisation not multiplied
    num_batch = all_frames_gt.shape[0]
    length_scale = torch.tensor(length_scale, device=rigids_mask.device)
    length_normalisation = (1 / length_scale).repeat(num_batch, 1, 1)
    # length_normalisation = (1 / length_scale).view(num_batch, 1, 1)

    # create masks for cdr & fw
    frames_masks = []
    positions_masks = []
    cdr_selection = torch.tensor(np.concatenate([reg_def['cdrs'], reg_def['cdrs']+chain_term])).to(batch['imgt_numeric'].device)
    fw_selection = torch.tensor(np.concatenate([reg_def['fw'], reg_def['fw']+chain_term])).to(batch['imgt_numeric'].device)
    
    for selection in [cdr_selection, fw_selection]:
        region_mask = torch.isin(batch['imgt_numeric'], selection).int() # [B, N]
        frames_mask = rigids_mask * loss_mask[...,None] * region_mask[...,None] # [B, N, 8]
        positions_mask = batch['atom14_atom_exists'] * loss_mask[...,None] * region_mask[...,None] # [B, N, 14]
        frames_masks.append(frames_mask.reshape(num_batch, -1)) # [B, 8N]
        positions_masks.append(positions_mask.reshape(num_batch, -1)) # [B, 14N]
    
    # compute intra/inter region fape, with different clamp distances
    fapes = {'intra': [], 'inter': []}
    for i, frames_mask in enumerate(frames_masks):
        for j, positions_mask in enumerate(positions_masks):
            clamp_distance = 30 if i != j else 10
            name = 'inter' if i != j else 'intra'
            fapes[name].append(compute_fape(
                all_frames_pred,
                all_frames_gt,
                frames_mask,
                all_atom_pred,
                all_atom_gt,
                positions_mask,
                length_normalisation,
                l1_clamp_distance=clamp_distance,
                )[..., None])

    assert torch.cat(fapes['intra'], dim=1).shape == (num_batch, 2)
    assert torch.cat(fapes['inter'], dim=1).shape == (num_batch, 2)
    intra_fape = torch.cat(fapes['intra'], dim=1).mean(dim=1)
    inter_fape = torch.cat(fapes['inter'], dim=1).mean(dim=1)
    return intra_fape + inter_weight * inter_fape


def mab_specific_BB_fape(
    pred,
    batch,
    loss_mask,
    length_scale,
    inter_weight, # weight for cdr-fw fape
    ):
    '''Compute FAPE for an antibody.
    FAPE between CDRs and Fw is clapmed at 30, otherwise at 10.
    '''
    # unpack
    gt_trans_1, gt_rotmats_1 = batch['trans_1'], batch['rotmats_1']
    pred_trans_1, pred_rotmats_1 = pred['pred_trans'], pred['pred_rotmats']
    imgt = batch['imgt_numeric']

    # here loss divided by normalisation not multiplied
    num_batch = gt_trans_1.shape[0]
    length_scale = torch.tensor(length_scale, device=gt_trans_1.device)
    length_normalisation = (1 / length_scale).repeat(num_batch, 1, 1)
    # length_normalisation = (1 / length_scale).view(num_batch, 1, 1)

    # get rigids and point
    gt_rigids = du.create_rigid(gt_rotmats_1, gt_trans_1)
    pred_rigids = du.create_rigid(pred_rotmats_1, pred_trans_1)
    gt_bb_points = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3].reshape(num_batch, -1, 3)
    pred_bb_points = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3].reshape(num_batch, -1, 3)

    # create masks for cdr & fw
    frames_masks = []
    positions_masks = []
    cdr_selection = torch.tensor(np.concatenate([reg_def['cdrs'], reg_def['cdrs']+chain_term])).to(imgt.device)
    fw_selection = torch.tensor(np.concatenate([reg_def['fw'], reg_def['fw']+chain_term])).to(imgt.device)
    
    for selection in [cdr_selection, fw_selection]:
        region_mask = torch.isin(imgt, selection).int()
        frames_mask = region_mask * loss_mask
        positions_mask = frames_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(num_batch, -1)
        frames_masks.append(frames_mask)
        positions_masks.append(positions_mask)
    
    # compute intra/inter region fape, with different clamp distances
    fapes = {'intra': [], 'inter': []}
    for i, frames_mask in enumerate(frames_masks):
        for j, positions_mask in enumerate(positions_masks):
            clamp_distance = 30 if i != j else 10
            name = 'inter' if i != j else 'intra'
            fapes[name].append(compute_fape(
                pred_rigids,
                gt_rigids,
                frames_mask,
                pred_bb_points,
                gt_bb_points,
                positions_mask,
                length_normalisation,
                l1_clamp_distance=clamp_distance,
                )[..., None])

    assert torch.cat(fapes['intra'], dim=1).shape == (num_batch, 2)
    assert torch.cat(fapes['inter'], dim=1).shape == (num_batch, 2)
    intra_fape = torch.cat(fapes['intra'], dim=1).mean(dim=1)
    inter_fape = torch.cat(fapes['inter'], dim=1).mean(dim=1)
    return intra_fape + inter_weight * inter_fape


def fape_losses(cfg, noisy_batch, model_output, loss_mask, calc_A_fape=False):#, norm_scales):
    if calc_A_fape: #all atom fape (eg torsions, sidechains, not just backbone)
        A_fape_loss = mab_specific_all_atom_fape(
            model_output, noisy_batch, loss_mask,
            cfg.bb_atom_scale,# / norm_scales[0][..., None],
            cfg.cdr_fw_fape_weight
            )
    else:
        A_fape_loss = torch.zeros(loss_mask.shape[0]).to(loss_mask.device) # empty tensor

    if cfg.BB_fape_cdr_weighting: #FAPE of backbone atoms with CDRs upweighted
        BB_fape_loss = mab_specific_BB_fape(
            model_output, noisy_batch, loss_mask,
            cfg.bb_atom_scale,# / norm_scales[0][..., None],
            cfg.cdr_fw_fape_weight
            )
    else:
        BB_fape_loss = BB_fape(
            model_output, noisy_batch, loss_mask,
            cfg.bb_atom_scale,# / norm_scales[0][..., None],
            )
    return A_fape_loss, BB_fape_loss


def torsion_losses(cfg, noisy_batch, model_output, loss_mask, loss_denom):
    # main torsion loss
    torsion_mask = loss_mask[...,None] * noisy_batch['torsion_mask']
    torsion_denom = noisy_batch['torsion_mask'].sum(axis=-1)
    # replace 0 denominator for padded residues / no 0 division
    torsion_denom = torch.where(torsion_denom == 0, torch.tensor(1, device=torsion_denom.device), torsion_denom)
    torsion_loss = torch.sum(
        ((model_output['pred_torsions'] - noisy_batch['torsion_angles']) ** 2 * torsion_mask[...,None]),
        dim=(-1,-2)
    ) / torsion_denom
    torsion_loss = torch.sum(torsion_loss, dim=-1) / loss_denom

    # torsion norm loss
    torsion_norm_loss = torch.mean(torch.abs(model_output['torsion_norm'] - 1), dim=(-1, -2))
    torsion_norm_loss = torch.sum(torsion_norm_loss * loss_mask, dim=-1) / loss_denom

    return torsion_loss, torsion_norm_loss


def bb_atom_losses(cfg, noisy_batch, model_output, loss_mask,
                   loss_denom, num_batch, num_res): #, norm_scales):
    # bb rmsd
    pred_bb_atoms = all_atom.to_atom37(model_output['pred_trans'], model_output['pred_rotmats'])[:, :, :3]
    gt_bb_atoms = all_atom.to_atom37(noisy_batch['trans_1'], noisy_batch['rotmats_1'])[:, :, :3]
    gt_bb_atoms *= cfg.bb_atom_scale # / norm_scales[0][..., None]
    pred_bb_atoms *= cfg.bb_atom_scale # / norm_scales[0][..., None]
    
    # bb rmsd with openfold utils for consistency with sample evaluation
    _, bb_rmsd_loss = superimpose_differentiable(
        gt_bb_atoms.reshape(num_batch,-1,3),
        pred_bb_atoms.reshape(num_batch,-1,3),
        mask=loss_mask.unsqueeze(2).repeat(1,1,3).reshape(num_batch,-1)
    )#convert predicted frames into 3d coords, superimpose with true structure and compute rmsd with gt

    # bb_rmsd_loss = torch.sum(
    #     (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
    #     dim=(-1, -2, -3)
    # ) / (loss_denom * 3)

    # distmat loss
    # local pairwise distance (auxiliary) loss 
    gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
    gt_pair_dists = torch.linalg.norm(
        gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
    pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
    pred_pair_dists = torch.linalg.norm(
        pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

    flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
    flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
    flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
    flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

    gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
    pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
    pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

    # enforce locality: no loss on anything >6A
    proximity_mask = gt_pair_dists < 6
    pair_dist_mask  = pair_dist_mask * proximity_mask

    #pairwise distances of nearby resides, enforces local backbone consistency
    dist_mat_loss = torch.sum(
        (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
        dim=(1, 2))
    dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)

    return bb_rmsd_loss, dist_mat_loss


def dist_between_ca_ca(points):
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    return torch.sqrt((points[:, 1:, 1, :] - points[:, :-1, 1, :]).pow(2).sum(-1) + eps).clamp(max=10)

def dist_between_c_n(points):
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    return torch.sqrt((points[:, 1:, 2, :] - points[:, :-1, 0, :]).pow(2).sum(-1) + eps).clamp(max=10)


def dist_check(pred_points, true_points, loss_mask, k=12):
    d_loss_mask = loss_mask[:, 1:] * loss_mask[:, :-1] # [B, N-1]
    d_loss_denom = torch.sum(d_loss_mask, dim=-1)

    ca_ca_dist = (
        ((dist_between_ca_ca(pred_points) - dist_between_ca_ca(true_points)).abs() - 0.075 * k) * d_loss_mask
        ).clamp(min=0).sum(dim=-1) / d_loss_denom

    c_n_dist = (
        ((dist_between_c_n(pred_points) - dist_between_c_n(true_points)).abs() - 0.04 * k) * d_loss_mask
        ).clamp(min=0).sum(dim=-1) / d_loss_denom

    return (ca_ca_dist + c_n_dist) / 4


def ang_between_ca_n_c(points):
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    nca = points[:, 1:, 1, :] - points[:, 1:, 0, :] # ca(i) - n(i)
    nc = points[:, :-1, 2, :] - points[:, 1:, 0, :] # c(i+1) - n(i)
    num = (nca * nc).sum(dim=-1)
    denom = (nca.norm(dim=-1) * nc.norm(dim=-1)) + eps # Avoid division by zero
    return num / denom


def ang_between_n_c_ca(points):
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    cca = points[:, :-1, 1, :] - points[:, :-1, 2, :] # ca(i+1) - c(i+1)
    cn = points[:, 1:, 0, :] - points[:, :-1, 2, :] # n(i) - c(i+1)
    num = (cca * cn).sum(dim=-1)
    denom = (cca.norm(dim=-1) * cn.norm(dim=-1)) + eps # Avoid division by zero
    return num / denom


def angle_check(pred_points, true_points, loss_mask, k=12):
    a_loss_mask = loss_mask[:, 1:] * loss_mask[:, :-1] # [B, N-1]
    a_loss_denom = torch.sum(a_loss_mask, dim=-1)
    ca_n_c = (
        ((ang_between_ca_n_c(pred_points) - ang_between_ca_n_c(true_points)).abs() - 0.02 * k) * a_loss_mask
        ).clamp(min=0).sum(-1) / a_loss_denom
    n_c_ca = (
        ((ang_between_n_c_ca(pred_points) - ang_between_n_c_ca(true_points)).abs() - 0.02 * k) * a_loss_mask
        ).clamp(min=0).sum(-1) / a_loss_denom
    return (n_c_ca + ca_n_c) / 4


def clashes_check(pred_points, aatypes, residue_index, atom14_mask, loss_mask, tolerance=1.5):
    # calculate distance between pairs of all atoms
    dists = ((
        pred_points[:, :, None, :, None, :] - pred_points[:, None, :, None, :, :]
        ).square().sum(-1) + eps).sqrt() # [B, N, N, 14, 14]
    
    # create distance masks
    # atom14 exits
    dists_mask = atom14_mask[:, :, None, :, None] * atom14_mask[:, None, :, None, :] # [B, N, N, 14, 14]

    # remove intra-residue and duplicates of inter-residue distances
    dists_mask = dists_mask * (residue_index[:, :, None, None, None] < residue_index[:, None, :, None, None])

    # don't count C - N bond as clash
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14) # C at index 2 in atom14
    c_one_hot = c_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape)
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14) # N at index 0 in atom14
    n_one_hot = n_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape)
    neighbour_mask = (residue_index[..., :, None, None, None] + 1) == residue_index[..., None, :, None, None]
    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # disulfide bridges
    cys_sg_one_hot = torch.zeros_like(atom14_mask)
    cys_sg_one_hot[:, :, 5] = (aatypes == restypes.index('C')).int() # SG at index 5 in atom14
    disulfide_bonds = cys_sg_one_hot[..., None, :, None] * cys_sg_one_hot[..., None, :, None,:]
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # unknown residue types
    unknowns = (aatypes != 20).int()
    unknowns = unknowns[..., None] * unknowns[..., None, :]
    dists_mask = dists_mask * unknowns[..., None, None]

    # calculate clashes
    res_vdw_tensor = torch.tensor([residue_van_der_waals_radius[res] for res in restypes + ['U']]).to(aatypes.device)
    atom_radius = res_vdw_tensor[aatypes]
    accepted_separation = atom_radius[:, :, None, :, None] + atom_radius[:, None, :, None, :]
    clashes = (accepted_separation - dists - tolerance).clamp(0)
    clashes = clashes * dists_mask

    return clashes.sum(dim=(-1, -2, -3, -4)) / loss_mask.sum(dim=-1)


def structure_violation_losses(noisy_batch, model_output, loss_mask, num_batch, calc_clashes=False):
    gt_atom14 = noisy_batch['atom14_1']
    pred_atom14 = all_atom.atom14_from_trans_rot_torsion(
        model_output['pred_trans'], model_output['pred_rotmats'], model_output['pred_torsions'], noisy_batch['aatypes_1']
        )
    
    dist_loss = dist_check(pred_atom14, gt_atom14, loss_mask)
    angle_loss = angle_check(pred_atom14, gt_atom14, loss_mask)
    if calc_clashes: # slow, don't calculate if not used
        clashes_loss = clashes_check(
            pred_atom14, noisy_batch['aatypes_1'], noisy_batch['res_idx'], noisy_batch['atom14_atom_exists'], loss_mask
            )
    else:
        clashes_loss = torch.zeros(num_batch).to(noisy_batch['aatypes_1'].device) # empty tensor for code to work

    return dist_loss, angle_loss, clashes_loss


def ensemble_rmsf_loss(noisy_batch, model_output, n_batch, n_ensemble, cfg):
    # align to first frame of each ensemble, combine batch and ensemble dims
    align_frames = model_output['pred_trans'][:, 0].unsqueeze(1).repeat(1, n_ensemble, 1, 1).reshape(n_batch * n_ensemble, -1, 3)
    pred_CA_coords = model_output['pred_trans'].reshape(n_batch * n_ensemble, -1, 3)
    align_mask = noisy_batch['res_mask'].reshape(n_batch * n_ensemble, -1)

    pred_CA_coords, _ = superimpose_differentiable(
            align_frames,
            pred_CA_coords,
            align_mask,
        )

    # separate batch and ensemble dims
    pred_CA_coords = pred_CA_coords.reshape(n_batch, n_ensemble, -1, 3)

    # calculate RMSF
    pred_CA_mean = torch.mean(pred_CA_coords, axis=1) # mean over ensemble
    pred_sq_dist = torch.sum(torch.pow(pred_CA_coords - pred_CA_mean[:, None, :, :], 2), axis=-1) # mean over x, y, z
    pred_rmsf = torch.sqrt(torch.mean(pred_sq_dist, axis=1) + eps) # mean over ensemble
    pred_rmsf = pred_rmsf[:, None, ...].repeat(1, pred_CA_coords.shape[1], 1) # repeat to match ensemble dim

    # calculate loss
    gt_rmsf = noisy_batch['rmsf'][:,:,:,0] # ground truth RMSF aligned on Fv at index 0
    sampling_variance =  noisy_batch['rmsf'][:,:,:,1] ** 2 # gt RMSF std aligned on Fv at index 1

    if cfg.rmsf_loss_region == 'cdrs':
        resi = torch.tensor(np.concatenate([reg_def['cdrs'], reg_def['cdrs'] + chain_term]))
        resi = resi.to(noisy_batch['imgt_numeric'].device)
        rmsf_loss_mask = torch.isin(noisy_batch['imgt_numeric'], resi).int() * noisy_batch['res_mask']
    elif cfg.rmsf_loss_region == 'cdrh3':
        resi = torch.tensor(reg_def['cdr3'])
        resi = resi.to(noisy_batch['imgt_numeric'].device)
        rmsf_loss_mask = torch.isin(noisy_batch['imgt_numeric'], resi).int() * noisy_batch['res_mask']
    elif cfg.rmsf_loss_region == 'all':
        rmsf_loss_mask = noisy_batch['res_mask']
    else:
        raise ValueError(f'Unknown RMSF loss region: {cfg.rmsf_loss_region}')

    loss_denom = rmsf_loss_mask.sum(axis=-1)

    if cfg.rmsf_loss == 'standard_mse':
        rmsf_loss = torch.sum((pred_rmsf - gt_rmsf) ** 2 * rmsf_loss_mask, axis=-1) / loss_denom
    elif cfg.rmsf_loss == 'chi2':
        model_variance = gt_rmsf * cfg.rmsf_loss_model_variance # unvertainity defined as a percentage of the RMSF
        total_variance = cfg.rmsf_loss_sampling_var_weight * sampling_variance + cfg.rmsf_loss_model_var_weight * model_variance

        rmsf_loss = torch.sum(
            (((pred_rmsf - gt_rmsf) ** 2) / (total_variance + eps)) * rmsf_loss_mask, axis=-1
            ) / loss_denom
        
        if torch.isnan(gt_rmsf).any():
            raise ValueError('NaN values found in ground truth RMSF during RMSF chi2 loss calculation')
        if torch.isnan(total_variance).any():
            raise ValueError('NaN values found in total variance during RMSF chi2 loss calculation')
        if torch.isnan(rmsf_loss).any():
            raise ValueError('NaN values found in RMSF loss during RMSF chi2 loss calculation')

    else:
        raise ValueError(f'Unknown RMSF loss type: {cfg.rmsf_loss}')

    # reshape to combine batch and ensemble dims
    rmsf_loss = rmsf_loss.reshape(n_batch * n_ensemble)

    return rmsf_loss

def all_losses(noisy_batch, model_output, batch_type, cfg):
    if batch_type == 'ensemble': # reshape to calculate structure losses accross batch and ensemble
        n_batch = noisy_batch['aatypes_1'].shape[0]
        n_ensemble = noisy_batch['aatypes_1'].shape[1]

        rmsf_loss = ensemble_rmsf_loss(noisy_batch, model_output, n_batch, n_ensemble, cfg)

        # reshape to combine batch and ensemble dims for remaining loss calculations
        noisy_batch = {k: v.reshape(n_batch * n_ensemble, *v.size()[2:]) for k,v in noisy_batch.items()}
        model_output = {k: v.reshape(n_batch * n_ensemble, *v.size()[2:]) for k,v in model_output.items()}

    loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
    num_batch, num_res = loss_mask.shape
    loss_denom = torch.sum(loss_mask, dim=-1)

    norm_scales = flow_normalisations(
        noisy_batch['t_trans'], noisy_batch['t_rot'], noisy_batch['t_seq'],
        cfg.t_normalize_clip, cfg.aatypes_loss_use_likelihood_weighting
        )

    aatypes_loss, trans_loss, rots_vf_loss = flow_losses(
        cfg, noisy_batch, model_output, num_batch,
        num_res, loss_mask, loss_denom, norm_scales
        )
    
    A_fape_loss, BB_fape_loss = fape_losses(
        cfg, noisy_batch, model_output, loss_mask, calc_A_fape=(cfg.A_fape_loss_weight > 0)
        ) #, norm_scales
        # )

    torsion_loss, torsion_norm_loss = torsion_losses(
        cfg, noisy_batch, model_output, loss_mask, loss_denom
        )#MSE of the predicted and true torsion losses

    bb_rmsd_loss, dist_mat_loss = bb_atom_losses(
        cfg, noisy_batch, model_output, loss_mask, loss_denom,
        num_batch, num_res)#, norm_scales
    # )

    dist_loss, angle_loss, clashes_loss = structure_violation_losses(
        noisy_batch, model_output, loss_mask, num_batch,
        calc_clashes=(cfg.clashes_loss_weight > 0)
        )

    t_trans = noisy_batch['t_trans'][:,0] # do this for t_trans (t_trans and t_rot are the same in training
    tot_loss = (
        trans_loss        * cfg.translation_loss_weight +
        rots_vf_loss      * cfg.rotation_loss_weight
    )

    if cfg.aatypes_loss_weight > 0:
        tot_loss += (aatypes_loss      * cfg.aatypes_loss_weight)
    if cfg.bb_loss_weight > 0:
        tot_loss += (bb_rmsd_loss      * cfg.bb_loss_weight           * (t_trans > cfg.bb_loss_t_pass))
    if cfg.pair_loss_weight > 0:
        tot_loss += (dist_mat_loss     * cfg.pair_loss_weight         * (t_trans > cfg.pair_loss_t_pass))
    if cfg.A_fape_loss_weight > 0:
        tot_loss += (A_fape_loss       * cfg.A_fape_loss_weight       * (t_trans > cfg.A_fape_loss_t_pass))
    if cfg.BB_fape_loss_weight > 0:
        tot_loss += (BB_fape_loss      * cfg.BB_fape_loss_weight      * (t_trans > cfg.BB_fape_loss_t_pass))
    if cfg.torsion_loss_weight > 0:
        tot_loss += (torsion_loss      * cfg.torsion_loss_weight      * (t_trans > cfg.torsion_loss_t_pass))
    if cfg.torsion_norm_loss_weight > 0:
        tot_loss += (torsion_norm_loss * cfg.torsion_norm_loss_weight * (t_trans > cfg.torsion_norm_t_pass))
    if cfg.dist_loss_weight > 0:
        tot_loss += (dist_loss         * cfg.dist_loss_weight         * (t_trans > cfg.dist_loss_t_pass))
    if cfg.angle_loss_weight > 0:
        tot_loss += (angle_loss        * cfg.angle_loss_weight        * (t_trans > cfg.angle_loss_t_pass))
    if cfg.clashes_loss_weight > 0:
        tot_loss += (clashes_loss      * cfg.clashes_loss_weight      * (t_trans > cfg.clashes_loss_t_pass))
    if cfg.rmsf_loss_weight > 0 and batch_type == 'ensemble':
        tot_loss += (rmsf_loss         * cfg.rmsf_loss_weight         * (t_trans > cfg.rmsf_loss_t_pass))

    # not correct if t_rots and t_trans are not set to be identical
    if cfg.loss_t_weighting == 'exp':
        weight = torch.exp(-t_trans * cfg.loss_t_weighting_exp_rate)
        tot_loss = tot_loss * weight

    loss_dict = {
            "aatypes_loss": aatypes_loss,
            "trans_loss": trans_loss,
            "rots_vf_loss": rots_vf_loss,
            "bb_rmsd_loss": bb_rmsd_loss,
            "dist_mat_loss": dist_mat_loss,
            "A_fape_loss": A_fape_loss,
            "BB_fape_loss": BB_fape_loss,
            "torsion_loss": torsion_loss,
            "torsion_norm_loss": torsion_norm_loss,
            "dist_loss": dist_loss,
            "angle_loss": angle_loss,
            "clashes_loss": clashes_loss,
            "tot_loss": tot_loss
        }
    if batch_type == 'ensemble':
        loss_dict['rmsf_loss'] = rmsf_loss

    if torch.isnan(tot_loss).any():
        pickle_dir = cfg.crash_dir
        os.makedirs(pickle_dir, exist_ok=True)

        items_to_pickle = [
            ("noisy_batch", {k: v.detach().cpu() for k, v in noisy_batch.items()}),
            ("model_output", {k: v.detach().cpu() for k, v in model_output.items()}),
            ("loss_dict", {k: v.detach().cpu() for k, v in loss_dict.items()}),
            ("loss_mask", loss_mask.detach().cpu()),
            ("trans_norm_scale", norm_scales[0]),
            ("rots_norm_scale", norm_scales[1]),
            ("seq_norm_scale", norm_scales[2]),
        ]
        for item_name, item in items_to_pickle:
            with open(os.path.join(pickle_dir, f'{item_name}.pkl'), 'wb') as file:
                pickle.dump(item, file)

        raise ValueError(f'NaN loss encountered in batch: {noisy_batch["csv_idx"]}.')

    return loss_dict

def calculate_region_rmds(coords1, coords2, imgt):
    '''Calculate RMSD for individual regions of an antibody'''
    region_mean_rmsds = {}

    # rmsd for the entire antibody
    sq_dist = (torch.sum(torch.pow((coords1 - coords2),2), dim=2))
    rmsds = torch.sqrt(torch.nanmean(sq_dist, dim=1))
    region_mean_rmsds['bb_rmsd'] = torch.mean(rmsds).item()

    # rmsd for individual regions
    for chain in ['h', 'l']:
        for region, resi in reg_def.items():
            resi = torch.tensor(resi).to(imgt.device)
            if chain == 'l':
                resi = resi + chain_term
            elif chain == 'h':
                resi = resi
            region_mask = torch.isin(imgt, resi) # also excludes padded residues
            
            region1 = coords1.clone().detach()
            region1[~region_mask] = torch.nan
            region2 = coords2.clone().detach()
            region2[~region_mask] = torch.nan

            sq_dist = (torch.sum(torch.pow((region1 - region2),2), dim=2))
            rmsds = torch.sqrt(torch.nanmean(sq_dist, dim=1))
            region_mean_rmsds[f'bb_rmsd_{chain}_{region}'] = torch.mean(rmsds).item()

    return region_mean_rmsds


def compute_folding_metrics(pred_atom37, gt_atom37, res_mask, imgt):
    '''Compute RMSD for individual regions of an antibody'''
    num_batch = gt_atom37.shape[0]
    n_resi = gt_atom37.shape[1]
    
    # reshape coordinates
    imgt = imgt.unsqueeze(2).repeat(1,1,3).reshape(num_batch,-1)
    res_mask = res_mask.unsqueeze(2).repeat(1,1,3).reshape(num_batch,-1)
    gt_bb_pos = gt_atom37[:,:,:3, :].reshape(num_batch,-1,3)
    pred_bb_pos = pred_atom37[:,:,:3, :].reshape(num_batch,-1,3)
    assert pred_bb_pos.shape[1] % 3 == 0

    # superimose, arguments: reference, coords to superimpose, mask
    _, global_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=res_mask)

    H_chain_mask = (imgt < 1000).int() * res_mask
    L_chain_mask = (imgt >= 1000).int() * res_mask
    
    h_imposed_pred_bb_pos, h_chain_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=H_chain_mask)
    h_results = calculate_region_rmds(gt_bb_pos, h_imposed_pred_bb_pos, imgt)
    
    l_imposed_pred_bb_pos, l_chain_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=L_chain_mask)
    l_results = calculate_region_rmds(gt_bb_pos, l_imposed_pred_bb_pos, imgt)

    rmsds = {'bb_rmsd': torch.mean(global_rmsd).item(),
             'bb_rmsd_h_chain': torch.mean(h_chain_rmsd).item(),
             'bb_rmsd_l_chain': torch.mean(l_chain_rmsd).item()}
    

    for region in ['cdr1', 'cdr2', 'cdr3', 'fw']:
        rmsds[f'bb_rmsd_h_{region}'] = h_results[f'bb_rmsd_h_{region}']
        rmsds[f'bb_rmsd_l_{region}'] = l_results[f'bb_rmsd_l_{region}']
   
    # calculate RMSD
    return rmsds


def rmsd_mat(region1, region2, num_batch):
    sq_dist = (torch.sum(torch.pow((region1 - region2),2), dim=2))
    rmsds = torch.sqrt(torch.nanmean(sq_dist, dim=1))
    return rmsds.reshape(num_batch, num_batch)


def rmsd_mat2accuracy_metrics(rmsd):
    # recall: distance from each GT to closest pred
    recall = torch.min(rmsd, dim=0).values.mean()

    # precision: distance from each GT to closest pred
    precision = torch.min(rmsd, dim=1).values.mean()

    # distributional accuracy
    rmsd = rmsd.numpy()
    uniform_dist = np.ones(rmsd.shape[0]) / rmsd.shape[0]
    W = pot.emd2(
        uniform_dist,
        uniform_dist,
        rmsd
    )
    return recall, precision, W


def rmsd_mat2rmsd_metrics(pred_rmsd, gt_rmsd_frames, gt_rmsd_sims):
    # max distance between any two 
    pred_rmsd_max = torch.max(pred_rmsd)
    pred_rmsd = pred_rmsd.fill_diagonal_(float('nan'))
    pred_rmsd_mean = torch.nanmean(pred_rmsd)

    # gt from frames
    gt_rmsd_frames_max = torch.max(gt_rmsd_frames)
    gt_rmsd_frames = gt_rmsd_frames.fill_diagonal_(float('nan'))
    gt_rmsd_frames_mean = torch.nanmean(gt_rmsd_frames)
    
    # gt from simulations
    gt_rmsd_sims_max = gt_rmsd_sims[0, 0] # shape [num_ensemble, 3], values identical across ensemble, index 0 for max
    gt_rmsd_sims_mean = gt_rmsd_sims[0, 1] # shape [num_ensemble, 3], values identical across ensemble, index 1 for mean

    return (pred_rmsd_max,
            pred_rmsd_mean,
            torch.sqrt((pred_rmsd_max - gt_rmsd_frames_max) ** 2), # root squared error
            torch.sqrt((pred_rmsd_mean - gt_rmsd_frames_mean) ** 2),
            torch.sqrt((pred_rmsd_max - gt_rmsd_sims_max) ** 2),
            torch.sqrt((pred_rmsd_mean - gt_rmsd_sims_mean) ** 2),
    )


def domain_accuracy_metrics(exp_gt_bb_pos, exp_pred_bb_pos, exp_res_mask, num_batch):
    _, rmsd = superimpose(exp_gt_bb_pos, exp_pred_bb_pos, mask=exp_res_mask)
    rmsd = rmsd.reshape(num_batch, num_batch) # [pred idx, gt idx]

    return rmsd_mat2accuracy_metrics(rmsd)

def domain_rmsd_metrics(exp_gt_bb_pos1, exp_gt_bb_pos2, 
                        exp_pred_bb_pos1, exp_pred_bb_pos2,
                        gt_rmsd_sims, exp_res_mask, num_batch):
    # pred rmsd matrix
    _, pred_rmsd = superimpose(exp_pred_bb_pos1, exp_pred_bb_pos2, mask=exp_res_mask)
    pred_rmsd = pred_rmsd.reshape(num_batch, num_batch)

    # gt rmsd matrix from frames
    _, gt_rmsd_frames = superimpose(exp_gt_bb_pos1, exp_gt_bb_pos2, mask=exp_res_mask)
    gt_rmsd_frames = gt_rmsd_frames.reshape(num_batch, num_batch)

    return rmsd_mat2rmsd_metrics(pred_rmsd, gt_rmsd_frames, gt_rmsd_sims)

def calculate_CA_rmsf(pred_atom37, gt_atom37, gt_rmsf_sims, region_mask, superimpose_mask):
    # convert to CA coordinate mask
    # only select every 3rd along axis -1
    region_mask = region_mask.bool()
    CA_region_mask = region_mask[0][None, ...].reshape(1, -1, 3)[0, :, 0]
    superimpose_mask = superimpose_mask[0][None, ...].reshape(1, -1, 3)[:, :, 0]

    # pred
    pred_CA_coords = pred_atom37[:, :, 1, :]
    # superimpose on first frame
    pred_CA_coords,  _ = superimpose(
        pred_CA_coords[0][None, ...].repeat(pred_CA_coords.shape[0],1,1),
        pred_CA_coords,
        superimpose_mask.repeat(pred_atom37.shape[0], 1)
    )

    assert (pred_CA_coords.shape[1] == CA_region_mask.shape[0] and
            pred_CA_coords.shape[1] == superimpose_mask.shape[1])
    pred_CA_mean = torch.mean(pred_CA_coords, axis=0)
    pred_sq_dist = torch.sum(torch.pow(pred_CA_coords - pred_CA_mean[None, :, :], 2), axis=-1)
    pred_rmsf = torch.sqrt(torch.mean(pred_sq_dist, axis=0))
    pred_rmsf = pred_rmsf[CA_region_mask]

    # gt from frames
    gt_CA_coords = gt_atom37[:, :, 1, :]
    # superimpose on first frame
    gt_CA_coords,  _ = superimpose(
        gt_CA_coords[0][None, ...].repeat(gt_CA_coords.shape[0],1,1),
        gt_CA_coords,
        superimpose_mask.repeat(gt_atom37.shape[0], 1)
    )
    assert (gt_CA_coords.shape[1] == CA_region_mask.shape[0] and
            gt_CA_coords.shape[1] == superimpose_mask.shape[1])

    gt_CA_mean = torch.mean(gt_CA_coords, axis=0)
    gt_sq_dist = torch.sum(torch.pow(gt_CA_coords - gt_CA_mean[None, :, :], 2), axis=-1)
    gt_rmsf_frames = torch.sqrt(torch.mean(gt_sq_dist, axis=0))
    gt_rmsf_frames = gt_rmsf_frames[CA_region_mask]

    # gt from simulations
    gt_rmsf_sims = gt_rmsf_sims[0, :][CA_region_mask] # shape [num_ensemble, N]
    assert pred_rmsf.shape == gt_rmsf_sims.shape

    # rmse
    rmsf_rmse_frames = torch.sqrt(torch.mean(torch.pow(pred_rmsf - gt_rmsf_frames, 2)))
    rmsf_rmse_sims = torch.sqrt(torch.mean(torch.pow(pred_rmsf - gt_rmsf_sims, 2)))

    return (pred_rmsf.max(), pred_rmsf.mean(),
            gt_rmsf_frames.max(), gt_rmsf_frames.mean(), rmsf_rmse_frames,
            gt_rmsf_sims.max(), gt_rmsf_sims.mean(), rmsf_rmse_sims)


def write_ensemble_metrics(
        recall, precision, W,
        max_pred_rmsd, mean_pred_rmsd,
        mae_max_rmsd_frames, mae_mean_rmsd_frames,
        mae_max_rmsd_sims, mae_mean_rmsd_sims,
        max_pred_rmsf, mean_pred_rmsf,
        max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
        max_gt_rmsf_sims, mean_gt_rmsf_sims, rmsf_rmse_sims,
        region, metrics
    ):
    metrics[f'{region}_recall'] = recall.item()
    metrics[f'{region}_precision'] = precision.item()
    metrics[f'{region}_W'] = W
    metrics[f'{region}_max_pred_rmsd'] = max_pred_rmsd.item()
    metrics[f'{region}_mean_pred_rmsd'] = mean_pred_rmsd.item()
    metrics[f'{region}_d_max_rmsd'] = mae_max_rmsd_frames.item()
    metrics[f'{region}_d_mean_rmsd'] = mae_mean_rmsd_frames.item()
    metrics[f'{region}_d_max_rmsd_sims'] = mae_max_rmsd_sims.item()
    metrics[f'{region}_d_mean_rmsd_sims'] = mae_mean_rmsd_sims.item()
    metrics[f'{region}_max_pred_rmsf'] = max_pred_rmsf.item()
    metrics[f'{region}_mean_pred_rmsf'] = mean_pred_rmsf.item()
    metrics[f'{region}_max_gt_rmsf'] = max_gt_rmsf_frames.item()
    metrics[f'{region}_mean_gt_rmsf'] = mean_gt_rmsf_frames.item()
    metrics[f'{region}_rmsf_rmse'] = rmsf_rmse_frames.item()
    metrics[f'{region}_max_gt_rmsf_sims'] = max_gt_rmsf_sims.item()
    metrics[f'{region}_mean_gt_rmsf_sims'] = mean_gt_rmsf_sims.item()
    metrics[f'{region}_rmsf_rmse_sims'] = rmsf_rmse_sims.item()
    return metrics

region2rmsdIdx = {
    'global': 0,
    'h_chain': 1,
    'l_chain': 2,
    'h': {
        'cdr1': 3,
        'cdr2': 4,
        'cdr3': 5,
    },
    'l': {
        'cdr1': 6,
        'cdr2': 7,
        'cdr3': 8,
    }
}

def compute_ensemble_prediction_metrics(pred_atom37s, gt_atom37s, res_masks, imgts, gt_rmsfs, gt_rmsds):
    num_batch = gt_atom37s.shape[0]
    num_ensemble = gt_atom37s.shape[1]
    n_resi = gt_atom37s.shape[2]

    combined_metrics = []
    for n in range(num_batch):
        pred_atom37 = pred_atom37s[n]
        gt_atom37 = gt_atom37s[n]
        imgt = imgts[n]
        res_mask = res_masks[n]
        gt_rmsf_sims = gt_rmsfs[n]
        gt_rmsd_sims = gt_rmsds[n]

        metrics = {}

        # reshape inputs
        expand_imgt = imgt.unsqueeze(2).repeat(num_ensemble,1,3).reshape(num_ensemble**2,-1)
        expand_res_mask = res_mask.unsqueeze(2).repeat(num_ensemble,1,3).reshape(num_ensemble**2,-1)
        H_chain_mask = (expand_imgt < 1000).int() * expand_res_mask
        L_chain_mask = (expand_imgt >= 1000).int() * expand_res_mask
        gt_bb_pos = gt_atom37[:,:,:3, :].reshape(num_ensemble,-1,3)
        pred_bb_pos = pred_atom37[:,:,:3, :].reshape(num_ensemble,-1,3)
        assert pred_bb_pos.shape[1] % 3 == 0

        # all combinations of pred and gt indices
        pred_idx, gt_idx = torch.where(
            torch.ones(num_ensemble, num_ensemble))
        expand_pred_bb_pos = pred_bb_pos[pred_idx]
        expand_gt_bb_pos = gt_bb_pos[gt_idx]

        # switch indexing to calcualte distance matrix with itself
        expand_pred_bb_pos2 = pred_bb_pos[gt_idx]
        expand_gt_bb_pos2 = gt_bb_pos[pred_idx]

        # domain metrics    
        for region, mask in zip(['global', 'h_chain', 'l_chain'], [expand_res_mask, H_chain_mask, L_chain_mask]):

            recall, precision, W = domain_accuracy_metrics(
                expand_gt_bb_pos, expand_pred_bb_pos, mask, num_ensemble
            )

            # rmsd
            (max_pred_rmsd, mean_pred_rmsd,
             mae_max_rmsd_frames, mae_mean_rmsd_frames,
             mae_max_rmsd_sims, mae_mean_rmsd_sims
            ) = domain_rmsd_metrics(
                expand_gt_bb_pos, expand_gt_bb_pos2, expand_pred_bb_pos, expand_pred_bb_pos2,
                gt_rmsd_sims[:, region2rmsdIdx[region], :], mask, num_ensemble
            )

            # rmsf
            idx = 0 if region == 'global' else 2 # 0 gobal aligned, 2 chain aligned
            gt_rmsf_sel = gt_rmsf_sims[:, :, idx] # shape [num_ensemble, N, 3]
            (max_pred_rmsf, mean_pred_rmsf,
             max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
             max_gt_rmsf_sims, mean_gt_rmsf_sims, rmsf_rmse_sims
            ) = calculate_CA_rmsf(
                pred_atom37, gt_atom37, gt_rmsf_sel, mask, mask,
            )

            metrics = write_ensemble_metrics(
                recall, precision, W,
                max_pred_rmsd, mean_pred_rmsd,
                mae_max_rmsd_frames, mae_mean_rmsd_frames, mae_max_rmsd_sims, mae_mean_rmsd_sims,
                max_pred_rmsf, mean_pred_rmsf,
                max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
                max_gt_rmsf_sims, mean_gt_rmsf_sims, rmsf_rmse_sims,
                region, metrics
            )

        # cdr metrics
        for chain, mask in zip(['h', 'l'], [H_chain_mask, L_chain_mask]):
            # align on whole chain
            pred_imposed_on_gt, _ = superimpose(expand_gt_bb_pos, expand_pred_bb_pos, mask=mask)
            pred_imposed_on_pred, _ = superimpose(expand_pred_bb_pos, expand_pred_bb_pos2, mask=mask)
            gt_imposed_on_gt, _ = superimpose(expand_gt_bb_pos, expand_gt_bb_pos2, mask=mask)

            for region, resi in reg_def.items():
                if region == 'fw' or region == 'cdrs': # ignore as no GT RMSD for these regions
                    continue
                resi = torch.tensor(resi).to(imgt.device)
                if chain == 'l':
                    resi = resi + chain_term
                elif chain == 'h':
                    resi = resi
                region_mask = torch.isin(expand_imgt, resi)

                # select gt regions
                gt = expand_gt_bb_pos.clone().detach()
                gt[~region_mask] = torch.nan
                gt_on_gt = gt_imposed_on_gt.clone().detach()
                gt_on_gt[~region_mask] = torch.nan
                
                # select pred regions
                pred = expand_pred_bb_pos.clone().detach()
                pred[~region_mask] = torch.nan
                pred_on_gt = pred_imposed_on_gt.clone().detach()
                pred_on_gt[~region_mask] = torch.nan
                pred_on_pred = pred_imposed_on_pred.clone().detach()
                pred_on_pred[~region_mask] = torch.nan

                rmsds = rmsd_mat(gt, pred_on_gt, num_ensemble)
                pred_rmsd = rmsd_mat(pred, pred_on_pred, num_ensemble)
                gt_rmsd_frames = rmsd_mat(gt, gt_on_gt, num_ensemble)
                gt_rmsd_sel = gt_rmsd_sims[:, region2rmsdIdx[chain][region], :] # shape [num_ensemble, 3]

                recall, precision, W = rmsd_mat2accuracy_metrics(rmsds)
                (max_pred_rmsd, mean_pred_rmsd,
                 mae_max_rmsd_frames, mae_mean_rmsd_frames,
                 mae_max_rmsd_sims, mae_mean_rmsd_sims
                ) = rmsd_mat2rmsd_metrics(pred_rmsd, gt_rmsd_frames, gt_rmsd_sel)

                gt_rmsf_sel = gt_rmsf_sims[:, :, 2] # shape [num_ensemble, N, 3], index 2 for chain aligned rmsf
                (max_pred_rmsf, mean_pred_rmsf,
                 max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
                 max_gt_rmsf_sims, mean_gt_rmsf_sims, rmsf_rmse_sims,
                ) = calculate_CA_rmsf(
                    pred_atom37, gt_atom37, gt_rmsf_sel, region_mask, mask
                )

                metrics = write_ensemble_metrics(
                    recall, precision, W,
                    max_pred_rmsd, mean_pred_rmsd,
                    mae_max_rmsd_frames, mae_mean_rmsd_frames,
                    mae_max_rmsd_sims, mae_mean_rmsd_sims,
                    max_pred_rmsf, mean_pred_rmsf,
                    max_gt_rmsf_frames, mean_gt_rmsf_frames, rmsf_rmse_frames,
                    max_gt_rmsf_sims, mean_gt_rmsf_sims, rmsf_rmse_sims,
                    f'{chain}_{region}', metrics
                )

        combined_metrics.append(metrics)

    return combined_metrics



######CODE FOR COMPUTING RMSDS WITH NANOBODIES IN DATASET SO NAN CHAINS DONT BREAK######
def calculate_region_rmds_withnano(coords1, coords2, imgt,chain):
    """Calculate RMSD for individual regions of an antibody for one chain at a time """
    region_mean_rmsds = {}

    # coords1/coords2: [1, N*3, 3], imgt: [1, N]
    sq_dist = torch.sum((coords1 - coords2) ** 2, dim=-1)
    region_mean_rmsds['bb_rmsd'] = torch.sqrt(torch.nanmean(sq_dist, dim=1)).mean().item()

    #for chain in ['h', 'l']:
    for region, resi in reg_def.items():
        resi = torch.tensor(resi, device=imgt.device)
        if chain == 'l':
            resi = resi + chain_term

        region_mask = torch.isin(imgt, resi)  # [1, N]
        #if chain=='l':
            #print('region mask sum', region_mask.sum().cpu().numpy())

        atom_mask = region_mask.repeat_interleave(3, dim=1)  # [1, N*3]

        if atom_mask.sum() == 0:
            region_mean_rmsds[f'bb_rmsd_{chain}_{region}'] = float('nan')
            continue

        region1 = coords1[:, atom_mask[0]]  # [1, n_atoms, 3]
        region2 = coords2[:, atom_mask[0]]  # [1, n_atoms, 3]

        sq_dist = torch.sum((region1 - region2) ** 2, dim=-1)
        rmsds = torch.sqrt(torch.mean(sq_dist, dim=1))
        region_mean_rmsds[f'bb_rmsd_{chain}_{region}'] = rmsds.mean().item()

    return region_mean_rmsds


def compute_folding_metrics_withnano(pred_atom37, gt_atom37, res_mask, imgt):
    """Compute RMSD metrics safely for batches that may contain nanobodies."""
    num_batch = gt_atom37.shape[0]
    out = []

    ####debug
    # make sure this is integer-based
    imgt = imgt.long()
    res_mask = res_mask.bool()


    for n in range(num_batch):
        # single-sample slices
        gt_bb_pos = gt_atom37[n:n+1, :, :3, :].reshape(1, -1, 3)
        #print('gt_bb_pos shape: ',gt_bb_pos.shape)
        pred_bb_pos = pred_atom37[n:n+1, :, :3, :].reshape(1, -1, 3)

        sample_imgt = imgt[n].repeat_interleave(3).unsqueeze(0)      # [1, N*3]
        #print(f'imgt shape: {imgt.shape}')
        #print(f'sample imgt shape: {sample_imgt.shape}')
        sample_mask = res_mask[n].repeat_interleave(3).unsqueeze(0)   # [1, N*3]

        metrics = {'bb_rmsd': float('nan'),
                   'bb_rmsd_h_chain': float('nan'),
                   'bb_rmsd_l_chain': float('nan')}

        # global RMSD
        _, global_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=sample_mask)
        metrics['bb_rmsd'] = global_rmsd.item()

        # heavy chain metrics
        H_chain_mask = ((sample_imgt < 1000).int() * sample_mask).bool()
        if H_chain_mask.sum() > 0:
            h_imposed_pred_bb_pos, h_chain_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=H_chain_mask.int())
            h_results = calculate_region_rmds_withnano(gt_bb_pos, h_imposed_pred_bb_pos, imgt[n:n+1],'h')
            metrics['bb_rmsd_h_chain'] = h_chain_rmsd.item()
            for region in ['cdr1', 'cdr2', 'cdr3', 'fw']:
                metrics[f'bb_rmsd_h_{region}'] = h_results[f'bb_rmsd_h_{region}']
        else:
            for region in ['cdr1', 'cdr2', 'cdr3', 'fw']:
                metrics[f'bb_rmsd_h_{region}'] = float('nan')

        # light chain metrics
        L_chain_mask = ((sample_imgt >= 1000).int() * sample_mask).bool()
        if L_chain_mask.sum() > 0:
            l_imposed_pred_bb_pos, l_chain_rmsd = superimpose(gt_bb_pos, pred_bb_pos, mask=L_chain_mask.int())
            l_results = calculate_region_rmds_withnano(gt_bb_pos, l_imposed_pred_bb_pos, imgt[n:n+1],'l')
            metrics['bb_rmsd_l_chain'] = l_chain_rmsd.item()
            for region in ['cdr1', 'cdr2', 'cdr3', 'fw']:
                metrics[f'bb_rmsd_l_{region}'] = l_results[f'bb_rmsd_l_{region}']
        else:
            for region in ['cdr1', 'cdr2', 'cdr3', 'fw']:
                metrics[f'bb_rmsd_l_{region}'] = float('nan')

        out.append(metrics)


    # average over batch, ignoring nan entries
    rmsds = {}
    keys = out[0].keys()
    #for k in keys:
        #vals = np.array([m[k] for m in out], dtype=float)
        #rmsds[k] = float(np.nanmean(vals))

    for k in keys:
        vals = np.array([m[k] for m in out], dtype=float)
        #print(f'{k}: {vals}')
        rmsds[k] = float(np.nanmean(vals))



    return rmsds
    #return None


def _region_res_mask(imgt, res_mask, chain, region):
    """Boolean residue mask for one chain + one IMGT region."""
    if region == 'global':
        base = res_mask.bool()
    else:
        resi = torch.tensor(reg_def[region], device=imgt.device)
        if chain == 'l':
            resi = resi + chain_term
        base = torch.isin(imgt, resi) & res_mask.bool()

    if chain == 'h':
        chain_mask = (imgt < chain_term) & res_mask.bool()
    elif chain == 'l':
        chain_mask = (imgt >= chain_term) & res_mask.bool()
    else:
        raise ValueError(f"Unknown chain {chain}")

    return base & chain_mask


def _safe_nanmean(x, dim=None):
    if dim is None:
        return torch.nanmean(x)
    return torch.nanmean(x, dim=dim)


def _region_backbone_rmsd(gt_bb_pos, pred_bb_pos, region_atom_mask):
    """
    gt_bb_pos/pred_bb_pos: [1, N*3, 3]
    region_atom_mask: [1, N] or [1, N*3] boolean
    """
    if region_atom_mask.sum() == 0:
        return float('nan')

    if region_atom_mask.shape[1] * 3 == gt_bb_pos.shape[1]:
        atom_mask = region_atom_mask
        if atom_mask.shape[1] != gt_bb_pos.shape[1]:
            atom_mask = atom_mask.repeat_interleave(3, dim=1)
    else:
        atom_mask = region_atom_mask.repeat_interleave(3, dim=1)

    region_gt = gt_bb_pos[:, atom_mask[0]]
    region_pred = pred_bb_pos[:, atom_mask[0]]
    sq_dist = torch.sum((region_gt - region_pred) ** 2, dim=-1)
    return torch.sqrt(torch.mean(sq_dist, dim=1)).mean().item()


def compute_region_loss_metrics_withnano(noisy_batch, model_output):
    """
    Nanobody-safe region metrics.

    Returns:
        dict with overall / chain / region metrics averaged over the batch,
        ignoring missing-chain entries.
    """
    pred_atom37 = all_atom.to_atom37(
        model_output["pred_trans"],
        model_output["pred_rotmats"],
    )
    gt_atom37 = all_atom.to_atom37(
        noisy_batch["trans_1"],
        noisy_batch["rotmats_1"],
    )

    imgt = noisy_batch["imgt_numeric"].long()
    res_mask = noisy_batch["res_mask"].bool()

    def _masked_mean_or_nan(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            return float("nan")
        return float(np.nanmean(values))

    batch_out = []

    for n in range(gt_atom37.shape[0]):
        sample = {}
        sample_imgt = imgt[n]
        sample_res_mask = res_mask[n]

        # Backbone atoms: N, CA, C
        gt_bb = gt_atom37[n : n + 1, :, :3, :].reshape(1, -1, 3)
        pred_bb = pred_atom37[n : n + 1, :, :3, :].reshape(1, -1, 3)

        full_atom_mask = sample_res_mask.repeat_interleave(3).unsqueeze(0)

        # Overall backbone RMSD
        if full_atom_mask.sum() > 0:
            _, bb_rmsd = superimpose(gt_bb, pred_bb, mask=full_atom_mask.int())
            sample["bb_rmsd"] = bb_rmsd.item()
        else:
            sample["bb_rmsd"] = float("nan")

        for chain, chain_sel in (
            ("h", sample_imgt < chain_term),
            ("l", sample_imgt >= chain_term),
        ):
            chain_res_mask = sample_res_mask & chain_sel
            chain_atom_mask = chain_res_mask.repeat_interleave(3).unsqueeze(0)

            sample[f"bb_rmsd_{chain}_chain"] = float("nan")

            if chain_atom_mask.sum() == 0:
                for region in ["cdr1", "cdr2", "cdr3", "fw"]:
                    sample[f"bb_rmsd_{chain}_{region}"] = float("nan")
                    #sample[f"bb_coord_mse_{chain}_{region}"] = float("nan")
                    sample[f"ca_pairdist_mse_{chain}_{region}"] = float("nan")
                continue

            # Align on the full chain first
            chain_imposed_pred_bb, chain_rmsd = superimpose(
                gt_bb, pred_bb, mask=chain_atom_mask.int()
            )
            sample[f"bb_rmsd_{chain}_chain"] = chain_rmsd.item()

            for region, resi in reg_def.items():
                resi = torch.tensor(resi, device=imgt.device)
                if chain == "l":
                    resi = resi + chain_term

                region_res_mask = chain_res_mask & torch.isin(sample_imgt, resi)
                region_atom_mask = region_res_mask.repeat_interleave(3)

                # Region RMSD / coordinate MSE on aligned backbone atoms
                if region_atom_mask.sum() == 0:
                    sample[f"bb_rmsd_{chain}_{region}"] = float("nan")
                    #sample[f"bb_coord_mse_{chain}_{region}"] = float("nan")
                    sample[f"ca_pairdist_mse_{chain}_{region}"] = float("nan")
                    continue

                region_gt = gt_bb[:, region_atom_mask]
                region_pred = chain_imposed_pred_bb[:, region_atom_mask]

                sq_dist = torch.sum((region_gt - region_pred) ** 2, dim=-1)  # [1, n_atoms]
                sample[f"bb_rmsd_{chain}_{region}"] = torch.sqrt(torch.mean(sq_dist, dim=1)).item()
                #sample[f"bb_coord_mse_{chain}_{region}"] = torch.mean(sq_dist).item()

                # Easy region-local geometry metric: CA pairwise-distance MSE
                region_ca_mask = region_res_mask
                if region_ca_mask.sum() > 1:
                    gt_ca = gt_atom37[n, region_ca_mask, 1, :]      # CA only
                    pred_ca = pred_atom37[n, region_ca_mask, 1, :]
                    gt_d = torch.cdist(gt_ca, gt_ca)
                    pred_d = torch.cdist(pred_ca, pred_ca)
                    sample[f"ca_pairdist_mse_{chain}_{region}"] = torch.mean((gt_d - pred_d) ** 2).item()
                else:
                    sample[f"ca_pairdist_mse_{chain}_{region}"] = float("nan")

        batch_out.append(sample)

    # Average across batch, ignoring NaNs
    out = {}
    for key in batch_out[0].keys():
        vals = [x[key] for x in batch_out]
        out[key] = _masked_mean_or_nan(vals)

    return out


###### Framework-aligned region RMSDs (matches postproc calculate_struc_pred_metrics.py) ######
_MIN_FW_BB_ATOMS = 9  # 3 backbone atoms (N, CA, C)


def _fw_region_res_mask(sample_imgt, sample_res_mask, chain, region):
    """Boolean residue mask for one chain and IMGT region."""
    sample_res_mask = sample_res_mask.bool()
    if chain == "h":
        chain_mask = (sample_imgt < chain_term) & sample_res_mask
    elif chain == "l":
        chain_mask = (sample_imgt >= chain_term) & sample_res_mask
    else:
        raise ValueError(f"Unknown chain {chain}")

    if region == "chain":
        return chain_mask

    if region not in reg_def:
        raise ValueError(f"Unknown region {region}")

    resi = torch.tensor(reg_def[region], device=sample_imgt.device)
    if chain == "l":
        resi = resi + chain_term
    return chain_mask & torch.isin(sample_imgt, resi)


def _fw_res_mask_to_atom_mask(res_mask):
    """Residue mask [N] -> backbone atom mask [1, N*3]."""
    return res_mask.repeat_interleave(3).unsqueeze(0)


def _fw_superimpose_fit_apply(reference, coords, fit_atom_mask, apply_atom_mask):
    """
    Fit Kabsch on fit_atom_mask atoms; apply transform to apply_atom_mask atoms.

    reference, coords: [1, N*3, 3]
    fit_atom_mask, apply_atom_mask: [1, N*3] bool

    Returns (aligned_coords [1, N*3, 3], fit_rmsd) or (None, None) if fit fails.
    """
    if fit_atom_mask.sum() < _MIN_FW_BB_ATOMS:
        return None, None

    _, fit_rmsd, rot, tran = superimpose(
        reference, coords, fit_atom_mask.int(), return_transform=True
    )
    rot = rot.reshape(-1, 3, 3)[0].to(coords.device).float()
    tran = tran.reshape(-1, 3)[0].to(coords.device).float()

    aligned_full = coords[0] @ rot + tran.unsqueeze(0)
    out = torch.zeros_like(coords)
    out[0, apply_atom_mask[0]] = aligned_full[apply_atom_mask[0]]
    return out, fit_rmsd


def _fw_bb_rmsd(gt_bb, pred_bb, atom_mask):
    """Backbone RMSD between gt and aligned pred for selected atoms."""
    if atom_mask.sum() == 0:
        return float("nan")
    gt = gt_bb[:, atom_mask[0]]
    pred = pred_bb[:, atom_mask[0]]
    sq_dist = torch.sum((gt - pred) ** 2, dim=-1)
    return torch.sqrt(torch.mean(sq_dist)).item()


def compute_fw_aligned_folding_metrics_withnano(pred_atom37, gt_atom37, res_mask, imgt):
    """
    Framework-aligned region RMSDs; nanobody-safe.

    Aligns each chain on its framework backbone atoms (matching
    calculate_struc_pred_metrics.py), then scores framework and CDR regions.
    Global bb_rmsd is computed after aligning on all framework atoms (both chains).
    """
    imgt = imgt.long()
    res_mask = res_mask.bool()
    num_batch = gt_atom37.shape[0]
    region_keys = ["cdr1", "cdr2", "cdr3", "fw"]
    out = []

    for n in range(num_batch):
        gt_bb = gt_atom37[n : n + 1, :, :3, :].reshape(1, -1, 3)
        pred_bb = pred_atom37[n : n + 1, :, :3, :].reshape(1, -1, 3)
        sample_imgt = imgt[n]
        sample_res_mask = res_mask[n]

        full_atom_mask = _fw_res_mask_to_atom_mask(sample_res_mask)

        metrics = {
            "bb_rmsd": float("nan"),
            "bb_rmsd_h_chain": float("nan"),
            "bb_rmsd_l_chain": float("nan"),
        }
        for chain in ("h", "l"):
            for region in region_keys:
                metrics[f"bb_rmsd_{chain}_{region}"] = float("nan")

        # Global RMSD: fit on all FW atoms, score all valid backbone atoms
        global_fw_res = (
            _fw_region_res_mask(sample_imgt, sample_res_mask, "h", "fw")
            | _fw_region_res_mask(sample_imgt, sample_res_mask, "l", "fw")
        )
        global_fw_atom = _fw_res_mask_to_atom_mask(global_fw_res)
        aligned_global, _ = _fw_superimpose_fit_apply(
            gt_bb, pred_bb, global_fw_atom, full_atom_mask
        )
        if aligned_global is not None:
            metrics["bb_rmsd"] = _fw_bb_rmsd(gt_bb, aligned_global, full_atom_mask)

        # Per-chain RMSDs: fit on chain FW, score chain and each region
        for chain in ("h", "l"):
            chain_res = _fw_region_res_mask(sample_imgt, sample_res_mask, chain, "chain")
            if chain_res.sum() == 0:
                continue

            fw_res = _fw_region_res_mask(sample_imgt, sample_res_mask, chain, "fw")
            chain_atom = _fw_res_mask_to_atom_mask(chain_res)
            fw_atom = _fw_res_mask_to_atom_mask(fw_res)

            aligned_chain, _ = _fw_superimpose_fit_apply(
                gt_bb, pred_bb, fw_atom, chain_atom
            )
            if aligned_chain is None:
                continue

            metrics[f"bb_rmsd_{chain}_chain"] = _fw_bb_rmsd(
                gt_bb, aligned_chain, chain_atom
            )
            for region in region_keys:
                region_res = _fw_region_res_mask(
                    sample_imgt, sample_res_mask, chain, region
                )
                region_atom = _fw_res_mask_to_atom_mask(region_res)
                metrics[f"bb_rmsd_{chain}_{region}"] = _fw_bb_rmsd(
                    gt_bb, aligned_chain, region_atom
                )

        out.append(metrics)

    rmsds = {}
    for k in out[0].keys():
        vals = np.array([m[k] for m in out], dtype=float)
        rmsds[k] = float(np.nanmean(vals))

    return rmsds


###### Fixed-protocol region RMSDs for training monitoring (uses batch region field) ######
_V2_REGION_IDS = {
    "h": {"fw": (0, 2, 4, 6), "cdr1": (1,), "cdr2": (3,), "cdr3": (5,)},
    "l": {"fw": (7, 9, 11, 13), "cdr1": (8,), "cdr2": (10,), "cdr3": (12,)},
}


def _v2_chain_res_mask(sample_region, sample_res_mask, sample_imgt, chain, region):
    """Residue mask for one chain/region using the batch region encoding."""
    sample_res_mask = sample_res_mask.bool()
    if chain == "h":
        chain_mask = (sample_imgt < chain_term) & sample_res_mask
    elif chain == "l":
        chain_mask = (sample_imgt >= chain_term) & sample_res_mask
    else:
        raise ValueError(f"Unknown chain {chain}")

    if region == "chain":
        return chain_mask

    if region not in _V2_REGION_IDS[chain]:
        raise ValueError(f"Unknown region {region}")

    region_ids = torch.tensor(
        _V2_REGION_IDS[chain][region], device=sample_region.device
    )
    return chain_mask & torch.isin(sample_region, region_ids)


def _v2_chain_region_rmsds(gt_atom37_n, pred_atom37_n, chain_res_mask, region_masks):
    """FW-align one chain and return backbone RMSDs for each region."""
    chain_idx = chain_res_mask.nonzero(as_tuple=True)[0]
    if chain_idx.numel() == 0:
        return None

    gt_chain = gt_atom37_n[chain_idx, :3, :].reshape(1, -1, 3)
    pred_chain = pred_atom37_n[chain_idx, :3, :].reshape(1, -1, 3)

    fw_on_chain = region_masks["fw"][chain_idx]
    if fw_on_chain.sum() == 0:
        return None

    fw_atom = fw_on_chain.repeat_interleave(3).unsqueeze(0)
    chain_atom = torch.ones(
        1, gt_chain.shape[1], dtype=torch.bool, device=gt_chain.device
    )
    aligned, _ = _fw_superimpose_fit_apply(gt_chain, pred_chain, fw_atom, chain_atom)
    if aligned is None:
        return None

    out = {}
    for region_name in ("fw", "cdr1", "cdr2", "cdr3", "chain"):
        region_on_chain = region_masks[region_name][chain_idx]
        region_atom = region_on_chain.repeat_interleave(3).unsqueeze(0)
        out[region_name] = _fw_bb_rmsd(gt_chain, aligned, region_atom)
    return out


def compute_fw_aligned_folding_metrics_v2(
    pred_atom37, gt_atom37, res_mask, imgt, region
):
    """
    Framework-aligned per-region RMSDs for training monitoring.

    Per chain: fit on framework backbone atoms (from batch region field),
    apply transform to the full chain, then score each region.
    """
    imgt = imgt.long()
    region = region.long()
    res_mask = res_mask.bool()
    num_batch = gt_atom37.shape[0]
    region_keys = ("cdr1", "cdr2", "cdr3", "fw")
    out = []

    for n in range(num_batch):
        sample_imgt = imgt[n]
        sample_region = region[n]
        sample_res_mask = res_mask[n]
        gt_n = gt_atom37[n]
        pred_n = pred_atom37[n]

        metrics = {
            "bb_rmsd": float("nan"),
            "bb_rmsd_h_chain": float("nan"),
            "bb_rmsd_l_chain": float("nan"),
            "h_cdr3_residue_count": float("nan"),
        }
        for chain in ("h", "l"):
            for reg in region_keys:
                metrics[f"bb_rmsd_{chain}_{reg}"] = float("nan")

        global_fw = (
            _v2_chain_res_mask(sample_region, sample_res_mask, sample_imgt, "h", "fw")
            | _v2_chain_res_mask(sample_region, sample_res_mask, sample_imgt, "l", "fw")
        )
        full_atom = sample_res_mask.repeat_interleave(3).unsqueeze(0)
        gt_bb = gt_n[:, :3, :].reshape(1, -1, 3)
        pred_bb = pred_n[:, :3, :].reshape(1, -1, 3)
        global_fw_atom = global_fw.repeat_interleave(3).unsqueeze(0)
        aligned_global, _ = _fw_superimpose_fit_apply(
            gt_bb, pred_bb, global_fw_atom, full_atom
        )
        if aligned_global is not None:
            metrics["bb_rmsd"] = _fw_bb_rmsd(gt_bb, aligned_global, full_atom)

        for chain in ("h", "l"):
            chain_res = _v2_chain_res_mask(
                sample_region, sample_res_mask, sample_imgt, chain, "chain"
            )
            region_masks = {
                reg: _v2_chain_res_mask(
                    sample_region, sample_res_mask, sample_imgt, chain, reg
                )
                for reg in ("fw", "cdr1", "cdr2", "cdr3")
            }
            region_masks["chain"] = chain_res

            if chain == "h":
                metrics["h_cdr3_residue_count"] = float(
                    region_masks["cdr3"].sum().item()
                )

            chain_scores = _v2_chain_region_rmsds(
                gt_n, pred_n, chain_res, region_masks
            )
            if chain_scores is None:
                continue

            metrics[f"bb_rmsd_{chain}_chain"] = chain_scores["chain"]
            for reg in region_keys:
                metrics[f"bb_rmsd_{chain}_{reg}"] = chain_scores[reg]

        out.append(metrics)

    rmsds = {}
    for k in out[0].keys():
        vals = np.array([m[k] for m in out], dtype=float)
        rmsds[k] = float(np.nanmean(vals))
    return rmsds