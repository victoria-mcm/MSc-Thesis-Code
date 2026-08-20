from typing import Any
import torch, time, os, random, pickle, wandb, numpy as np, pandas as pd, logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from abb4.analysis import utils as au
from abb4.models.flow_model import FlowModel
import abb4.models.losses as losses
from abb4.models import utils as mu
from abb4.data.interpolant import Interpolant
from abb4.data.data_module import heterogenos_length_collate, subsample_one_per_cluster
from abb4.data import utils as du
from abb4.data import all_atom
from abb4.data import so3_utils
from abb4.data import residue_constants
from abb4.experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import grad_norm
from contextlib import nullcontext


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)

        self.model = FlowModel(cfg.model)                   # joint CNF/CTMC model
        self.interpolant = Interpolant(cfg.interpolant)     # handles noising and ODE propagation

        self._interpolant_cfg = cfg.interpolant
        ### FOR DEBUG
        #self._cached_debug_batch = None
        #####
        self._exp_cfg = cfg.experiment

        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens

        self.task = cfg.data.task
        self.gen_extent = cfg.data.gen_extent

        self._valid_sample_write_dir = self._exp_cfg.checkpointer.dirpath
        self._test_sample_write_dir = os.path.join(self._exp_cfg.checkpointer.dirpath, 'test')
        if 'inference' in self._exp_cfg:
            self._output_dir = self._exp_cfg.inference.output_dir
        os.makedirs(self._valid_sample_write_dir, exist_ok=True)
        os.makedirs(self._test_sample_write_dir, exist_ok=True)

        self.epoch_metrics = {'valid': [], 'extra_valid': [], 'test': [], 'extra_test': []}
        self.epoch_samples = {'valid': [], 'extra_valid': [], 'test': [], 'extra_test': []}
        self.val_H3_d_rmsd_sims = []
        self.save_hyperparameters() # saves to self.hparams (also in model checkpoints)
        # self.log_gradient_norms = True

    def configure_optimizers(self):
        optim = {}
        if self._exp_cfg.optimizer.which == 'AdamW':
            optim['optimizer'] = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=self._exp_cfg.optimizer.lr,
                                                   weight_decay=self._exp_cfg.optimizer.weight_decay)
        elif self._exp_cfg.optimizer.which == 'RAdam':
            optim['optimizer'] = torch.optim.RAdam(params=self.model.parameters(),
                                                   lr=self._exp_cfg.optimizer.lr,
                                                   weight_decay=self._exp_cfg.optimizer.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer.which: {self._exp_cfg.optimizer.which}')

        if self._exp_cfg.lr_scheduler.use == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                            optim['optimizer'],
                                            factor=self._exp_cfg.lr_scheduler.factor,
                                            patience=self._exp_cfg.lr_scheduler.patience)
            optim['lr_scheduler'] = { 'scheduler': lr_scheduler,
                                      'monitor': self._exp_cfg.lr_scheduler.monitor,  
                                      'interval': self._exp_cfg.lr_scheduler.interval,
                                      'frequency': self._exp_cfg.lr_scheduler.frequency,
                                      'strict': self._exp_cfg.lr_scheduler.strict
                                    }

        elif self._exp_cfg.lr_scheduler.use == 'cosine_annealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim['optimizer'],
                                                                                T_0=self._exp_cfg.lr_scheduler.T_0,
                                                                                T_mult=self._exp_cfg.lr_scheduler.T_mult,
                                                                                eta_min=self._exp_cfg.lr_scheduler.eta_min,
                                                                last_epoch=-1)
            optim['lr_scheduler'] = { 'scheduler': lr_scheduler, 
                                      'interval': self._exp_cfg.lr_scheduler.interval,
                                      'frequency': self._exp_cfg.lr_scheduler.frequency,
                                      'strict': self._exp_cfg.lr_scheduler.strict
                                    }
        elif self._exp_cfg.lr_scheduler.use == 'warmup_cosine_annealing':
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim['optimizer'],
                start_factor=self._exp_cfg.lr_scheduler.warmup_start_factor,
                end_factor=self._exp_cfg.lr_scheduler.warmup_end_factor,
                total_iters=self._exp_cfg.lr_scheduler.warmup_epochs
            )

            # Main scheduler after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim['optimizer'],
                                                                                T_0=self._exp_cfg.lr_scheduler.T_0,
                                                                                T_mult=self._exp_cfg.lr_scheduler.T_mult,
                                                                                eta_min=self._exp_cfg.lr_scheduler.eta_min,
                                                                last_epoch=-1)
            # Combine warmup and main scheduler
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim['optimizer'],
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self._exp_cfg.lr_scheduler.warmup_epochs]  # switch between schedulers
            )

            optim['lr_scheduler'] = { 'scheduler': lr_scheduler, 
                                      'interval': self._exp_cfg.lr_scheduler.interval,
                                      'frequency': self._exp_cfg.lr_scheduler.frequency,
                                      'strict': self._exp_cfg.lr_scheduler.strict
                                    }
        elif self._exp_cfg.lr_scheduler.use == 'cosine_annealing_no_restart':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim['optimizer'],
                                                                T_max=self._exp_cfg.lr_scheduler.T_max,
                                                                eta_min=self._exp_cfg.lr_scheduler.eta_min,
                                                                last_epoch=-1)
            optim['lr_scheduler'] = { 'scheduler': lr_scheduler, 
                                      'interval': self._exp_cfg.lr_scheduler.interval,
                                      'frequency': self._exp_cfg.lr_scheduler.frequency,
                                      'strict': self._exp_cfg.lr_scheduler.strict
                                    }

        elif self._exp_cfg.lr_scheduler.use == 'warmup_cosine_annealing_no_restart':
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim['optimizer'],
                start_factor=self._exp_cfg.lr_scheduler.warmup_start_factor,
                end_factor=self._exp_cfg.lr_scheduler.warmup_end_factor,
                total_iters=self._exp_cfg.lr_scheduler.warmup_epochs
            )

            # Main scheduler after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim['optimizer'],
                T_max=self._exp_cfg.lr_scheduler.T_max,
                eta_min=self._exp_cfg.lr_scheduler.eta_min,
                last_epoch=-1
            )

            # Combine warmup and main scheduler
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim['optimizer'],
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self._exp_cfg.lr_scheduler.warmup_epochs]  # switch between schedulers
            )

            # Update your dictionary
            optim['lr_scheduler'] = {
                'scheduler': lr_scheduler,
                'interval': self._exp_cfg.lr_scheduler.interval,
                'frequency': self._exp_cfg.lr_scheduler.frequency,
                'strict': self._exp_cfg.lr_scheduler.strict
            }

        elif not self._exp_cfg.lr_scheduler.use:
            pass
        else:
            raise ValueError(f'Unknown lr_scheduler.use: {self._exp_cfg.lr_scheduler}')

        return optim
    
    def _log_scalar(
            self,
            key,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=None,

            # ! WARNING ============================================================
            # The original values here (also in multiflow) were:
            #       sync_dist=False, rank_zero_only=True
            # The latter is possible because @rank_zero_only decorator in WandbLogger implements guard 
            # equivalent to: https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#rank-zero-only

            # The defaults for both are False in Lightning Docs; Wandb doesn't suggest changing them.
            # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-wb

            # Passing sync_dist=True (and rank_zero_only=False) seems to cause issues with stalling on multi-GPU runs.

            sync_dist=True,        # whether to reduce metric across devices (adds communication overhead)
            rank_zero_only=False    # logged only from rank 0
            # =======================================================================
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self.model, norm_type=2)
        for k, v in norms.items():
            self._log_scalar(k, v,
                sync_dist=False, 
                rank_zero_only=True,
                on_step=True,
                on_epoch=False)

        # Log learning rate(s)
        try:
            lr_values = [group.get('lr', None) for group in optimizer.param_groups]
            if len(lr_values) == 1:
                self._log_scalar(
                    'train/lr', lr_values[0],
                    sync_dist=False,
                    rank_zero_only=True,
                    on_step=True,
                    on_epoch=False
                )
            else:
                for i, lr in enumerate(lr_values):
                    self._log_scalar(
                        f'train/lr_group_{i}', lr,
                        sync_dist=False,
                        rank_zero_only=True,
                        on_step=True,
                        on_epoch=False
                    )
        except Exception:
            # Silently skip LR logging if optimizer doesn't expose param_groups
            pass

    def _should_log_region_rmsd(self, stage, batch_type, cfg):
        if batch_type != "single":
            return False
        if stage in ("valid", "extra_valid", "test", "extra_test"):
            return True
        if stage == "train" and getattr(cfg, "log_train_region_rmsd", False):
            return True
        return False

    def _log_region_rmsd_metrics(self, clean_batch, stage, cfg):
        """Fixed-t one-step forward pass for per-region RMSD monitoring."""
        metrics_t = getattr(cfg, "metrics_t", 0.05)
        batch_size = clean_batch["res_mask"].shape[0]

        with torch.no_grad():
            metrics_batch = self.interpolant.corrupt_batch_at_fixed_t(
                clean_batch, metrics_t, stage=stage
            )
            metrics_output = self.model(metrics_batch)
            pred_atom37 = all_atom.to_atom37(
                metrics_output["pred_trans"], metrics_output["pred_rotmats"]
            )
            gt_atom37 = all_atom.to_atom37(
                clean_batch["trans_1"], clean_batch["rotmats_1"]
            )
            rmsd_metrics = losses.compute_fw_aligned_folding_metrics_v2(
                pred_atom37=pred_atom37,
                gt_atom37=gt_atom37,
                res_mask=clean_batch["res_mask"],
                imgt=clean_batch["imgt_numeric"],
                region=clean_batch["region"],
            )

        for key, val in rmsd_metrics.items():
            if np.isnan(val) or not np.isfinite(val):
                continue
            self.log(
                f"{stage}/{key}",
                torch.tensor(val, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=(stage.startswith("valid") and key == "bb_rmsd_h_cdr3"),
                batch_size=batch_size,
                sync_dist=True,
            )

    def model_step(self, noisy_batch: Any, batch_type: str, stage: str = 'train'):
        """ Forward pass and loss computation for a train/val/test batch of strucs and seqs. 
        Not called during inference/sampling.
        See eq. (16) in MultiFlow paper for loss components.
        Every antibody in a batch has the same length. """
        assert stage in ['train', 'valid',  'extra_valid', 'test', 'extra_test', 'sample'], f'Unknown stage {stage}.'

        if stage == 'train':
            cfg = self._exp_cfg.training
        elif stage == 'valid' or stage == 'extra_valid':
            cfg = self._exp_cfg.validation
        elif stage == 'test' or stage == 'extra_test':
            cfg = self._exp_cfg.testing
        elif stage == 'sample':
            cfg = self._exp_cfg.sampling

        context = torch.no_grad() if stage != 'train' else nullcontext()

        if batch_type == 'single':
            with context:
                model_output = self.model(noisy_batch)
                assert model_output['pred_trans'].dim() == 3

        elif batch_type == 'ensemble':
            n_batch = noisy_batch['aatypes_1'].shape[0]
            n_ensemble = noisy_batch['aatypes_1'].shape[1]
            with context:
                # combine batch and ensemble dimensions for model pass
                noisy_batch = {k: v.reshape(n_batch * n_ensemble, *v.size()[2:]) for k,v in noisy_batch.items()}

                model_output = self.model(noisy_batch)

                # reshape to seperate dims for batch and ensmeble
                noisy_batch = {k: v.reshape(n_batch, n_ensemble, *v.size()[1:]) for k,v in noisy_batch.items()}
                model_output = {k: v.reshape(n_batch, n_ensemble, *v.size()[1:]) for k,v in model_output.items()}
               
                assert model_output['pred_trans'].dim() == 4
                assert model_output['pred_trans'].size()[:2] == (n_batch, n_ensemble)

        if self._should_log_region_rmsd(stage, batch_type, cfg):
            self._log_region_rmsd_metrics(noisy_batch, stage, cfg)

        loss_dict = losses.all_losses(noisy_batch, model_output, batch_type, cfg.training_losses)

        return loss_dict

    def process_data_batch(self, data_batch, batch_type, stage):
        assert stage in ['train', 'valid', 'extra_valid', 'test', 'extra_test', 'sample'], f'Unknown stage {stage}.'
        assert 'diffuse_mask' in data_batch and data_batch['diffuse_mask'] is not None, 'No diffuse mask in data_batch.'
        self.interpolant.set_device(data_batch['res_mask'].device)
        
        # noise batch
        if batch_type == 'single':
            '''
            ##### FOR DEBUG ONLY########
            if self._cached_debug_batch is None:
                self._cached_debug_batch = self.interpolant.corrupt_batch(data_batch, stage=stage)
            noisy_batch = self._cached_debug_batch
            '''
            noisy_batch = self.interpolant.corrupt_batch(data_batch, stage=stage)
            assert noisy_batch['t_trans'].dim() == 2
        elif batch_type == 'ensemble':
            # noise each ensemble separately
            batch_size = data_batch['aatypes_1'].shape[0]
            ensemble_size = data_batch['aatypes_1'].shape[1]
            noisy_batch = {}
            for i in range(batch_size):
                ensemble = {k: v[i] for k,v in data_batch.items()}
                noisy_ensemble = self.interpolant.corrupt_batch(ensemble, stage=stage)
                for k in noisy_ensemble.keys():
                    if k not in noisy_batch:
                        noisy_batch[k] = [noisy_ensemble[k]]
                    else:
                        noisy_batch[k].append(noisy_ensemble[k])

            noisy_batch = {k: torch.stack(v) for k,v in noisy_batch.items()}
            assert noisy_batch['t_trans'].dim() == 3
            assert noisy_batch['t_trans'].size() == (batch_size, ensemble_size, 1)

        # generate self-conditioning edge-rep distogram and sequence
        # TODO: currently doesn't work for ensemble
        if self._interpolant_cfg.self_condition and batch_type == 'single':
            if stage in ['valid', 'extra_valid', 'test', 'extra_test'] or random.random() > 0.5:
                with torch.no_grad():
                    initial_preds = self.model(noisy_batch)
                    # adds predicted aatypes to self conditioning
                    noisy_batch = self.interpolant.update_self_conditioning(noisy_batch, initial_preds)
        elif self._interpolant_cfg.self_condition and batch_type == 'ensemble':
            raise NotImplementedError(
                f'Self condition not implemented for ensemble batch')

        # forward pass and per-item losses
        batch_losses = self.model_step(noisy_batch, batch_type, stage=stage)

        return batch_losses, noisy_batch

    def loss_agg_and_log(self, batch_losses, noisy_batch, stage):
        assert stage in ['train', 'valid', 'extra_valid', 'test', 'extra_test'], f'Invalid stage {stage}.'

        # mean loss across loss types per item in batch
        num_batch = batch_losses['bb_rmsd_loss'].shape[0]
        total_losses = { 
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"{stage}/{k}", v, prog_bar=False, batch_size=num_batch,
                sync_dist=False, 
                rank_zero_only=True,
                on_step=True,
                on_epoch=False)
        
        # mean loss across time-bin; per loss type
        for t_name in ['t_rot', 't_trans', 't_seq']:
            t_value = torch.squeeze(noisy_batch[t_name])
            self._log_scalar(
                f"{stage}/{t_name}",
                np.mean(du.to_numpy(t_value)),
                prog_bar=False, batch_size=num_batch,
                sync_dist=False, 
                rank_zero_only=True,
                on_step=True,
                on_epoch=False)
            
            # log stratified losses
            for loss_name, loss_dict in batch_losses.items():
                match loss_name:
                    case 'rots_vf_loss' if t_name == 't_rot':
                        batch_t = t_value
                    case 'aatypes_loss' if t_name == 't_seq':
                        batch_t = t_value
                    case 'trans_loss' if t_name == 't_trans':
                        batch_t = t_value
                    case _ if t_name == 't_trans':
                        batch_t = t_value
                    case _ if t_name != 't_trans':
                        continue

                stratified_losses = mu.t_stratified_loss(
                    batch_t, loss_dict, loss_name=loss_name)
                for k,v in stratified_losses.items():
                    self._log_scalar(
                        f"{stage}/{k}", v, prog_bar=False, batch_size=num_batch,
                        sync_dist=False, 
                        rank_zero_only=True,
                        on_step=True,
                        on_epoch=False)
                
        # throughput
        self._log_scalar(
            f"{stage}/length", noisy_batch['res_mask'].shape[1], 
            prog_bar=False, batch_size=num_batch,
            sync_dist=False, rank_zero_only=True, on_step=True, on_epoch=False)
        self._log_scalar(
            f"{stage}/batch_size", num_batch, prog_bar=False,
            sync_dist=False, rank_zero_only=True, on_step=True, on_epoch=False)
        
        # total loss
        if stage == 'train':
            specified_loss = self._exp_cfg.training.loss
        elif stage == 'valid' or stage == 'extra_valid':
            specified_loss = self._exp_cfg.validation.loss
        elif stage == 'test' or stage == 'extra_test':
            specified_loss = self._exp_cfg.testing.loss
        on_epoch = stage in ['valid', 'extra_valid', 'test', 'extra_test']
        on_step = stage == 'train'

        loss = total_losses[specified_loss]
        self._log_scalar(f"{stage}/loss", loss, 
                         batch_size=num_batch,
                         on_step=on_step,
                         on_epoch=on_epoch)
        
        return loss

    def data_step(self, data_batch, batch_type, stage='train'):
        """Training-like step (loss computation on strucs and seqs)."""

        # noising, fwd pass, individual losses
        batch_losses, noisy_batch = self.process_data_batch(data_batch, batch_type, stage)
        
        # loss logging and aggregation
        num_batch = batch_losses['bb_rmsd_loss'].shape[0]
        loss = self.loss_agg_and_log(batch_losses, noisy_batch, stage=stage)

        return loss, num_batch

    def move_batch_to_device(self, batch: Any):
        self.interpolant.set_device(f'cuda:{torch.cuda.current_device()}')
        batch = {k: v.to(self.interpolant._device) for k, v in batch.items()}
        return batch

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()

        single_batch, ensemble_batch = self.split_batch_types(batch)

        if single_batch is not None: # if single batch present, training on single structures
            data_batch = single_batch
            batch_type = 'single'
        elif ensemble_batch is not None: # elif ensemble batch present, training on ensemble structures
            data_batch = ensemble_batch
            batch_type = 'ensemble'
        else:
            raise ValueError('No training samples found!')

        data_batch = self.move_batch_to_device(data_batch)
        train_loss, num_batch = self.data_step(data_batch, batch_type=batch_type, stage='train')

        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time,
                         on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=False)

        return train_loss

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log('train/epoch_time_minutes', 
                 epoch_time,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 sync_dist=True,
        )
        self._epoch_start_time = time.time()

    def sample_step(self, gen_batch):
        """Inference-like step (de novo generation)."""

        self.interpolant.set_device(f'cuda:{torch.cuda.current_device()}')
        return self.interpolant.sample(gen_batch, self.model)

    def rollout_single_sample_validation(self, single_batch):
        """Full reverse-diffusion sample; return per-region FW-aligned backbone RMSDs."""
        single_batch = self.move_batch_to_device(single_batch)
        with torch.no_grad():
            returndict = self.sample_step(single_batch)
            # sample traj is detached to CPU in convert_to_coords_and_detach
            pred_atom37 = returndict['clean_atom37_traj'][-1]
            device = pred_atom37.device
            gt_atom37 = all_atom.to_atom37(
                single_batch['trans_1'], single_batch['rotmats_1']
            ).to(device)
            return losses.compute_fw_aligned_folding_metrics_v2(
                pred_atom37=pred_atom37,
                gt_atom37=gt_atom37,
                res_mask=single_batch['res_mask'].to(device),
                imgt=single_batch['imgt_numeric'].to(device),
                region=single_batch['region'].to(device),
            )

    def _sync_sample_validation_metrics(self, metrics_list):
        """All-reduce finite per-structure metrics into global nanmeans."""
        if len(metrics_list) == 0:
            return {}

        synced = {}
        for key in metrics_list[0].keys():
            vals = [m[key] for m in metrics_list if np.isfinite(m[key])]
            local_sum = float(np.sum(vals)) if vals else 0.0
            local_count = float(len(vals))
            total_sum = torch.tensor(local_sum, device=self.device, dtype=torch.float64)
            total_count = torch.tensor(local_count, device=self.device, dtype=torch.float64)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            if total_count.item() == 0:
                synced[key] = float('nan')
            else:
                synced[key] = (total_sum / total_count).item()
        return synced

    def _run_single_sample_validation(self):
        valid_cfg = self._exp_cfg.validation
        if not getattr(valid_cfg, 'single_sample_validation', False):
            return
        if self.task != 'fwd_fold':
            self._print_logger.warning(
                'single_sample_validation is only implemented for fwd_fold; skipping.'
            )
            return

        datamodule = self.trainer.datamodule
        valid_dataset = datamodule.singles.get('valid')
        if valid_dataset is None:
            self._print_logger.warning('No validation single-structure dataset; skipping sample validation.')
            return

        pdb_csv = valid_dataset.pdb_csv
        if getattr(valid_cfg, 'single_sample_use_cluster_subsample', True):
            pdb_csv = subsample_one_per_cluster(pdb_csv, seed=self.current_epoch)
        else:
            pdb_csv = pdb_csv.copy()
            pdb_csv['index'] = list(range(len(pdb_csv)))

        indices = pdb_csv['index'].tolist()
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            indices = indices[rank::world_size]

        num_timesteps_override = getattr(valid_cfg, 'single_sample_num_timesteps', None)
        old_timesteps = None
        if num_timesteps_override is not None:
            old_timesteps = self.interpolant._sample_cfg.num_timesteps
            self.interpolant._sample_cfg.num_timesteps = num_timesteps_override

        metrics_list = []
        try:
            for idx in indices:
                sample_batch = heterogenos_length_collate([valid_dataset[int(idx)]])
                metrics_list.append(self.rollout_single_sample_validation(sample_batch))
        finally:
            if old_timesteps is not None:
                self.interpolant._sample_cfg.num_timesteps = old_timesteps

        aggregated = self._sync_sample_validation_metrics(metrics_list)
        if not aggregated:
            return

        for key, val in aggregated.items():
            if np.isnan(val) or not np.isfinite(val):
                continue
            self.log(
                f'valid/sample_{key}',
                torch.tensor(val, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=(key == 'bb_rmsd_h_cdr3'),
                sync_dist=False,
            )

        if self.trainer.is_global_zero:
            self._print_logger.info(
                f'Sample validation on {len(pdb_csv)} cluster-representative structures: '
                f"bb_rmsd_h_cdr3={aggregated.get('bb_rmsd_h_cdr3', float('nan')):.4f}"
            )

    def split_batch_types(self, batch: Any):
        return (
            batch.get('single_struc'),
            batch.get('ensemble')
        )

    def rollout_validation(self, ensemble_batch):
        n_batch = ensemble_batch['aatypes_1'].shape[0]
        n_ensemble = ensemble_batch['aatypes_1'].shape[1]
        n_total = n_batch * n_ensemble
        # reshape to combined batch and ensemble dimensions before passing through model
        ensemble_batch = {k: v.reshape(n_total, *v.size()[2:]) for k,v in ensemble_batch.items()}

        if self.task == 'fwd_fold':
            returndict = self.sample_step(ensemble_batch) # moves batch to device
            clean_atom37_traj = returndict['clean_atom37_traj'] # clean atom37 traj includes sidechain atoms
            int_traj = returndict['atom37_traj']
            S = self._interpolant_cfg.sampling.num_timesteps
            assert clean_atom37_traj.shape == (S, n_total, ensemble_batch['res_idx'].shape[1], 37, 3) # (S, B*E, N, 37, 3)
            assert int_traj.shape == (S+1, n_total, ensemble_batch['res_idx'].shape[1], 37, 3) # (S+1, B*E, N, 37, 3)
            gt_atom37 = all_atom.atom14_to_atom37(ensemble_batch['atom14_1'], ensemble_batch).detach().cpu()
            assert gt_atom37.shape == (n_total, ensemble_batch['res_idx'].shape[1], 37, 3) # (B*E, N, 37, 3)
            ensemble_batch = {k: v.to(gt_atom37.device) for k, v in ensemble_batch.items()}

            # reshape to separate batch and ensemble dimensions
            clean_atom37_traj = clean_atom37_traj.reshape(S, n_batch, n_ensemble, *clean_atom37_traj.shape[2:]) # (S, B, E, N, 37, 3)
            int_traj = int_traj.reshape(S+1, n_batch, n_ensemble, *int_traj.shape[2:]) # (S+1, B, E, N, 37, 3)
            gt_atom37 = gt_atom37.reshape(n_batch, n_ensemble, *gt_atom37.shape[1:]) # (B, E, N, 37, 3)
            ensemble_batch = {k: v.reshape(n_batch, n_ensemble, *v.size()[1:]) for k,v in ensemble_batch.items()}

            folding_metrics = losses.compute_ensemble_prediction_metrics(
                clean_atom37_traj[-1], gt_atom37, ensemble_batch['res_mask'].detach(), ensemble_batch['imgt_numeric'].detach(), ensemble_batch['rmsf'].detach(), ensemble_batch['rmsd'].detach()
                )

            # append metrics for each ensemble seperately
            samples = []
            for i in range(n_batch):
                samples.append([
                    clean_atom37_traj[:, i],
                    int_traj[:, i],
                    gt_atom37[i],
                    ensemble_batch['aatypes_1'][i],
                    ensemble_batch['pdb_name_enc'][i],
                    ensemble_batch['res_mask'][i],
                    ensemble_batch['imgt_numeric'][i]
                ])

        elif self.task == 'inv_fold':
            # TODO: implement sensible metrics to evaluate antibody-likeness or protein-likeness
            raise NotImplementedError('Generation-based evaluation at validation/test phase is not yet implemented for inverse folding.')
        elif self.task == 'cogen':
            # TODO: implement sensible metrics to evaluate antibody-likeness or protein-likeness
            raise NotImplementedError('Generation-based evaluation at validation/test phase is not yet implemented for cogen.')

        return samples, folding_metrics


    def data_and_gen_step(self, batch: Any, batch_idx: int, stage='valid'):
        """
        Performs a training-like loss computation followed by sample generation
        on a joint batch of complete (training-like) data and inference-like data.
        Logs everything appropriately.
        """
        single_batch, ensemble_batch = self.split_batch_types(batch)
        loss = None

        samples_list = self.epoch_samples[stage]
        metrics_list = self.epoch_metrics[stage]

        if stage == 'valid' or stage == 'extra_valid':
            step_eval = self._exp_cfg.validation.step_validation
            sample_eval = self._exp_cfg.validation.sample_validation
        elif stage == 'test' or stage == 'extra_test':
            step_eval = self._exp_cfg.testing.step_validation
            sample_eval = self._exp_cfg.testing.sample_validation

        # training-like/step level loss computation
        if step_eval:
            step_batch = single_batch if single_batch is not None else ensemble_batch
            batch_type = 'single' if single_batch is not None else 'ensemble'
            step_batch = self.move_batch_to_device(step_batch)
            loss, len_step_batch = self.data_step(step_batch, batch_type=batch_type, stage=stage)

        # inference/complete sample rollout loss computation
        if sample_eval and ensemble_batch is not None:
            samples, folding_metrics = self.rollout_validation(ensemble_batch)
            samples_list += samples
            metrics_list += folding_metrics

        return loss, len_step_batch

    def val_or_test_step(self, batch: Any, batch_idx: int, stage='valid'):
        step_start_time = time.time()

        loss, combined_batchlen = self.data_and_gen_step(batch, batch_idx, stage=stage)

        step_time = time.time() - step_start_time
        self._log_scalar(f"{stage}/examples_per_second", combined_batchlen / step_time,
                         on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=False)

        return loss
        
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if type(batch) is tuple: # when two loaders are used a tuple (batch, index) is recieved
            batch = batch[0]
        if dataloader_idx == 0:
            # MAIN validation
            return self.val_or_test_step(batch, batch_idx, stage="valid")
        elif dataloader_idx == 1:
            # EXTRA validation
            self.val_or_test_step(batch, batch_idx, stage="extra_valid")
            return None
        else:
            raise ValueError(f'Ubexpected dataloader_idx {dataloader_idx} in validation_step.')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if type(batch) is tuple: # when two loaders are used a tuple (batch, index) is recieved
            batch = batch[0]
        if dataloader_idx == 0:
            # MAIN test
            return self.val_or_test_step(batch, batch_idx, stage="test")
        elif dataloader_idx == 1:
            # EXTRA test
            self.val_or_test_step(batch, batch_idx, stage="extra_test")
            return None
        else:
            raise ValueError(f'Ubexpected dataloader_idx {dataloader_idx} in test_step.')

    def end_val_test_epoch(self, stage='valid'):
        if stage == 'valid':
            samples_list = self.epoch_samples['valid'] + self.epoch_samples['extra_valid']
            metrics_list = self.epoch_metrics['valid']
            metrics_list_extra = self.epoch_metrics['extra_valid']
        elif stage == 'test':
            samples_list = self.epoch_samples['test'] + self.epoch_samples['extra_test']
            metrics_list = self.epoch_metrics['test']
            metrics_list_extra = self.epoch_metrics['extra_test']

        if len(metrics_list_extra) > 0:
            epoch_metrics = pd.DataFrame(metrics_list_extra)
            for metric_name, metric_val in epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f'extra_{stage}/{metric_name}',
                    metric_val,
                    batch_size=len(epoch_metrics),
                )
            metrics_list_extra.clear()

        if len(metrics_list) > 0:
            epoch_metrics = pd.DataFrame(metrics_list)
            for metric_name, metric_val in epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f'{stage}/{metric_name}',
                    metric_val,
                    batch_size=len(epoch_metrics),
                )

            if stage == 'valid':
                val_H3_d_rmsd_sims = torch.tensor(epoch_metrics.mean().to_dict()['h_cdr3_d_mean_rmsd_sims']).to(self.interpolant._device)
                dist.all_reduce(val_H3_d_rmsd_sims, op=dist.ReduceOp.SUM) # sync across devices
                val_H3_d_rmsd_sims /= dist.get_world_size()
                self.val_H3_d_rmsd_sims.append(val_H3_d_rmsd_sims.item())
    
                self._log_scalar(
                    f'{stage}/best_h_cdr3_d_mean_rmsd_sims',
                    min(self.val_H3_d_rmsd_sims),
                    batch_size=len(epoch_metrics),
                    sync_dist=False, # manually synced
                    rank_zero_only=True,
                )
            metrics_list.clear()

        if len(samples_list) > 0:
            # save best for best validation epoch and during training
            if stage == 'test':
                save = self._exp_cfg.testing.save_sampled_strucs
                write_dir = self._test_sample_write_dir
                epoch_n = self._exp_cfg.testing.ckpt_name.split('=')[-1].split('.')[0]
                suffix = f'epoch_{epoch_n}_steps_{self._interpolant_cfg.sampling.num_timesteps}_i_{self._exp_cfg.test_number}'
                self._print_logger.info(f'Saving test samples for epoch {epoch_n}')
            elif stage == 'valid' and self.val_H3_d_rmsd_sims[-1] == min(self.val_H3_d_rmsd_sims):
                save = self._exp_cfg.validation.save_sampled_strucs
                write_dir = self._valid_sample_write_dir
                suffix = f'best_val_epoch'
            elif self.current_epoch % self._exp_cfg.validation.sample_every_n_epochs == 0:
                save = self._exp_cfg.validation.save_sampled_strucs
                write_dir = self._valid_sample_write_dir
                suffix = f'epoch_{self.current_epoch}'
            else:
                save = False
            
            if save:
                for (atom37_traj_b, int_traj_b, gt_atom37_b, aatypes_b, pdb_idx_b, res_mask_b, imgt_b) in samples_list: # batch
                    for i in range(gt_atom37_b.shape[0]): # structure
                        res_mask = res_mask_b[i].bool()
                        atom37_traj = atom37_traj_b[:,i][..., res_mask, :, :].detach().numpy()
                        int_traj = int_traj_b[:,i][..., res_mask, :, :].detach().numpy()
                        gt_atom37 = gt_atom37_b[i][res_mask].detach().numpy()
                        aatypes = aatypes_b[i][res_mask].detach().numpy()
                        pdb_idx = pdb_idx_b[i].detach().numpy()
                        imgt = imgt_b[i].detach().numpy()
                        sample_dir = os.path.join(write_dir, f'sample_{pdb_idx.item()}')
                        os.makedirs(sample_dir, exist_ok=True)

                        true_struc_path = os.path.join(sample_dir, 'true_struc.pdb')
                        if not os.path.exists(true_struc_path):
                            au.write_prot_to_pdb(
                                prot_pos=gt_atom37,
                                file_path=os.path.join(sample_dir, 'true_struc.pdb'),
                                aatype=aatypes,
                                overwrite=True
                                )
                            with open(os.path.join(sample_dir, 'true_struc.pkl'), "wb") as f:
                                pickle.dump({"gt_atom37": gt_atom37, "res_mask": res_mask, 'imgt': imgt}, f)

                        au.write_prot_to_pdb(
                            prot_pos=atom37_traj[-1],
                            file_path=os.path.join(sample_dir, f'pred_struc_{suffix}.pdb'),
                            aatype=aatypes,
                            overwrite=True
                            )
                        with open(os.path.join(sample_dir, f'pred_struc_{suffix}.pkl'), "wb") as f:
                            pickle.dump({"pred_atom37": atom37_traj[-1], "res_mask": res_mask, 'imgt': imgt}, f)

                        if self._exp_cfg.validation.save_traj:
                            aatypes_save = np.repeat(aatypes[None, ...], atom37_traj.shape[0], axis=0) # repeat aatypes for each frame
                            au.write_prot_to_pdb(
                                prot_pos=atom37_traj,
                                file_path=os.path.join(sample_dir, f'pred_struc_{suffix}_traj.pdb'),
                                aatype=aatypes_save,
                                overwrite=True
                                )

                        if self._exp_cfg.validation.save_int_traj:
                            aatypes_save = np.repeat(aatypes[None, ...], int_traj.shape[0], axis=0) # repeat aatypes for each frame
                            au.write_prot_to_pdb(
                                prot_pos=int_traj,
                                file_path=os.path.join(sample_dir, f'pred_struc_{suffix}_int.pdb'),
                                aatype=aatypes_save,
                                overwrite=True
                                )
            self.epoch_samples[f'{stage}'].clear()
            self.epoch_samples[f'extra_{stage}'].clear()

    def on_validation_epoch_end(self):
        self.end_val_test_epoch(stage='valid')
        self._run_single_sample_validation()
    
    def on_test_epoch_end(self):
        self.end_val_test_epoch(stage='test')

    def predict_step(self, pred_batch, batch_idx: int):
        num_batch, num_res = pred_batch['res_idx'].shape
        write_dir = self._exp_cfg.prediction.output_dir

        # sample
        returndict = self.sample_step(pred_batch)

        for i in range(num_batch):
            pdb_name = bytes(pred_batch['pdb_name'][i].tolist()).decode("utf-8")
            pdb_name = pdb_name.replace('\x00', '')
            res_mask = du.to_numpy(pred_batch['res_mask'][i].bool())
            copy_index = pred_batch['copy_index'][i].item()

            pred_atom37 = du.to_numpy(returndict['clean_atom37_traj'][-1, i][res_mask, :, :])
            pred_traj = du.to_numpy(returndict['clean_atom37_traj'][:, i][..., res_mask, :, :])
            pred_int_traj = du.to_numpy(returndict['atom37_traj'][:, i][..., res_mask, :, :])
            # gt_atom37 = du.to_numpy(all_atom.atom14_to_atom37(pred_batch['atom14_1'], pred_batch)[i][res_mask, :, :])
            imgt = du.to_numpy(pred_batch['imgt_numeric'][i][res_mask])
            aatypes = du.to_numpy(pred_batch['aatypes_1'][i][res_mask])

            os.makedirs(os.path.join(write_dir, pdb_name), exist_ok=True)
            au.write_prot_to_pdb(
                                prot_pos=pred_atom37,
                                file_path=os.path.join(write_dir, pdb_name, f'sample_{copy_index}.pdb'),
                                aatype=aatypes,
                                overwrite=True,
                                no_indexing=True,
                                imgt_numeric=imgt,
                                )
            # with open(os.path.join(write_dir, pdb_name, f'sample_{copy_index}.pkl'), "wb") as f:
            #     pickle.dump({"pred_atom37": pred_atom37, "res_mask": res_mask, 'imgt': imgt, 'aatypes': aatypes}, f)

            # true_struc_path = os.path.join(write_dir, pdb_name, 'true_struc.pdb')
            # if not os.path.exists(true_struc_path):
            #     au.write_prot_to_pdb(
            #         prot_pos=gt_atom37,
            #         file_path=os.path.join(true_struc_path),
            #         aatype=aatypes,
            #         overwrite=True,
            #         no_indexing=True,
            #         )
            #     with open(true_struc_path.replace('.pdb', '.pkl'), "wb") as f:
            #         pickle.dump({"gt_atom37": gt_atom37, "res_mask": res_mask, 'imgt': imgt, 'aatypes': aatypes}, f)

            if self._exp_cfg.prediction.save_traj:
                aatypes_save = np.repeat(aatypes[None, ...], pred_traj.shape[0], axis=0) # repeat aatypes for each frame
                au.write_prot_to_pdb(
                    prot_pos=pred_traj,
                    file_path=os.path.join(write_dir, pdb_name, f'sample_{copy_index}_clean_traj.pdb'),
                    aatype=aatypes_save,
                    overwrite=True,
                    imgt_numeric=imgt,
                    )

            if self._exp_cfg.prediction.save_int_traj:
                aatypes_save = np.repeat(aatypes[None, ...], pred_int_traj.shape[0], axis=0) # repeat aatypes for each frame
                au.write_prot_to_pdb(
                    prot_pos=pred_int_traj,
                    file_path=os.path.join(write_dir, pdb_name, f'sample_{copy_index}_int_traj.pdb'),
                    aatype=aatypes_save,
                    overwrite=True,
                    imgt_numeric=imgt,
                    )

        return
    
    def forward(self, batch):
        return self.model(batch)
