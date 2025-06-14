import numpy as np
import torch
from data import MocapSequenceDataset
import os.path as osp
from loguru import logger
from omegaconf import OmegaConf
from smpl import Smpl
class moshpp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = MoSh.prepare_cfg(dict_cfg=dict_cfg, **kwargs)
        num_train_markers = 46  # constant

        if self.cfg.surface_model.type == 'smplx' and cfg.moshpp.optimize_betas and (
                cfg.mocap.exclude_marker_types is not None and 'face' in cfg.mocap.exclude_marker_types):
            logger.info(
                'For optimizing face markers in smplx chumpy implementation you should set optimize_betas to False.\n'
                'Otherwise face markers must be excluded and optimize_face must be false. '
                'Chumpy implementation does not allow shared betas and '
                'separate facial expressions for first stage. You can run stagei twice as a fix. \n'
                'In the first run you get the shape parameters and in the second run you get face marker placement.'
                'This wont be accurate.')
            logger.info('Adding face to mocap.exclude_markers')
            if cfg.mocap.exclude_marker_types is None:
                cfg.mocap.exclude_marker_types = ['face']
            else:
                cfg.mocap.exclude_marker_types.append(['face'])
            logger.info('Setting moshpp.optimize_face to False')
            cfg.moshpp.optimize_face = False

    logger.info(f'Loading model: {surface_model_fname}')

    if surface_model_type == 'object':
        from moshpp.models.object_model import RigidObjectModel
        can_model = RigidObjectModel(ply_fname=surface_model_fname)
        beta_shared_models = [RigidObjectModel(ply_fname=surface_model_fname) for _ in
                                               range(num_beta_shared_models)]
    else:
        from moshpp.prior.gmm_prior_ch import create_gmm_body_prior

        sm_temp = load_surface_model(surface_model_fname=surface_model_fname,
                                     pose_hand_prior_fname=pose_hand_prior_fname,
                                     use_hands_mean=use_hands_mean,
                                     dof_per_hand=dof_per_hand,
                                     v_template_fname=v_template_fname,
                                     surface_model_type=surface_model_type,
                                     #this is to address the models that have the same number of jonts i.e. dog, rat
                                     )

        betas = ch.array(np.zeros(len(sm_temp.betas)))

        can_model = Smplx(trans=ch.array(np.zeros(sm_temp.trans.size)),
                                 pose=ch.array(np.zeros(sm_temp.pose.size)),
                                 betas=betas,
                                 temp_model=sm_temp)

        assert can_model.model_type == surface_model_type, ValueError(f'{can_model.model_type} != {surface_model_type}')
        priors = {}
        if surface_model_type == 'animal_horse':
            from moshpp.prior.horse_body_prior import smal_horse_prior, smal_horse_joint_angle_prior
            if pose_body_prior_fname:
                priors['pose'] = smal_horse_prior(pose_body_prior_fname)
            priors['pose_jangles'] = smal_horse_joint_angle_prior()
        elif surface_model_type == 'animal_dog':
            from moshpp.prior.dog_body_prior import MaxMixtureDog
            # from moshpp.prior.smal_dog_prior import smal_dog_prior, smal_dog_joint_angle_prior
            if pose_body_prior_fname:
                priors['pose'] = MaxMixtureDog(prior_pklpath=pose_body_prior_fname).get_gmm_prior()
            # priors['pose_jangles'] = smal_dog_joint_angle_prior()
        else:
            if pose_body_prior_fname:
                priors['pose'] = create_gmm_body_prior(pose_body_prior_fname=pose_body_prior_fname,
                                                   exclude_hands=surface_model_type in ['smplh', 'smplx'])
        if not optimize_face:
            priors['betas'] = AliasedBetas(sv=can_model, surface_model_type=surface_model_type)

        can_model.priors = priors

        # do not share betas when optimizing betas to go around an issue of chumpy
        # with it current implementation of SMPL-X body shape betas and facial expressions share the same parameter; i.e. betas
        # now mosh requires betas to be shared across the canonical model and the 12 frames
        # now those 12 frames cannot share expressions though!
        # and that requires double indexing the same array; i.e. once for body shape and once for expressions
        # that messes up chumpy since it holds a grudge on double indexing the same optimization free variable
        # solution is to precompute betas and to not lock it at all when face needs to be optimized
        beta_shared_models = [SmplModelLBS(pose=ch.array(np.zeros(can_model.pose.size)),
                                           trans=ch.array(np.zeros(can_model.trans.size)),
                                           betas=can_model.betas if not optimize_face else ch.array(np.zeros(can_model.betas.size)),
                                           temp_model=can_model) for _ in range(num_beta_shared_models)]

    return can_model, beta_shared_models






    def forward(self, x: torch.Tensor):
        x = self.stagei(x)
        y = self.stageii(x)
        return y

    def fit_sequence(self, sequence_data):
        pass

    def stagei(self):
        pass

    def stageii(self):
        pass

class StageOneFitter(torch.nn.Module):
    def __init__(self,
                 num_frames = 12,
                 num_betas=10,
                 device='cuda'):
        self.model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz')
        self.betas = torch.nn.Parameter(torch.zeros(num_betas))
        self.body_pose = torch.nn.Parameter(torch.zeros([num_frames,23*3])) #看看后期要不要改成自动适配
        self.global_orient = torch.nn.Parameter(torch.zeros([num_frames,3]))
        self.transl = torch.nn.Parameter(torch.zeros([num_frames,3]))
        tpose = torch.zeros([num_frames,23*3])
        self.register_buffer('tpose', tpose)
        self.device = device

    def fit(self, train_loader, num_epochs=10):
        print(f"Starting Stage i training for {num_epochs} epochs.")



        opt = torch.optim.LBFGS(parameters,
                                max_iter=maxiter,
                                line_search_fn='strong_wolfe')
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader, epoch)
            if val_loader is not None:
                self.validate(val_loader, epoch)
            if self.scheduler:
                self.scheduler.step()

    def train_one_epoch(self, dataloader, epoch, stage):
        self.model.train()
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model.training_step(batch, stage=stage)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class StageTwoFitter:
    def __init__(self, model, optimizer, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def fit(self, train_loader, val_loader=None, stage='I', num_epochs=10):
        print(f"Starting Stage {stage} training for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader, epoch, stage)
            if val_loader is not None:
                self.validate(val_loader, epoch, stage)
            if self.scheduler:
                self.scheduler.step()

    def train_one_epoch(self, dataloader, epoch, stage):
        self.model.train()
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model.training_step(batch, stage=stage)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader, epoch, stage):
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model.validation_step(batch, stage=stage)



def main():

    # 1. load data
    cfg = OmegaConf.load('./config.yaml')

    dataset = MocapSequenceDataset(cfg.paths.dataset_dir)


    # 2. load model
    stageonefitter = StageOneFitter()

    stageonefitter.fit()

    logger.debug('Final mosh stagei loss: {}'.format(' | '.join(
        [f'{k} = {np.sum(v ** 2):2.2e}' for k, v in mp.stagei_data['stagei_debug_details']['stagei_errs'].items()])))

    if not mp.cfg.runtime.stagei_only:
        mp.mosh_stageii(mosh_stageii)
        logger.debug('Final mosh stageii loss: {}'.format(' | '.join(
            [f'{k} = {np.sum(v ** 2):2.2e}' for k, v in
             mp.stageii_data['stageii_debug_details']['stageii_errs'].items()])))

    # 3. define loss and training (stage i and ii)

    for data in dataset:
        pass

    # 4. test model



if __name__ == '__main__':
    main()