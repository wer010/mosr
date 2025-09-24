import os
import pickle
import torch
import numpy as np
import argparse
import json
from glob import glob
import os.path as osp
from data import rigidbody_marker_id, moshpp_marker_id, BabelDataset, MetaBabelDataset, virtual_marker, MetaCollate
from torch.utils.data import DataLoader, random_split
from models import Moshpp, FrameModel, SequenceModel
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from utils import visualize, vis_diff
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
import higher
from geo_utils import estimate_lcs_with_faces


def loss_rec_fn(output, gt, smpl_model=None):
    pos_offset = 0.0095
    ori_offset = 0.0095
    B = output["poses"].shape[0]
    L = output["poses"].shape[1]
    device = output["poses"].device


    output = smpl_model(betas=output["betas"],
                     body_pose=output["poses"][:,3:],
                     global_orient=output["poses"][:,0:3],
                     transl=output["trans"])
    v_posed = output['vertices']
    joints = output['joints']

    lcs = estimate_lcs_with_faces(vid=vid,
                                  fid=smpl_model.vertex_faces[vid],
                                  vertices=v_posed,
                                  faces=model.faces_tensor)

    marker_pos = torch.matmul(lcs, pos_offset[None, ..., None])[:, :, 0:3, 0]
    marker_ori = torch.matmul(lcs[:, :, 0:3, 0:3], ori_offset)

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    pose_loss = mse_loss(output["poses"], gt["poses"])
    shape_loss = l1_loss(output["betas"].view(B, -1), gt["betas"])
    tran_loss = mse_loss(output["trans"], gt["trans"])
    return pose_loss + shape_loss + tran_loss

def loss_fn(output, gt, smpl_model=None, do_fk=True):
    B = output["poses"].shape[0]
    L = output["poses"].shape[1]
    device = output["poses"].device

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    pose_loss = mse_loss(output["poses"], gt["poses"])
    shape_loss = l1_loss(output["betas"].view(B, -1), gt["betas"])
    tran_loss = mse_loss(output["trans"], gt["trans"])

    if do_fk:
        joints_hat = smpl_model(
            betas=output["betas"].expand(-1, L, -1).reshape(B * L, -1),
            body_pose=output["poses"].reshape(B * L, -1)[:, 3:],
            global_orient=output["poses"].reshape(B * L, -1)[:, 0:3],
            transl=output["trans"].reshape(B * L, -1),
        )["joints"].reshape(B, L, -1, 3)
        fk_loss = mse_loss(joints_hat, gt["joints"])
    else:
        fk_loss = torch.zeros(1, device=device)
    total_loss = pose_loss + shape_loss + tran_loss + 0.1 * fk_loss

    losses = {
        "pose": pose_loss,
        "shape": shape_loss,
        "tran": tran_loss,
        "fk": fk_loss,
        "total_loss": total_loss,
    }
    return losses


def train(
    train_dataset,
    test_dataset,
    model,
    save_dir,
    metrics_engine,
    batch_size=5,
    device="cuda",
    lr=5e-4,
    epochs=400,
    smpl_model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
):
    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    best_mpjpe = torch.inf
    # 普通训练
    smpl_model = Smpl(
        model_path=smpl_model_path,
        device=device,
    )
    model.train()

    collate_fn = MetaCollate()

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    train_optimizer = optim.Adam(model.parameters(), lr=lr)
    val_optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(train_optimizer, step_size=epochs//10, gamma=0.8)
    global_step = 0
    print("Begin training.")
    for epoch in tqdm(range(epochs)):

        for data in trainloader:
            B = data["marker_info"].shape[0]
            L = data["marker_info"].shape[1]
            x = data["marker_info"].contiguous().view(B, L, -1)

            train_optimizer.zero_grad()
            global_step += 1

            output = model(x)

            losses = loss_fn(output, data, smpl_model)
            if writer is not None:
                mode_prefix = "train"
                for k in losses:
                    prefix = "{}/{}".format(k, mode_prefix)
                    writer.add_scalar(prefix, losses[k].cpu().item(), global_step)

            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            losses["total_loss"].backward()
            train_optimizer.step()

        scheduler.step()
        # evaluate the query set (support set)
        finetune = True
        mpjpe = torch.tensor(0.0).to(device)
        metrics_tasks = {}
        for data in testloader:
            num_tasks = data[0]["marker_info"].shape[0]
            for t in range(num_tasks):
                supp_set = {key: value[t] for key, value in data[0].items()}
                qry_set = {key: value[t] for key, value in data[1].items()}
                if finetune:
                    # print('Begin finetuning')
                    B = supp_set["marker_info"].shape[0]
                    L = supp_set["marker_info"].shape[1]

                    ft_model = model.clone()

                    ft_model.train()
                    for ft_i in range(B // batch_size):
                        val_optimizer.zero_grad()
                        x = (
                            supp_set["marker_info"][
                                batch_size * ft_i : (ft_i + 1) * batch_size
                            ]
                            .contiguous()
                            .view(batch_size, L, -1)
                        )
                        output = ft_model(x)
                        gt = {
                            key: value[batch_size * ft_i : (ft_i + 1) * batch_size]
                            for key, value in supp_set.items()
                            if key != "task_name"
                        }
                        losses = loss_fn(output, gt, smpl_model)
                        losses["total_loss"].backward()
                        val_optimizer.step()
                else:
                    ft_model = model

                ft_model.eval()
                B = qry_set["marker_info"].shape[0]
                L = qry_set["marker_info"].shape[1]

                output_list = []
                for val_i in range(B):
                    x = qry_set["marker_info"][val_i].contiguous().view(1, L, -1)
                    output = ft_model(x)
                    output["joints"] = smpl_model(
                        betas=output["betas"].reshape(-1),
                        body_pose=output["poses"].reshape(L, -1)[:, 3:],
                        global_orient=output["poses"].reshape(L, -1)[:, 0:3],
                        transl=output["trans"].reshape(L, -1),
                    )["joints"].reshape(1, L, -1, 3)
                    output_list.append(output)
                # vis_diff_aitviewer('smpl', gt_full_poses=support_set['poses'][i], gt_betas=support_set['betas'][i], gt_trans=support_set['trans'][i],
                #                    pred_full_poses=output['poses'].squeeze(), pred_betas=output['betas'].squeeze(),
                #                    pred_trans=output['trans'].squeeze())
                output = {}
                for key in output_list[0].keys():
                    output[key] = torch.concatenate([item[key] for item in output_list])

                metrics = metrics_engine.compute(output, qry_set)
                metrics_tasks[supp_set["task_name"]] = metrics
                for k in metrics.keys():
                    prefix = f"{k}/test"
                    writer.add_scalar(prefix, metrics[k], global_step)
                mpjpe += metrics["MPJPE [mm]"]
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            for key in metrics_tasks:
                print(
                    metrics_engine.to_pretty_string(
                        metrics_tasks[key],
                        f"Task {key}-{model.model_name()}-SupportSet",
                    )
                )
            torch.save(model.state_dict(), osp.join(save_dir, "model.pth"))


def test(
    test_dataset,
    model,
    metrics_engine,
    model_path=None,
    device="cuda",
    vis=False,
    smpl_model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
    epochs_ft=1,
    batch_size=5
):

    collate_fn = MetaCollate()
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    smpl_model = Smpl(
        model_path=smpl_model_path,
        device=device,
    )
    model.eval()
    supp_results = []
    qry_results = []
    for data in testloader:
        supp_set = {key: value[0] for key, value in data[0].items()}
        qry_set = {key: value[0] for key, value in data[1].items()}
        # print('Begin finetuning')
        B = supp_set["marker_info"].shape[0]
        L = supp_set["marker_info"].shape[1]
        ft_model = model.clone()
        optimizer = optim.Adam(ft_model.parameters(), lr=5e-4)

        ft_model.train()
        for e in range(epochs_ft):
            for ft_i in range(B // batch_size):
                optimizer.zero_grad()
                x = (
                    supp_set["marker_info"][batch_size * ft_i : (ft_i + 1) * batch_size]
                    .contiguous()
                    .view(batch_size, L, -1)
                )
                output = ft_model(x)
                gt = {
                    key: value[batch_size * ft_i : (ft_i + 1) * batch_size]
                    for key, value in supp_set.items()
                    if key != "task_name"
                }
                losses = loss_fn(output, gt, smpl_model)
                losses["total_loss"].backward()
                optimizer.step()

        ft_model.eval()
        
        # evaluate the support set
        B = supp_set["marker_info"].shape[0]
        L = supp_set["marker_info"].shape[1]
        supp_output_list = []
        for val_i in range(B):
            x = supp_set["marker_info"][val_i].contiguous().view(1, L, -1)
            output = ft_model(x)
            output["joints"] = smpl_model(
                betas=output["betas"].reshape(-1),
                body_pose=output["poses"].reshape(L, -1)[:, 3:],
                global_orient=output["poses"].reshape(L, -1)[:, 0:3],
                transl=output["trans"].reshape(L, -1),
            )["joints"].reshape(1, L, -1, 3)
            supp_output_list.append(output)
            if vis:
                vis_diff_aitviewer(
                    "smpl",
                    gt_full_poses=supp_set["poses"][val_i],
                    gt_betas=supp_set["betas"][val_i],
                    gt_trans=supp_set["trans"][val_i],
                    pred_full_poses=output["poses"].squeeze(),
                    pred_betas=output["betas"].squeeze(),
                    pred_trans=output["trans"].squeeze(),
                )
        output = {}
        for key in supp_output_list[0].keys():
            output[key] = torch.concatenate([item[key] for item in supp_output_list])
        metrics = metrics_engine.compute(output, supp_set)
        supp_results.append(metrics)
        print(
            metrics_engine.to_pretty_string(
                metrics,
                f"Task {supp_set['task_name']}-{model.model_name()}-SupportSet",
            )
        )

        # evaluate the query set
        B = qry_set["marker_info"].shape[0]
        L = qry_set["marker_info"].shape[1]
        qry_output_list = []
        for val_i in range(B):
            x = qry_set["marker_info"][val_i].contiguous().view(1, L, -1)
            output = ft_model(x)
            output["joints"] = smpl_model(
                betas=output["betas"].reshape(-1),
                body_pose=output["poses"].reshape(L, -1)[:, 3:],
                global_orient=output["poses"].reshape(L, -1)[:, 0:3],
                transl=output["trans"].reshape(L, -1),
            )["joints"].reshape(1, L, -1, 3)
            qry_output_list.append(output)
            if vis:
                vis_diff_aitviewer(
                    "smpl",
                    gt_full_poses=qry_set["poses"][val_i],
                    gt_betas=qry_set["betas"][val_i],
                    gt_trans=qry_set["trans"][val_i],
                    pred_full_poses=output["poses"].squeeze(),
                    pred_betas=output["betas"].squeeze(),
                    pred_trans=output["trans"].squeeze(),
                )
        output = {}
        for key in qry_output_list[0].keys():
            output[key] = torch.concatenate([item[key] for item in qry_output_list])

        metrics = metrics_engine.compute(output, qry_set)
        qry_results.append(metrics)
        print(
            metrics_engine.to_pretty_string(
                metrics,
                f"Task {supp_set['task_name']}-{model.model_name()}-QuerySet",
            )
        )
        
    oa_results = {}
    for key in supp_results[0].keys():
        oa_results[key] = np.mean([item[key] for item in supp_results])
    print(metrics_engine.to_pretty_string(oa_results, f"Overall {model.model_name()}-SupportSet"))

    for key in qry_results[0].keys():
        oa_results[key] = np.mean([item[key] for item in qry_results])
    print(metrics_engine.to_pretty_string(oa_results, f"Overall {model.model_name()}-QuerySet"))
    return supp_results, qry_results


def metatrain(
    train_dataset,
    test_dataset,
    model,
    save_dir,
    metrics_engine,
    device="cuda",
    tasks_num=1,
    inner_loop_num=1,
    support_set_ratio=0.5,
    meta_batch_size=5,
    meta_lr=1e-4,
    inner_lr=5e-4,
    epochs=1000,
    smpl_model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
):
    smpl_model = Smpl(
        model_path=smpl_model_path,
        device=device,
    )

    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    collate_fn = MetaCollate(shuffle=True, support_ratio=support_set_ratio)

    trainloader = DataLoader(train_dataset, batch_size=tasks_num, collate_fn=collate_fn)
    model.train()
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
    scheduler = optim.lr_scheduler.StepLR(meta_opt, step_size=epochs//10, gamma=0.8)

    for epoch in tqdm(range(epochs)):
        for data in trainloader:
            task_num = data[0]["marker_info"].shape[0]

            inner_batch_size = meta_batch_size
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            qry_losses = []
            meta_opt.zero_grad()
            for t in range(task_num):
                supp_set = {key: value[t] for key, value in data[0].items()}
                qry_set = {key: value[t] for key, value in data[1].items()}
                B = supp_set["marker_info"].shape[0]
                L = supp_set["marker_info"].shape[1]
                with torch.backends.cudnn.flags(enabled=False):
                    with higher.innerloop_ctx(
                        model, inner_opt, copy_initial_weights=False, track_higher_grads=True
                    ) as (fnet, diffopt):
                        # Optimize the likelihood of the support set by taking
                        # gradient steps w.r.t. the model's parameters.
                        # This adapts the model's meta-parameters to the task.
                        # higher is able to automatically keep copies of
                        # your network's parameters as they are being updated.
                        for inner_l in range(inner_loop_num):
                            for inner_i in range(B // inner_batch_size):
                                x = (
                                    supp_set["marker_info"][
                                        inner_batch_size
                                        * inner_i : (inner_i + 1)
                                        * inner_batch_size
                                    ]
                                    .contiguous()
                                    .view(inner_batch_size, L, -1)
                                )
                                output = fnet(x)
                                gt = {
                                    key: value[
                                        inner_batch_size
                                        * inner_i : (inner_i + 1)
                                        * inner_batch_size
                                    ]
                                    for key, value in supp_set.items()
                                    if key != "task_name"
                                }
                                spt_loss = loss_fn(output, gt, smpl_model)
                                diffopt.step(spt_loss["total_loss"])

                        # The final set of adapted parameters will induce some
                        # final loss and accuracy on the query dataset.
                        # These will be used to update the model's meta-parameters.
                        total_qry_loss = torch.tensor(0.0, device=device)

                        for inner_i in range(B // inner_batch_size):
                            x = (
                                qry_set["marker_info"][
                                    inner_batch_size
                                    * inner_i : (inner_i + 1)
                                    * inner_batch_size
                                ]
                                .contiguous()
                                .view(inner_batch_size, L, -1)
                            )
                            output = fnet(x)
                            gt = {
                                key: value[
                                    inner_batch_size
                                    * inner_i : (inner_i + 1)
                                    * inner_batch_size
                                ]
                                for key, value in qry_set.items()
                                if key != "task_name"
                            }
                            qry_loss = loss_fn(output, gt, smpl_model)
                            total_qry_loss += qry_loss["total_loss"]
                            qry_losses.append(qry_loss)
                        total_qry_loss.backward()

            if writer is not None:
                mode_prefix = "train"
                for k in qry_losses[0]:
                    prefix = "{}/{}".format(k, mode_prefix)
                    loss = sum([item[k].cpu().item() for item in qry_losses]) / len(
                        qry_losses
                    )
                    writer.add_scalar(prefix, loss, epoch)

                    # Update the model's meta-parameters to optimize the query
                    # losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.

            meta_opt.step()
        scheduler.step()
    torch.save(model.state_dict(), osp.join(save_dir, "model.pth"))


def main(config):
    # 加载设备
    device = config.device if hasattr(config, "device") else "cuda"
    marker_type = config.marker_type if hasattr(config, "marker_type") else "moshpp"
    if marker_type == "moshpp":
        vid = [value for value in moshpp_marker_id.values()]
        config.data_path = osp.join(config.data_path, "moshpp")
        input_dim = 3
    elif marker_type == "rbm":
        vid = [value for value in rigidbody_marker_id.values()]
        config.data_path = osp.join(config.data_path, "rbm")
        input_dim = 6
    else:
        raise ValueError(f"未知的marker_type: {marker_type}")

    n_marker = len(vid)

    # 保存目录
    save_dir = osp.join("./results", f'{datetime.now().strftime("%Y%m%d-%H%M")}-{config.base_model}-{config.train_mode}-{config.marker_type}-{config.epochs}epochs')

    # 根据配置选择模型
    if config.base_model == "sequence":
        model = SequenceModel(
            input_size=input_dim * n_marker,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
            m_bidirectional=True,
            model_type=config.model_type
        ).to(device)
    elif config.base_model == "frame":
        model = FrameModel(
            input_size=input_dim * n_marker,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
            only_pose=True
        ).to(device)
    else:
        raise ValueError(f"未知的base_model: {config.base_model}")

    metrics_engine = MetricsEngine()

    train_fp = osp.join(config.data_path, "meta_train_data_with_normalize_betas_marker.pkl")
    test_fp = osp.join(config.data_path, "meta_val_data_with_normalize_betas_marker.pkl")

    # 根据训练模式选择训练方式
    if config.train_mode == "pretrain":
        # 保存目录
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
            with open(osp.join(save_dir, "config.json"), "w") as f:
                json.dump(config.__dict__, f)
        train_dataset = BabelDataset(train_fp, device=device)
        test_dataset = MetaBabelDataset(test_fp, device=device)
        train(
            train_dataset,
            test_dataset,
            model,
            save_dir,
            metrics_engine,
            batch_size=config.batch_size,
            device=device,
            lr=config.inner_lr,
            epochs=config.epochs,
            smpl_model_path=config.smpl_model_path,
        )
        test_dir = save_dir
    elif config.train_mode == "meta":
        # 假设metatrain函数支持这些参数
        # 保存目录
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
            with open(osp.join(save_dir, "config.json"), "w") as f:
                json.dump(config.__dict__, f)
        train_dataset = MetaBabelDataset(train_fp, device=device)
        test_dataset = MetaBabelDataset(test_fp, device=device)
        metatrain(
            train_dataset,
            test_dataset,
            model,
            save_dir,
            metrics_engine,
            device=device,
            tasks_num=config.tasks_num,
            inner_loop_num=config.inner_loop_num,
            support_set_ratio=config.support_set_ratio,
            meta_batch_size=config.meta_batch_size,
            meta_lr=config.meta_lr,
            inner_lr=config.inner_lr,
            epochs=config.epochs,
            smpl_model_path=config.smpl_model_path
        )
        test_dir = save_dir
    elif config.train_mode == "test":
        # 如果有指定测试目录则用，否则用当前save_dir
        assert config.model_path is not None, "model_path is required for test mode"
        test_dataset = MetaBabelDataset(test_fp, device=device)
        test_dir = config.model_path
    else:
        raise ValueError(f"未知的train_mode: {config.train_mode}")

    # 测试模型
    test(
        test_dataset,
        model,
        metrics_engine,
        test_dir,
        device,
        vis=False,
        smpl_model_path=config.smpl_model_path,
        epochs_ft=config.epochs_ft,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -----------------------
    # General experiment setup
    # -----------------------

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/lanhai/restore/dataset/mocap/mosr/",
        help="Path to dataset.",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
        help="Path to smpl model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )

    # -----------------------
    # Training mode
    # -----------------------
    parser.add_argument(
        "--train_mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "meta", "test"],
        help="Training mode: pretrain (standard) or meta (meta-learning).",
    )

    # -----------------------
    # Model config
    # -----------------------
    parser.add_argument(
        "--base_model",
        type=str,
        default="sequence",
        choices=["frame", "sequence"],
        help="Backbone model type.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru"],
        help="Model type.",
    )

    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size for RNN/MLP layers."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of RNN layers."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability."
    )

    # -----------------------
    # Pretrain config
    # -----------------------
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs for pretraining."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for pretraining."
    )
    parser.add_argument(
        "--epochs_ft", type=int, default=1, help="Number of epochs for finetuning."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "plateau", "none"],
        help="Learning rate scheduler type.",
    )

    # -----------------------
    # Meta-learning config
    # -----------------------
    parser.add_argument(
        "--tasks_num",
        type=int,
        default=1,
        help="Number of tasks per meta-learning episode.",
    )
    parser.add_argument(
        "--inner_loop_num",
        type=int,
        default=1,
        help="Number of inner loop epochs on support set.",
    )
    parser.add_argument(
        "--support_set_ratio",
        type=float,
        default=0.5,
        help="Ratio of support set vs query set per task.",
    )
    parser.add_argument(
        "--meta_batch_size",
        type=int,
        default=5,
        help="Number of episodes per meta-optimization step.",
    )
    parser.add_argument(
        "--meta_lr", type=float, default=1e-4, help="Learning rate for meta-optimizer."
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=5e-4,
        help="Learning rate for inner loop updates.",
    )

    # Input data.
    parser.add_argument(
        "--marker_type",
        type=str,
        default="moshpp",
        choices=["moshpp", "rbm"],
        help="Marker type.",
    )

    parser.add_argument(
        "--use_real_offsets",
        action="store_true",
        help="Sampling is informed by real offset distribution.",
    )
    parser.add_argument(
        "--offset_noise_level",
        type=int,
        default=0,
        help="How much noise to add to real offsets.",
    )

    # Data augmentation.
    parser.add_argument(
        "--noise_num_markers",
        type=int,
        default=1,
        help="How many markers are affected by the noise.",
    )
    parser.add_argument(
        "--spherical_noise_strength",
        type=float,
        default=0.0,
        help="Magnitude of noise in %.",
    )
    parser.add_argument(
        "--spherical_noise_length",
        type=float,
        default=0.0,
        help="Temporal length of noise in %.",
    )
    parser.add_argument(
        "--suppression_noise_length",
        type=float,
        default=0.0,
        help="Marker suppression length.",
    )
    parser.add_argument(
        "--suppression_noise_value",
        type=float,
        default=0.0,
        help="Marker suppression value.",
    )
    
    config = parser.parse_args()
    main(config)
