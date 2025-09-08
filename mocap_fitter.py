import os
import pickle
import torch
import numpy as np
import argparse
from glob import glob
import os.path as osp
from data import BabelDataset, MetaBabelDataset, virtual_marker, MetaCollate
from torch.utils.data import DataLoader, random_split
from models import Moshpp, SimpleRNN, ResNet
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from utils import visualize, vis_diff
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
import higher

mosr_marker_id = {
    "head": 335,
    "chest": 3073,
    "left_arm": 2821,
    "left_forearm": 1591,
    "left_hand": 2000,
    "left_leg": 981,
    "left_shin": 1115,
    "left_foot": 3341,
    "right_arm": 4794,
    "right_forearm": 5059,
    "right_hand": 5459,
    "right_leg": 4465,
    "right_shin": 4599,
    "right_foot": 6742,
}
moshpp_marker_id = {
    "ARIEL": 411,
    "C7": 3470,
    "CLAV": 3171,
    "LANK": 3327,
    "LBHD": 182,
    "LBSH": 2940,
    "LBWT": 3122,
    "LELB": 1666,
    "LELBIN": 1725,
    "LFHD": 0,
    "LFIN": 2174,
    "LFRM": 1568,
    "LFSH": 1317,
    "LFWT": 857,
    "LHEE": 3387,
    "LIWR": 2112,
    "LKNE": 1053,
    "LKNI": 1058,
    "LMT1": 3336,
    "LMT5": 3346,
    "LOWR": 2108,
    "LSHN": 1082,
    "LTHI": 1454,
    "LTHMB": 2224,
    "LTOE": 3233,
    "LUPA": 1443,
    "MBWT": 3022,
    "MFWT": 3503,
    "RANK": 6728,
    "RBHD": 3694,
    "RBSH": 6399,
    "RBWT": 6544,
    "RELB": 5135,
    "RELBIN": 5194,
    "RFHD": 3512,
    "RFIN": 5635,
    "RFRM": 5037,
    "RFSH": 4798,
    "RFWT": 4343,
    "RHEE": 6786,
    "RIWR": 5573,
    "RKNE": 4538,
    "RKNI": 4544,
    "RMT1": 6736,
    "RMT5": 6747,
    "ROWR": 5568,
    "RSHN": 4568,
    "RTHI": 4927,
    "RTHMB": 5686,
    "RTOE": 6633,
    "RUPA": 4918,
    "STRN": 3506,
    "T10": 3016,
}


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


def train(model, save_dir, metrics_engine, batch_size=5, device="cuda", epochs=400):
    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    best_mpjpe = torch.inf
    # 普通训练
    smpl_model = Smpl(
        model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
        device=device,
    )
    model.train()

    train_dataset = BabelDataset(
        "/home/lanhai/restore/dataset/mocap/mosr/metatrain.pkl", device=device
    )
    test_dataset = MetaBabelDataset(
        "/home/lanhai/restore/dataset/mocap/mosr/metatest.pkl", device=device
    )
    collate_fn = MetaCollate()

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    global_step = 0
    print("Begin training.")
    for epoch in tqdm(range(epochs)):

        for data in trainloader:
            B = data["marker_pos"].shape[0]
            L = data["marker_pos"].shape[1]
            x = data["marker_pos"].contiguous().view(B, L, -1)

            optimizer.zero_grad()
            global_step += 1

            output = model(x)

            losses = loss_fn(output, data, smpl_model)
            if writer is not None:
                mode_prefix = "train"
                for k in losses:
                    prefix = "{}/{}".format(k, mode_prefix)
                    writer.add_scalar(prefix, losses[k].cpu().item(), global_step)

            losses["total_loss"].backward()
            optimizer.step()

        # evaluate the query set (support set)
        finetune = True
        mpjpe = torch.tensor(0.0).to(device)
        metrics_tasks = {}
        for data in testloader:
            num_tasks = data[0]["marker_pos"].shape[0]
            for t in range(num_tasks):
                supp_set = {key: value[t] for key, value in data[0].items()}
                qry_set = {key: value[t] for key, value in data[1].items()}
                if finetune:
                    # print('Begin finetuning')
                    B = supp_set["marker_pos"].shape[0]
                    L = supp_set["marker_pos"].shape[1]

                    ft_model = model.clone()

                    ft_model.train()
                    for ft_i in range(B // batch_size):
                        optimizer.zero_grad()
                        global_step += 1
                        x = (
                            supp_set["marker_pos"][
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
                        optimizer.step()
                else:
                    ft_model = model

                ft_model.eval()
                B = qry_set["marker_pos"].shape[0]
                L = qry_set["marker_pos"].shape[1]

                output_list = []
                for val_i in range(B):
                    x = qry_set["marker_pos"][val_i].contiguous().view(1, L, -1)
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
                    writer.add_scalar(prefix, metrics[k].cpu().item(), global_step)
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


def test(model, metrics_engine, model_path=None, device="cuda", vis=False):
    finetune = True
    test_dataset = MetaBabelDataset(
        "/home/lanhai/restore/dataset/mocap/mosr/metatest.pkl", device=device
    )
    collate_fn = MetaCollate()
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    smpl_model = Smpl(
        model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
        device=device,
    )
    model.eval()

    for data in testloader:
        num_tasks = data[0]["marker_pos"].shape[0]
        for t in range(num_tasks):
            supp_set = {key: value[t] for key, value in data[0].items()}
            qry_set = {key: value[t] for key, value in data[1].items()}
            if finetune:
                # print('Begin finetuning')
                B = supp_set["marker_pos"].shape[0]
                L = supp_set["marker_pos"].shape[1]
                ft_model = model.clone()

                ft_model.train()
                for e in range(5):
                    for ft_i in range(B // 5):
                        optimizer.zero_grad()
                        x = (
                            supp_set["marker_pos"][5 * ft_i : (ft_i + 1) * 5]
                            .contiguous()
                            .view(5, L, -1)
                        )
                        output = ft_model(x)
                        gt = {
                            key: value[5 * ft_i : (ft_i + 1) * 5]
                            for key, value in supp_set.items()
                            if key != "task_name"
                        }
                        losses = loss_fn(output, gt, smpl_model)
                        losses["total_loss"].backward()
                        optimizer.step()
            else:
                ft_model = model

            ft_model.eval()
            B = qry_set["marker_pos"].shape[0]
            L = qry_set["marker_pos"].shape[1]

            output_list = []
            for val_i in range(B):
                x = qry_set["marker_pos"][val_i].contiguous().view(1, L, -1)
                output = ft_model(x)
                output["joints"] = smpl_model(
                    betas=output["betas"].reshape(-1),
                    body_pose=output["poses"].reshape(L, -1)[:, 3:],
                    global_orient=output["poses"].reshape(L, -1)[:, 0:3],
                    transl=output["trans"].reshape(L, -1),
                )["joints"].reshape(1, L, -1, 3)
                output_list.append(output)
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
            for key in output_list[0].keys():
                output[key] = torch.concatenate([item[key] for item in output_list])

            metrics = metrics_engine.compute(output, qry_set)
            print(
                metrics_engine.to_pretty_string(
                    metrics,
                    f"Task {supp_set['task_name']}-{model.model_name()}-SupportSet",
                )
            )


def metatrain(
    model,
    save_dir,
    metrics_engine,
    device="cuda",
    tasks_num=3,
    inner_loop_num=1,
    support_set_ratio=0.5,
    meta_batch_size=5,
    meta_lr=1e-3,
    inner_lr=1e-2,
    epochs=1000
):
    smpl_model = Smpl(
        model_path="/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz",
        device=device,
    )

    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    collate_fn = MetaCollate(shuffle=True, support_ratio=support_set_ratio)
    train_dataset = MetaBabelDataset(
        "/home/lanhai/restore/dataset/mocap/mosr/metatrain.pkl", device=device
    )
    trainloader = DataLoader(train_dataset, batch_size=tasks_num, collate_fn=collate_fn)
    model.train()
    meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
    for epoch in tqdm(range(400)):
        for data in trainloader:
            task_num = data[0]["marker_pos"].shape[0]

            inner_batch_size = meta_batch_size
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)

            qry_losses = []
            meta_opt.zero_grad()
            for t in range(task_num):
                supp_set = {key: value[t] for key, value in data[0].items()}
                qry_set = {key: value[t] for key, value in data[1].items()}
                B = supp_set["marker_pos"].shape[0]
                L = supp_set["marker_pos"].shape[1]

                with higher.innerloop_ctx(
                    model, inner_opt, copy_initial_weights=True, track_higher_grads=True
                ) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for inner_l in range(inner_loop_num):
                        for inner_i in range(B // inner_batch_size):
                            x = (
                                supp_set["marker_pos"][
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
                            qry_set["marker_pos"][
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

    torch.save(model.state_dict(), osp.join(save_dir, "model.pth"))


def main(config):
    # 加载设备
    device = config.device if hasattr(config, "device") else "cuda"
    marker_type = config.marker_type if hasattr(config, "marker_type") else "moshpp"
    if marker_type == "moshpp":
        vid = [value for value in moshpp_marker_id.values()]
    elif marker_type == "mosr":
        vid = [value for value in mosr_marker_id.values()]
    else:
        raise ValueError(f"未知的marker_type: {marker_type}")

    n_marker = len(vid)

    # 保存目录
    save_dir = osp.join("./results", datetime.now().strftime("%Y%m%d-%H%M"))
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    # 根据配置选择模型
    if config.base_model == "rnn":
        model = SimpleRNN(
            input_size=3 * n_marker,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
            m_bidirectional=True
        ).to(device)
    elif config.base_model == "resnet":
        model = ResNet(
            input_size=3 * n_marker,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
        ).to(device)
    else:
        raise ValueError(f"未知的base_model: {config.base_model}")

    metrics_engine = MetricsEngine()

    # 根据训练模式选择训练方式
    if config.train_mode == "pretrain":
        train(
            model, save_dir, metrics_engine, batch_size=config.batch_size, device=device, epochs=config.epochs
        )
    elif config.train_mode == "meta":
        # 假设metatrain函数支持这些参数
        metatrain(
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
            epochs=config.epochs
        )
    else:
        raise ValueError(f"未知的train_mode: {config.train_mode}")

    # 测试模型
    # 如果有指定测试目录则用，否则用当前save_dir
    # test_dir = "/home/lanhai/PycharmProjects/mosr/results/20250907-1049"
    test_dir = None
    if test_dir is not None:
        test(model, metrics_engine, test_dir, device, vis=False)
    else:
        test(model, metrics_engine, save_dir, device, vis=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -----------------------
    # General experiment setup
    # -----------------------
    # parser.add_argument('--data_path', type=str, default=None,
    #                     help='Path to dataset.')
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
        choices=["pretrain", "meta"],
        help="Training mode: pretrain (standard) or meta (meta-learning).",
    )

    # -----------------------
    # Model config
    # -----------------------
    parser.add_argument(
        "--base_model",
        type=str,
        default="rnn",
        choices=["rnn", "resnet"],
        help="Backbone model type.",
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
        default=4,
        help="Number of episodes per meta-optimization step.",
    )
    parser.add_argument(
        "--meta_lr", type=float, default=1e-3, help="Learning rate for meta-optimizer."
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=1e-2,
        help="Learning rate for inner loop updates.",
    )

    # Input data.
    parser.add_argument(
        "--marker_type",
        type=str,
        default="moshpp",
        choices=["moshpp", "mosr"],
        help="Marker type.",
    )
    parser.add_argument(
        "--use_marker_pos", action="store_true", help="Feed marker positions."
    )
    parser.add_argument(
        "--use_marker_ori", action="store_true", help="Feed marker orientations."
    )
    parser.add_argument(
        "--use_marker_nor",
        action="store_true",
        help="Feed marker normal instead of orientation.",
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
