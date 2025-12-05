# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

from knn_eval_custom import run_knn_eval
from eval_linear_custom import run_linear_eval
import webdataset as wds
import glob



torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help="Total number of training images when using WebDataset (e.g., 414000 for inat_414k)."
    )
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    # ===== kNN eval related =====
    parser.add_argument('--eval_data_path', default=None, type=str,
        help='Root of eval data. Should contain train/ and val/ subfolders.')
    parser.add_argument('--eval_every', default=0, type=int,
        help='Run k-NN eval every N epochs. 0 means disable.')
    parser.add_argument('--eval_knn_k', default=20, type=int,
        help='k for k-NN eval.')

    parser.add_argument('--eval_cub_path', default="/home/ubuntu/data/eval_cub/data", type=str,
        help='Root of CUB eval data.')
    parser.add_argument('--eval_imgnet_path', default="/home/ubuntu/data/eval_imgnet/data", type=str,
        help='Root of ImageNet-mini eval data.')
    parser.add_argument('--eval_sun_path', default="/home/ubuntu/data/eval_sun/data", type=str,
        help='Root of SUN eval data.')
    
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    #   /home/ubuntu/data/train_ccXm_clean
    tar_pattern = os.path.join(args.data_path, "shard-*.tar")
    shard_urls = sorted(glob.glob(tar_pattern))

    if len(shard_urls) == 0:
        print(f"[WARN] No shards found under {tar_pattern}, falling back to ImageFolder.")
        dataset = SafeImageFolder(args.data_path, transform=transform)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Data loaded: there are {len(dataset)} images (ImageFolder).")
    else:
        print(f"Found {len(shard_urls)} tar shards in {args.data_path}, using WebDataset with DDP sharding.")

        if args.num_samples is None or args.num_samples <= 0:
            raise ValueError(
                "Using WebDataset but --num_samples is not set or <= 0. "
                "For ccXm_clean, set e.g. --num_samples 1000000."
            )

        # WebDataset DataPipeline with proper sharding:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shard_urls),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
            wds.decode("pil"),
            wds.to_tuple("jpg"),
            wds.map(lambda sample: (transform(sample[0]), 0)),
        )

        dataset = dataset.with_length(args.num_samples)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        sampler = None
        print(f"Data loaded from WebDataset: pattern = {tar_pattern}, num_samples = {args.num_samples}.")



    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        # 只有在使用 DistributedSampler 时才需要 set_epoch
        if isinstance(getattr(data_loader, "sampler", None),
                      torch.utils.data.distributed.DistributedSampler):
            data_loader.sampler.set_epoch(epoch)


        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
                     
        # if utils.is_main_process():
        #     with (Path(args.output_dir) / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        
        if utils.is_main_process():

            # ====== eval logic: kNN every eval_every, linear every (5 * eval_every) ======
            if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:

                # 当前是第几次 eval（从 1 开始算）
                eval_index = (epoch + 1) // args.eval_every

                eval_datasets = {
                    "cub": args.eval_cub_path,
                    "imgnet": args.eval_imgnet_path,
                    "sun": args.eval_sun_path,
                }

                for name, root in eval_datasets.items():
                    if root is None:
                        continue

                    # -------------------- kNN eval --------------------
                    try:
                        print(f"[kNN-{name}] Running kNN eval at epoch {epoch+1} on {root} ...")
                        knn_top1 = run_knn_eval(
                            ckpt_path=os.path.join(args.output_dir, "checkpoint.pth"),
                            eval_root=root,
                            device="cuda",
                            k=args.eval_knn_k,
                        )
                        print(f"[kNN-{name}] Epoch {epoch+1}: Top1={knn_top1:.2f}%")
                        log_stats[f'knn_top1_{name}_k{args.eval_knn_k}'] = float(knn_top1)
                    except Exception as e:
                        print(f"[kNN-{name}] Eval failed at epoch {epoch+1}: {e}")

                    # -------------------- linear probe eval --------------------
                    # 当 eval_index 是 5 的倍数 → 跑 linear probe（也即 epoch 多了 5 * eval_every）
                    if eval_index % 5 == 0:
                        try:
                            print(f"[Linear-{name}] Running linear probe eval at epoch {epoch+1} on {root} ...")
                            linear_res = run_linear_eval(
                                ckpt_path=os.path.join(args.output_dir, "checkpoint.pth"),
                                eval_root=root,
                                device="cuda",
                                n_last_blocks=4,
                                avgpool_patchtokens=False,
                                epochs=100,
                                batch_size=1024,
                                num_workers=args.num_workers,
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=0.0,
                                use_precomputed_features=True,
                            )
                            log_stats[f'linear_best_val_{name}'] = float(linear_res["best_val_acc"])
                            log_stats[f'linear_test_{name}'] = float(linear_res["test_acc"])
                        except Exception as e:
                            print(f"[Linear-{name}] Linear eval failed at epoch {epoch+1}: {e}")

            # write log
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


                
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args):
    """
    关键改动：
    - 不再 `for (images, _) in data_loader`，改成手动控制迭代次数 = len(data_loader)
    - 如果某个 rank 的 DataLoader 提前 StopIteration，就重新创建 iter 继续取数据
    - 这样每个 rank 在每个 epoch 都执行完全相同数量的 step / all_reduce 调用
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # 每个 epoch 固定的 iteration 数（所有 rank 一样）
    iters_per_epoch = len(data_loader)

    # 为当前 epoch 创建一个迭代器
    data_iter = iter(data_loader)

    # 用 range(iters_per_epoch) 包一层，方便继续用 MetricLogger.log_every
    for it in metric_logger.log_every(range(iters_per_epoch), 10, header):
        # global training iteration index，和 lr_schedule / wd_schedule 对齐
        global_it = epoch * iters_per_epoch + it

        # 这里手动从 data_iter 取 batch，如耗尽则重新开始
        try:
            images, _ = next(data_iter)
        except StopIteration:
            # 当前 rank 的数据耗尽，重新开始一个新的 epoch 流
            data_iter = iter(data_loader)
            images, _ = next(data_iter)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[global_it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                # unscale the gradients of optimizer's assigned params in-place
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[global_it]  # momentum parameter for this step
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class SafeImageFolder(torch.utils.data.Dataset):
    """
    ImageFolder that silently skips empty class folders.
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, extensions=IMG_EXTENSIONS):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions

        # å…ˆæ‰¾å‡ºæ‰€æœ‰ class å­ç›®å½•ï¼ˆåå­—å’Œ ImageFolder ä¸€æ ·ï¼‰
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()

        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        skipped = []

        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            cls_samples = []
            for r, _, fnames in os.walk(cls_dir):
                for fname in sorted(fnames):
                    if fname.lower().endswith(tuple(self.extensions)):
                        path = os.path.join(r, fname)
                        cls_samples.append(path)

            if len(cls_samples) == 0:
                skipped.append(cls_name)
                continue

            cls_idx = len(self.classes)
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = cls_idx
            for path in cls_samples:
                self.samples.append((path, cls_idx))

        self.targets = [s[1] for s in self.samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {root} (all classes empty?).")

        if len(skipped) > 0:
            print(f"[SafeImageFolder] Skipped {len(skipped)} empty classes, e.g.: {skipped[:10]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

        # ===== 两个 global view：对应原版的 224×224，这里变成 96×96 =====
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                96,                          # 输出分辨率：96×96（你的最大分辨率）
                scale=global_crops_scale,    # 默认 (0.4, 1.0)，和原版一致
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                96,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        # ===== local views：对应原版的 96×96，这里缩成 64×64 =====
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                64,                          # 小图：64×64
                scale=local_crops_scale,     # 默认 (0.05, 0.4)，和原版一致
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # 两个 global crop
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        # 多个 local crop
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
