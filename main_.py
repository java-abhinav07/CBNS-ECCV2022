import logging
import os
import sys
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models.pointnet_utils import visualize_data
from utils import options, save_checkpoint, get_datasets
import torch.nn.functional as F
import torch.distributions as tod

from models import pointnet_cls, dgcnn_cls, line_cloud
from src import FPSSampler, RandomSampler, SampleNet
from src import sputils
from models.entropy import EntropyLoss

torch.manual_seed(0)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.NullHandler())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# dump to GLOBALS dictionary
GLOBALS = None

torch.autograd.set_detect_anomaly(True)


def append_to_GLOBALS(key, value):
    try:
        GLOBALS[key].append(value)
    except KeyError:
        GLOBALS[key] = []
        GLOBALS[key].append(value)


def main(args, dbg=False):
    global GLOBALS
    if dbg:
        GLOBALS = {}

    LOGGER.info(os.environ["CUDA_VISIBLE_DEVICES"])

    trainset, testset = get_datasets(args)
    action = Action(args)

    if args.train_dgcnn or args.train_pointnet or args.plot_saliency:
        train(args, trainset, testset, action)

    elif args.test or args.test_dgcnn:
        test(args, testset, action)

    return GLOBALS


def test(args, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    model, attacker, sampler = action.create_model()

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # load attacker
    if args.attacker != "":
        assert os.path.isfile(args.attacker)
        LOGGER.info("Loading attacker model from " + str(args.attacker))
        attacker.load_state_dict(torch.load(args.attacker))
        attacker.to(args.device)
        attacker.eval()
    else:
        LOGGER.info("Pretrained attacker model not found")
        attacker = None

    if args.sampler_model != "":
        assert os.path.isfile(args.sampler_model)
        LOGGER.info("Loading sampler model from" + str(args.sampler_model))
        sampler_dict = sampler.state_dict()
        pretrained_dict = torch.load(args.sampler_model)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in sampler_dict
        }
        sampler_dict.update(pretrained_dict)
        sampler.load_state_dict(pretrained_dict, strict=False)
        sampler.to(args.device)
        sampler.eval()

    elif args.sampler in ["fps", "random"]:
        LOGGER.info("Using FPS/Random Sampler (w/o noise)")
        sampler = sampler

    else:
        LOGGER.info("Sampler not found")
        if args.attacker != "":
            raise ()
        else:
            LOGGER.info("Testing without sampler")
            sampler = None

    # Batch norms etc. configured for testing mode.
    model.to(args.device)
    model.eval()

    # Dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    action.test_1(model, attacker, sampler, testloader)


def train(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = torch.device(args.device)

    model, attacker, sampler = action.create_model()

    if args.pretrained:
        LOGGER.info("Loading pretrained pointnet (task)...")
        assert os.path.isfile(args.pretrained)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(args.device)
    model.requires_grad_(True)
    model.train()

    if args.attacker != "":
        assert os.path.isfile(args.attacker)
        # do partial load if attacker.sampler was also stored
        attacker_dict = attacker.state_dict()
        pretrained_dict = torch.load(args.attacker)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in attacker_dict
        }
        attacker_dict.update(pretrained_dict)
        attacker.load_state_dict(pretrained_dict)
        attacker.to(args.device)
        attacker.requires_grad_(True)
        if args.attacker_discrim:
            attacker.train()
        else:
            attacker.eval()

        LOGGER.info("Loaded attacker")
    else:
        LOGGER.info("Training without attacker")

    if args.sampler_model != "":
        assert os.path.isfile(args.sampler_model)
        sampler_dict = sampler.state_dict()
        pretrained_dict = torch.load(args.sampler_model)

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in sampler_dict
        }
        sampler_dict.update(pretrained_dict)
        sampler.load_state_dict(pretrained_dict, strict=False)
        sampler.to(args.device)
    else:
        LOGGER.info("No pretrained sampler")
        if args.sampler != "none":
            LOGGER.info("Training a sampler from scratch")
            sampler.to(args.device)
            sampler.train()

    checkpoint = None
    if args.resume:
        sampler_ = (
            "/".join(args.resume.split("/")[:-1]) + "/train_sampler_snap_last.pth"
        )
        assert os.path.isfile(args.resume)
        assert os.path.isfile(sampler_)

        checkpoint = torch.load(args.resume)
        sampler_checkpoint = torch.load(sampler_)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        LOGGER.info("Loaded task and sampler model")
        LOGGER.info("Task: " + args.resume)

        if args.sampler_model == "" or args.train_samplenet:
            sampler.load_state_dict(sampler_checkpoint["sampler"])
            LOGGER.info("Sampler: " + sampler_)

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # ignore optimizer etc and plot using model gradients
    if args.plot_saliency:
        action.saliency_1(model, attacker, sampler, testloader)
        return

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    # Task and Sampler Optimizer
    if args.sampler == "samplenet" and args.train_samplenet and (not args.finetune):
        sampler.requires_grad_(True)
        sampler.train()

        if args.task == "priv":
            # Finetune sampler (privacy aware) and tune utility model
            params = list(model.parameters()) + list(sampler.parameters())
        else:
            # Finetune vanilla sampler
            params = list(sampler.parameters())

        learnable_params = filter(lambda p: p.requires_grad, params)

    elif args.sampler == "fps" and args.learn_noise and (not args.finetune):
        sampler.requires_grad_(True)
        sampler.train()
        if args.task == "priv":
            params = list(model.parameters()) + list(sampler.parameters())
        else:
            params = list(sampler.parameters())

        learnable_params = filter(lambda p: p.requires_grad, params,)

    else:
        learnable_params = filter(lambda p: p.requires_grad, list(model.parameters()))

    if args.attacker_discrim:
        attacker.requires_grad_(True)
        attacker.train()
        learnable_paramsD = filter(
            lambda p: p.requires_grad, list(attacker.parameters())
        )
        optimizerD = torch.optim.Adam(learnable_paramsD, lr=1e-3)

    else:
        optimizerD = None

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=1e-3)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=1e-3)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=1e-3, momentum=0.9)

    if checkpoint is not None:
        max_acc = checkpoint["max_acc"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    max_acc = -float("inf")

    # training
    LOGGER.debug("train, begin")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = action.train_1(
            model, attacker, sampler, trainloader, optimizer, optimizerD=optimizerD,
        )
        eval_ = True
        # reduce testing frequency with contrastive training for optimizing train time
        if args.contrastive_feat:
            eval_ = True if epoch % 2 == 0 else False

        if eval_:
            eval_out = action.eval_1(model, attacker, sampler, testloader)

            # parse eval dict
            (
                val_loss,
                val_loss_task,
                val_loss_attack,
                val_acc,
                val_acc_attacker,
                reg,
                cont,
            ) = (
                eval_out["loss"],
                eval_out["task"],
                eval_out["attack"],
                eval_out["acc"],
                eval_out["acc_attack"],
                eval_out["reg"],
                eval_out["cont"],
            )

            LOGGER.info(
                "epoch, %04d, train_loss=%f, val_loss=%f, val_loss_task=%f, val_loss_attack=%f, val_acc=%f, val_acc_attack=%f, reg=%f, cont=%f",
                epoch + 1,
                train_loss,
                val_loss,
                val_loss_task,
                val_loss_attack,  # discriminator loss if args.attacker_discrim
                val_acc,
                val_acc_attacker,
                reg,
                cont,
            )

            # save task model and sampler at that state
            if args.task == "priv" and val_acc is not None:
                assert val_acc_attacker > 0.0
                tradeoff = val_acc / (val_acc_attacker + 1e-8)
                is_best = tradeoff >= max_acc
                max_acc = max(tradeoff, max_acc)

            elif args.task == "vanilla":
                is_best = val_acc >= max_acc
                max_acc = max(val_acc, max_acc)

            model_snap = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "max_acc": max_acc,
                "optimizer": optimizer.state_dict(),
            }
            if args.sampler != "none":
                sampler_snap = {
                    "epoch": epoch + 1,
                    "sampler": sampler.state_dict(),
                    "max_acc": max_acc,
                    "optimizer": optimizer.state_dict(),
                }

            # TASK + Best Sampler
            if is_best:
                save_checkpoint(model_snap, args.outfile, "snap_best")
                save_checkpoint(model.state_dict(), args.outfile, "model_best")

                if args.sampler != "none":
                    save_checkpoint(sampler_snap, args.outfile, "sampler_snap_best")
                    save_checkpoint(sampler.state_dict(), args.outfile, "sampler_best")

            else:
                save_checkpoint(model_snap, args.outfile, "snap_last")
                save_checkpoint(model.state_dict(), args.outfile, "model_last")

                if args.sampler != "none":
                    save_checkpoint(sampler_snap, args.outfile, "sampler_snap_last")
                    save_checkpoint(sampler.state_dict(), args.outfile, "sampler_last")

    LOGGER.debug("train, end")


class ContrastiveLossFeat(nn.Module):
    def __init__(self, args):
        super(ContrastiveLossFeat, self).__init__()
        self.margin = 1.0
        self.args = args
        self.encoder = None

    def extract_features(self, x):
        if self.args.train_pointnet:
            x = x.permute(0, 2, 1)
            return self.encoder(x)[0]  # bs, NumInPoints
        else:
            return self.encoder.get_feat(x)

    def forward(self, anchor, positive, negative):
        # get features
        feat_anchor = self.extract_features(anchor)
        feat_positive = self.extract_features(positive)
        feat_negative = self.extract_features(negative)

        # pairwise distance
        euc_pos = torch.pow(
            torch.clamp(
                self.margin - F.pairwise_distance(feat_anchor, feat_positive), min=0.0
            ),
            2,
        )
        euc_neg = torch.pow(F.pairwise_distance(feat_anchor, feat_negative), 2)

        # increase distance between anchor and positive (same class: maximize)
        # decrease distance between anchor and negative (different class: minimize)
        contrastive_loss = torch.mean(F.relu(euc_neg) + euc_pos)

        return contrastive_loss * self.args.cont_scale


class Action:
    def __init__(self, args):
        self.logfile = args.outfile + ".log"
        self.vis_file = os.path.join(
            "/".join(args.outfile.split("/")[:-2]), "visualize"
        )
        self.experiment_name = args.pretrained

        self.transfer_from_pointnet = args.transfer_from_pointnet
        self.DATASET = args.dataset

        self.p0_zero_mean = True
        self.batch_size = args.batch_size

        # k map
        if self.DATASET == "facescape":
            self.task_dict = {
                "cls_exp": 20,
                "cls_gender": 2,
                "cls_identity": 847,
                "cls_age": 69,
            }

            self.target_map = {
                "cls_exp": -1,
                "cls_gender": -2,
                "cls_identity": 0,
                "cls_age": -3,
            }
        elif self.DATASET == "modelnet":
            self.task_dict = {
                "cls_living": 2,
                "cls_all": 4,
            }

            # hard coding to avoid bugs
            args.num_in_points = 2048

        # SampleNet:
        self.ALPHA = args.alpha  # Sampling loss
        self.LMBDA = args.lmbda  # Projection loss
        self.GAMMA = args.gamma  # Inside sampling loss - linear.
        # Inside sampling loss - point cloud size factor.
        self.DELTA = args.delta
        self.NUM_IN_POINTS = args.num_in_points
        self.NUM_OUT_POINTS = args.num_out_points
        self.BOTTLNECK_SIZE = args.bottleneck_size
        self.GROUP_SIZE = args.projection_group_size

        self.SKIP_PROJECTION = args.skip_projection
        self.SAMPLER = args.sampler

        self.TRAIN_SAMPLENET = args.train_samplenet
        self.TRAIN_POINTNET = args.train_pointnet
        self.TRAIN_DGCNN = args.train_dgcnn
        self.TASK = args.task
        self.adv_weight = float(args.adv_weight / 100)
        self.scale = args.scale
        self.USE_STN = args.use_STN
        self.ATTACKER_TASK = args.attacker_task
        self.BASE_TASK = args.base_task
        self.use_enc_stn = args.use_enc_stn
        self.EPOCHS = args.epochs
        self.DEVICE = args.device
        self.CONTRASTIVE_FEAT = args.contrastive_feat
        self.ADV_CONTRASTIVE = args.adv_contrastive
        self.VISUALIZE = args.visualize
        self.SIGMA = args.std_noise
        self.NO_ADV = args.no_adv
        self.LEARN_NOISE = args.learn_noise
        self.RESAMPLE = args.resample
        self.ATTACKER_DISCRIM = args.attacker_discrim
        self.FINETUNE = args.finetune
        self.GAUSSIAN = args.gaussian
        self.REG_WEIGHT = args.reg_weight
        self.TEST_DGCNN = args.test_dgcnn
        self.PLOT_SALIENCY = args.plot_saliency
        self.TEST = args.test
        self.LINE_CLOUD = args.line_cloud
        self.MAX_ENTROPY = args.max_entropy

        if self.MAX_ENTROPY:
            self.max_entropy_loss = EntropyLoss()

        if args.contrastive_feat:
            feat_extractor = self.load_feat_extractor(args.cont_feat_extractor)
            self.CONT_FEAT_EXTRACTOR = feat_extractor

        if self.VISUALIZE > 0:
            if not os.path.isdir(self.vis_file):
                os.mkdir(self.vis_file)
            if not os.path.isdir(os.path.join(self.vis_file, "sampled")):
                os.mkdir(os.path.join(self.vis_file, "sampled"))
            if not os.path.isdir(os.path.join(self.vis_file, "sampled")):
                os.mkdir(os.path.join(self.vis_file, "sampled"))
            if not os.path.isdir(os.path.join(self.vis_file, "original")):
                os.mkdir(os.path.join(self.vis_file, "original"))
            if not os.path.isdir(os.path.join(self.vis_file, "combined")):
                os.mkdir(os.path.join(self.vis_file, "combined"))

            LOGGER.info("Visualization directories created")

        if self.CONTRASTIVE_FEAT:
            self.closs = ContrastiveLossFeat(args=ARGS).to(self.DEVICE)

        self.POINTWISE_DIST = args.pointwise_dist

    def load_feat_extractor(self, path):
        k = self.task_dict[self.ATTACKER_TASK]
        if self.TRAIN_POINTNET:
            if isinstance(k, int):
                pointnet_model = pointnet_cls.get_model(
                    k=k,
                    use_enc_stn=self.use_enc_stn,
                    normal_channel=False,
                    num_in_points=self.NUM_IN_POINTS,
                ).to(self.DEVICE)
            else:
                pointnet_model = pointnet_ae.PointNetAutoEncoder(
                    use_feat_stn=self.use_enc_stn, num_points=self.NUM_IN_POINTS
                ).to(self.DEVICE)

        elif self.TRAIN_DGCNN or self.TEST_DGCNN:
            pointnet_model = dgcnn_cls.DGCNN(
                k=k, output_channels=self.NUM_OUT_POINTS
            ).to(self.DEVICE)

        self.try_transfer(pointnet_model, path)

        return pointnet_model

    def create_model(self) -> Tuple:
        """Create Task network and load pretrained feature weights if requested"""

        k = self.task_dict[self.BASE_TASK]

        if self.TRAIN_DGCNN or self.TEST_DGCNN:
            LOGGER.info("Loading DGCNN...")
            pointnet_model = dgcnn_cls.DGCNN(k=k, output_channels=self.NUM_OUT_POINTS)

        else:
            if isinstance(k, int):
                pointnet_model = pointnet_cls.get_model(
                    k=k,
                    use_enc_stn=self.use_enc_stn,
                    normal_channel=False,
                    num_in_points=self.NUM_IN_POINTS,
                )
            else:
                raise ()

        # Load pointnet_model baseline weights
        self.try_transfer(pointnet_model, self.transfer_from_pointnet)

        if self.TRAIN_POINTNET:
            pointnet_model.requires_grad_(True)
            pointnet_model.train()
        else:
            pointnet_model.requires_grad_(False)
            pointnet_model.eval()

        if self.TASK == "priv":
            k = self.task_dict[self.ATTACKER_TASK]
            if not (self.TRAIN_DGCNN or self.TEST_DGCNN):
                if isinstance(k, int):
                    attack_model = pointnet_cls.get_model(
                        k=k,
                        use_enc_stn=self.use_enc_stn,
                        normal_channel=False,
                        num_in_points=self.NUM_IN_POINTS,
                    )

                else:
                    raise ()
            else:
                attack_model = dgcnn_cls.DGCNN(k=k, output_channels=self.NUM_OUT_POINTS)

            # requires to propogate gradients but remains frozen
            attack_model.requires_grad_(True)
            if self.ATTACKER_DISCRIM:
                attack_model.train()
            else:
                attack_model.eval()

            if self.ATTACKER_TASK not in list(self.task_dict.keys()):
                attack_model = None
        else:
            attack_model = None

        # Create sampling network
        if self.SAMPLER == "samplenet":
            sampler = SampleNet(
                num_out_points=self.NUM_OUT_POINTS,
                bottleneck_size=self.BOTTLNECK_SIZE,
                group_size=self.GROUP_SIZE,
                initial_temperature=1.0,
                input_shape="bnc",
                output_shape="bnc",
                skip_projection=self.SKIP_PROJECTION,
                learn_noise=self.LEARN_NOISE,
                pointwise_dist=self.POINTWISE_DIST,
            )

            if self.TRAIN_SAMPLENET:
                sampler.requires_grad_(True)
                sampler.train()
            else:
                sampler.requires_grad_(False)
                sampler.eval()

        elif self.SAMPLER == "fps":
            sampler = FPSSampler(
                self.NUM_OUT_POINTS,
                permute=True,
                input_shape="bnc",
                output_shape="bnc",
                learn_noise=self.LEARN_NOISE,
                pointwise_dist=self.POINTWISE_DIST,
            )

            if self.LEARN_NOISE or self.POINTWISE_DIST:
                if not self.FINETUNE:
                    sampler.requires_grad_(True)
                    sampler.train()
                else:
                    sampler.requires_grad_(False)
                    sampler.eval()
            else:
                sampler.requires_grad_(False)

        elif self.SAMPLER == "random":
            sampler = RandomSampler(
                self.NUM_OUT_POINTS, input_shape="bnc", output_shape="bnc"
            )
        else:
            sampler = None

        if self.CONTRASTIVE_FEAT:
            self.closs.encoder = self.CONT_FEAT_EXTRACTOR
            # keep the feature extractor fixed
            if not self.ATTACKER_DISCRIM:
                self.closs.encoder = self.closs.encoder.requires_grad_(False)
            # use the updated attacker as feature extractor
            else:
                if self.TRAIN_POINTNET:
                    self.closs.encoder = attack_model.feat
                else:  # DGCNN
                    self.closs.encoder = attack_model
                self.closs.encoder = self.closs.encoder.requires_grad_(True)

        return pointnet_model, attack_model, sampler

    @staticmethod
    def try_transfer(model, path):
        if path is not None:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(path, map_location="cpu")
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)

            model.load_state_dict(model_dict)
            LOGGER.info(f"Model loaded from {path}")

    @staticmethod
    def detach_data(data):
        data_new = []
        for dta in data:
            if not isinstance(dta, list):
                data_new.append(dta.detach())
            else:
                c_dta_new = []
                for c_dta in dta:
                    c_dta_new.append(c_dta.detach())

                data_new.append(c_dta_new)

        return data_new

    def train_1(
        self, model, attacker, sampler, trainloader, optimizer, optimizerD=None,
    ):
        total_loss = 0.0
        count = 0
        reg_loss_pointwise = 0.0

        for i, data in enumerate(tqdm(trainloader)):
            # condition only on task
            if sampler is not None and sampler.name == "samplenet":
                (
                    sampler_loss,
                    sampled_data,
                    sampler_loss_info,
                    sampler_reg,
                    contrastive_loss,
                ) = self.compute_samplenet_loss(sampler, data)

                if self.GAUSSIAN or self.LEARN_NOISE:
                    if self.POINTWISE_DIST or self.RESAMPLE:
                        sampled_data, reg_loss_pointwise = self.add_noise(
                            sampled_data, sampler
                        )
                    else:
                        sampled_data = self.add_noise(sampled_data, sampler)

            elif sampler is not None and sampler.name == "fps":
                sampled_data = self.non_learned_sampling(sampler, data)
                sampler_loss = torch.tensor(0, dtype=torch.float32)
                sampler_reg = torch.tensor(0, dtype=torch.float32)
                contrastive_loss = torch.tensor(0, dtype=torch.float32)

                if self.TASK == "priv" or self.FINETUNE:
                    if self.POINTWISE_DIST or self.RESAMPLE:
                        sampled_data, reg_loss_pointwise = self.add_noise(
                            sampled_data, sampler
                        )
                    else:
                        sampled_data = self.add_noise(sampled_data, sampler)
            else:
                sampled_data = data
                sampler_loss = torch.tensor(0, dtype=torch.float32)
                sampler_reg, contrastive_loss = 0.0, 0.0

            if self.LINE_CLOUD:
                sampled_data = list(sampled_data)
                sampled_data[0] = line_cloud.get_line_cloud(sampled_data[0])

            if self.TASK == "priv":
                pointnet_loss, _, attacker_loss, _, _ = self.compute_combined_loss(
                    model, attacker, data, sampled_data
                )

            else:
                if self.BASE_TASK.startswith("cls"):
                    pointnet_loss, correct = self.compute_pointnet_loss(
                        model, sampled_data
                    )

            # SampleNet loss is already factorized by ALPHA and LMBDA hyper parameters.
            loss = (
                pointnet_loss
                + (1000 * sampler_reg)
                + sampler_loss
                + (self.LMBDA * contrastive_loss)
                + (self.REG_WEIGHT * reg_loss_pointwise)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.ATTACKER_DISCRIM:
                # need to calculate loss again or specify retain_graph=True
                sampled_data = self.detach_data(sampled_data)
                attacker.train()
                lossD, attacker_acc = self.compute_pointnet_loss(
                    attacker, sampled_data, att=True
                )

                # apply scaling
                lossD = lossD * self.adv_weight

                optimizerD.zero_grad()
                lossD.backward()
                optimizerD.step()

                data = self.detach_data(data)
                sampled_data = self.detach_data(sampled_data)

                # change attacker state after update
                attacker.eval()

            vloss1 = loss.item()
            total_loss += vloss1
            count += 1

        avg_loss = float(total_loss) / count
        return avg_loss

    def sphere_attack(self, loss, sampled_data, batch_idx):
        """
        computes saliency maps for test pointclouds sampled from 
        private sampler and public sampler for visualization and saves it to logdir/snapshots.
        Batch Size = 1
        sampled data must be in B*N*C format
        """

        dirname1 = "/".join(self.logfile.split("/")[:-1])
        dirname = os.path.join(dirname1, "SaliencySnapshots")
        indices = os.path.join(dirname1, "indices")
        points_saver = os.path.join(dirname1, "Points")

        Path(dirname).mkdir(parents=True, exist_ok=True)
        Path(indices).mkdir(parents=True, exist_ok=True)
        Path(points_saver).mkdir(parents=True, exist_ok=True)

        if self.DATASET == "modelnet":
            name = sampled_data[1][0][0].item()
        else:
            k = self.target_map["cls_exp"]  # use only only task name for comparison
            target = sampled_data[1]
            target = target[:, k] if k != None else target

            name = target[0].item()

        output_filename = os.path.join(dirname, str(name) + str(batch_idx) + ".png")

        points = sampled_data[0]

        # compute gradients of loss wrt sampled data, original data
        grad = torch.autograd.grad(outputs=loss, inputs=points)[0]
        grad = grad.cpu().numpy()
        # print("Sum of gradients: ", np.sum(grad))

        points = points.detach().cpu().numpy()
        points_adv = points.copy()

        # change grad to spherical coordinates
        sphere_core = np.median(points_adv, axis=1, keepdims=True)
        r2 = np.square(points_adv - sphere_core)
        r2 = np.sum(r2, axis=2)
        sphere_r = np.sqrt(r2)

        sphere_axis = points_adv - sphere_core  ## BxNx3

        # raw scores
        sphere_map = -np.multiply(
            np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, 1)
        )

        # argsort to get rankings
        sorted_indices = np.argsort(sphere_map, axis=1)  # check axis

        # save points as well for distance
        np.save(os.path.join(points_saver, str(name) + str(batch_idx)), points)

        # top 100 points
        top_indices = sorted_indices.squeeze(0)[-100:]

        # save index ranking to compare later
        np.save(
            os.path.join(indices, str(name) + str(batch_idx)), sorted_indices.squeeze(0)
        )

        # plotting
        cmap = plt.get_cmap("viridis", sorted_indices.shape[1])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        points = points.squeeze(0)

        xmin, xmax = np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1
        ymin, ymax = np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1
        zmin, zmax = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1

        if self.DATASET == "facescape":
            ax.scatter(points[:, 1], points[:, 0], points[:, 2], s=5, c="lightgrey")
            ax.scatter(
                points[top_indices, 1],
                points[top_indices, 0],
                points[top_indices, 2],
                s=25,
                c="r",
            )
        #             sm = plt.cm.ScalarMappable(cmap=cmap)
        #             plt.colorbar(sm)
        else:
            ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=2, c="lightgrey")

            if self.BASE_TASK == "cls_living":
                ax.scatter(
                    points[top_indices, 2],
                    points[top_indices, 0],
                    points[top_indices, 1],
                    s=25,
                    c="g",
                )
            else:
                ax.scatter(
                    points[top_indices, 2],
                    points[top_indices, 0],
                    points[top_indices, 1],
                    s=25,
                    c="r",
                )

        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_zlabel("z")
        ax.set_xlim(xmin, xmax)
        ax.set_zlim(zmin, zmax)
        ax.set_ylim(ymin, ymax)

        ax.axis("off")

        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()

    def eval_1(self, model, attacker, sampler, testloader):
        (
            total_loss,
            attack_total,
            attack_acc_total,
            task_total,
            correct_total,
            sampler_reg_total,
            contrastive_total,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        reg_loss_pointwise = 0.0

        # Shift to eval mode for BN / Projection layers
        task_state = model.training
        if sampler != None:
            sampler_state = sampler.training
            sampler.eval()

        model.eval()
        if attacker:
            attack_state = attacker.training
            attacker.eval()

        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # Sample using one of the samplers:
                if sampler is not None and sampler.name == "samplenet":
                    (
                        sampler_loss,
                        sampled_data,
                        sampler_loss_info,
                        sampler_reg,
                        contrastive_loss,
                    ) = self.compute_samplenet_loss(sampler, data)
                    if self.GAUSSIAN:
                        if self.POINTWISE_DIST or self.RESAMPLE:
                            sampled_data, reg_loss_pointwise = self.add_noise(
                                sampled_data, sampler
                            )
                        else:
                            sampled_data = self.add_noise(sampled_data, sampler)

                elif sampler is not None and sampler.name == "fps":
                    sampled_data = self.non_learned_sampling(sampler, data)
                    sampler_loss = torch.tensor(0, dtype=torch.float32)
                    sampler_reg, contrastive_loss, reg_loss_pointwise = 0.0, 0.0, 0.0

                    if self.TASK == "priv" or self.FINETUNE:
                        if self.POINTWISE_DIST or self.RESAMPLE:
                            sampled_data, reg_loss_pointwise = self.add_noise(
                                sampled_data, sampler
                            )
                        else:
                            sampled_data = self.add_noise(sampled_data, sampler)
                else:
                    sampled_data = data
                    sampler_loss = torch.tensor(0, dtype=torch.float32)
                    sampler_reg, contrastive_loss = 0.0, 0.0

                if self.TASK == "priv":
                    (
                        pointnet_loss,
                        task_loss,
                        attack_loss,
                        correct,
                        attack_acc,
                    ) = self.compute_combined_loss(model, attacker, data, sampled_data)
                    task_total += task_loss.item()
                    attack_total += attack_loss.item()
                    attack_acc_total += attack_acc
                else:
                    if self.BASE_TASK.startswith("cls"):
                        pointnet_loss, correct = self.compute_pointnet_loss(
                            model, sampled_data
                        )
                        task_total += pointnet_loss.item()

                # samplenet loss is already factorized by ALPHA and LMBDA hyper parameters.
                loss = (
                    pointnet_loss
                    + sampler_loss
                    + (sampler_reg * 1000)
                    + (self.LMBDA * contrastive_loss)
                    + (self.REG_WEIGHT * reg_loss_pointwise)
                )
                correct_total += correct

                vloss1 = loss.item()
                sampler_reg_total += sampler_reg
                contrastive_total += contrastive_loss

                total_loss += vloss1

                count += 1

        val_loss = float(total_loss) / count
        val_loss_task = float(task_total) / count
        val_acc = float(correct_total) / count
        val_loss_attack = float(attack_total) / count
        val_acc_attack = float(attack_acc_total) / count
        sampler_reg_total = float(sampler_reg_total) / count
        contrastive_total = float(contrastive_total) / count

        model.train(task_state)
        if attacker:
            attacker.train(attack_state)
        if sampler is not None:
            sampler.train(sampler_state)

        loss = {
            "loss": val_loss,
            "task": val_loss_task,
            "attack": val_loss_attack,
            "acc": val_acc,
            "acc_attack": val_acc_attack,
            "reg": sampler_reg_total,
            "cont": contrastive_total,
        }

        return loss

    def saliency_1(self, model, attacker, sampler, testloader):
        model.eval()
        model.requires_grad_(True)

        for i, data in enumerate(tqdm(testloader)):
            # make sure data requires grad
            data[0].requires_grad_(True)
            if sampler is not None and sampler.name == "samplenet":
                sampler.requires_grad_(True)
                sampler.eval()

                (_, sampled_data, _, _, _,) = self.compute_samplenet_loss(sampler, data)

                if self.GAUSSIAN or self.LEARN_NOISE:
                    if self.POINTWISE_DIST or self.RESAMPLE:
                        sampled_data, _ = self.add_noise(sampled_data, sampler)
                    else:
                        sampled_data = self.add_noise(sampled_data, sampler)

            elif sampler is not None and sampler.name == "fps":
                sampled_data = self.non_learned_sampling(sampler, data)
                if self.TASK == "priv" or self.FINETUNE:
                    if self.POINTWISE_DIST or self.RESAMPLE:
                        sampled_data, _ = self.add_noise(sampled_data, sampler)
                    else:
                        sampled_data = self.add_noise(sampled_data, sampler)
            else:
                sampled_data = data

            sampled_data[0].requires_grad_(True)

            if self.TASK == "priv":
                pointnet_loss, _, _, _, _ = self.compute_combined_loss(
                    model, attacker, data, sampled_data
                )

            else:
                if self.BASE_TASK.startswith("cls"):
                    pointnet_loss, correct = self.compute_pointnet_loss(
                        model, sampled_data
                    )

            self.sphere_attack(pointnet_loss, sampled_data, i)

    #             if i >= 1:
    #                 break

    def test_1(self, model, attacker, sampler, testloader):
        pointnet_errors, task_loss_total, norm_recon_error, reg_loss_pointwise = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        accuracy, attack_acc_total, attack_loss_total = 0.0, 0.0, 0.0
        count = 0

        if self.VISUALIZE > 0:
            LOGGER.info(f"Visualizing {self.VISUALIZE} files")
            view_array = np.random.randint(
                low=0, high=len(testloader), size=self.VISUALIZE
            )

        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                # Sample using one of the samplers:
                if sampler is not None and sampler.name == "samplenet":
                    (
                        sampler_loss,
                        sampled_data,
                        sampler_loss_info,
                        sampler_reg,
                        contrastive_loss,
                    ) = self.compute_samplenet_loss(sampler, data)
                    if self.GAUSSIAN:
                        if self.POINTWISE_DIST or self.RESAMPLE:
                            sampled_data, _ = self.add_noise(sampled_data, sampler)
                        else:
                            sampled_data = self.add_noise(sampled_data, sampler)

                elif sampler is not None and (sampler.name in ["fps", "random"]):
                    sampled_data = self.non_learned_sampling(sampler, data)
                    if self.TASK == "priv" or self.FINETUNE:
                        if self.POINTWISE_DIST or self.RESAMPLE:
                            sampled_data, _ = self.add_noise(sampled_data, sampler)
                        else:
                            sampled_data = self.add_noise(sampled_data, sampler)

                else:
                    sampled_data = data

                if self.TASK == "priv":
                    (
                        pointnet_loss,
                        task_loss,
                        attack_loss,
                        correct,
                        attack_acc,
                    ) = self.compute_combined_loss(model, attacker, data, sampled_data)
                    attack_acc_total += attack_acc
                    attack_loss_total += attack_loss.item()
                    task_loss_total += task_loss.item()

                else:
                    if self.BASE_TASK.startswith("cls"):
                        pointnet_loss, correct = self.compute_pointnet_loss(
                            model, sampled_data
                        )

                    task_loss_total += pointnet_loss.item()

                # save visualized data
                if self.VISUALIZE > 0 and i in view_array:
                    sample_number = np.random.randint(
                        0, sampled_data[0].shape[0], size=1
                    )[0]
                    s_data = sampled_data[0][sample_number, :, :].unsqueeze(0).cpu()
                    d_data = data[0][sample_number, :, :].unsqueeze(0).cpu()

                    # save both sampled data and original data
                    visualize_data(
                        s_data, os.path.join(self.vis_file, "sampled", f"{i}")
                    )
                    visualize_data(
                        d_data, os.path.join(self.vis_file, "original", f"{i}")
                    )
                    visualize_data(
                        [d_data, s_data],
                        os.path.join(self.vis_file, "combined", f"{i}"),
                        combined=True,
                    )

                accuracy += correct
                pointnet_errors += pointnet_loss.item()
                count += 1.0

        # Compute Precision curve and AUC.
        accuracy = float(accuracy) / count
        pointnet_errors = float(pointnet_errors) / count

        attack_acc_total = float(attack_acc_total) / count

        with open(self.logfile, "a") as f:
            print(f"Experiment name: {self.experiment_name}", file=f)
            print(f"Accuracy = {accuracy}", file=f)
            if self.TASK == "priv":
                print(f"Attacker Accuracy = {attack_acc_total}", file=f)
                print(f"Attacker Loss = {attack_loss_total/count}", file=f)

            print(f"Mean Pointnet Error = {pointnet_errors}", file=f)
            print(f"STD Pointnet Error = {np.std(pointnet_errors)}", file=f)

        print(f"Experiment name: {self.experiment_name}")
        print(f"Accuracy = {accuracy}")
        if self.TASK == "priv":
            print(f"Attacker Accuracy = {attack_acc_total}")
            print(f"Attacker Loss = {attack_loss_total/count}")

        print(f"Mean Pointnet Error = {pointnet_errors}")
        print(f"STD Pointnet Error = {np.std(pointnet_errors)}")

    def non_learned_sampling(self, sampler, data):
        """Sample p1 point cloud using FPS."""
        p0, igt, contrastive_data = data
        p0 = p0.to(self.DEVICE)  # pts
        igt = igt.to(self.DEVICE)

        p0_samp = sampler(p0)
        sampled_data = (p0_samp, igt)

        return sampled_data

    def add_noise(self, data: torch.Tensor, model: nn.Module) -> Tuple:
        if self.LEARN_NOISE:
            if self.RESAMPLE:
                scale = model.noise_std[0].abs()
                loc = model.noise_mean[0]

            elif self.POINTWISE_DIST:
                # use feat extractor to get stds
                scale, loc = model.get_std(data[0])
            else:
                new_data = data[0] + model.noise[0]

            if self.RESAMPLE or self.POINTWISE_DIST:
                with torch.enable_grad():
                    noise = tod.Normal(loc=loc, scale=scale.abs() + 1e-10)
                    resampled = noise.rsample()
                    new_data = data[0] + resampled
                    regularization = F.mse_loss(
                        resampled, torch.zeros(resampled.shape).to(self.DEVICE)
                    )
                    return (new_data, data[1]), regularization
        else:
            new_data = data[0] + torch.normal(
                mean=0.0, std=self.SIGMA, size=data[0].shape
            ).to(self.DEVICE)

        return (new_data, data[1])

    # Losses
    def compute_pointnet_loss(self, model, data, att=None):
        try:
            inputs, target, _ = data
        except:
            inputs, target = data

        inputs, target = inputs.to(self.DEVICE), target.to(self.DEVICE)

        if self.DATASET == "facescape":
            # pick correct target for classification
            if att == True:
                k = self.target_map[self.ATTACKER_TASK]
            elif att == None:
                k = self.target_map[self.BASE_TASK]

            target = target[:, k] if k != None else target

        elif self.DATASET == "modelnet":

            if (att == True and self.ATTACKER_TASK == "cls_living") or (
                att == None and self.BASE_TASK == "cls_living"
            ):
                target = target[:, 1:][:]
            else:
                target = target[:, :1][:]

            target = target.squeeze(1)

        if self.TRAIN_POINTNET or self.TEST:
            x, trans_feat = model(inputs)
            _, pred = torch.max(x.data, 1)

            if self.MAX_ENTROPY and att is not None:
                loss = self.max_entropy_loss(x)
            else:
                loss = pointnet_cls.get_loss()(x, target, trans_feat)

        elif self.TRAIN_DGCNN or self.TEST_DGCNN:
            logits = model(inputs)
            pred = logits.max(dim=1)[1]
            if self.MAX_ENTROPY and att is not None:
                loss = self.max_entropy_loss(x)
            else:
                loss = dgcnn_cls.get_loss()(logits, target)

        correct = (pred == target).sum()

        total = target.size(0)
        correct = float(correct.item()) / total

        return loss, correct

    def compute_samplenet_loss(self, sampler, data, attacker=None):
        """Sample point clouds using SampleNet and compute sampling associated losses."""
        p0, igt, contrastive_data = data[0], data[1], data[2]

        p0, igt = p0.to(self.DEVICE), igt.to(self.DEVICE)  # pts
        sampler = sampler.to(self.DEVICE)

        p0_simplified, p0_projected, original = sampler(p0)

        # Sampling loss
        p0_simplification_loss = sampler.get_simplification_loss(
            p0, p0_simplified, self.NUM_OUT_POINTS, self.GAMMA, self.DELTA
        )

        simplification_loss = p0_simplification_loss
        sampled_data = (p0_projected, igt)

        # Projection loss
        projection_loss = sampler.get_projection_loss()

        if self.ADV_CONTRASTIVE or self.CONTRASTIVE_FEAT:
            p_points, p_label, n_points, n_label = contrastive_data
            p_points, p_label, n_points, n_label = (
                p_points.to(self.DEVICE),
                p_label.to(self.DEVICE),
                n_points.to(self.DEVICE),
                n_label.to(self.DEVICE),
            )

            # sample positive
            p_simplified, p_projected, p_original = sampler(p_points)

            # sample negative
            n_simplified, n_projected, n_original = sampler(n_points)

            contrastive_loss = self.closs(p0_projected, p_projected, n_projected)

        else:
            contrastive_loss = 0.0

        samplenet_loss = (self.ALPHA * simplification_loss) + (
            self.LMBDA * projection_loss
        )

        samplenet_loss_info = {
            "simplification_loss": simplification_loss,
            "projection_loss": projection_loss,
            "contrastive_loss": contrastive_loss,
        }

        return (
            samplenet_loss,
            sampled_data,
            samplenet_loss_info,
            0,
            contrastive_loss,
        )

    def compute_combined_loss(self, model, attacker, data, sampled_data):
        p_data, gt_data, contrastive_data = data[0], data[1], data[2]
        p_data, gt_data = p_data.to(self.DEVICE), gt_data.to(self.DEVICE)

        if self.ATTACKER_TASK.startswith("cls"):
            attacker_loss, attacker_acc = self.compute_pointnet_loss(
                attacker, sampled_data, att=True
            )

        if self.BASE_TASK.startswith("cls"):
            base_loss, base_acc = self.compute_pointnet_loss(model, sampled_data)

        # do adversarial training
        if not self.NO_ADV:
            final_loss = ((1 - self.adv_weight) * base_loss) - (
                attacker_loss * self.adv_weight * self.scale
            )

        # only base loss (no adversarial training)
        else:
            final_loss = base_loss

        if self.ADV_CONTRASTIVE and self.CONTRASTIVE_FEAT:
            return final_loss, base_loss, attacker_loss, base_acc, attacker_acc

        elif self.CONTRASTIVE_FEAT:
            return (
                base_loss,
                base_loss,
                torch.tensor([0.0]).to(self.DEVICE),
                base_acc,
                attacker_acc,
            )

        return final_loss, base_loss, attacker_loss, base_acc, attacker_acc


if __name__ == "__main__":
    ARGS = options(parser=sputils.get_parser())

    logging.basicConfig(
        format="%(levelname)s:%(name)s, %(asctime)s, %(message)s",
        filename=f"{ARGS.outfile}.log",
    )
    LOGGER.debug("Training (PID=%d), %s", os.getpid(), ARGS)

    _ = main(ARGS)
    LOGGER.debug("done (PID=%d)", os.getpid())
