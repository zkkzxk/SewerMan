import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd
from collections import Counter

print(torch.cuda.is_available())
print("\nCUDA is available:{}, version is {}".format(torch.cuda.is_available(), torch.version.cuda))
print("\ndevice_name: {}".format(torch.cuda.get_device_name(0)))
print(torch.version.cuda)
print(torch.backends.cudnn.version())

from inference import load_model
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

import torch.nn.functional as F
from torch import nn
from torchvision import models as torch_models
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiLabelDataModule, BinaryDataModule, BinaryRelevanceDataModule
import sewer_models
import ml_models

import clip


class MultiLabelModel(pl.LightningModule):
    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if
                                     name.islower() and not name.startswith("__") and callable(
                                         torch_models.__dict__[name]))
    SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if
                               name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
    MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if
                                    name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
    MODEL_NAMES = TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES

    def __init__(self, model="resnet18", num_classes=2, learning_rate=1e-2, momentum=0.9, weight_decay=0.0001,
                 criterion=torch.nn.BCEWithLogitsLoss, **kwargs):
        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        # 初始化CLIP模型
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # 初始化主模型（但不用于训练）
        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(model))

        self.aux_logits = hasattr(self.model, "aux_logits")
        self.criterion = criterion

    def forward(self, x):
        # 在统计模式下，我们不需要实际的forward
        return torch.randn(x.size(0), self.num_classes, device=x.device)

    def get_text_features(self):
        """为每个类别生成CLIP文本特征"""
        class_names = ["Cracks, breaks, and collapses", "Surface damage", "Production error",
                       "Deformation", "Displaced joint", "Displaced joint", "Roots", "Inﬁltration", "Settled deposits",
                       "Settled deposits", "Obstacle", "Branch pipe", "Branch pipe", "Drilled connection",
                       "Lateral reinstatement cuts", "Lateral reinstatement cuts",
                       "Connection with construction changes"]

        templates = [
            "a photo of a defect {}",
            "a bad sewer with {}",
            "an infrastructure issue of type {}"
        ]

        text_features = []
        for name in class_names:
            texts = [t.format(name) for t in templates]
            text_inputs = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                features = self.clip_model.encode_text(text_inputs)
                features = features.mean(dim=0)
                features /= features.norm()
            text_features.append(features)
        return torch.stack(text_features)

    def multi_label_ot_partition(self, sim_matrix, labels, reg=0.1, threshold=0.5):
        """
        sim_matrix: (bs, num_classes) 图像-文本相似度
        labels: (bs, num_classes) 多标签矩阵
        """
        cost_matrix = 1 - sim_matrix.T
        Q = torch.exp(-cost_matrix / reg)
        labels_T = labels.T
        class_marginals = labels_T.sum(dim=1)

        # Sinkhorn迭代
        for _ in range(10):
            Q = Q * (class_marginals.unsqueeze(1) / (Q.sum(dim=1, keepdim=True) + 1e-8))
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)

        confidence = (Q.T * labels).sum(dim=1)
        clean_mask = confidence > threshold
        clean_idx = torch.where(clean_mask)[0]

        return clean_idx

    def analyze_training_data_distribution(self, dm):
        """分析训练数据分布"""
        print("开始分析训练数据分布...")

        # 设置模型为评估模式
        self.eval()

        # 统计数据
        total_samples = 0
        clean_samples = 0
        class_distribution_before = torch.zeros(self.num_classes)
        class_distribution_after = torch.zeros(self.num_classes)
        confidence_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dm.train_dataloader()):
                x, y, _ = batch
                x = x.to(self.device)
                y = y.to(self.device)

                batch_size = x.size(0)
                total_samples += batch_size

                # 计算CLIP特征和相似度
                image_features = self.clip_model.encode_image(x)
                image_features = F.normalize(image_features, dim=1)
                text_features = self.get_text_features()
                sim_matrix = image_features @ text_features.T

                # OT清理
                clean_idx = self.multi_label_ot_partition(sim_matrix, y.float(), threshold=0.2)

                # 统计清理前后的数据
                clean_samples += len(clean_idx)

                # 统计类别分布（清理前）
                class_distribution_before += y.sum(dim=0).cpu()

                # 统计类别分布（清理后）
                if len(clean_idx) > 0:
                    class_distribution_after += y[clean_idx].sum(dim=0).cpu()

                # 计算置信度分数用于分析
                cost_matrix = 1 - sim_matrix.T
                Q = torch.exp(-cost_matrix / 0.1)
                labels_T = y.T.float()
                class_marginals = labels_T.sum(dim=1)

                for _ in range(10):
                    Q = Q * (class_marginals.unsqueeze(1) / (Q.sum(dim=1, keepdim=True) + 1e-8))
                    Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)

                confidence = (Q.T * y.float()).sum(dim=1)
                confidence_scores.extend(confidence.cpu().numpy())

                if batch_idx % 10 == 0:
                    print(f"已处理 {batch_idx} 个批次, {total_samples} 个样本")

        # 输出统计结果
        print("\n" + "=" * 50)
        print("训练数据分布统计结果")
        print("=" * 50)
        print(f"总样本数: {total_samples}")
        print(f"清理后保留样本数: {clean_samples}")
        print(f"清理比例: {clean_samples / total_samples * 100:.2f}%")
        print(f"移除样本数: {total_samples - clean_samples}")
        print(f"移除比例: {(total_samples - clean_samples) / total_samples * 100:.2f}%")

        print("\n清理前各类别样本分布:")
        class_names = ["Cracks, breaks, and collapses", "Surface damage", "Production error",
                       "Deformation", "Displaced joint", "Displaced joint", "Roots", "Inﬁltration",
                       "Settled deposits", "Settled deposits", "Obstacle", "Branch pipe",
                       "Branch pipe", "Drilled connection", "Lateral reinstatement cuts",
                       "Lateral reinstatement cuts", "Connection with construction changes"]

        for i, (count_before, count_after, name) in enumerate(
                zip(class_distribution_before, class_distribution_after, class_names)):
            retention_rate = count_after / count_before * 100 if count_before > 0 else 0
            print(
                f"类别 {i:2d} ({name[:30]:30s}): 清理前 {count_before:6.0f} | 清理后 {count_after:6.0f} | 保留率 {retention_rate:6.2f}%")

        # 置信度分析
        confidence_scores = np.array(confidence_scores)
        print(f"\n置信度统计:")
        print(f"平均置信度: {confidence_scores.mean():.4f}")
        print(f"置信度标准差: {confidence_scores.std():.4f}")
        print(f"最小置信度: {confidence_scores.min():.4f}")
        print(f"最大置信度: {confidence_scores.max():.4f}")
        print(f"中位数置信度: {np.median(confidence_scores):.4f}")

        # 置信度分布
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(confidence_scores, bins=bins)
        print(f"\n置信度分布:")
        for i in range(len(bins) - 1):
            print(
                f"{bins[i]:.1f}-{bins[i + 1]:.1f}: {hist[i]:6d} 样本 ({hist[i] / len(confidence_scores) * 100:5.1f}%)")

        return {
            'total_samples': total_samples,
            'clean_samples': clean_samples,
            'class_distribution_before': class_distribution_before.numpy(),
            'class_distribution_after': class_distribution_after.numpy(),
            'confidence_scores': confidence_scores
        }

    def training_step(self, batch, batch_idx):
        # 在统计模式下不需要训练
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        # 在统计模式下不需要优化器
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--model', type=str, default="zhong2023_multilabel", choices=MultiLabelModel.MODEL_NAMES)
        return parser


def main(args):
    pl.seed_everything(1234567890)

    # 初始化数据
    img_size = 299 if args.model in ["inception_v3", "chen2018_multilabel"] else 224

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    if args.training_mode == "e2e":
        dm = MultiLabelDataModule(batch_size=args.batch_size, workers=args.workers, ann_root=args.ann_root,
                                  data_root=args.data_root, train_transform=train_transform,
                                  eval_transform=eval_transform, only_defects=False)
    elif args.training_mode == "defect":
        dm = MultiLabelDataModule(batch_size=args.batch_size, workers=args.workers, ann_root=args.ann_root,
                                  data_root=args.data_root, train_transform=train_transform,
                                  eval_transform=eval_transform, only_defects=True)
    elif args.training_mode == "binary":
        dm = BinaryDataModule(batch_size=args.batch_size, workers=args.workers, ann_root=args.ann_root,
                              data_root=args.data_root, train_transform=train_transform, eval_transform=eval_transform)
    elif args.training_mode == "binaryrelevance":
        assert args.br_defect is not None, "Training mode is 'binary_relevance', but no 'br_defect' was stated"
        dm = BinaryRelevanceDataModule(batch_size=args.batch_size, workers=args.workers, ann_root=args.ann_root,
                                       data_root=args.data_root, train_transform=train_transform,
                                       eval_transform=eval_transform, defect=args.br_defect)
    else:
        raise Exception("Invalid training_mode '{}'".format(args.training_mode))

    dm.prepare_data()
    dm.setup("fit")

    # 初始化模型（用于统计，不训练）
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dm.class_weights)
    light_model = MultiLabelModel(num_classes=dm.num_classes, criterion=criterion, **vars(args))

    # 将模型移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    light_model = light_model.to(device)

    # 分析数据分布
    stats = light_model.analyze_training_data_distribution(dm)

    # 保存统计结果到文件
    logger_path = os.path.join(args.log_save_dir, args.model, f"data_analysis_version_{args.log_version}")
    os.makedirs(logger_path, exist_ok=True)

    # 保存详细统计结果
    with open(os.path.join(logger_path, "data_distribution_analysis.txt"), "w") as f:
        f.write("训练数据OT清理分布分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"总样本数: {stats['total_samples']}\n")
        f.write(f"清理后保留样本数: {stats['clean_samples']}\n")
        f.write(f"清理比例: {stats['clean_samples'] / stats['total_samples'] * 100:.2f}%\n")
        f.write(f"移除样本数: {stats['total_samples'] - stats['clean_samples']}\n")
        f.write(
            f"移除比例: {(stats['total_samples'] - stats['clean_samples']) / stats['total_samples'] * 100:.2f}%\n\n")

        f.write("各类别清理前后分布:\n")
        class_names = ["Cracks, breaks, and collapses", "Surface damage", "Production error",
                       "Deformation", "Displaced joint", "Displaced joint", "Roots", "Inﬁltration",
                       "Settled deposits", "Settled deposits", "Obstacle", "Branch pipe",
                       "Branch pipe", "Drilled connection", "Lateral reinstatement cuts",
                       "Lateral reinstatement cuts", "Connection with construction changes"]

        for i, (count_before, count_after, name) in enumerate(zip(stats['class_distribution_before'],
                                                                  stats['class_distribution_after'],
                                                                  class_names)):
            retention_rate = count_after / count_before * 100 if count_before > 0 else 0
            f.write(
                f"类别 {i:2d} ({name[:30]:30s}): 清理前 {count_before:6.0f} | 清理后 {count_after:6.0f} | 保留率 {retention_rate:6.2f}%\n")

    print(f"\n分析完成！结果已保存到: {logger_path}")


def run_cli():
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations1')
    parser.add_argument('--data_root', type=str, default='D:/data')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--training_mode', type=str, default="e2e",
                        choices=["e2e", "binary", "binaryrelevance", "defect"])
    parser.add_argument('--br_defect', type=str, default=None,
                        choices=[None, "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH",
                                 "PB", "OS", "OP", "OK"])

    # 添加必要的参数
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MultiLabelModel.add_model_specific_args(parser)
    args = parser.parse_args(args=[])

    main(args)


if __name__ == "__main__":
    run_cli()