
import os
from argparse import ArgumentParser
import numpy as np
import torch

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


from torchvision import models as torch_models
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
#from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_datamodules import MultiLabelDataModule, BinaryDataModule, BinaryRelevanceDataModule
import sewer_models
import ml_models

import torch





class MultiLabelModel(pl.LightningModule):
    '''这段代码的作用是获取可用的模型名称列表。它首先从torch_models、sewer_models和ml_models模块中获取所有小写且不以双下划线开头的可调用对象的名称，并将它们按字母顺序排序。然后，它将这些名称列表合并到MODEL_NAMES中。'''
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
        # def __init__(self, model="resnet18", num_classes=2, learning_rate=1e-2, momentum=0.9, weight_decay=0.0001,criterion=torch.nn.BCEWithLogitsLoss, **kwargs):

        super(MultiLabelModel, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        if model in MultiLabelModel.TORCHVISION_MODEL_NAMES:
            '''这行代码的作用是根据给定的model名称从torch_models模块中选择相应的模型，并将num_classes作为参数传递给该模型的构造函数，然后将实例化的模型赋值给self.model变量。 '''
            self.model = torch_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[model](num_classes=self.num_classes)
        elif model in MultiLabelModel.MULTILABEL_MODEL_NAMES:
            self.model = ml_models.__dict__[model](num_classes=self.num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(model))
        '''这行代码的作用是检查self.model是否具有属性aux_logits。它使用了hasattr()函数来判断是否存在该属性。如果self.model具有aux_logits属性，那么self.aux_logits将被设置为True，否则将被设置为False。 '''
        self.aux_logits = hasattr(self.model, "aux_logits")

        if self.aux_logits:
            self.train_function = self.aux_loss
        else:
            self.train_function = self.normal_loss
        self.criterion = criterion
        '''这段代码检查self.criterion是否具有set_device方法，并且该方法是可调用的。如果是这样，代码将执行self.criterion.set_device()。这个检查是为了确保self.criterion对象具有set_device方法，并且可以在代码中正确地调用它。 '''
        if callable(getattr(self.criterion, "set_device", None)):
            self.criterion.set_device(self.device)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def aux_loss(self, x, y):
        y = y.float()
        '''y_hat和y_aux_hat是通过对输入x进行模型推理得到的输出结果。
        - y_hat是主要的预测结果，通常是一个张量或数组，表示模型对输入x的主要预测。GroundTruth
        - y_aux_hat是辅助的预测结果，通常也是一个张量或数组，表示模型对输入x的辅助预测。Pre'''
        y_hat, y_aux_hat = self(x)
        loss = self.criterion(y_hat, y) + 0.4 * self.criterion(y_aux_hat, y)

        return loss

    def normal_loss(self, x, y):
        y = y.float()
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        return loss

    '''这段代码是一个PyTorch Lightning模型的训练步骤函数。在每个训练步骤中，它接收一个批次数据batch和批次索引batch_idx作为输入。'''

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.train_function(x, y)

        # .log sends to tensorboard/logger, prog_bar also sends to the progress bar
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, sync_dist=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss = self.normal_loss(x, y)

        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, sync_dist=True)
        return result

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
                                weight_decay=self.hparams.weight_decay)

        #print(optim )

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30,60], gamma=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[15,30, 80], gamma=0.1)




        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        #parser.add_argument('--model', type=str, default="resnet50", choices=MultiLabelModel.MODEL_NAMES)
        parser.add_argument('--model', type=str, default="zhong2023_multilabel", choices=MultiLabelModel.MODEL_NAMES)
        return parser


def main(args):
    pl.seed_everything(1234567890)

    # Init data with transforms
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
        '''这段代码是在一个类的构造函数中，根据传入的参数args.br_defect来拼接一个前缀字符串prefix。拼接的方式是将args.br_defect添加到prefix的末尾。 '''
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


    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=dm.class_weights)


    light_model = MultiLabelModel(num_classes=dm.num_classes, criterion=criterion, **vars(args))

    # train
    prefix = "{}-".format(args.training_mode)
    if args.training_mode == "binaryrelevance":
        prefix += args.br_defect

    logger = TensorBoardLogger(save_dir=args.log_save_dir, name=args.model,
                               version=prefix + "version_" + str(args.log_version))

    logger_path = os.path.join(args.log_save_dir, args.model, prefix + "version_" + str(args.log_version))

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logger_path, '{epoch:02d}-{val_loss:.2f}'),
        #dirpath=os.path.join(logger_path, '{epoch:02d}-{val_loss:.2f}'),
        save_top_k=90,
        save_last=True,
        verbose=False,
        monitor="val_loss",
        mode='min',
        prefix='',
        period=1
        # every_n_epochs = 1
    )

    lr_monitor = LearningRateLogger(logging_interval='step')
    '''这行代码创建了一个trainer对象，使用了pl.Trainer.from_argparse_args方法，并传入了一些参数.terminate_on_nan=True：这个参数指定当训练过程中出现NaN值时是否终止训练。如果设置为True，训练会在出现NaN值时立即停止。
        benchmark=True：这个参数指定是否在训练过程中启用性能基准测试。如果设置为True，训练过程中会记录一些性能指标，如每个批次的训练时间等。checkpoint_callback=checkpoint_callback：这个参数指定训练过程中的检查点回调函数。checkpoint_callback是一个检查点回调函数对象，用于在训练过程中保存模型的检查点。
        callbacks=[lr_monitor]：这个参数指定训练过程中的回调函数列表。lr_monitor是一个学习率监控回调函数对象，用于监控学习率的变化情况。'''

    trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan=True, benchmark=True, max_epochs=90, logger=logger,
                                            checkpoint_callback=checkpoint_callback, callbacks=[lr_monitor], gpus=1)
    # trainer = pl.Trainer.from_argparse_args(args, terminate_on_nan=True, benchmark=True, max_epochs=args.max_epochs,logger=logger, checkpoint_callback=checkpoint_callback,callbacks=[lr_monitor], gpus=1)
    '''这段代码尝试使用trainer.fit()方法来训练light_model模型，并将数据模块dm传递给它。如果在训练过程中出现任何异常，将会捕获该异常并执行以下操作：
    1. 打印异常信息。
    2. 将异常信息写入名为"error.txt"的文件中，该文件位于logger_path指定的路径下。
    请注意，trainer.fit()方法用于训练模型，light_model是要训练的模型，dm是数据模块。异常处理部分的目的是在训练过程中捕获任何错误，并将错误信息记录下来以供后续分析和调试。 '''


    # updated_state_dict, model_name, num_classes, training_mode, br_defect = load_model('./best_model/.ckpt', True)
    # light_model.model.load_state_dict(updated_state_dict,False)



    try:
        trainer.fit(light_model, dm)
    except Exception as e:
        print(e)
        with open(os.path.join(logger_path, "error.txt"), "w") as f:
            f.write(str(e))


def run_cli():
    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations')
    parser.add_argument('--data_root', type=str, default='D:/data')
    parser.add_argument('--batch_size', type=int, default=98, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--log_save_dir', type=str, default="./logs")
    parser.add_argument('--log_version', type=int, default=1)
    parser.add_argument('--training_mode', type=str, default="e2e",
                        choices=["e2e", "binary", "binaryrelevance", "defect"])
    parser.add_argument('--br_defect', type=str, default=None,
                        choices=[None, "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH",
                                 "PB", "OS", "OP", "OK"])

    # add TRAINER level args
    parser = pl.Trainer.add_argparse_args(parser)

    # add MODEL level args
    parser = MultiLabelModel.add_model_specific_args(parser)
    args = parser.parse_args(args=[])

    # Adjust learning rate to amount of nodes/GPUs
    # args.workers =  max(0, min(8, 4*args.gpus))
    # args.learning_rate = args.learning_rate * (args.gpus * args.num_nodes * args.batch_size) / 256

    main(args)


if __name__ == "__main__":
    run_cli()