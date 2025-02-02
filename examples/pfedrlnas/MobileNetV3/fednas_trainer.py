"""
Customized Trainer for PerFedRLNAS.
"""
import fednas_specific

from plato.trainers import basic
from plato.config import Config


if Config().trainer.lr_scheduler == "timm":
    BasicTrainer = basic.TrainerWithTimmScheduler
else:
    BasicTrainer = basic.Trainer


class Trainer(BasicTrainer):
    """Use special optimizer and loss criterion specific for NASVIT."""

    def get_loss_criterion(self):
        return fednas_specific.get_nasvit_loss_criterion()

    def get_optimizer(self, model):
        optimizer = fednas_specific.get_optimizer(model)
        return optimizer
