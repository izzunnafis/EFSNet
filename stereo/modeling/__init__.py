from .models.efsnet.trainer import Trainer as EFSNetTrainer


__all__ = {
    'EFSNet': EFSNetTrainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
