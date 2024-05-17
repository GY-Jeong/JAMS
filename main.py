import logging
import math
import os
from pathlib import Path
import itertools

import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint

from src.data.data_pipeline import data_pipeline
from src.factories import (
    get_callbacks,
    get_dataloaders,
    get_datasets,
    get_lookups,
    get_lr_scheduler,
    get_metric_collections,
    get_model,
    get_optimizer,
    get_text_encoder,
    get_transform,
)
from src.trainer.trainer import Trainer
from src.utils.seed import set_seed

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


def deterministic() -> None:
    """Run experiment deterministically. There will still be some randomness in the backward pass of the model."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()
    else:
        import torch

    set_seed(cfg.seed)

    # Check if CUDA_VISIBLE_DEVICES is set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if cfg.gpu != -1 and cfg.gpu is not None and cfg.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ",".join([str(gpu) for gpu in cfg.gpu])
                if isinstance(cfg.gpu, list)
                else str(cfg.gpu)
            )

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(f"Device: {device}")
    pprint(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Da
    all_data = {
        data_name: data_pipeline(config=cfg.data[data_name], version=data_name)
        for data_name in cfg.data.keys()
    }

    all_train_documents = list(
        itertools.chain.from_iterable(
            [all_data[data_name].get_train_documents for data_name in all_data.keys()]
        )
    )

    text_encoder = get_text_encoder(
        config=cfg.text_encoder,
        data_dir='files/data/mimiciv_integrate',
        texts=all_train_documents
    )

    # One-hot encoder for icd9 target : 6150
    # One-hot encoder for icd10 target : 7942
    all_label_transform = {
        data_name: get_transform(
            config=cfg.label_transform,
            targets=all_data[data_name].all_targets,
            load_transform_path=cfg.load_model,
            version=data_name
        )
        for data_name in all_data.keys()
    }

    text_transform = get_transform(
        config=cfg.text_transform,
        texts=all_train_documents,
        text_encoder=text_encoder,
        load_transform_path=cfg.load_model,
    )

    del all_train_documents

    for data_name in all_data.keys():
        all_data[data_name].truncate_text(cfg.data_max_length)
        all_data[data_name].transform_text(text_transform.batch_transform)

    all_lookups = {
        data_name: get_lookups(
            config=cfg.lookup,
            data=all_data[data_name],
            label_transform=all_label_transform[data_name],
            text_transform=text_transform,
        )
        for data_name in all_data.keys()
    }

    num_classes_info = {
        'num_classes_'+data_name: all_lookups[data_name].data_info["num_classes"]
        for data_name in all_data.keys()

    }

    # 같은 인코더를 쓰기 때문에 vocab size는 동일함
    num_classes_info['vocab_size'] = all_lookups['mimiciv_icd9'].data_info["vocab_size"]
    
    model = get_model(
        config=cfg.model, data_info=num_classes_info, text_encoder=text_encoder
    )
    model.to(device)

    metric_collections = {
        data_name: get_metric_collections(
            config=cfg.metrics,
            number_of_classes=all_lookups[data_name].data_info["num_classes"],
            code_system2code_indices=all_lookups[data_name].code_system2code_indices,
            split2code_indices=all_lookups[data_name].split2code_indices,
        )
        for data_name in all_data.keys()
    }

    datasets = get_datasets(
        config=cfg.dataset,
        data=all_data,
        text_transform=text_transform,
        label_transform=all_label_transform,
        lookups=all_lookups
    )

    dataloaders = get_dataloaders(config=cfg.dataloader, datasets_dict=datasets)

    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )

    optimizer = get_optimizer(config=cfg.optimizer, model=model)

    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )
    
    lr_scheduler = get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    callbacks = get_callbacks(config=cfg.callbacks)

    trainer = Trainer(
        config=cfg,
        data=all_data,
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        all_lookups=all_lookups,
        accumulate_grad_batches=accumulate_grad_batches,
    ).to(device)

    if cfg.load_model:
        trainer.experiment_path = Path(cfg.load_model)

    trainer.fit()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
