import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import  pprint
from rich.progress import track
from torch.utils.data import DataLoader

from src.data.datatypes import Data, Lookups
from src.metrics import  MetricCollection
from src.models import BaseModel
from src.settings import ID_COLUMN, TARGET_COLUMN
from src.trainer.callbacks import BaseCallback
from src.utils.decision_boundary import f1_score_db_tuning
import wandb


# dataloaders, metric_collections, lookups : 데이터 2개
class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        data: Data,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections,
        callbacks: BaseCallback,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        all_lookups = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.all_lookups = all_lookups
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.experiment_path = Path(mkdtemp())
        self.current_val_results = {'mimiciv_icd9':None, 'mimiciv_icd10':None}
        self.stop_training = False
        self.best_db_icd9 = 0.5
        self.best_db_icd10 = 0.5
        self.on_initialisation_end()
    
    def fit(self) -> None:
        """Train and validate the model."""
        try:
            self.on_fit_begin()            
            for _ in range(self.epoch, self.epochs):
                if self.stop_training:
                    break
                self.on_epoch_begin()
                self.train_one_epoch(self.epoch)
                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val")
                self.val(self.epoch, split_name="val", version='mimiciv_icd9')
                self.val(self.epoch, split_name="val", version='mimiciv_icd10')
                self.on_epoch_end()
                self.epoch += 1
            self.on_fit_end()
            self.val(self.epoch, split_name="val", version='mimiciv_icd9', evaluating_best_model=True)
            self.val(self.epoch, split_name="test", version='mimiciv_icd9', evaluating_best_model=True)
            self.val(self.epoch, split_name="val", version='mimiciv_icd10', evaluating_best_model=True)
            self.val(self.epoch, split_name="test", version='mimiciv_icd10', evaluating_best_model=True)
            self.save_final_model()
        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        num_batches = len(self.dataloaders["train"])
        for batch_idx, batch in enumerate(
            track(self.dataloaders["train"], description=f"Epoch: {epoch}Training")
        ):
            batch = batch.to(self.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                output = self.model.training_step(batch)
                wandb.log({"Train loss": output["loss"].item()})
                loss = output["loss"] / self.accumulate_grad_batches
            self.gradient_scaler.scale(loss).backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches
            ):
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()

            output = {
                'icd9':{
                    'logits':output['logits_icd9'],
                    'loss': output['loss'],
                    'targets': output['targets_icd9'],
                },
                'icd10': {
                    'logits':output['logits_icd10'],
                    'loss': output['loss'],
                    'targets': output['targets_icd10'],
                }
            }
            self.update_metrics(output, "train")
        self.on_train_end(epoch)

    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.

        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()
        with torch.no_grad():
            for batch in track(
                self.dataloaders[split_name],
                description=f"Epoch: {epoch} | Validating on training data",
            ):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device))
                self.update_metrics(output, split_name)
            self.on_val_end(split_name, epoch)

    def val(
        self, epoch, version: str, split_name: str = "val", evaluating_best_model: bool = False, 
    ) -> None:
        self.model.eval()
        self.on_val_begin()
        logits = []
        targets = []
        logits_cpu = []
        targets_cpu = []
        ids = []
        
        edition = version.split('_')[1]
        with torch.no_grad():
            for idx, batch in enumerate(
                track(
                    self.dataloaders[version][split_name],
                    description=f"Epoch: {epoch} | Validating on {split_name} & {version}",
                )
            ):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device), version)

                output = {
                    'logits':output[f"logits_{edition}"],
                    'loss': output["loss"],
                    'targets': output[f"targets_{edition}"]
                }

                self.update_metrics(output, split_name, version)
                # TODO : 전체 output dict중에서 필요한 정보만을 추출
                logits.append(output["logits"])
                targets.append(output["targets"])
                ids.append(batch.ids)
                if idx % 1000 == 0:
                    # move to cpu to save gpu memory
                    logits_cpu.append(torch.cat(logits, dim=0).cpu())
                    targets_cpu.append(torch.cat(targets, dim=0).cpu())
                    logits = []
                    targets = []
            logits_cpu.append(torch.cat(logits, dim=0).cpu())
            targets_cpu.append(torch.cat(targets, dim=0).cpu())

            logits = torch.cat(logits_cpu, dim=0)
            targets = torch.cat(targets_cpu, dim=0)
            ids = torch.cat(ids, dim=0)
        # TODO : 정리해야됨
        self.on_val_end(split_name, epoch, version, logits, targets, ids, evaluating_best_model)

    def update_metrics(self, outputs: dict[str, torch.Tensor], split_name: str, version: Optional[str] = None) -> None:
        if version:
            # TODO : version이 명시적으로 주어진 경우 (validation, test) 처리
            if version == 'mimiciv_icd9':
                for target_name in self.metric_collections['mimiciv_icd9'][split_name].keys():
                    self.metric_collections['mimiciv_icd9'][split_name][target_name].update(outputs)
            else:
                for target_name in self.metric_collections['mimiciv_icd10'][split_name].keys():
                    self.metric_collections['mimiciv_icd10'][split_name][target_name].update(outputs)
        else:
            if type(outputs['icd9']['loss']) is not float:
                for target_name in self.metric_collections['mimiciv_icd9'][split_name].keys():
                    self.metric_collections['mimiciv_icd9'][split_name][target_name].update(outputs['icd9'])
            
            if type(outputs['icd10']['loss']) is not float:
                for target_name in self.metric_collections['mimiciv_icd10'][split_name].keys():
                    self.metric_collections['mimiciv_icd10'][split_name][target_name].update(outputs['icd10'])



    def calculate_metrics(
        self,
        split_name: str,
        version: Optional[str] = None,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        
        results_dict = defaultdict(dict)
        
        if split_name == "val":
            for target_name in self.metric_collections[version][split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[version][
                    split_name
                ][target_name].compute()
        else:
            for target_name in self.metric_collections[version][split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[version][
                    split_name
                ][target_name].compute(logits, targets)

        if self.threshold_tuning and split_name == "val":
            if version == 'mimiciv_icd9':
                best_result_icd9, best_db_icd9 = f1_score_db_tuning(logits, targets)
                results_dict[split_name]["all"] |= {"f1_micro_tuned": best_result_icd9}
                if evaluating_best_model:
                    pprint(f"Best threshold @{version}: {best_db_icd9}")
                    pprint(f"Best result @{version}: {best_result_icd9}")
                    for target_name in self.metric_collections[version]["test"]:
                        self.metric_collections[version]["test"][target_name].set_threshold(best_db_icd9)
                self.best_db_icd9 = best_db_icd9
            else:
                best_result_icd10, best_db_icd10 = f1_score_db_tuning(logits, targets)
                results_dict[split_name]["all"] |= {"f1_micro_tuned": best_result_icd10}
                if evaluating_best_model:
                    pprint(f"Best threshold @{version}: {best_db_icd10}")
                    pprint(f"Best result @{version}: {best_result_icd10}")
                    for target_name in self.metric_collections[version]["test"]:
                        self.metric_collections[version]["test"][target_name].set_threshold(best_db_icd10)
                self.best_db_icd10 = best_db_icd10
        return results_dict

    def reset_metric(self, split_name: str, version: str) -> None:
        for target_name in self.metric_collections[version][split_name].keys():
            self.metric_collections[version][split_name][target_name].reset_metrics()

    def reset_metrics(self) -> None:
        versions = ['mimiciv_icd9', 'mimiciv_icd10']
        for version in versions:
            for split_name in self.metric_collections[version].keys():
                for target_name in self.metric_collections[version][split_name].keys():
                    self.metric_collections[version][split_name][target_name].reset_metrics()
            for split_name in self.metric_collections[version].keys():
                for target_name in self.metric_collections[version][split_name].keys():
                    self.metric_collections[version][split_name][target_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict_icd9 = self.calculate_metrics(split_name="train", version='mimiciv_icd9')
        results_dict_icd10 = self.calculate_metrics(split_name="train", version='mimiciv_icd10')
        # print(results_dict_icd9)
        results_dict = {
            'mimiciv_icd9': results_dict_icd9,
            'mimiciv_icd10': results_dict_icd10,
        }
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        # print(results_dict)
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_train_end()

    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        version: str,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
        evaluating_best_model: bool = False
    ) -> None:
        results_dict = self.calculate_metrics(
            split_name=split_name,
            version=version,
            logits=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        # TODO : 겹치는 부분
        self.current_val_results[version] = results_dict
        self.log_dict(results_dict, epoch, version)
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model:
            self.save_predictions(
                split_name=split_name, logits=logits, targets=targets, ids=ids, version=version
            )

    def save_predictions(
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
        version: str = None
    ):
        from time import time

        tic = time()
        pprint(f"Saving predictions @ {self.experiment_path}")
        label_transform = self.dataloaders[version][split_name].dataset.label_transform[version]
        code_names = label_transform.get_classes()
        logits = logits.numpy()
        pprint("Building dataframe")
        df = pd.DataFrame(logits, columns=code_names)
        pprint("Adding targets")
        df[TARGET_COLUMN] = list(map(label_transform.inverse_transform, targets))
        pprint("Adding ids")
        df[ID_COLUMN] = ids.numpy()
        pprint("Saving dataframe")
        df.to_feather(self.experiment_path / f"predictions_{split_name}_{version}.feather")
        # df.to_csv(self.experiment_path / f"predictions_{split_name}_{version}.csv")
        pprint("Saved predictions in {:.2f} seconds".format(time() - tic))

    def on_epoch_begin(self) -> None:
        self.reset_metrics()
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):  
                # TODO : 여기도 기준 잡아야됨
                self.lr_scheduler.step(
                    self.current_val_results["val"]["all"]["f1_micro"]
                )

        for callback in self.callbacks:
            callback.on_epoch_end(self, epoch=self.epoch)

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int, version: Optional[str] = None
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch, version)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        
        for split_name in self.metric_collections['mimiciv_icd10'].keys():
            for target_name in self.metric_collections['mimiciv_icd10'][split_name].keys():
                self.metric_collections['mimiciv_icd10'][split_name][target_name].to(device)

        for split_name in self.metric_collections['mimiciv_icd9'].keys():
            for target_name in self.metric_collections['mimiciv_icd9'][split_name].keys():
                self.metric_collections['mimiciv_icd9'][split_name][target_name].to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db_icd9": self.best_db_icd9,
            "db_icd10": self.best_db_icd10,
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:
        checkpoint = torch.load(self.experiment_path / file_name)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db_icd9 = checkpoint["db_icd9"]
        self.best_db_icd10 = checkpoint["db_icd10"]
        pprint("Loaded checkpoint from {}".format(self.experiment_path / file_name))

    def save_transforms(self) -> None:
        """Save text tokenizer and label encoder"""
        self.dataloaders['mimiciv_icd9']["val"].dataset.text_transform.save(self.experiment_path)
        label_transforms = self.dataloaders['mimiciv_icd9']["val"].dataset.label_transform
        label_transforms['mimiciv_icd9'].save(self.experiment_path, 'mimiciv_icd9')
        label_transforms['mimiciv_icd10'].save(self.experiment_path, 'mimiciv_icd10')

    def save_final_model(self) -> None:
        self.save_checkpoint("final_model.pt")
        self.save_transforms()
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")
