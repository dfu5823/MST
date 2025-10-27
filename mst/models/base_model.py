from pathlib import Path
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, Accuracy, AUROC

class VeryBasicModel(pl.LightningModule):
    def __init__(self, save_hyperparameters=True):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

    def forward(self, x, cond=None):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError
    
    def _epoch_end(self, state:str):
        return 

    def training_step(self, batch: dict, batch_idx: int ):
        self._step_train += 1 
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val )

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None: 
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self, outputs) -> None:
        self._epoch_end("test")

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)  

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return self.load_weights(checkpoint["state_dict"], **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        flt = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if flt(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 


class BasicModel(VeryBasicModel):
    def __init__(
        self, 
        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-3, 'weight_decay':1e-2},
        lr_scheduler=None, 
        lr_scheduler_kwargs={},
        save_hyperparameters=True
    ):
        super().__init__(save_hyperparameters=save_hyperparameters)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            lr_scheduler_config  = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [lr_scheduler_config]
        else:
            return [optimizer]


class BasicClassifier(BasicModel):
    def __init__(
        self, 
        in_ch,
        out_ch,
        spatial_dims,
        loss=None,                        # <-- auto-select if None (BCE for binary, CE for multiclass)
        loss_kwargs={},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
        lr_scheduler=None, 
        lr_scheduler_kwargs={},
        aucroc_kwargs=None,               # <-- auto-select task if None
        acc_kwargs=None,                  # <-- auto-select task if None
        save_hyperparameters=True,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, save_hyperparameters)
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.spatial_dims = spatial_dims

        # -------- Loss selection --------
        if loss is None:
            if int(out_ch) == 1:
                self.loss_func = nn.BCEWithLogitsLoss(**loss_kwargs)
            else:
                self.loss_func = nn.CrossEntropyLoss(**loss_kwargs)
        else:
            self.loss_func = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs 

        # -------- Metrics (task-aware) --------
        if aucroc_kwargs is None:
            aucroc_kwargs = {}
        if acc_kwargs is None:
            acc_kwargs = {}

        if int(out_ch) == 1:
            # Binary classification
            auc_defaults = {"task": "binary"}
            acc_defaults = {"task": "binary"}
        else:
            # Multiclass classification
            auc_defaults = {"task": "multiclass", "num_classes": int(out_ch), "average": "macro"}
            acc_defaults = {"task": "multiclass", "num_classes": int(out_ch)}

        # User-provided kwargs override defaults where provided
        auc_defaults.update(aucroc_kwargs)
        acc_defaults.update(acc_kwargs)

        self.auc_roc = nn.ModuleDict({
            state: AUROC(**auc_defaults) for state in ["train_", "val_", "test_"]
        })  # 'train' not allowed as key
        self.acc = nn.ModuleDict({
            state: Accuracy(**acc_defaults) for state in ["train_", "val_", "test_"]
        })

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        target = batch['target']
        batch_size = target.shape[0]
        self.batch_size = batch_size 

        # Run Model (expects model(**batch) to return raw logits)
        pred = self(**batch)

        # ------------------------- Compute Loss ---------------------------
        logging_dict = {}
        logging_dict['loss'] = self.compute_loss(pred, target)

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            if int(self.out_ch) == 1:
                # Binary: logits shape [B] or [B,1]; targets {0,1}
                logits = pred.squeeze(-1) if pred.ndim == 2 and pred.shape[1] == 1 else pred
                targ = target
                if targ.ndim > 1:
                    targ = targ.squeeze(-1)
                # AUROC/Accuracy accept logits or probs for binary
                self.acc[state+"_"].update(logits, targ.int())
                self.auc_roc[state+"_"].update(logits, targ.int())
            else:
                # Multiclass: logits [B, C]; targets long class indices or one-hot
                targ = target
                if targ.ndim > 1 and targ.shape[-1] == int(self.out_ch):
                    targ = targ.argmax(dim=-1)
                # For AUROC multiclass, pass probabilities
                probs = F.softmax(pred, dim=1)
                self.acc[state+"_"].update(pred, targ)       # Accuracy can use logits
                self.auc_roc[state+"_"].update(probs, targ)  # AUROC expects probs for multiclass
            
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val, batch_size=batch_size, on_step=True, on_epoch=True, 
                         sync_dist=False)

        return logging_dict['loss'] 

    def _epoch_end(self, state):
        for name, value in [("ACC", self.acc[state+"_"]), ("AUC_ROC", self.auc_roc[state+"_"])]:
            self.log(f"{state}/{name}", value.compute(), batch_size=self.batch_size, on_step=False, on_epoch=True, 
                     sync_dist=True)
            value.reset()

    def compute_loss(self, pred, target):
        if int(self.out_ch) == 1:
            # BCE: ensure shapes align and dtype is float
            logits = pred
            targ = target
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # shape [B]
            if targ.ndim > 1:
                targ = targ.squeeze(-1)     # shape [B]
            targ = targ.float()
            return self.loss_func(logits, targ)
        else:
            # CE: target should be long class indices
            targ = target
            if targ.ndim > 1 and targ.shape[-1] == int(self.out_ch):
                targ = targ.argmax(dim=-1)
            targ = targ.long()
            return self.loss_func(pred, targ)
