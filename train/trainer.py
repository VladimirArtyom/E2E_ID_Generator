import torch
from numpy import sum
from metrics_calculator import MetricsCalc, MetricsResult, MetricEnum
from torch.optim import AdamW
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, precision_score, recall_score)
from typing import Mapping, List

class Trainer:
    def __init__(
        this, device: str,
        epochs: int,
        learning_rate: float,
        tokenizer: AutoTokenizer,
        model: torch.nn.Module,
        optimizer: AdamW, 
        save_dir: str,

        train_batch_size: int,
        train_set: Dataset,
        validation_batch_size: int,
        validation_set: Dataset,
        test_batch_size: int,
        test_set: Dataset,

        workers: int, 
        pin_memory: bool,
        metrics_evaluation: bool,
        do_validate: bool = False,
        ) -> None:
        this.device = device
        this.epochs = epochs
        this.learning_rate = learning_rate
        this.tokenizer: AutoTokenizer = tokenizer
        this.model = model.to(this.device)
        this.optimizer = optimizer
        this.save_dir = save_dir

        this.train_batch_size = train_batch_size
        this.validation_batch_size = validation_batch_size
        this.test_batch_size = test_batch_size
        this.metrics_evaluation = metrics_evaluation

        this.train_data_loader = DataLoader(train_set,
                                            batch_size=this.train_batch_size,
                                            num_workers=workers,
                                            pin_memory=pin_memory,
                                            shuffle=True)

        this.validation_data_loader = DataLoader(validation_set,
                                                 batch_size=this.train_batch_size,
                                                 num_workers=workers,
                                                 pin_memory=pin_memory,
                                                 shuffle=False)

        this.test_data_loader = DataLoader(test_set,
                                           batch_size=this.test_batch_size,
                                           num_workers=workers,
                                           pin_memory=pin_memory,
                                           shuffle=False)
        this.train_metrics: List[MetricsResult] = []
        this.train_loss = MetricsCalc(name="loss_train")
        this.validation_loss = MetricsCalc(name="validation_loss")
        this.do_validate = do_validate

        if metrics_evaluation:
            this.best_valid_score: float = 0
            this.best_loss_score: float = 2**31 - 1
        else:
            this.best_valid_score = float("inf")
            this.best_loss_score = float("inf")

    def train(this):
        for epoch in range(1, this.epochs + 1):
            this.model.train()
            this.train_loss.reset()
            with tqdm(total=len(this.train_data_loader), unit="batches") as tq:
                tq.set_description(f"epoch {epoch}")

                for batch in this.train_data_loader:
                    this.optimizer.zero_grad()
                    data = {key: value.to(this.device) for key, value in batch.items()}
                    output = this.model(**data)
                    loss = output.loss
                    loss.backward()
                    this.optimizer.step()
                    this.train_loss.update(loss.item(), this.train_batch_size)
                    tq.set_postfix({"train_loss": this.train_loss.avg})
                    tq.update(1)
                this._save()
            if this.do_validate:
                if this.metrics_evaluation:
                    print("Train Loss : ", loss.item())
                    valid_metrics = this._single_value_metrics(this.validate_metrics(this.validation_data_loader, this.validation_batch_size))
                    if valid_metrics[MetricEnum.BAS.name] > this.best_valid_score:
                        print(f"The best validation improved from {this.best_valid_score} to {valid_metrics[MetricEnum.BAS.name]}")
                        this.best_valid_score = valid_metrics[MetricEnum.BAS.name]
                        this._save()
                    this._show_metrics_results(valid_metrics)
                else:
                    valid_loss = this.validate(this.validation_data_loader)
                    if this.best_loss_score > valid_loss:
                        print(f"The best loss decreased from {this.best_loss_score} to {valid_loss}")
                        this.best_loss_score = valid_loss
                        this._save()
                    print("Train Loss : ", valid_loss)
            else:
                print("Skipping the validation part, hemat duit cok")

    def validate(this, dataloader: DataLoader):
        this.model.eval()
        losses : List[float] = []
        with tqdm(total=len(dataloader), unit="batches") as tq:
            tq.set_description("validation")
            for batch in dataloader:
                data = {key: value.to(this.device) for key, value in batch.items()}
                output = this.model(**data)
                loss: Tensor = output.loss
                this.validation_loss.update(loss.item(), this.validation_batch_size)
                tq.set_postfix({"valid_loss": this.validation_loss.avg})
                tq.update(1)
                losses.append(this.validation_loss.avg)
        return sum(losses) / len(losses)


    def _save(this) -> None:
        this.tokenizer.save_pretrained(this.save_dir)
        this.model.save_pretrained(this.save_dir)


    def _show_metrics_results(this, out: Mapping[str, float]) -> None:
        print("Validation Loss : ",out[MetricEnum.LOSS.name])
        print("Validation Accuracy : ",out[MetricEnum.ACCURACY.name])
        print("Validation F1 : ", out[MetricEnum.F1.name])
        print("Validation BAS : ", out[MetricEnum.BAS.name])
        print("Vaidation Matthew: ", out[MetricEnum.MATTHEW.name])
        print("Validation Recall: ", out[MetricEnum.RECALL.name]) 
        print("Validation Precision: ",out[MetricEnum.PRECISION.name])


    def _single_value_metrics(this,
                              listOfMetrics: List[MetricsResult])-> Mapping[str, float]:
        out: Mapping[str, float] = {
            MetricEnum.LOSS.name: 0,
            MetricEnum.ACCURACY.name: 0,
            MetricEnum.F1.name: 0,
            MetricEnum.BAS.name: 0,
            MetricEnum.MATTHEW.name: 0,
            MetricEnum.RECALL.name: 0,
            MetricEnum.PRECISION.name: 0,
        }

        n = len(listOfMetrics)

        for la in listOfMetrics:
            acc = la.get_update_avg(MetricEnum.ACCURACY.name)   
            f1 = la.get_update_avg(MetricEnum.F1.name)
            bas = la.get_update_avg(MetricEnum.BAS.name)
            matthew = la.get_update_avg(MetricEnum.MATTHEW.name)
            recall = la.get_update_avg(MetricEnum.RECALL.name)
            prec = la.get_update_avg(MetricEnum.PRECISION.name)
            loss = la.get_update_avg(MetricEnum.LOSS.name)


            out[MetricEnum.LOSS.name] += loss
            out[MetricEnum.ACCURACY.name] += acc
            out[MetricEnum.F1.name] += f1
            out[MetricEnum.BAS.name] += bas
            out[MetricEnum.MATTHEW.name] += matthew
            out[MetricEnum.RECALL.name] += recall
            out[MetricEnum.PRECISION.name] += prec

        out[MetricEnum.LOSS.name] = out[MetricEnum.LOSS.name] / n
        out[MetricEnum.ACCURACY.name] = out[MetricEnum.ACCURACY.name] / n
        out[MetricEnum.F1.name] = out[MetricEnum.F1.name] / n
        out[MetricEnum.BAS.name] = out[MetricEnum.BAS.name] / n
        out[MetricEnum.MATTHEW.name] = out[MetricEnum.MATTHEW.name] / n
        out[MetricEnum.RECALL.name] = out[MetricEnum.RECALL.name] / n
        out[MetricEnum.PRECISION.name] = out[MetricEnum.PRECISION.name] / n

        return out

    def _calculate_metrics( this,
                           true: torch.Tensor,
                           pred: torch.Tensor,
                           ) -> Mapping[str, float]:
        acc_score = accuracy_score(true, pred)
        f1 = f1_score(true, pred)
        bas = balanced_accuracy_score(true, pred)
        matthew = matthews_corrcoef(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)

        return {
            MetricEnum.ACCURACY.name: acc_score,
            MetricEnum.F1.name: f1,
            MetricEnum.BAS.name: bas,
            MetricEnum.MATTHEW.name: matthew,
            MetricEnum.RECALL.name: recall,
            MetricEnum.PRECISION.name: precision
        }

    @torch.no_grad()
    def validate_metrics(this,
                         dataloader: DataLoader,
                         batch_size: int) -> List[MetricsResult]:
        this.model.eval()
        validation_metrics: List[MetricsResult] = []
        with tqdm(total=len(dataloader), unit="batches") as tq:
            tq.set_description(f"validation_result")
            for batch in dataloader:
                data = {key: value.to(this.device) for key, value in batch.items()}
                output = this.model(**data)
                preds = torch.argmax(output.logits, dim=1)
                scores = this._calculate_metrics(
                    data["labels"].cpu(),
                    preds.cpu()
                )
                scores[MetricEnum.LOSS.name] = output.loss.item()
                mcResult = MetricsResult()
                mcResult.update(scores, batch_size)
                validation_metrics.append(mcResult)
                """
                tq.set_postfix({
                      "loss": mcResult.get_update_avg(MetricEnum.LOSS.name),
                      "accuracy": mcResult.get_update_avg(MetricEnum.ACCURACY.name),
                      "f1": mcResult.get_update_avg(MetricEnum.F1.name),
                      "bas": mcResult.get_update_avg(MetricEnum.BAS.name),
                      "matthew": mcResult.get_update_avg(MetricEnum.MATTHEW.name),
                      "recall": mcResult.get_update_avg(MetricEnum.RECALL.name),
                      "precision": mcResult.get_update_avg(MetricEnum.PRECISION.name)
                      })
                """
                tq.update(1)
        return validation_metrics


