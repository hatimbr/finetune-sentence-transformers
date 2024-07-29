import mlflow
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module, Softmax
from torch.profiler import profile
from torch.utils.data import DataLoader
from torchmetrics.regression import PearsonCorrCoef
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

from .config import OptimizerConfig, ProfilerConfig
from .optimizer import get_optimizer_scheduler
from .track_prof import TorchProfilerContext


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        optimizer_config: OptimizerConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            model,
            total_train_step=len(train_loader) * epochs,
            **optimizer_config.export()
        )
        self.step = 0

    def infer(self, model_inp: BatchEncoding) -> Tensor:
        model_inp = model_inp.to("cuda")
        embedded_sent = self.model(**model_inp)["pooler_output"]

        embedded_query = embedded_sent[::2]
        embedded_output = embedded_sent[1::2]
        similarities_matrix = embedded_query @ embedded_output.T
        return similarities_matrix

    def train_loop(
        self,
        dev_test: bool = False,
        track: bool = False,
        profiler: profile | None = None
    ) -> Tensor:
        self.model.train()
        list_loss = Tensor([]).to("cuda")
        loop = tqdm(self.train_loader, ascii=True)
        criterion = CrossEntropyLoss()
        labels = torch.arange(0, self.train_loader.batch_size, device="cuda")

        for i, model_inp in enumerate(loop):
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            similarities_matrix = self.infer(model_inp)

            loss_query = criterion(similarities_matrix, labels)
            loss_output = criterion(similarities_matrix.T, labels)
            loss = (loss_query + loss_output) / 2

            loss.backward()
            self.optimizer.step()

            list_loss = torch.cat((list_loss, loss.detach().data.view(1)))
            avg_loss = list_loss.mean().item()

            if track:
                mlflow.log_metrics(
                    {"loss": loss.item(), "avg_loss": avg_loss},
                    step=self.step,
                )
            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)
            self.step += 1

            if profiler is not None:
                profiler.step()

            if dev_test and i == 20:
                loop.close()
                print(
                    "Max memory allocated:",
                    torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
                )
                break

        return list_loss

    @torch.no_grad()
    def valid_loop(
        self, dev_test: bool = False, track: bool = False
    ) -> PearsonCorrCoef:
        self.model.eval()
        loop = tqdm(self.valid_loader, ascii=True)
        metric = PearsonCorrCoef().to("cuda")
        softmax = Softmax(dim=1)
        labels = torch.eye(self.valid_loader.batch_size, device="cuda").flatten()

        for i, model_inp in enumerate(loop):
            similarities_matrix = self.infer(model_inp)
            pred = softmax(similarities_matrix)
            score = metric(pred.flatten(), labels)

            loop.set_postfix(
                it_score=score.cpu().item(), avg_score=metric.compute().cpu().item()
            )

            if dev_test and i == 20:
                loop.close()
                break

        if track:
            mlflow.log_metric(
                "pearson_corr_coef", metric.compute().cpu().item(), step=self.step
            )

        return metric

    def train(
        self,
        epochs: int | None = None,
        dev_test: bool = False,
        track: bool = False,
        profiler_config: ProfilerConfig | None = None
    ) -> Module:
        if epochs is None:
            epochs = self.epochs

        metric = self.valid_loop(dev_test=dev_test, track=track)
        print(f"Initial f1 score: {metric.compute().cpu().item()}")

        for epoch in range(epochs):
            print(
                "*"*40, f"Epoch {epoch+1}/{epochs}", "*"*40
            )

            with TorchProfilerContext(
                **({} if profiler_config is None else profiler_config.export()),
            ) as profiler:
                list_loss = self.train_loop(
                    dev_test=dev_test, track=track, profiler=profiler
                )

            metric = self.valid_loop(dev_test=dev_test, track=track)

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"f1 score: {metric.compute().cpu().item()}"
            )

        return self.model
