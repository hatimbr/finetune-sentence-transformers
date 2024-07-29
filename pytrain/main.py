from .config import GlobalConfig
from .data import get_dataloader
from .track_prof import MlTrackContext
from .trainer import Trainer

from transformers import AutoTokenizer, AutoModel


def run():
    config = GlobalConfig()
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModel.from_pretrained(config.model_path).to("cuda")

    train_loader, valid_loader = get_dataloader(
        config.parquet_path, config.batch_size, tokenizer
    )

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        config.epochs,
        config.optimizer_config,
    )

    with MlTrackContext(config, track=config.track):
        model = trainer.train(
            dev_test=config.dev_test,
            track=config.track,
            profiler_config=config.profiler_config
        )


if __name__ == "__main__":
    run()
