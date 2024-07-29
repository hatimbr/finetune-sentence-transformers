from functools import partial
from math import cos, pi

from torch.nn import Module
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _linear_cosine_with_warmup(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    anneal_strategy: str,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    elif anneal_strategy == "linear":
        return max(
            0.0,
            (
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps))
            ),
        )

    elif anneal_strategy == "cosine":
        return max(
            0.0,
            (
                cos(
                    (
                        float(max(0, current_step - num_warmup_steps))
                        / float(max(1, num_training_steps - num_warmup_steps))
                    )
                    * pi
                )
                + 1
            )
            / 2,
        )

    else:
        raise ValueError(f"Unknown anneal strategy (lr scheduler): {anneal_strategy}")


def get_scheduler(
    optimizer: Optimizer,
    total_train_step: int,
    num_warmup_steps: int,
    lr_scheduler_name: str,
) -> LambdaLR:
    lr_lambda = partial(
        _linear_cosine_with_warmup,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_train_step,
        anneal_strategy=lr_scheduler_name,
    )
    return LambdaLR(optimizer, lr_lambda, -1)


def get_optimizer_scheduler(
    model: Module,
    optimizer_name: str,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    lr_scheduler_name: str | None,
    total_train_step: int,
    num_warmup_steps: int,
) -> tuple[Optimizer, LambdaLR | None]:
    match optimizer_name:
        case "adamw":
            optimizer = AdamW(
                params=model.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )
        case "sgd":
            optimizer = SGD(
                params=model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if lr_scheduler_name is not None:
        lr_scheduler = get_scheduler(
            optimizer, total_train_step, num_warmup_steps, lr_scheduler_name
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler
