from pathlib import Path

from mlflow import (
    ActiveRun,
    log_params,
    search_runs,
    set_experiment,
    set_tracking_uri,
    start_run,
)
from torch import profiler

from .config import GlobalConfig


class MlTrackContext:
    """Wrapper for mlflow tracking context manager."""

    def __init__(self, config: GlobalConfig, track: bool = False):
        self.config = config
        self.track = track

    def __enter__(self) -> ActiveRun | None:
        if self.track:
            set_tracking_uri(self.config.mlflow_dir)
            set_experiment(self.config.experiment_name)

            if search_runs(
                filter_string=f"run_name='{self.config.run_name}'"
            ).empty:
                run_id = None
            else:
                run_id = search_runs(
                    filter_string=f"run_name='{self.config.run_name}'"
                ).iloc[0]["run_id"]

            self.run = start_run(
                run_name=self.config.run_name, run_id=run_id, log_system_metrics=True
            )
            self.run.__enter__()
            log_params(self.config.export())
            return self.run

        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.track:
            self.run.__exit__(exc_type, exc_val, exc_tb)
        else:
            return exc_type is None


class TorchProfilerContext:
    """Wrapper for PyTorch profiler context manager."""

    def __init__(
        self,
        profiler_path: Path = Path().cwd() / "profiler",
        profile: bool = False,
        profiler_schedule_wait: int = 1,
        profiler_schedule_warmup: int = 1,
        profiler_schedule_active: int = 5,
        profiler_schedule_repeat: int = 1,
        profiler_profile_memory: bool = True,
        profiler_with_stack: bool = False,
        profiler_record_shapes: bool = False,
        profiler_with_flops: bool = False,
        profiler_with_modules: bool = False,
    ):
        self.profiler_path = profiler_path
        self.profile = profile
        self.profiler_schedule_wait = profiler_schedule_wait
        self.profiler_schedule_warmup = profiler_schedule_warmup
        self.profiler_schedule_active = profiler_schedule_active
        self.profiler_schedule_repeat = profiler_schedule_repeat
        self.profiler_profile_memory = profiler_profile_memory
        self.profiler_with_stack = profiler_with_stack
        self.profiler_record_shapes = profiler_record_shapes
        self.profiler_with_flops = profiler_with_flops
        self.profiler_with_modules = profiler_with_modules

    def __enter__(self) -> profiler.profile | None:
        if self.profile:
            self.profiler = profiler.profile(
                schedule=profiler.schedule(
                    wait=self.profiler_schedule_wait,
                    warmup=self.profiler_schedule_warmup,
                    active=self.profiler_schedule_active,
                    repeat=self.profiler_schedule_repeat,
                ),
                on_trace_ready=profiler.tensorboard_trace_handler(
                    str(self.profiler_path)
                ),
                profile_memory=self.profiler_profile_memory,
                with_stack=self.profiler_with_stack,
                record_shapes=self.profiler_record_shapes,
                with_flops=self.profiler_with_flops,
                with_modules=self.profiler_with_modules
            )
            self.profiler.__enter__()
            return self.profiler
        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
