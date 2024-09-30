import click
import time
from typing import List, Optional, Dict, Any

from rayllm_batch import RayLLMBatch, init_engine_from_config
from rayllm_batch import CNNDailySummary, SyntheticWithSharedPrefix

WORKLOAD_METADATA = {}


def register_workload(workload_cls, **default_configs):
    name = workload_cls.__name__
    if name in WORKLOAD_METADATA:
        raise ValueError(f"Workload {name} is already registered")
    WORKLOAD_METADATA[workload_cls.__name__] = {
        "cls": workload_cls,
        "default_configs": default_configs,
    }


def get_workload_names():
    return list(WORKLOAD_METADATA.keys())


def parse_workload_args(workload_arg: List[str]):
    workload_args = {}
    for arg in workload_arg:
        key, value = [t.strip() for t in arg.split("=")]
        try:
            workload_args[key] = float(value)
        except ValueError:
            workload_args[key] = value
    return workload_args


def init_workload(workload_name: str, workload_arg: List[str]):
    if workload_name not in WORKLOAD_METADATA:
        raise ValueError(f"Unknown workload name: {workload_name}")

    workload_metadata = WORKLOAD_METADATA[workload_name]
    workload_cls = workload_metadata["cls"]
    workload_kwargs: Dict[str, Any] = workload_metadata["default_configs"].copy()
    workload_kwargs.update(parse_workload_args(workload_arg))
    return workload_cls(**workload_kwargs)


register_workload(CNNDailySummary, dataset_fraction=0.004)
register_workload(
    SyntheticWithSharedPrefix,
    num_synthetic_requests=1000,
    num_synthetic_prefixes=1,
    num_synthetic_prefix_tokens=0,
    num_unique_synthetic_prompt_tokens=2000,
    max_tokens=100,
)


def main(
    engine_cfg_file: str, workload_name: str, workload_arg: Optional[List[str]] = None,
    num_replicas: int = 1
):
    engine = init_engine_from_config(engine_cfg_file)
    workload = init_workload(workload_name, workload_arg or [])

    batch = RayLLMBatch(
        engine,
        workload,
        batch_size=None,
        num_replicas=num_replicas,
    )
    start = time.perf_counter()
    ds = batch.run()
    elapsed_time = time.perf_counter() - start
    total_tokens = ds.sum("num_input_tokens") + ds.sum("num_generated_tokens")
    engine_time = batch.get_avg_engine_time_per_replica()
    engine_thrpt = total_tokens / engine_time
    proj_1m_time = batch.project_1m_token_time(total_tokens, elapsed_time, engine_time)
    num_input_tokens = ds.sum("num_input_tokens")
    num_generated_tokens = ds.sum("num_generated_tokens")

    print(f"Total elapsed time: {elapsed_time:.2f}s")
    print(f"Total tokens processed: {total_tokens}")
    print(f'Total input tokens: {num_input_tokens}')
    print(f'Total generated tokens: {num_generated_tokens}')
    print(f"Engine throughput (tokens/s): {engine_thrpt:.2f}")
    print(f"1M token projection time: {proj_1m_time:.2f}s")



@click.command()
@click.option(
    "--engine-cfg-file",
    type=str,
    required=True,
    help="Path to the engine config file",
)
@click.option(
    "--workload-name",
    type=click.Choice(get_workload_names()),
    required=True,
    help="Name of the workload to run",
)
@click.option(
    "--workload-arg",
    "-w",
    type=str,
    multiple=True,
    help="Override default arguments for the workload",
)
@click.option(
    "--num-replicas",
    type=int,
    required=False,
    default=1,
    help="Number of replicas to run the workload."
)
def cli(engine_cfg_file: str, workload_name: str, workload_arg: List[str], num_replicas: int):
    main(engine_cfg_file, workload_name, workload_arg, num_replicas)

if __name__ == "__main__":
    cli()
