#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger

import torch
from tqdm import trange
from transformers import AutoModel, TensorType

from backends import Backend, BackendConfig
from benchmark import Benchmark
from config import BenchmarkConfig
from utils import SEC_TO_NS_SCALE

try:
    import pcl_bert
    print("PCL_BERT is available")
except ImportError:
    pcl_bert = None
    print("PCL_BERT not found")

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    use_torchscript: bool = False

    @staticmethod
    def version() -> str:
        return torch.__version__


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        use_pcl = self.use_pcl if hasattr(self, 'use_pcl') else False
        if use_pcl:
            assert pcl_bert is not None, "pcl backend requested but pcl_bert module not found"
            with pcl_bert.pcl_impl(use_pcl):
                self.model = AutoModel.from_pretrained(model)
        else:
            self.model = AutoModel.from_pretrained(model)
        self.use_pcl = use_pcl

        LOGGER.info(f"Allocated PyTorch Backend for model: {model}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        backend = cls(config.model)
        backend.configure(config.backend)

        return backend

    def configure(self, config: PyTorchConfig):
        super().configure(config)

        LOGGER.info("Configuring PyTorch Benchmark:")

        # Disable gradients
        torch.set_grad_enabled(False)
        LOGGER.info("\t+ Disabled gradients")

        self.model.eval()
        LOGGER.info("\t+ Turning eval mode on Module (model.eval())")

        if config.num_threads is not None:
            if torch.get_num_threads() != config.num_threads:
                torch.set_num_threads(config.num_threads)

            LOGGER.info(f"\t+ Number of threads (torch.set_num_threads({config.num_threads}))")

        if config.num_interops_threads is not None:
            # TODO: Setting this value multiple times between PyTorch & TorchScript runs raise a C error

            if torch.get_num_interop_threads() != config.num_interops_threads:
                torch.set_num_interop_threads(config.num_interops_threads)

            LOGGER.info(
                f"\t+ Number of interop threads (torch.set_num_interop_threads({config.num_interops_threads}))"
            )

        if config.use_torchscript:
            self.model.config.return_dict = False
            LOGGER.info("\t+ Disabling dictionary output for TorchScript")

    def execute(self, config: BenchmarkConfig) -> Benchmark:
        if config.backend.use_torchscript:
            return self._run_torchscript(config)
        else:
            return self._run_pytorch(config)

    def _run_pytorch(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        if self.use_pcl:
            LOGGER.info("Running PyTorch PCL benchmark")
        else:
            LOGGER.info("Running PyTorch Eager benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.PYTORCH,
        )

        inputs = inputs.to(config.device)
        self.model = self.model.to(config.device)

        # Warmup
        for _ in trange(config.warmup_runs, desc="Warming up"):
            self.model(**inputs)

        # Run benchmark
        benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
        while sum(benchmark.latencies) < benchmark_duration_ns:
            with benchmark.track():
                self.model(**inputs)

        benchmark.finalize(benchmark_duration_ns)

        return benchmark

    def _run_torchscript(self, config: BenchmarkConfig) -> Benchmark:
        """
        :return:
        """
        LOGGER.info("Running TorchScript benchmark")
        benchmark = Benchmark()

        dummy_inputs = self._get_dummy_inputs(
            batch_size=config.batch_size,
            seq_len=(config.sequence_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        )

        inputs = self.tokenizer(
            dummy_inputs,
            is_split_into_words=True,
            return_tensors=TensorType.PYTORCH,
        )

        inputs.to(config.device)
        self.model = self.model.to(config.device)

        # To be sure inputs will be presented with the right prototype
        ordered_inputs = OrderedDict({
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "token_type_ids": inputs.token_type_ids,
        })

        LOGGER.debug("Calling torch JIT on model (optimize=True)")
        model_scripted = torch.jit.trace(self.model, tuple(ordered_inputs.values()))

        with torch.jit.optimized_execution(True):
            for _ in trange(config.warmup_runs, desc="Warming up"):
                model_scripted(*ordered_inputs.values())

            benchmark_duration_ns = config.benchmark_duration * SEC_TO_NS_SCALE
            while sum(benchmark.latencies) < benchmark_duration_ns:
                with benchmark.track():
                    model_scripted(*ordered_inputs.values())

            benchmark.finalize(benchmark_duration_ns)
        return benchmark

class PclBackend(PyTorchBackend):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        self.use_pcl = True
        super().__init__(model)

