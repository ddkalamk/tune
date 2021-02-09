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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Generic, TypeVar, ClassVar, List

from hydra.types import TargetConf
from omegaconf import MISSING
from psutil import cpu_count
from transformers import AutoTokenizer

from benchmark import Benchmark

LOGGER = getLogger("backends")


@dataclass
class BackendConfig(TargetConf):
    name: str = MISSING
    num_threads: int = cpu_count()
    num_interops_threads: int = cpu_count()


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, model: str):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    @classmethod
    @abstractmethod
    def allocate(cls, config: 'BenchmarkConfig'):
        raise NotImplementedError()

    @abstractmethod
    def configure(self, config: BackendConfigT):
        raise NotImplementedError()

    @abstractmethod
    def execute(self, config: 'BenchmarkConfig') -> Benchmark:
        raise NotImplementedError()

    def clean(self, config: 'BenchmarkConfig'):
        pass

    def _get_dummy_token(self) -> str:
        if self.tokenizer.pad_token is not None:
            return self.tokenizer.pad_token
        else:
            return self.tokenizer.convert_tokens_to_string([0])

    def _get_dummy_inputs(self, batch_size: int, seq_len: int) -> List[List[str]]:
        return [[self._get_dummy_token()] * seq_len] * batch_size

