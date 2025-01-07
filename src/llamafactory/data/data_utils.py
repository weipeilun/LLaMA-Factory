# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from PIL import Image
import numpy as np
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset"]]


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    
    
def resize_and_cut(frame: np.ndarray, **kwargs):
    width_times = int(kwargs.get("width_times", 3))
    height_times = int(kwargs.get("height_times", 2))
    video_processor = kwargs.get("video_processor")
    width, height = kwargs.get("width", 336 * width_times), kwargs.get("height", 336 * height_times)
    frame = video_processor.resize(frame, {"shortest_edge": height})
    
    # padding the long edge
    if frame.shape[1] < width:
        new_image = np.zeros_like(frame, shape=(height, width, 3))

        # If the image is too small, pad it with zeros
        top_pad = math.ceil((height - frame.shape[0]) / 2)
        bottom_pad = top_pad + frame.shape[0]
        left_pad = math.ceil((width - frame.shape[1]) / 2)
        right_pad = left_pad + frame.shape[1]
        new_image[top_pad:bottom_pad, left_pad:right_pad, ...] = frame
        frame = new_image
        
    frames = []
    width_start = (frame.shape[1] - width) // 2
    height_start = (frame.shape[0] - height) // 2
    for height_idx in range(0, height_times):
        for width_idx in range(0, width_times):
            frame_crop = frame[height_start + height_idx * 336:height_start + height_idx * 336 + 336, width_start + width_idx * 336:width_start + width_idx * 336 + 336, :]
            # for debugging
            # frame_show = Image.fromarray(frame_crop)
            # frame_show.show(f'test_{width_idx}_{height_idx}')
            frames.append(frame_crop)
    return frames
