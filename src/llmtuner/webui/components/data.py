import json
import os
from typing import TYPE_CHECKING, Dict, Tuple

import gradio as gr

from ...extras.constants import DATA_CONFIG


if TYPE_CHECKING:
    from gradio.components import Component


PAGE_SIZE = 2


def prev_page(page_index: int) -> int:
    return page_index - 1 if page_index > 0 else page_index


def next_page(page_index: int, total_num: int) -> int:
    return page_index + 1 if (page_index + 1) * PAGE_SIZE < total_num else page_index


def can_preview(dataset_dir: str, dataset: list) -> "gr.Button":
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception:
        return gr.Button(interactive=False)

    if len(dataset) == 0 or "file_name" not in dataset_info[dataset[0]]:
        return gr.Button(interactive=False)

    local_path = os.path.join(dataset_dir, dataset_info[dataset[0]]["file_name"])
    if (os.path.isfile(local_path)
            or (os.path.isdir(local_path) and len(os.listdir(local_path)) != 0)):
        return gr.Button(interactive=True)
    else:
        return gr.Button(interactive=False)


def load_single_data(data_file_path):
    with open(os.path.join(data_file_path), "r", encoding="utf-8") as f:
        if data_file_path.endswith(".json"):
            data = json.load(f)
        elif data_file_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            data = [line for line in f]  # noqa: C416
        return data


def get_preview(dataset_dir: str, dataset: list, page_index: int) -> Tuple[int, list, "gr.Column"]:
    with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    data_file: str = dataset_info[dataset[0]]["file_name"]
    local_path = os.path.join(dataset_dir, data_file)
    if os.path.isdir(local_path):
        data = []
        for file_name in os.listdir(local_path):
            data.extend(load_single_data(os.path.join(local_path, file_name)))
    else:
        data = load_single_data(local_path)

    return len(data), data[PAGE_SIZE * page_index: PAGE_SIZE * (page_index + 1)], gr.Column(visible=True)


def create_preview_box(dataset_dir: "gr.Textbox", dataset: "gr.Dropdown") -> Dict[str, "Component"]:
    data_preview_btn = gr.Button(interactive=False, scale=1)
    with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0)
            page_index = gr.Number(value=0, interactive=False, precision=0)

        with gr.Row():
            prev_btn = gr.Button()
            next_btn = gr.Button()
            close_btn = gr.Button()

        with gr.Row():
            preview_samples = gr.JSON()

    dataset.change(can_preview, [dataset_dir, dataset], [data_preview_btn], queue=False).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    data_preview_btn.click(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    prev_btn.click(prev_page, [page_index], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    next_btn.click(next_page, [page_index, preview_count], [page_index], queue=False).then(
        get_preview, [dataset_dir, dataset, page_index], [preview_count, preview_samples, preview_box], queue=False
    )
    close_btn.click(lambda: gr.Column(visible=False), outputs=[preview_box], queue=False)
    return dict(
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        preview_samples=preview_samples,
    )
