from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import extract_vision_info
import yaml
import torch
import av
from io import BytesIO
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject
from typing import List

# Read config from yaml file
base_dir = "/home/weipeilun/playground/racing/LLaMA-Factory"
with open(f'{base_dir}/examples/train_lora/qwen2_vl.yaml', 'r') as f:
    config = yaml.safe_load(f)
model_id = config['output_dir']
min_pixels = config['min_pixels']
max_pixels = config['max_pixels']


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    f"{base_dir}/{model_id}/checkpoint-500",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(f"{base_dir}/{model_id}", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "resized_height": 1080,
                "resized_width": 1920,
                "video": "/home/weipeilun/playground/racing/racing_status_estimation/data_formatted_v2/video_2s/mugello-mclaren_mp412c/video_3013.mp4",
            },
            {"type": "text", "text": "# Answer the follwing questions.\nWhich track I'm on?\n# Answer the following questions from frame 10 to 19.\nWhat's the current speed shown in speedometer? Use km/h.\nWhat is the current speed based on the driving trajectory? Use km/h.\nWhich gear I'm currently at?\nWhich section I'm currently in the track?\nWhat's my normalized track position?\nWhat is my normalized distance from the left of the track?\nWhat's my current steering angle? Use degree measure, clockwise is positive."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

def _preprocess_image(image: "ImageObject", **kwargs) -> "ImageObject":
    resized_height: int = kwargs.get("resized_height", None)
    resized_width: int = kwargs.get("resized_width", None)
    if resized_height is not None and resized_width is not None:
        if image.width != resized_width or image.height != resized_height:
            image = image.resize((resized_width, resized_height), resample=Image.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

def _regularize_images(images, **kwargs) -> List["ImageObject"]:
    results = []
    for image in images:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])

        if not isinstance(image, ImageObject):
            raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

        results.append(_preprocess_image(image, **kwargs))

    return results


def fetch_video(vision_info):
    frames = []
    container = av.open(vision_info["video"])
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, 1).astype(int)
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            image = frame.to_image()
            frames.append(image)
    frames = _regularize_images(frames, **vision_info)
    return frames

def read_video_pyav(messages):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    vision_infos = extract_vision_info(messages)
    video_inputs = []
    for vision_info in vision_infos:
        if "video" in vision_info:
            vision_info["video"]
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("video should in content.")
        
    return None, video_inputs

image_inputs, video_inputs = read_video_pyav(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
