import av
import torch
import numpy as np
from src.llamafactory.data.data_utils import resize_and_cut
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


import yaml

# Read config from yaml file
with open('examples/train_lora/llava1_5_lora_sft.yaml', 'r') as f:
    config = yaml.safe_load(f)
model_id = config['output_dir']

# model_id = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices, video_processor):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    kwargs = {"width_times": 3, "height_times": 2, "video_processor": video_processor}
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.extend(resize_and_cut(frame.to_ndarray(format="rgb24"), **kwargs))
            # frames.append(frame)
    return np.stack(frames)


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "# Answer the follwing questions.\nWhich track I'm on?\n# Answer the following questions from frame 2 to 4.\nWhat's the current speed shown in speedometer? Use km/h.\nWhat is the current speed based on the driving trajectory? Use km/h.\nWhich gear I'm currently at?\nWhich section I'm currently in the track?\nWhat's my normalized track position?\nWhat is my normalized distance from the left of the track?\nWhat's my current steering angle? Use degree measure, clockwise is positive."},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

video_path = '/home/weipeilun/playground/racing/racing_status_estimation/data_formatted/video/mugello_maclarn-mp4-12c/video_4155.mp4'
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, 1).astype(int)
clip = read_video_pyav(container, indices, processor.video_processor)
inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

output = model.generate(**inputs_video, max_new_tokens=1024, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
