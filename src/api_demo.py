# coding=utf-8
# Implements API for fine-tuned models.
# Usage: python api_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint


import json
import time
import torch
import uvicorn
from fastapi import FastAPI
from threading import Thread
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer
from starlette.responses import StreamingResponse
from typing import Any, Dict, List, Literal, Optional, Union

from utils import (
    Template,
    load_pretrained,
    prepare_infer_args,
    get_logits_processor
)


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer, source_prefix

    query = request.messages[-1].content
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        source_prefix = prev_messages.pop(0).content

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    inputs = tokenizer([prompt_template.get_prompt(query, history, source_prefix)], return_tensors="pt")
    inputs = inputs.to(model.device)

    gen_kwargs = generating_args.to_dict()
    gen_kwargs.update({
        "input_ids": inputs["input_ids"],
        "temperature": request.temperature if request.temperature else gen_kwargs["temperature"],
        "top_p": request.top_p if request.top_p else gen_kwargs["top_p"],
        "max_new_tokens": request.max_new_tokens if request.max_new_tokens else gen_kwargs["max_new_tokens"],
        "logits_processor": get_logits_processor()
    })

    if request.stream:
        generate = predict(gen_kwargs, request.model)
        return StreamingResponse(generate, media_type="text/event-stream")

    generation_output = model.generate(**gen_kwargs)
    outputs = generation_output.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(gen_kwargs: Dict[str, Any], model_id: str):
    global model, tokenizer

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object = "chat.completion.chunk")
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    for new_text in streamer:
        if len(new_text) == 0:
            continue

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object = "chat.completion.chunk")
        yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object = "chat.completion.chunk")
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))


if __name__ == "__main__":
    model_args, data_args, finetuning_args, generating_args = prepare_infer_args()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    prompt_template = Template(data_args.prompt_template)
    source_prefix = data_args.source_prefix if data_args.source_prefix else ""

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
