from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field


class RequestModel(BaseModel):
    text: str = Field("", example="")

class GenerationRequestModel(BaseModel):
    file: UploadFile = File(description="一个二进制文件"),
    task_id: str = Field(description="Field"),
    hf_dataset: str = Field("123", example="")
    hf_dataset_config: str = Field("", example="")
    hf_dataset_split: str = Field("train", example="")
    prompt_column: str = Field("prompt", example="")
    prompt_template: str = Field("{{ instruction }}", example="")
    model: str = Field("", example="")
    vllm_server_url: str = Field("http://localhost:8000/v1", example="")
    api_key: str = Field("")
    template: str = Field("qwen", example="qwen")
    temperature: float = Field(0.1, example=0.1)
    top_p: float = Field(0.9, example=0.9)
    max_new_tokens: int = Field(8192, example=8192)
    num_generations: int = Field(1, example=1)
    input_batch_size: int = Field(64, example=64)
    client_replicas: int = Field(1, example=1)
    timeout: int = Field(600, example=600)
    retries: int = Field(0, example=0)
    hf_output_dataset: str = Field("", example="")
    split_train_test: float = Field(1, example=0.9)


class ResponseModel(BaseModel):
    isSuc: bool = Field(True, example=True)
    code: int = Field(0, example=0)
    msg: str = Field("Succeed~", example="Succeed~")
    res: dict = Field({}, example={})





# 路由
utils_router = APIRouter()
openr1_router = APIRouter()

# utils_api
from server.utils_api import upload_file
from server.openr1_api import async_sft_distilled, async_grpo, async_distill_data
# from server.distill_api import distill_data, sft_distilled
