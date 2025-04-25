import os
from typing import Optional
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi import FastAPI, UploadFile, File, Form, status, Body, Path, Query
from pydantic import BaseModel, Field
from tools.error_define import BinaryDecodingError, FileConversionError, DataDistillationError
from server.router import openar1_router, ResponseModel, RequestModel, GenerationRequestModel
import pandas as pd
import uuid
import datasets
from datasets import Dataset, load_from_disk, load_dataset
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


def excel_2_arrow(data_path, distill_path):
    '''excel 转蒸馏需要的格式'''
    data = pd.read_excel(data_path)

    data_list = []
    for i in range(data.shape[0]):
        data_list.append({
            "problem": data.iloc[i, 0],
            "text": data.iloc[i, 1] if data.iloc[i, 1] is not None else "",
        })

    da = Dataset.from_list(data_list)
    da.save_to_disk(distill_path)
    # for i in range(da.shape[0]):
    #     print(da[i])


def build_distilabel_pipeline(
        model: str,
        base_url: str = "http://localhost:8000/v1",
        prompt_column: Optional[str] = None,
        prompt_template: str = "{{ instruction }}",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 8192,
        num_generations: int = 1,
        input_batch_size: int = 64,
        client_replicas: int = 1,
        timeout: int = 900,
        retries: int = 0,
) -> Pipeline:
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline().ray() as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key="something",
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=prompt_template,
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


def distill(args):
    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset...")
    try:
        dataset = datasets.load_from_disk(args.hf_dataset)
        if "train" in dataset:
            dataset = dataset["train"]

    except Exception as e:
        print(e)
        # dataset = load_dataset(r"F:\inspur\GPU\code\open-r1\open-r1\datasets\da-test", args.hf_dataset_config, split=args.hf_dataset_split)
        dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    print("Dataset loaded!")

    print("dataset:")
    print(dataset)

    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")

    # distilled to sft
    tmp = distiset["default"]["train"].map(qwen_template)

    #
    tmp.save_to_disk(args.hf_output_dataset)
    # tmp.save_to_disk(distiset_path=args.hf_output_dataset)

    try:
        dataset = load_from_disk(dataset_path=args.hf_output_dataset)
    except Exception as e:
        print(f"{e}")
        dataset = load_dataset(path=args.hf_output_dataset)


    print("dataset:")
    print(dataset)

    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.save_to_disk(distiset_path=args.hf_output_dataset)
        print(distiset)
        print("distilled data saved!")


def qwen_template(data):
    data["text"] = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
                   f"{data['problem']}<|im_end|>\n<|im_start|>assistant\n{data['generation']}<|im_end|>"
    return data


'''蒸馏数据'''
@openar1_router.post(path="/distill_data", response_model=ResponseModel, tags=["蒸馏数据"])
async def distill_data(
        file: UploadFile = File(description="一个二进制文件"),
        hf_dataset: str = Form("1.txt", example=""),
        # hf_dataset_config: str = Form("", example=""),+
        # hf_dataset_split: str = Form("train", example=""),
        prompt_column: str = Form("problem", example=""),
        prompt_template: str = Form("{{ instruction }}", example=""),
        model: str = Form("", example=""),
        vllm_server_url: str = Form("http://localhost:8000/v1", example=""),
        temperature: float = Form(0.1, example=0.1),
        top_p: float = Form(0.9, example=0.9),
        max_new_tokens: int = Form(8192, example=8192),
        num_generations: int = Form(1, example=1),
        input_batch_size: int = Form(64, example=64),
        # client_replicas: int = Field(1, example=1),
        timeout: int = Form(600, example=600),
        # retries: int = Field(0, example=0),
        hf_output_dataset: str = Form("", example="")
):
    # 验证文件
    args = GenerationRequestModel()
    args.file = file
    args.hf_dataset = hf_dataset
    # args.hf_dataset_config = ""
    # args.hf_dataset_split = "train"
    args.prompt_column = prompt_column
    args.prompt_template = prompt_template
    args.model = model
    args.vllm_server_url = vllm_server_url
    args.temperature = temperature
    args.top_p = top_p
    args.max_new_tokens = max_new_tokens
    args.num_generations = num_generations
    args.input_batch_size = input_batch_size
    # args.client_replicas = 1
    args.timeout = timeout
    # args.retries = 0
    args.hf_output_dataset = hf_output_dataset

    uid = uuid.uuid4()

    # 将文件保存
    dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{uid}"
    distill_data_path = dir_path
    distilled_data_path = os.path.join(dir_path, "distilled")
    tmp_path = os.path.join(dir_path, str("tmp_" + args.file.filename))
    if not os.path.exists(distill_data_path):
        os.makedirs(distill_data_path)
    elif not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    try:
        file_content = await args.file.read()  # 读取上传文件的内容
        with open(tmp_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        raise BinaryDecodingError(e)

    try:
        # 转为蒸馏前格式
        excel_2_arrow(tmp_path, distill_data_path)
    except Exception as e:
        raise FileConversionError(e)

    try:
        # 蒸馏数据
        args.hf_dataset = distill_data_path
        args.hf_output_dataset = distilled_data_path
        distill(args)
    except Exception as e:
        raise DataDistillationError(e)

    # try:
    #     # 读取蒸馏后数据，并返回
    #     try:
    #         dataset = load_from_disk(dataset_path=os.path.join(distilled_data_path, "default", "train"))
    #     except Exception as e:
    #         print(f"{e}")
    #         dataset = load_dataset(path=distilled_data_path)
    #
    #     for i in range(dataset.shape[0]):
    #         if args.template == "qwen":
    #             dataset_map = dataset.map(qwen_template)
    #     print(dataset_map)
    # except Exception as e:
    #     raise BinaryDecodingError(e)

    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"uid": str(uid), "distilled_data_path":
        distilled_data_path}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


'''使用蒸馏数据 SFT'''


@openar1_router.post(path="/sft_distilled", response_model=ResponseModel, tags=["使用蒸馏数据 SFT"])
async def sft_distilled(
        accelerate_configs_file: UploadFile = File(description="accelerate 配置文件"),
        sft_configs_file: UploadFile = File(description="sft 配置文件，包涵相关微调参数"),
        uid: str = Form(description="生成蒸馏数据返回的uid"),
):
    try:
        dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{uid}"
        accelerate_configs_file_content = await accelerate_configs_file.read()  # 读取上传文件的内容
        sft_configs_file_content = await sft_configs_file.read()  # 读取上传文件的内容
        with open(os.path.join(dir_path, "accelerate_configs.yaml"), "wb") as f:
            f.write(accelerate_configs_file_content)
        with open(os.path.join(dir_path, "sft_configs.yaml"), "wb") as f:
            f.write(sft_configs_file_content)

        # 更新数据路径
        import yaml
        with open(os.path.join(dir_path, "sft_configs.yaml"), 'r') as f:
            data = yaml.safe_load(f)
            data["dataset_name"] = os.path.join(dir_path, "distilled/default/train")

        with open(os.path.join(dir_path, "sft_configs.yaml"), 'w', encoding='utf-8') as f:
            # 使用 yaml.dump() 方法将字典 d 转换为 YAML 格式，并将其写入文件中
            f.write(yaml.dump(data))

    except Exception as e:
        raise BinaryDecodingError(e)

    # 更新数据路径
    system_cmd_str = f'accelerate launch --config_file {str(os.path.join(dir_path, "accelerate_configs.yaml"))} {os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "third_party_tools", "open-r1/src/open_r1/sft.py")} --config {str(os.path.join(dir_path, "sft_configs.yaml"))}'
    print("system_cmd_str:", system_cmd_str)

    import subprocess
    commands = [
        [system_cmd_str]
        # ["accelerate", "launch", "--config_file",
        #  f"{os.path.join(dir_path, 'accelerate_configs.yaml')}",
        #  "src/open_r1/sft.py",
        #  "--config", f'{os.path.join(dir_path, "sft_configs.yaml")}']
    ]
    for cmd in commands:
        result = subprocess.run(
            cmd,
            capture_output=True,  # 捕获输出
            text=True,  # 返回字符串（而非字节）
            shell=True  # 在 Windows 下可能需要
        )
        print(f"命令: {' '.join(cmd)}")
        print("退出码:", result.returncode)
        print("输出:\n", result.stdout)  # 标准输出
        print("错误:\n", result.stderr)  # 标准错误

    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": ""}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)
