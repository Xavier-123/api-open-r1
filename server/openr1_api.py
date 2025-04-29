import json
import os

import requests
import yaml
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form, status
from multiprocessing import Process, Event
from tools.error_define import BinaryDecodingError, FileConversionError, DataDistillationError
from tools.distill import openR1_distill, excel_2_arrow, distill, distilled_sft, _grpo
from tools.log import logger
from server.router import openr1_router, ResponseModel, GenerationRequestModel
import uuid
import time
from threading import Thread, Event

task_dict = {}
sft_task_dict = {}
grpo_task_dict = {}
threads_list = []

'''蒸馏数据'''


@openr1_router.post(path="/async_distill_data", response_model=ResponseModel, tags=["蒸馏数据"])
async def async_distill_data(
        file: UploadFile = File(description="一个二进制文件"),
        task_id: str = Form("1234567890"),
        hf_dataset: str = Form("1.txt", example=""),
        prompt_column: str = Form("problem", example=""),
        prompt_template: str = Form("{{ instruction }}", example=""),
        model: str = Form("", example=""),
        vllm_server_url: str = Form("http://localhost:8000/v1", example=""),
        temperature: float = Form(0.1, example=0.1),
        top_p: float = Form(0.9, example=0.9),
        max_new_tokens: int = Form(8192, example=8192),
        num_generations: int = Form(1, example=1),
        input_batch_size: int = Form(64, example=64),
        timeout: int = Form(600, example=600),
        hf_output_dataset: str = Form("", example="")
):
    '''异步接口'''
    # 验证文件
    args = GenerationRequestModel()
    args.file = file
    args.task_id = task_id
    args.hf_dataset = hf_dataset
    args.prompt_column = prompt_column
    args.prompt_template = prompt_template
    args.model = model
    args.vllm_server_url = vllm_server_url
    args.temperature = temperature
    args.top_p = top_p
    args.max_new_tokens = max_new_tokens
    args.num_generations = num_generations
    args.input_batch_size = input_batch_size
    args.timeout = timeout
    args.hf_output_dataset = hf_output_dataset

    # uid = uuid.uuid4()

    task_id = args.task_id
    print("task_id:", task_id)
    if task_id == "" or task_id is None:
        task_id = uuid.uuid4()

    # 判断task_id是否存在
    if task_id in task_dict:
        content = {"isSuc": False, "code": 0, "msg": f"task_id {task_id} is exist in task", "res": {}}
        return JSONResponse(status_code=status.HTTP_200_OK, content=content)

    try:
        # 将文件保存
        dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{task_id}"
        distill_data_path = dir_path
        distilled_data_path = os.path.join(dir_path, "distilled")
        tmp_path = os.path.join(dir_path, str("tmp_" + args.file.filename))
        if not os.path.exists(distill_data_path):
            os.makedirs(distill_data_path)
        elif not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        # 将数据保存到本地
        try:
            file_content = await args.file.read()  # 读取上传文件的内容
            with open(tmp_path, "wb") as f:
                f.write(file_content)
            logger.info("data save success.")
        except Exception as e:
            del task_dict[args.task_id]
            os.remove(distill_data_path)
            os.remove(tmp_path)
            raise BinaryDecodingError(e)

        # 调用 distill 获取
        task_dict[task_id] = {"time": time.time(), "code": 2, "msg": ""}
        # thr = Thread(target=openR1_distill, args=(args, tmp_path, distill_data_path, distilled_data_path, task_dict))
        thr = Process(target=openR1_distill, args=(args, tmp_path, distill_data_path, distilled_data_path, task_dict))
        thr.daemon = True
        threads_list.append(thr)
        thr.start()

        content = {"isSuc": True, "code": 2, "msg": "任务已开启~", "res": {"task_id": task_id}}
        logger.info(f">>> task_id:{task_id}, response:{content}")

        return JSONResponse(status_code=status.HTTP_200_OK, content=content)

    except Exception as e:
        content = {"isSuc": False, "code": -1, "msg": str(e), "res": {}}
        return JSONResponse(status_code=status.HTTP_200_OK, content=content)


# '''使用蒸馏数据 SFT'''
# @openr1_router.post(path="/sft_distilled", response_model=ResponseModel, tags=["使用蒸馏数据 SFT"])
# async def sft_distilled(
#         accelerate_configs_file: UploadFile = File(description="accelerate 配置文件"),
#         sft_configs_file: UploadFile = File(description="sft 配置文件，包涵相关微调参数"),
#         uid: str = Form(description="生成蒸馏数据返回的uid"),
# ):
#     try:
#         dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{uid}"
#         accelerate_configs_file_content = await accelerate_configs_file.read()  # 读取上传文件的内容
#         sft_configs_file_content = await sft_configs_file.read()  # 读取上传文件的内容
#         with open(os.path.join(dir_path, "accelerate_configs.yaml"), "wb") as f:
#             f.write(accelerate_configs_file_content)
#         with open(os.path.join(dir_path, "sft_configs.yaml"), "wb") as f:
#             f.write(sft_configs_file_content)
#
#         # 更新数据路径
#         import yaml
#         with open(os.path.join(dir_path, "sft_configs.yaml"), 'r') as f:
#             data = yaml.safe_load(f)
#             data["dataset_name"] = os.path.join(dir_path, "distilled")
#
#         with open(os.path.join(dir_path, "sft_configs.yaml"), 'w', encoding='utf-8') as f:
#             # 使用 yaml.dump() 方法将字典 d 转换为 YAML 格式，并将其写入文件中
#             f.write(yaml.dump(data))
#
#     except Exception as e:
#         raise BinaryDecodingError(e)
#
#     # 更新数据路径
#     system_cmd_str = f'accelerate launch --config_file {str(os.path.join(dir_path, "accelerate_configs.yaml"))} {os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "third_party_tools", "open-r1/src/open_r1/sft.py")} --config {str(os.path.join(dir_path, "sft_configs.yaml"))}'
#     print("system_cmd_str:", system_cmd_str)
#     os.system(system_cmd_str)
#
#     # commands = [
#     #     [system_cmd_str]
#     # ]
#     # for cmd in commands:
#     #     result = subprocess.run(
#     #         cmd,
#     #         capture_output=True,  # 捕获输出
#     #         text=True,  # 返回字符串（而非字节）
#     #         shell=True  # 在 Windows 下可能需要
#     #     )
#     #     print(f"命令: {' '.join(cmd)}")
#     #     print("退出码:", result.returncode)
#     #     print("输出:\n", result.stdout)  # 标准输出
#     #     print("错误:\n", result.stderr)  # 标准错误
#
#     content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": ""}
#     return JSONResponse(status_code=status.HTTP_200_OK, content=content)

'''使用蒸馏数据 SFT'''


@openr1_router.post(path="/async_sft_distilled", response_model=ResponseModel, tags=["使用蒸馏数据 SFT"])
async def async_sft_distilled(
        accelerate_configs_file: UploadFile = File(description="accelerate 配置文件"),
        sft_configs_file: UploadFile = File(description="sft 配置文件，包涵相关微调参数"),
        task_id: str = Form(description="生成蒸馏数据返回的uid"),
):
    try:
        dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{task_id}"
        accelerate_configs_file_content = await accelerate_configs_file.read()  # 读取 accelerate 配置文件
        sft_configs_file_content = await sft_configs_file.read()  # 读取 sft 配置文件
        with open(os.path.join(dir_path, "accelerate_configs.yaml"), "wb") as f:
            f.write(accelerate_configs_file_content)
        with open(os.path.join(dir_path, "sft_configs.yaml"), "wb") as f:
            f.write(sft_configs_file_content)

        # 更新数据路径
        with open(os.path.join(dir_path, "sft_configs.yaml"), 'r') as f:
            data = yaml.safe_load(f)
            data["dataset_name"] = os.path.join(dir_path, "distilled")

        with open(os.path.join(dir_path, "sft_configs.yaml"), 'w', encoding='utf-8') as f:
            # 使用 yaml.dump() 方法将字典 d 转换为 YAML 格式，并将其写入文件中
            f.write(yaml.dump(data))

    except Exception as e:
        logger.info("File read/write error。")
        raise BinaryDecodingError(e)

    # 开始微调
    try:
        sft_task_dict[task_id] = {"time": time.time(), "code": 2, "msg": ""}
        system_cmd_str = f'accelerate launch --config_file {str(os.path.join(dir_path, "accelerate_configs.yaml"))} {os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "third_party_tools", "open-r1/src/open_r1/sft.py")} --config {str(os.path.join(dir_path, "sft_configs.yaml"))}'
        thr = Process(target=distilled_sft, args=(system_cmd_str, task_id, data["output_dir"]))
        thr.daemon = True
        threads_list.append(thr)
        thr.start()

    except Exception as e:
        logger.info(e)

    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"task_id": task_id}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@openr1_router.post(path="/async_grpo", response_model=ResponseModel, tags=["使用蒸馏数据 SFT"])
async def async_grpo(
        accelerate_configs_file: UploadFile = File(description="accelerate 配置文件"),
        grpo_configs_file: UploadFile = File(description="grpo 配置文件，包涵相关微调参数"),
        task_id: str = Form(description="生成蒸馏数据返回的uid"),
        trl_vllm_devices: str = Form(default=None),
        cuda_visible_devices: str = Form(default=None),
):
    try:
        import chardet
        dir_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{task_id}"
        if not os.path.exists(dir_path):
            logger.info(f"create dir {task_id}.")
            os.mkdir(dir_path)

        accelerate_configs_file_content = await accelerate_configs_file.read()  # 读取 accelerate 配置文件
        accelerate_configs_file_path = os.path.join(dir_path, "accelerate_configs.yaml")
        accelerate_configs_file_dict = yaml.safe_load(accelerate_configs_file_content)
        grpo_configs_file_content = await grpo_configs_file.read()  # 读取 grpo 配置文件
        grpo_configs_file_path = os.path.join(dir_path, "grpo_configs.yaml")
        grpo_configs_file_dict = yaml.safe_load(grpo_configs_file_content)
        # grpo_configs_file_dict["dataset_name"] = os.path.join(dir_path, "distilled")

        with open(accelerate_configs_file_path, "w", encoding='utf-8') as f:
            f.write(str(accelerate_configs_file_dict))

        with open(grpo_configs_file_path, "w", encoding='utf-8') as f:
            f.write(str(grpo_configs_file_dict))


    except Exception as e:
        logger.info("File read/write error。")
        raise BinaryDecodingError(e)

    # 启动vllm
    try:
        if grpo_configs_file_dict["use_vllm"] == True:
            if trl_vllm_devices is not None and len(trl_vllm_devices) > 0:
                trl_vllm_cmd = f"CUDA_VISIBLE_DEVICES={trl_vllm_devices} trl vllm-serve --model {grpo_configs_file_dict['model_name_or_path']} --gpu_memory_utilization 0.8 --max_model_len 8192"
            else:
                trl_vllm_cmd = f"trl vllm-serve --model {grpo_configs_file_dict['model_name_or_path']} " \
                      f"--gpu_memory_utilization 0.9 --max_model_len 8192"

            os.system(trl_vllm_cmd)

    except Exception as e:
        logger.info(e)

    # grpo
    try:
        if task_id in grpo_task_dict:
            logger.info(f"{task_id} is existed.")
            content = {"isSuc": False, "code": -1, "msg": f"{task_id} is existed.", "res": ""}
            return JSONResponse(status_code=status.HTTP_200_OK, content=content)
        grpo_task_dict[task_id] = {"time": time.time(), "code": 2, "msg": ""}

        if cuda_visible_devices is not None and len(cuda_visible_devices) > 0:
            system_cmd_str = f'CUDA_VISIBLE_DEVICES={cuda_visible_devices} accelerate launch --config_file' \
                             f' {str(os.path.join(dir_path, "accelerate_configs.yaml"))} ' \
                             f'{os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "third_party_tools", "open-r1/src/open_r1/grpo.py")} --config {str(os.path.join(dir_path, "grpo_configs.yaml"))}'
        thr = Process(target=_grpo, args=(system_cmd_str, task_id, grpo_configs_file_dict["output_dir"]))
        thr.daemon = True
        threads_list.append(thr)
        thr.start()

    except Exception as e:
        logger.info(e)

    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"task_id": task_id}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@openr1_router.post(path="/test")
async def test(
        task_id: str = Form("1234567890", example=""),
        distilled_data_path: str = Form("", example=""),
):
    content = {"isSuc": True, "code": 0, "msg": "Success ~",
               "res": {"task_id": task_id, "distilled_data_path": distilled_data_path}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)
