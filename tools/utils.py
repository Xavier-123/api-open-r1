import os
import json
import requests
from tools.log import logger

def data_callback(args):
    _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
    data = {
        "task_id": args.task_id,
        "distilled_data_path": args.hf_output_dataset
    }
    req = requests.post(_url, json=json.dumps(data))
    if req.status_code == 200:
        logger.info(f"数据蒸馏完成, data be saved {args.hf_output_dataset}")
        # logger.info(data)
    else:
        raise f"Connection {_url} failed."


def sft_callback(task_id, output_dir, sft_task_dict):
    data = {
        "task_id": task_id,
        "sft_model_path": output_dir,
        "code": 0,
        "msg": "",
    }

    # 判断训练结果，返回不同状态码
    if sft_task_dict[task_id]["code"] == 0:
        data["code"] = 0
    else:
        data["code"] = -1
        data["msg"] = sft_task_dict[task_id]["msg"]

    # _callback(args)
    _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
    req = requests.post(_url, json=json.dumps(data))

    if req.status_code == 200 and data["code"] == 0:
        logger.info(f"sft completed. sft model be saved {output_dir}")
    elif req.status_code != 200 and data["code"] == 0:
        logger.info(f"Connection {_url} failed.")
    else:
        logger.info(f"task_id: {task_id}, distilled data failed.")


def grpo_callback(task_id, output_dir):
    # 调用接口，返回模型路径
    data = {
        "task_id": task_id,
        "sft_model_path": output_dir
    }
    _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
    req = requests.post(_url, json=json.dumps(data))
    if req.status_code == 200:
        logger.info(f"GRPO completed. GRPO model be saved {output_dir}")
    else:
        raise f"Connection {_url} failed."