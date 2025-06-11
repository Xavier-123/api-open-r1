import os
import json
import requests
from tools.log import logger

def data_callback(args, ifSuccess=True, msg=""):
    _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/distillationTaskRun/getOperationEventInfo")
    data = {
        "taskRunId": args.task_id,
        "pathKey": args.hf_output_dataset,
        "ifSuccess": ifSuccess,
        "msg": msg,
    }

    try:
        req = requests.post(_url, data=json.dumps(data))
        # req = requests.post(_url, json=json.dumps(data))
        if req.status_code == 200:
            logger.info(f"数据蒸馏完成, data be saved {args.hf_output_dataset}")
        else:
            raise f"Connection {_url} failed."
    except Exception as e:
        logger.info(e)



def sft_callback(task_id, output_dir, ifSuccess=True, msg=""):
    data = {
        "taskRunId": task_id,
        "pathKey": output_dir,
        "ifSuccess": ifSuccess,
        "msg": msg,
    }

    # _callback(args)
    try:
        _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/distillationTaskRun/getOperationEventInfo")
        req = requests.post(_url, data=json.dumps(data))
        # req = requests.post(_url, json=json.dumps(data))

        if req.status_code == 200 and data["code"] == 0:
            logger.info(f"sft completed. sft model be saved {output_dir}")
        elif req.status_code != 200 and data["code"] == 0:
            logger.info(f"Connection {_url} failed.")

    except Exception as e:
        raise e


def grpo_callback(task_id, output_dir, ifSuccess=True, msg=""):
    # 调用接口，返回模型路径
    data = {
        "taskRunId": task_id,
        "pathKey": output_dir,
        "ifSuccess": ifSuccess,
        "msg": msg,
    }
    _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/distillationTaskRun/getOperationEventInfo")
    req = requests.post(_url, data=json.dumps(data))
    # req = requests.post(_url, json=json.dumps(data))
    if req.status_code == 200:
        logger.info(f"GRPO completed. GRPO model be saved {output_dir}")
    else:
        raise f"Connection {_url} failed."