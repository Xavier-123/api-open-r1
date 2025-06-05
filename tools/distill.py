import os
import json
import time
import subprocess
import pandas as pd
import requests
import datasets
from typing import Optional
from datasets import Dataset, load_from_disk, load_dataset
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration
from tools.log import logger
from tools.error_define import FileConversionError, DataDistillationError


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


def qwen_template(data):
    data["text"] = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
                   f"{data['problem']}<|im_end|>\n<|im_start|>assistant\n{data['generation']}<|im_end|>"
    return data


def distill(args):
    logger.info("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    logger.info(f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset...")
    try:
        dataset = datasets.load_from_disk(args.hf_dataset)
        if "train" in dataset:
            dataset = dataset["train"]

    except Exception as e:
        logger.info(e)
        # dataset = load_dataset(r"F:\inspur\GPU\code\open-r1\open-r1\datasets\da-test", args.hf_dataset_config, split=args.hf_dataset_split)
        dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    logger.info("Dataset loaded!")

    logger.info("dataset:")
    logger.info(dataset)

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

    logger.info("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    logger.info("Generation pipeline finished!")

    # distilled to sft
    dataset = distiset["default"]["train"].map(qwen_template)

    #
    if args.hf_output_dataset:
        logger.info(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        dataset.save_to_disk(args.hf_output_dataset)
        logger.info(dataset)
        logger.info("distilled data saved!")

    try:
        dataset = load_from_disk(dataset_path=args.hf_output_dataset)
    except Exception as e:
        logger.info(f"{e}")
        dataset = load_dataset(path=args.hf_output_dataset)

    logger.info("distilled dataset:")
    # print(dataset)
    for i in range(dataset.shape[0]):
        logger.info(dataset[i])


def openR1_distill(args, tmp_path, distill_data_path, distilled_data_path, task_dict):

    # 转为蒸馏前格式
    try:
        # if args.file:
        #     pass
        # 将excel转换为dataset格式
        excel_2_arrow(tmp_path, distill_data_path)
        logger.info("excel to arrow Format conversion successful.")
    except Exception as e:
        del task_dict[args.task_id]
        os.remove(distill_data_path)
        os.remove(tmp_path)
        raise FileConversionError(e)

    # 蒸馏数据
    try:
        args.hf_dataset = distill_data_path
        args.hf_output_dataset = distilled_data_path
        distill(args)
        logger.info("distilled success.")
    except Exception as e:
        del task_dict[args.task_id]
        os.remove(distill_data_path)
        os.remove(tmp_path)
        raise DataDistillationError(e)

    try:

        _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
        data = {
            "task_id": args.task_id,
            "distilled_data_path": args.hf_output_dataset
        }
        req = requests.post(_url, json=json.dumps(data))
        if req.status_code == 200:
            logger.info(f"数据蒸馏完成, data be saved {args.hf_output_dataset}")
        else:
            raise f"Connection {_url} failed."

    except Exception as e:
        del task_dict[args.task_id]
        os.remove(distill_data_path)
        os.remove(tmp_path)
        raise DataDistillationError(e)


def distilled_sft(system_cmd_str_list, task_id, output_dir, sft_task_dict):
    try:
        logger.info(system_cmd_str_list)
        result = subprocess.run(system_cmd_str_list, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"成功！输出：{result.stdout}")
        else:
            logger.info(f"失败！输出：{result.stderr}")
            sft_task_dict[task_id]["time"] = time.time()
            sft_task_dict[task_id]["code"] = -1
            sft_task_dict[task_id]["msg"] = result.stderr

    except Exception as e:
        logger.info(e)
        sft_task_dict[task_id]["time"] = time.time()
        sft_task_dict[task_id]["code"] = -1
        sft_task_dict[task_id]["msg"] = "失败"

    try:
        # 调用接口，返回模型路径
        data = {
            "task_id": task_id,
            "sft_model_path": output_dir,
            "code": 0,
            "msg": "",
        }
        _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")

        # 判断训练结果，返回不同状态码
        if sft_task_dict[task_id]["code"] == 0:
            data["code"] = 0
        else:
            data["code"] = -1
            data["msg"] = sft_task_dict[task_id]["msg"]
        req = requests.post(_url, json=json.dumps(data))

        if req.status_code == 200 and data["code"] == 0:
            logger.info(f"sft completed. sft model be saved {output_dir}")
        elif req.status_code != 200 and data["code"] == 0:
            logger.info(f"Connection {_url} failed.")
        else:
            logger.info(f"task_id: {task_id}, distilled data failed.")
    except Exception as e:
        logger.info(e)


def _grpo(system_cmd_grpo, task_id, output_dir):
    try:
        logger.info(system_cmd_grpo)
        os.system(system_cmd_grpo)
    except Exception as e:
        logger.info(e)

    try:
        # 调用接口，返回模型路径
        data = {
            "task_id": task_id,
            "sft_model_path": output_dir
        }
        _url = os.environ.get("_GRPO_URL", "http://127.0.0.1:8018/test")
        req = requests.post(_url, json=json.dumps(data))
        if req.status_code == 200:
            logger.info(f"GRPO completed. GRPO model be saved {output_dir}")
        else:
            raise f"Connection {_url} failed."
    except Exception as e:
        logger.info(e)


def _sft_grpot():
    return
