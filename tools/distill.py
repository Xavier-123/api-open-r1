import os
import json
import time
import subprocess
import pandas as pd
import requests
import datasets
from typing import Optional
from datasets import Dataset, load_from_disk, load_dataset, DatasetDict
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration
from tools.log import logger
from tools.utils import data_callback, sft_callback, grpo_callback
from tools.error_define import FileConversionError, DataDistillationError


def build_distilabel_pipeline(
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
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
                # api_key="something",
                api_key=api_key if len(api_key) > 0 else "something",
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


def excel_2_arrow(data_path, distill_path, args):
    '''excel 转蒸馏需要的格式'''
    data = pd.read_excel(data_path)

    # 随机打乱数据
    df = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 计算分割点
    split_idx = int(len(df) * args.split_train_test)

    # 划分训练集和测试集
    train_set = df.iloc[:split_idx]
    test_set = df.iloc[split_idx:]

    train_list, test_list = [], []
    for i in range(train_set.shape[0]):
        train_list.append({
            "problem": train_set.iloc[i, 0],
            "text": train_set.iloc[i, 1] if train_set.iloc[i, 1] is not None else "",
        })

    for i in range(test_set.shape[0]):
        test_list.append({
            "problem": test_set.iloc[i, 0],
            "text": test_set.iloc[i, 1] if test_set.iloc[i, 1] is not None else "",
        })

    # 合并为DatasetDict
    train_dataset = Dataset.from_list(train_list)
    test_dataset = Dataset.from_list(test_list)
    if args.split_train_test == 1:
        da = DatasetDict({
            "train": train_dataset,
        })
    else:
        da = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
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

    logger.info(
        f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset...")
    try:
        dataset = datasets.load_from_disk(args.hf_dataset)

    except Exception as e:
        logger.info(e)
        dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    logger.info("Dataset loaded!")
    logger.info("dataset:")
    logger.info(dataset)

    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        api_key=args.api_key,
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
    distiset_train = pipeline.run(
        dataset=dataset["train"],
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    train_dataset = distiset_train["default"]["train"].map(qwen_template)

    if 0 < args.split_train_test < 1:
        pipeline_test = build_distilabel_pipeline(
            model=args.model,
            base_url=args.vllm_server_url,
            api_key=args.api_key,
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

        distiset_test = pipeline_test.run(
            dataset=dataset["test"],
            dataset_batch_size=args.input_batch_size * 1000,
            use_cache=False,
        )
        test_dataset = distiset_test["default"]["train"].map(qwen_template)

        # 合并为DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

    else:
        # 合并为DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
        })

    logger.info("Generation pipeline finished!")

    if args.hf_output_dataset:
        logger.info(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        dataset.save_to_disk(args.hf_output_dataset)
        logger.info(dataset)
        logger.info("distilled data saved!")

    '''测试数据是否正常'''
    try:
        dataset = load_from_disk(dataset_path=args.hf_output_dataset)
    except Exception as e:
        logger.info(f"{e}")
        dataset = load_dataset(path=args.hf_output_dataset)

    logger.info("distilled dataset:")
    logger.info(dataset)


def openR1_distill(args, tmp_path, distill_data_path, distilled_data_path, task_dict):
    # 转为蒸馏前格式
    try:
        excel_2_arrow(tmp_path, distill_data_path, args)
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
        data_callback(args)
        # _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
        # data = {
        #     "task_id": args.task_id,
        #     "distilled_data_path": args.hf_output_dataset
        # }
        # req = requests.post(_url, json=json.dumps(data))
        # if req.status_code == 200:
        #     logger.info(f"数据蒸馏完成, data be saved {args.hf_output_dataset}")
        # else:
        #     raise f"Connection {_url} failed."

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
        sft_callback(task_id, output_dir, sft_task_dict)
        # data = {
        #     "task_id": task_id,
        #     "sft_model_path": output_dir,
        #     "code": 0,
        #     "msg": "",
        # }
        #
        # # 判断训练结果，返回不同状态码
        # if sft_task_dict[task_id]["code"] == 0:
        #     data["code"] = 0
        # else:
        #     data["code"] = -1
        #     data["msg"] = sft_task_dict[task_id]["msg"]
        #
        # # _callback(args)
        # _url = os.environ.get("CALLBACK_URL", "http://127.0.0.1:8018/test")
        # req = requests.post(_url, json=json.dumps(data))
        #
        # if req.status_code == 200 and data["code"] == 0:
        #     logger.info(f"sft completed. sft model be saved {output_dir}")
        # elif req.status_code != 200 and data["code"] == 0:
        #     logger.info(f"Connection {_url} failed.")
        # else:
        #     logger.info(f"task_id: {task_id}, distilled data failed.")
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
        grpo_callback(task_id, output_dir)
        # data = {
        #     "task_id": task_id,
        #     "sft_model_path": output_dir
        # }
        # _url = os.environ.get("_GRPO_URL", "http://127.0.0.1:8018/test")
        # req = requests.post(_url, json=json.dumps(data))
        # if req.status_code == 200:
        #     logger.info(f"GRPO completed. GRPO model be saved {output_dir}")
        # else:
        #     raise f"Connection {_url} failed."
    except Exception as e:
        logger.info(e)


def _sft_grpot():
    return
