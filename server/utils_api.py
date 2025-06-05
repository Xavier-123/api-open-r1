import os
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi import FastAPI, UploadFile, File, Form, status, Body
from tools.error_define import BinaryDecodingError
from server.router import utils_router, ResponseModel, RequestModel
import pandas as pd
import uuid
from datasets import Dataset, load_from_disk

def excel_2_arrow(data_path):
    '''excel 转蒸馏需要的格式'''
    data = pd.read_excel(data_path)

    data_list = []
    for i in range(data.shape[0]):
        data_list.append({
            "problem": data.iloc[i, 0],
            "text": data.iloc[i, 1] if data.iloc[i, 1] is not None else "",
        })

    da = Dataset.from_list(data_list)
    da.save_to_disk(data_path[:-4])
    for i in range(da.shape[0]):
        print(da[i])


'''上传文件'''
@utils_router.post(path="/upload_file", summary="bytes", response_model=ResponseModel, tags=["上传文件"])
async def upload_file(
        file: UploadFile = File(description="一个二进制文件"),
):
    # 验证文件
    uid = uuid.uuid4()

    # 将文件保存
    saved_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{uid}/{file.filename}"
    tmp_path = saved_path + ".tmp"
    try:
        file_content = await file.read()  # 读取上传文件的内容
        with open(tmp_path, "wb") as f:
            f.write(file_content)

        # 转为蒸馏前格式
        excel_2_arrow(data_path=tmp_path)


    except Exception as e:
        raise BinaryDecodingError(e)
    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"uid": uid}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


'''上传数据集'''
@utils_router.post(path="/upload_datasets", summary="bytes", response_model=ResponseModel, tags=["上传文件"])
async def upload_file(
        file: UploadFile = File(description="excel、csv或jsonl文件"),
):
    # 验证文件
    uid = uuid.uuid4()

    # 将文件保存
    saved_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0] + f"/file_save/{uid}/{file.filename}"
    tmp_path = saved_path + ".tmp"
    try:
        file_content = await file.read()  # 读取上传文件的内容
        with open(tmp_path, "wb") as f:
            f.write(file_content)

        # 转为蒸馏前格式
        excel_2_arrow(data_path=tmp_path)
        os.remove(tmp_path)

    except Exception as e:
        raise BinaryDecodingError(e)
    content = {"isSuc": True, "code": 0, "msg": "Success ~", "res": {"uid": uid}}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)