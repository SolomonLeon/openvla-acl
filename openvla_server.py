import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from starlette.responses import JSONResponse


from acl_infer import (
    Net,
    check_ret,
    acl,
    SUCCESS,
    FAILED,
    DEVICE_ID,
    ACL_MEM_MALLOC_HUGE_FIRST,
)
from acl_infer import VLAllama, VLAfeature_extractor, VLAprocessor

ret = acl.init()
check_ret("acl.init", ret)

# 设置模型文件路径
vbb1_path = "/path/to/vbb1.om"
vbb2_path = "/path/to/vbb2.om"
proj_path = "/path/to/projector.om"
llama_paths = ["/path/to/llama_1.om", "/path/to/llama_2.om", ..., "/path/to/llama_7.om"]

# 创建FastAPI实例
app = FastAPI()

# 初始化模型
# 假设VLAllama和VLAfeature_extractor类的实现与之前提供的示例代码相同
llama = VLAllama(llama_paths)
feature_extractor = VLAfeature_extractor(vbb1_path, vbb2_path, proj_path)


# 定义处理函数
@app.post("/act")
async def predict_action(image: Image.Image, instruction: str):
    try:
        # 前处理
        processor = VLAprocessor()
        input_ids, attn_mask, pixel_value = processor.preprocess(instruction, image)

        # 特征提取
        embed, mask = feature_extractor.extract(input_ids, pixel_value)

        # 动作预测
        predict_tokens = llama.generate(embed, mask)

        # 结果解码
        actions = processor.decode(predict_tokens, unnorm_key="bridge_orig")

        # 返回结果
        return JSONResponse(content=actions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 定义一个运行服务的函数
def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


# 如果是主程序，则运行服务
if __name__ == "__main__":
    run_server()
