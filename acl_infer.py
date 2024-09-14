from typing import List
from transformers import AutoProcessor  # 用于预处理
from PIL import Image  # 输入图像
import numpy as np
import acl
from openvla_7b.modeling_prismatic_modify import ActionDecoder
from openvla_7b.configuration_prismatic import OpenVLAConfig

DEVICE_ID = 0  # 设备id
SUCCESS = 0  # 资源初始函数成功状态值
FAILED = 1  # 资源初始函数失败状态值
ACL_MEM_MALLOC_HUGE_FIRST = 0  # 内存分配策略


# ret只是用来接受各类资源初始化函数的返回值，这些函数返回0代表正常，由此定义这个检查函数
def check_ret(message, ret):
    """用于检查各个返回值是否正常，若否，则抛出对应异常信息"""
    if ret != SUCCESS:
        raise Exception("{} failed ret={}".format(message, ret))


class Net(object):  # 模型初始化、输入输出基础类
    def __init__(self, device_id, model_path):
        self.device_id = device_id  # 设备id
        self.model_path = model_path  # 模型路径
        self.model_id = None  # 模型id
        self.context = None  # 用于管理资源

        self.model_desc = (
            None  # 模型描述信息，包括模型输入个数、输入维度、输出个数、输出维度等信息
        )
        self.load_input_dataset = None  # 输入数据集，aclmdlDataset类型
        self.load_output_dataset = None  # 输出数据集，aclmdlDataset类型

        self.init_resource()

    def init_resource(self):
        """初始化 acl 相关资源"""
        ret = acl.rt.set_device(self.device_id)  # 指定 device
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)  # 创建 context
        check_ret("acl.rt.create_context", ret)

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)  # 加载模型
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()  # 创建描述模型基本信息的数据类型
        print("init resource success")

        ret = acl.mdl.get_desc(
            self.model_desc, self.model_id
        )  # 根据模型ID获取模型基本信息
        check_ret("acl.mdl.get_desc", ret)

    def _gen_input_dataset(self, input_list):
        """组织输入数据的dataset结构"""
        input_num = acl.mdl.get_num_inputs(
            self.model_desc
        )  # 根据模型信息得到模型输入个数
        self.load_input_dataset = acl.mdl.create_dataset()  # 创建输入dataset结构
        for i in range(input_num):
            item = input_list[i]  # 获取第 i 个输入数据
            data = acl.util.bytes_to_ptr(item.tobytes())  # 获取输入数据字节流
            size = item.size * item.itemsize  # 获取输入数据字节数
            dataset_buffer = acl.create_data_buffer(
                data, size
            )  # 创建输入dataset buffer结构, 填入输入数据
            _, ret = acl.mdl.add_dataset_buffer(
                self.load_input_dataset, dataset_buffer
            )  # 将dataset buffer加入dataset
        print("create model input dataset success")

    def _gen_output_dataset(self):
        """组织输出数据的dataset结构"""
        output_num = acl.mdl.get_num_outputs(
            self.model_desc
        )  # 根据模型信息得到模型输出个数
        self.load_output_dataset = acl.mdl.create_dataset()  # 创建输出dataset结构
        for i in range(output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(
                self.model_desc, i
            )  # 获取模型输出个数
            temp_buffer, ret = acl.rt.malloc(
                temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST
            )  # 为每个输出申请device内存
            dataset_buffer = acl.create_data_buffer(
                temp_buffer, temp_buffer_size
            )  # 创建输出的data buffer结构,将申请的内存填入data buffer
            _, ret = acl.mdl.add_dataset_buffer(
                self.load_output_dataset, dataset_buffer
            )  # 将 data buffer 加入输出dataset
        print("create model output dataset success")

    def run(self, inputs):
        """数据集构造、模型推理、解析输出"""
        self._gen_input_dataset(inputs)  # 构造输入数据集
        self._gen_output_dataset()  # 构造输出数据集

        # 模型推理，推理完成后，输出会放入 self.load_output_dataset
        ret = acl.mdl.execute(
            self.model_id, self.load_input_dataset, self.load_output_dataset
        )
        check_ret("acl.mdl.execute", ret)

        # 解析输出
        result = []
        output_num = acl.mdl.get_num_outputs(
            self.model_desc
        )  # 根据模型信息得到模型输出个数
        for i in range(output_num):
            buffer = acl.mdl.get_dataset_buffer(
                self.load_output_dataset, i
            )  # 从输出dataset中获取buffer
            data = acl.get_data_buffer_addr(buffer)  # 获取输出数据内存地址
            size = acl.get_data_buffer_size(buffer)  # 获取输出数据字节数
            narray = acl.util.ptr_to_bytes(data, size)  # 将指针转为字节流数据

            # 根据模型输出的维度和数据类型,将字节流数据解码为numpy数组
            dims, ret = acl.mdl.get_cur_output_dims(
                self.model_desc, i
            )  # 得到当前输出的维度
            out_dim = dims["dims"]  # 提取维度信息
            output_nparray = np.frombuffer(narray, dtype=np.float16).reshape(
                tuple(out_dim)
            )  # 解码为numpy数组
            result.append(output_nparray)

        # 释放模型输入输出数据集
        self._destroy_dataset()
        print("execute stage success")

        return result

    def _destroy_dataset(self):
        """释放模型输入输出数据"""
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)  # 获取输入buffer个数
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)  # 获取每个输入buffer
                if data_buf:
                    ret = acl.destroy_data_buffer(
                        data_buf
                    )  # 销毁每个输入buffer (销毁 aclDataBuffer 类型)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(
                dataset
            )  # 销毁输入数据 (销毁 aclmdlDataset类型的数据)
            check_ret("acl.mdl.destroy_dataset", ret)

    def release_resource(self):
        """释放 acl 相关资源"""
        ret = acl.mdl.unload(self.model_id)  # 卸载模型
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)  # 释放模型描述信息
            self.model_desc = None

        if self.context:
            ret = acl.rt.destroy_context(self.context)  # 释放 Context
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)  # 释放 device 资源
        check_ret("acl.rt.reset_device", ret)

        print("Resources released successfully.")


class VLAllama:  # VLA专用llama，embedding层/最后一层单独定义
    def __init__(self, model_path: List[str]):
        self.model_path = model_path

    def embedding(self, token):
        embedding_weights = np.load("embed.npy")
        embedded_output = embedding_weights[token.flatten()]
        return embedded_output

    def infer(self, input_embed, attn_mask):
        partial_output = input_embed
        for path in self.model_path:
            print("model ", path, "is running")
            partial_llama = Net(DEVICE_ID, path)
            partial_output = partial_llama.run([partial_output, attn_mask])[0]
            partial_llama.release_resource()
        return partial_output

    def detoken(self, result):
        # 此函数用于从模型输出解出预测的token值，源码do_sample=False，故采用最直接的贪心解码, 选择概率最高的token
        trans_weights = np.load("final_trans.npy")
        logits = np.dot(result[:, -1, :], trans_weights)
        exp_logits = np.exp(logits - np.max(logits))  # 减去最大值以提高数值稳定性
        probs = exp_logits / np.sum(exp_logits)
        predicted_token_id = np.argmax(probs)
        print(predicted_token_id)
        return predicted_token_id

    def generate(self, trig_embed, trig_mask):
        input_embed = trig_embed
        attn_mask = trig_mask
        predict_tokens = []
        for i in range(1, 8):
            # 预测得到一个新token
            single_token = self.detoken(self.infer(input_embed, attn_mask))
            predict_tokens += [single_token]
            # 准备下一次预测的输入,丢掉上一时刻的第一个token（相当于固定长度279的attn_mask)
            token_embed = self.embedding(np.array([single_token])).reshape(1, 1, 4096)
            input_embed = np.concatenate((input_embed[:, 1:, :], token_embed), axis=1)
            attn_mask = np.concatenate(
                (attn_mask[:, 1:], np.ones((1, 1), dtype=np.int16)), axis=1
            )
        return predict_tokens


class VLAfeature_extractor:  # 图像文字特征提取
    def __init__(self, vbb1_path, vbb2_path, proj_path):
        self.vbb1_path = vbb1_path
        self.vbb2_path = vbb2_path
        self.proj_path = proj_path

    def embedding(self, token):
        embedding_weights = np.load("embed.npy")
        embedding_dim = embedding_weights.shape[1]
        embedded_output = embedding_weights[token.flatten()].reshape(
            1, 23, embedding_dim
        )
        return embedded_output

    def extract(self, input_ids, pixel_value):
        array_part1 = pixel_value[:, :3, :, :]  # 获取前 3 个通道
        array_part2 = pixel_value[:, 3:, :, :]  # 获取后 3 个通道
        # 获取文字embedding，此处添加的token是为与训练数据对齐
        input_ids = np.concatenate((input_ids, np.array([29871]).reshape(1, 1)), axis=1)
        txt_embeds = self.embedding(input_ids)
        # 图片提取图像特征并做embedding
        vbb1 = Net(DEVICE_ID, self.vbb1_path)
        feature_1 = vbb1.run([array_part1])[0]
        vbb1.release_resource()
        vbb2 = Net(DEVICE_ID, self.vbb2_path)
        feature_2 = vbb2.run([array_part2])[0]
        vbb2.release_resource()
        projector = Net(DEVICE_ID, self.proj_path)
        feature = np.concatenate((feature_1, feature_2), axis=2)
        image_embed = projector.run(feature)[0]
        projector.release_resource()
        # 合并文字+图像的embedding，制作配套attn_mask：除了补空的token被mask（0），其他为true（1）
        embed = np.concatenate(
            (
                txt_embeds[:, 0, :].reshape(1, 1, 4096),
                image_embed,
                txt_embeds[:, 1:, :],
            ),
            axis=1,
        )
        mask = np.concatenate(
            (np.ones((1, 278), dtype=np.int16), np.zeros((1, 1), dtype=np.int16)),
            axis=1,
        )
        return embed, mask


class VLAprocessor:  # 图像文字前处理与动作解码
    def __init__(self):
        self.preprocessor = AutoProcessor.from_pretrained(
            "/root/ms-VLA/openvla_7b", trust_remote_code=True
        )
        self.config = OpenVLAConfig.from_pretrained("/root/ms-VLA/openvla_7b")
        self.decoder = ActionDecoder(self.config)

    def preprocess(self, prompt, image):
        # 预处理函数，调用processor
        token_mask_pixel = self.preprocessor(prompt, image)
        return (
            token_mask_pixel.input_ids.numpy().astype(np.int16),
            token_mask_pixel.attention_mask.numpy().astype(np.float16),
            token_mask_pixel.pixel_values.numpy().astype(np.float16),
        )

    def decode(self, predicted_action, unnorm_key):
        actions = self.decoder.predict_action(
            predicted_action_token_ids=np.array(predicted_action), unnorm_key=unnorm_key
        )
        return actions


if __name__ == "__main__":
    ret = acl.init()  # 初始化/去初始化acl，一个文件一次就行
    check_ret("acl.init", ret)
    # 设置模型文件路径
    vbb1_path = "/root/ms-VLA/om/vbb1_fp.om"
    vbb2_path = "/root/ms-VLA/om/vbb2_fp.om"
    proj_path = "/root/ms-VLA/om/projector_fp.om"
    llama_path = [
        "/root/ms-VLA/om/llama_1_mixhead1_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead2_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead3_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead4_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead5_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead6_modify.om",
        "/root/ms-VLA/om/llama_1_mixhead7_modify.om",
    ]
    # 输入
    image: Image.Image = Image.open(
        "/root/ms-VLA/test_image/test_img.png"
    )  # 后期可修改为摄像头图片
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    # 前处理
    processor = VLAprocessor()
    input_ids, attn_mask, pixel_value = processor.preprocess(prompt, image)
    # 特征提取
    feature_extractor = VLAfeature_extractor(vbb1_path, vbb2_path, proj_path)
    embed, mask = feature_extractor.extract(input_ids, pixel_value)
    # 动作预测
    llama = VLAllama(llama_path)
    predict_tokens = llama.generate(embed, mask)
    # 结果解码
    actions = processor.decode(predict_tokens, unnorm_key="bridge_orig")
    print(actions)
    # ACL去初始化
    ret = acl.finalize()
    check_ret("acl.finalize", ret)
