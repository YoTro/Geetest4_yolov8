# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import cv2
import math
import os
import json
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators
from paddle import get_device


class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        # [主要作用] 初始化数据集对象。
        # 该函数会读取配置文件，加载标签文件列表，并根据配置进行数据集的初始化设置，
        # 例如设置数据目录、是否打乱、数据增强操作等。
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.delimiter = dataset_config.get("delimiter", "\t")
        label_file_list = dataset_config.pop("label_file_list")
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert (
            len(ratio_list) == data_source_num
        ), "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        logger.info("Initialize indexes of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)
        self.need_reset = True in [x < 1 for x in ratio_list]

    def get_image_info_list(self, file_list, ratio_list):
        # [主要作用] 读取并解析标签文件。
        # 该函数会打开指定的标签文件列表，读取所有行（即图片路径和标签），
        # 并根据给定的比例（ratio_list）进行采样，最终返回一个包含所有数据信息的列表。
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        # [主要作用] 随机打乱数据。
        # 在训练模式下，用于将 self.data_lines 列表中的数据顺序进行随机化，以增加训练的随机性。
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        # [主要作用] 解析特殊的文件名格式。
        # 用于处理一个标签对应多个图片的情况（文件名以'['开头，为JSON格式的列表）。
        # 如果遇到这种格式，会从中随机选择一个文件名返回。
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        # [主要作用] 获取用于数据增强的额外数据。
        # 例如，在实现 CutMix、Mixup 等需要混合多张图片的数据增强方法时，此函数用于随机获取额外的图片和标签。
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__())]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {"img_path": img_path, "label": label}
            if not os.path.exists(img_path):
                continue
            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if "polys" in data.keys():
                if data["polys"].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx, _recursive_count=0):
        # [主要作用] 获取并处理单个数据样本。
        # 这是 Pytorch/Paddle.io.Dataset 的核心函数，根据索引（idx）获取对应的数据行，
        # 读取图片，然后应用一系列的数据变换（如增强、归一化），最后返回处理好的数据。
        # 包含了异常处理和重试机制，确保在数据处理出错时能够跳过并获取下一个样本。
        if _recursive_count > 10:
            self.logger.error("FATAL: Failed to get a valid sample after 10 retries. Please check your dataset and file paths in label files.")
            raise RuntimeError("Failed to get a valid sample after 10 retries. Check logs for details.")
            
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {"img_path": img_path, "label": label}
            
            # self.logger.debug(f"Attempting to load image: {img_path}") # Enhanced logging

            if not os.path.exists(img_path):
                # Log the problematic path and line
                self.logger.error(f"Image file does not exist: {img_path}. From label line: '{data_line.strip()}'")
                raise FileNotFoundError(f"{img_path} does not exist!")
            
            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img

            data["ext_data"] = self.get_ext_data()
            data["filename"] = data["img_path"]
            outs = transform(data, self.ops)
        except Exception as e:
            self.logger.error(
                "When parsing line '{}', an error occurred: {}\nTraceback: {}".format(
                    data_line.strip(), e, traceback.format_exc()
                )
            )
            outs = None
            
        if outs is None:
            self.logger.warning(f"Failed to process line, trying another sample. Failed line: {data_line.strip()}")
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx, _recursive_count + 1)
        return outs

    def __len__(self):
        # [主要作用] 返回数据集中样本的总数。
        return len(self.data_idx_order_list)


class MultiScaleDataSet(SimpleDataSet):
    def __init__(self, config, mode, logger, seed=None):
        # [主要作用] 初始化多尺度数据集。
        # 继承自 SimpleDataSet，并在其基础上增加对多尺度训练的特定支持（例如基于宽高比的排序）。
        super(MultiScaleDataSet, self).__init__(config, mode, logger, seed)
        self.ds_width = config[mode]["dataset"].get("ds_width", False)
        if self.ds_width:
            self.wh_aware()

    def wh_aware(self):
        # [主要作用] 实现宽高比感知的数据排序。
        # 为了提高多尺度训练的效率，此函数计算数据集中每张图片的宽高比，并按此比例对数据进行排序。
        # 这样可以将形状相似的图片分到同一个批次中，减少批处理时填充（padding）的开销。
        data_line_new = []
        wh_ratio = []
        for line in self.data_lines:
            data_line_new.append(line)
            line = line.decode("utf-8")
            name, label, w, h = line.strip("\n").split(self.delimiter)
            wh_ratio.append(float(w) / float(h))

        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(self.data_lines)))

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        # [主要作用] 对图像进行归一化和缩放。
        # 这是一个特定的图像变换函数，它将图像缩放到指定的高度（imgH）和宽度（imgW），
        # 并进行归一化处理（数值缩放到[-1, 1]范围）。支持可选的填充（padding）。
        img = data["image"]
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")

        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data["image"] = padding_im
        data["valid_ratio"] = valid_ratio
        if "iluvatar_gpu" in get_device():
            data["valid_ratio"] = np.float32(valid_ratio)
        return data

    def __getitem__(self, properties):
        # [主要作用] 获取并处理单个多尺度数据样本。
        # 针对多尺度训练重写了__getitem__方法。它接收一个包含（宽度、高度、索引）的元组作为参数，
        # 根据给定的尺寸和索引来获取和处理数据，支持动态调整图像的输入尺寸。
        # properties is a tuple, contains (width, height, index)
        img_height = properties[1]
        idx = properties[2]
        if self.ds_width and properties[3] is not None:
            wh_ratio = properties[3]
            img_width = img_height * (
                1 if int(round(wh_ratio)) == 0 else int(round(wh_ratio))
            )
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]
            wh_ratio = None

        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode("utf-8")
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {"img_path": img_path, "label": label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data["img_path"], "rb") as f:
                img = f.read()
                data["image"] = img
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops[:-1])
            if outs is not None:
                outs = self.resize_norm_img(outs, img_width, img_height)
                outs = transform(outs, self.ops[-1:])
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        return outs