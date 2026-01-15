
# 环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 其余依赖直接 pip install
```

# 训练

分为两种：从头训练、微调 Real-ESRGAN。

## 准备元信息 `meta_info.txt`

```bash
python scripts/generate_meta_info.py \
  --input datasets/DF2K \
  --root datasets \
  --meta_info datasets/DF2K/meta_info/meta_info.txt
```

参数说明：

- `--input`：数据集路径
- `--root`：数据集所在根目录
- `--meta_info`：生成 txt 路径

------

## 从头训练

### 训练 Real-ESRNet 模型

修改选项文件 `options/train_realesrnet_x4plus.yml` 中的内容：

```yaml
train:
  name: DF2K+OST
  type: RealESRGANDataset
  dataroot_gt: datasets/DF2K  # 修改为你的数据集文件夹根目录
  meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # 修改为你自己生成的元信息txt
  io_backend:
    type: disk
```

开始训练：

```bash
python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
```

### 训练 Real-ESRGAN 模型

Real-ESRNet 训练后会在 `experiments/` 下生成训练结果 `train/model/.pth` 文件。
如需指定预训练路径到其他文件，请修改选项文件 `train_realesrgan_x4plus.yml` 中 `pretrain_network_g` 的值。

修改选项文件 `train_realesrgan_x4plus.yml`（与上节类似）。

------

## 使用自有数据集微调

分两类：动态生成降级图像、使用已配对图像。

### 动态生成降级图像

下载预训练模型：

- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_netD.pth`

放入目录：`experiments/pretrained_models`

微调：修改选项文件 `options/finetune_realesrgan_x4plus.yml`，特别是 `datasets` 部分：

```yaml
train:
  name: DF2K+OST
  type: RealESRGANDataset
  dataroot_gt: datasets/DF2K  # 修改为你的数据集文件夹根目录
  meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # 修改为你自己生成的元信息txt
```

### 使用已配对的数据

设置两个文件夹：

- `gt folder`：标准参考（高分辨率图像）
- `lq folder`：低质量（低分辨率图像）

使用脚本 `scripts/generate_meta_info_pairdata.py` 生成元信息（`meta_info`）txt 文件。

微调：修改选项文件 `options/finetune_realesrgan_x4plus_pairdata.yml`，特别是 `datasets` 部分：

```yaml
train:
  name: DIV2K
  type: RealESRGANPairedDataset
  dataroot_gt: datasets/DF2K  # 修改为你的 gt folder 文件夹根目录
  dataroot_lq: datasets/DF2K  # 修改为你的 lq folder 文件夹根目录
  meta_info: datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt  # 修改为你自己生成的元信息txt
```

开始训练：

```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```

## 训练yml文件修改
除了基本的元信息、模型路径外，可以修改以下参数。
### the first/second degradation process
涉及训练过程中添加高斯、灰度噪声强度以及resize程度

### data loader
num_worker/batch_size设计GPU显存，可根据实际情况调整（我当时设备情况：RTX 3050）

### network structures
type以及具体数字，可参考inference_realesrgan.py中model_name处数字，也是我仿照写出reslesr-x4v3配置文件的依据

### training settings
可修改学习率、损失函数以及迭代次数 total_iter，每次训练的配置文件都有在experiments中保存

# 推理

```bash
python inference_realesrgan.py -i <input> -o <output> -n <model_name> \
  --model_path <model_path> -dn -dns <denoise_strength> --wdn_model_path <wdn_model_path> \
  -s <outscale> -t <tile>
```

参数说明：

- `-i`：输入图片路径
- `-o`：输出图片路径
- `-n`：模型名称（见代码注释），决定构建哪个网络架构；权重文件必须与架构匹配，默认使用`RealESRGAN_x4plus`
若出现类似‘RuntimeError: Error(s) in loading state_dict for RRDBNet:Missing key(s) in state_dict: "conv_first.weight", "conv_first.bias",’的错误，说明-n给的模型结构与model_path不符合
- `--model_path`：模型路径
- `-dn`：`store_true`，加上时启用 denoise 功能
- `-dns`：denoise 强度（0~1）
- `--wdn_model_path`：denoise 时使用两个网络 `net_a`、`net_b` 做（DNI）

DNI 说明：

- `net_a`：偏“锐利/细节增强”（可能更容易出噪点、纹理更激进）
- `net_b`：偏“去噪/更保守”（更干净但更平滑）
- 直接使用训练模型可能生成白线或不存在结构，可使用 DNI 将 base 模型与训练模型按比例融合，以达到更居中的效果并节省处理时间
- ![image-20260112155048572](C:\Users\Dev-01\AppData\Roaming\Typora\typora-user-images\image-20260112155048572.png)
从左往右，图一到图四分别为原图、base模型处理、训练模型处理、利用base模型以及训练模型dni强度为0.5处理结果。注意看图三为微调模型直接处理结果，在元件边缘处出现明显的白线，并且在铺铜区域出现不存在的形似电容等结构。

计算方式：
```text
net_a = dns * net_a + (1 - dns) * net_b
```

注意事项：

- 使用时需要注意 `net_a`、`net_b` 的 key：有两种分别为 `params`、`params_ema`
- 需在 `realesrgan/utils.py` 的 `dni` 函数定义中修改 `key1`、`key2`

改进：新建dni_save.py，将dni插值后的模型save为新模型，之后用的时候不用wdn_model_path参数，直接在model_path中指明插值后新模型路径

其他参数：

- `-s`：最终上采样倍数，例如 `s=4`：input 4x4 -> output 16x16
- `-t`：tile 分割，`0` 表示不分割；图片过大 `t=0` 可能 OOM，需适当增大 `t`

------

# 部署

## 模型转换

使用 `scripts/pytorch2onnx.py` 将 `.pth` 转换为 `.onnx`。注意model建立时的类型，不同model不一样，如x4plus就是RRDBNet
```python
model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
```
使用命令
```python
python scripts/pytorch2onnx.py --input --output --params
```
注意：params参数是否添加取决于model的键名

`.onnx` 是通用格式；之后需在不同机器上转换为 TensorRT `.engine`，因为不同机器的 TensorRT/CUDA 版本差异会影响 `.engine`，但对 `.onnx` 不影响。

将 onnx 转为 engine命令：

```bash
trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16
```

## 创建 `.dll`

在 C++文件夹下的SR 项目中将上述 Python 代码改写为 C++，创建 `TRTRunner` 类，导出：

- `SR.dll`
- `SR.lib`
- `TRTRunner.h`

	在新项目Projec1（与项目SR处在同一解决方案下，手动点击导入类TRTRunner）中导入 `TRTRunner.h`、`SR.dll`，生成对应 `exe` 后，将 `SR.dll` 放在 `exe` 所在目录。并且生成的SR.dll，SR.lib，Project1.exe在`Visual Studio\repos\SR\SR\x64\Release`下。
	之后若要修改.dll文件和.exe文件，添加路径按照下图所示：
SR项目：
包含目录：![image-20260114111558106](C:\Users\Dev-01\Desktop\超分-ESRGAN\README.assets\image-20260114111558106.png)

库目录：![image-20260114111634842](C:\Users\Dev-01\Desktop\超分-ESRGAN\README.assets\image-20260114111634842.png)

附加依赖项：![image-20260114111717777](C:\Users\Dev-01\Desktop\超分-ESRGAN\README.assets\image-20260114111717777.png)

Project1项目：
包含目录：TRTRunner,h所在路径

库目录：SR.lib所在路径

附加依赖项目:SR.lib

修改完生成新的exe,lib,dll文件记得迁移到exe_test中重新测试
PS：不同模型.engine的outname不同，可用netron打开.onnx格式查看后，修改Project项目中test.cpp的outname再重新生成.exe文件

## 使用

在exe_test文件中打开cmd，然后运行：
（内含说明文档）
```bash
Project1.exe --engine realesrgan-x4-fp16.engine --input input1 --output output
```

# 实验结果说明
## 初始阶段
起初拿图片训练出模型，直接处理图片，会生成白线和不存在电路结构，于是猜测是enhance强度过大，放大了不必要噪声。于是先经过官方的x4plus，再经过训练模型

## 后来阶段
增加了dni函数功能，但要注意不用模型的key name不同，增加dns调节强度。之后增加了dni_save.py，保存修改后模型。

## 实验结果
类似100kv-0.2指的是100kv图片经过dns强度为0.2处理结果
100kv-0.2-90mins指上述图片再经过90mins图片训练得到模型的处理
之后不用这么繁琐，利用dni_save.py直接保存调节后的模型，再用新模型进行处理