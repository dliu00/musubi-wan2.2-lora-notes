# musubi\-wan2\.2\-lora\-notes

*issues and some process\-notes for training wan2\.2\-lora based on musubi\-tuner*

---

## 仓库简介

本仓库仅为个人自用学习备份笔记，用于记录基于 Musubi\-Tuner 框架，实操 Wan2\.2\-I2V\-A14B 图生视频模型 LoRA 微调的完整流程、固定配置、常用指令、踩坑问题以及对应的解决方案。所有内容仅服务于本人日常查阅，非公开教程，无任何商用属性。

---

## 一、基础前置信息

### 1\.1 运行基础参数

- 训练框架：Musubi\-Tuner

- 适配模型：Wan2\.2 I2V\-A14B（仅支持图生视频，不兼容文生视频模型）

- 运行环境：独立 Conda 虚拟环境 `musubi\_env`

- 硬件门槛：≥24G 独立显存，训练强制开启 FP8 显存优化 \+ CPU梯度卸载

- 训练模式：轻量化 LoRA 微调，全程不篡改原始大模型权重，24G显存最优适配方案

### 1\.2 必备依赖模型文件

- 文本编码器：`models\_t5\_umt5\-xxl\-enc\-bf16\.pth`

- 专属VAE：`Wan2\.1\_VAE\.pth`

- DiT主干模型：多份dit系列safetensors分片文件（需手动合并后方可使用）

---

## 二、数据集搭建规范

### 2\.1 固定目录结构

目录名称、大小写、层级不可随意修改，否则直接触发报错：

```plaintext
train_data/
├─ img/         # 存放所有训练原图，格式支持jpg/png
└─ cache/       # 潜像、文本缓存目录，无需手动创建，脚本自动生成

```

### 2\.2 数据集制作硬性规则

1. 一一对应原则：每一张训练原图，必须配套同名txt标签文件（例：001\.jpg 对应 001\.txt），禁止空文件、重复文件、无效垃圾素材；

2. 触发词规则：所有标签txt文件，首位关键词统一设置为 **manblue**，作为LoRA专属激活触发词；

3. 素材建议：单角色定向训练推荐15\-30张高清素材，画面聚焦人物全身/面部，画风统一、背景简洁，禁止多人物同框画面。

### 2\.3 数据集配置文件

配置文件 `dataset\.toml` 存放于musubi\-tuner项目根目录，参数为实测最优值，无需二次调整：

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "./train_data/img"
cache_directory = "./train_data/cache"
num_repeats = 1

```

---

## 三、Accelerate 环境初始化

### 3\.1 功能说明

首次训练必须初始化配置，用于规避多进程报错、显存分配异常、训练无故中断、精度丢失等问题；后续重复训练无需重新配置。

### 3\.2 初始化命令

```bash
accelerate config

```

### 3\.3 逐项配置答案

1. This machine

2. No distributed training

3. NO

4. 1

5. bf16

6. Default

7. no

8. no

9. no

10. no

### 3\.4 配置验证

执行下方命令，输出内容包含 `mixed\_precision=bf16`、`num\_processes=1` 即为配置成功：

```bash
accelerate env

```

---

## 四、Pre\-Caching 预缓存操作

**重要备注**：Wan2\.2 I2V图生视频缓存与T2V文生视频缓存不互通，禁止混用；全新训练前务必清空旧缓存，未添加 `\-\-i2v` 参数会直接抛出KeyError报错。

### 4\.1 清空历史旧缓存

```bash
rm -rf ./train_data/cache/*

```

### 4\.2 T5文本编码缓存生成

```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config dataset.toml --t5 /root/ai-models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth --batch_size 1
```

### 4\.3 I2V专属VAE潜像缓存

该命令必须携带 `\-\-i2v` 参数，适配图生视频首帧专属潜像逻辑：

```bash
python src/musubi_tuner/wan_cache_latents.py --dataset_config dataset.toml --vae /root/ai-models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth --i2v --batch_size 1

```

---

## 五、DiT切片模型合并（服务器专属）

### 5\.1 注意事项

服务器 `/mnt/scratch` 为临时挂载磁盘，服务器重启后内部数据自动清空；每次重启服务器后，必须重新执行合并脚本，否则无法启动训练。

### 5\.2 一键合并脚本

```bash
python -c "
from safetensors.torch import load_file, save_file
import os
input_dir = '/root/ai-models/Wan-AI/Wan2.2-I2V-A14B/'
output_path = '/mnt/scratch/wan_merged_model/dit.safetensors'
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.safetensors') and ('dit' in f)])
full_sd = {}
for f in files:
    full_sd.update(load_file(os.path.join(input_dir, f)))
save_file(full_sd, output_path)
print('✅ DiT模型合并完成')
"

```

### 5\.3 合并完成固定路径

```plaintext
/mnt/scratch/wan_merged_model/dit.safetensors

```

---

## 六、LoRA正式训练指令

### 6\.1 启动命令（24G显存专属优化版）

整合FP8权重压缩、CPU梯度卸载全部优化参数，直接完整复制运行，禁止拆分换行：

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task i2v-A14B --dit /mnt/scratch/wan_merged_model/dit.safetensors --vae /root/ai-models/Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth --t5 /root/ai-models/Wan-AI/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth --dataset_config dataset.toml --sdpa --mixed_precision bf16 --fp8_base --fp8_scaled --gradient_checkpointing --gradient_checkpointing_cpu_offload --max_data_loader_n_workers 0 --network_module networks.lora_wan --network_dim 32 --timestep_sampling shift --discrete_flow_shift 5.0 --min_timestep 0 --max_timestep 900 --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 --output_dir /mnt/scratch/lora_output --output_name my_i2v_lora --optimizer_type adamw8bit --learning_rate 2e-4

```

### 6\.2 核心参数释义

- `\-\-task i2v\-A14B`：绑定图生视频专属训练任务，适配模型底层网络

- `\-\-network\_dim 32`：LoRA维度，平衡拟合度与过拟合风险，当前最优参数

- `\-\-fp8\_base / \-\-fp8\_scaled`：开启FP8权重压缩，降低60%左右显存占用

- `gradient\_checkpointing\_cpu\_offload`：梯度迁移至CPU，解决24G显存OOM溢出问题

- `\-\-discrete\_flow\_shift 5\.0`：Wan2\.2官方默认扩散偏移值，禁止随意修改

- `\-\-optimizer\_type adamw8bit`：8位优化器，提升训练稳定性、节省显存

- `\-\-max\_train\_epochs 16`：总迭代轮次，每1个epoch自动保存一份LoRA权重

---

## 七、训练产物与格式转换

### 7\.1 权重保存路径

```plaintext
/mnt/scratch/lora_output/my_i2v_lora.safetensors

```

### 7\.2 产物基础属性

文件格式：safetensors；文件大小：30\~80MB；原生为单分支LoRA，无法直接用于双分支工作流，需格式转换，否则会出现雪花屏、画面崩坏问题。

### 7\.3 高低噪分支转换命令

转换仅修改文件内部标记，不改动原始权重，适配ComfyUI双分支工作流：

#### 低噪分支（管控人脸、细节、人物身份）

```bash
python src/musubi_tuner/convert_lora.py \
--input /root/workspace/musubi-tuner/lora/my_i2v_lora.safetensors \
--output /root/workspace/musubi-tuner/lora/my_i2v_lora_low.safetensors \
--target other

```

#### 高噪分支（管控构图、姿势、整体动作）

```bash
python src/musubi_tuner/convert_lora.py \
--input /root/workspace/musubi-tuner/lora/my_i2v_lora.safetensors \
--output /root/workspace/musubi-tuner/lora/my_i2v_lora_high.safetensors \
--target attn

```

**使用规范**：是ai总结生成的，实际上运行一次的结果可以直接分别用于两个分支，不用区分low noise和high noise

**把Lora复制到comfyui目录**：
```bash
cp ~/workspace/lora/manblue_i2v_lora.safetensors /root/ComfyUI/models/loras/
```

---

## 八、触发词使用规范

- 专属激活词：**manblue**

- 使用规则：推理视频时，触发词必须放置在提示词首位，后置或缺位会导致LoRA生效微弱、完全失效

- 通用提示词模板：`manblue, realistic texture, smooth video movement, high detail, natural lighting`

---

## 九、ComfyUI 推理流程

1. 加载全套基础模型：DiT主干模型 \+ Wan2\.1 VAE \+ T5文本编码器；

2. 替换专用节点：废弃通用加载LoRA节点，使用I2V专属双分支加载节点；

3. 分别挂载转换后的high、low双分支LoRA，初始权重统一设置0\.5；

4. 导入参考首图，提示词首位填写专属触发词manblue；

5. 执行推理，根据画面状态微调权重参数。

### 9\.1 效果微调方案

- 人物特征微弱：逐步上调权重，上限1\.0

- 画面闪烁、人脸畸形、雪花屏：下调权重至0\.25\~0\.4

- 风格跑偏：精简提示词，删除与训练角色冲突的描述词汇

---

## 十、常见报错及解决方案

1. **KeyError: latents\_image**：复用T2V缓存 / 预缓存命令未添加\-\-i2v参数；清空全部缓存，重新执行I2V专属潜像缓存命令。

2. **显存溢出OOM**：未开启FP8压缩、未开启梯度CPU卸载；直接使用文档内置完整训练命令。

3. **服务器重启模型丢失**：临时磁盘特性；服务器重启后重新执行DiT合并脚本。

4. **LoRA完全不生效**：使用普通LoRA节点、未填写前置触发词、单文件直插双分支、混用T2V工作流；更换双分支节点、规范提示词、转换高低噪文件。

5. **命令提示找不到文件**：自定义错误路径；直接使用文档内绝对路径指令。

6. **加载LoRA出现雪花屏**：单文件直接接入双分支、权重数值过高；拆分高低噪分支，权重下调至0\.3\~0\.5。

---

## 十一、训练成功判定标准

1. 预缓存指令执行完毕，cache目录自动生成完整的文本、潜像缓存文件；

2. DiT合并脚本正常输出✅标识，生成完整可用的safetensors主干文件；

3. 训练日志初始化完成，成功加载400\+LoRA模块与8位优化器；

4. 程序自动迭代epoch、step，无报错、无强制闪退退出；

5. 每轮epoch自动保存LoRA权重，文件大小维持在30\-80MB区间。

---

## 版权声明 \&amp; 开源协议

版权声明 本仓库内所有内容（包含文字笔记、配置参数、运行指令）仅为个人学习自用备份。未经本人书面明确许可，任何第三方禁止商用全部内容。仓库内提及的开源框架、AI模型版权归原作者所有。

