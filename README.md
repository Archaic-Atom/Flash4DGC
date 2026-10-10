# Flash4DGC

Flash4DGC 面向闭环 4DGC 参考模型，提供精确的运动网格转码与 GPU 解码。
本仓库包含 4DGC 的训练、渲染源码以及统一的 Flash 运行入口，
运行时不依赖原有的 `flash_experiments` 目录。

**当前冻结配置：**运动量化 q=1、球谐（SH）阶数为 1、SH 数据包量化 q=2，不进行 SH 旋转。
粗网格采用稀疏 int8 分页表示，细网格采用无损的符号—幅值位平面表示。
网络权重与空间边界保持 FP32 精度。最终编解码器不使用 `clip_1` 截断或周期性状态刷新。

配对参考方法为 **4DGC-CL**，即采用本项目闭环训练适配器的 4DGC，
并非未经修改的原论文复现版本。Flash 精确保留的是参考方法的解码张量，
不代表对量化前训练模型或原始视频无损。

## 1. 环境安装

使用 Linux 系统、NVIDIA GPU、与 PyTorch 匹配的 CUDA Toolkit，以及该工具链支持的编译器。
已有实验使用 RTX 3090 和 CUDA 11.8 环境。
以下为新环境的安装方案，尚未对这套整合版执行全新环境安装验证。

```bash
conda create -n flash4dgc python=3.10 -y
conda activate flash4dgc
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m flash4dgc self-test --gpu
```

请在本仓库根目录执行命令，通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU。
Flash 扩展会在首次使用时编译。LPIPS 质量评测首次运行时会下载官方预训练权重，
已有本地缓存时无需重复下载。
原始环境依赖清单保存在 `upstream/`，不作为跨环境直接安装的依赖文件。
公开上传项目前，请确认[第三方代码与许可条款](THIRD_PARTY_NOTICES.md)。

## 2. 输入数据与首帧模型

预处理后的场景目录结构如下。本次编解码器发布不包含数据预处理流程。

```text
SCENE/colmap_0/images/<camera>.png
SCENE/colmap_0/sparse/0/{cameras,images,points3D}.bin
SCENE/colmap_1/images/<camera>.png
...
SCENE/colmap_299/images/<camera>.png
INITIAL/point_cloud/iteration_10000/point_cloud.ply
```

数据加载器复用第 0 帧的相机标定，并将排序后索引为 0 的相机留作测试视角。
`--test-camera` 用于核验测试相机名称，不会改变训练/测试划分。
N3DV 使用 `cam00` 和 `resolution=2`；预处理后的 MeetRoom 使用 `cam_0` 和 `resolution=1`。
每个场景必须使用自身的相机标定、首帧 PLY 和运动预热权重。

可以通过仓库内的上游训练入口获得 SH 阶数为 1 的首帧高斯模型。
首帧初始化质量会影响整个序列。请将下面的 `/absolute/...` 替换为实际绝对路径。

```bash
python train_initial.py -s /absolute/SCENE/colmap_0 -m /absolute/INITIAL \
  --eval --resolution 2 --sh_degree 1 --iterations 10000 \
  --test_iterations 10000 --save_iterations 10000
python -m flash4dgc warmup \
  --ply /absolute/INITIAL/point_cloud/iteration_10000/point_cloud.ply \
  --output /absolute/RUN/motion_warmup.pth
python -m flash4dgc config --dataset /absolute/SCENE --initial /absolute/INITIAL \
  --warmup /absolute/RUN/motion_warmup.pth --output /absolute/RUN/config.json \
  --resolution 2 --test-camera cam00
```

## 3. 闭环训练

```bash
python -m flash4dgc train --config /absolute/RUN/config.json \
  --output /absolute/RUN/closedloop300 --frames 300 --port 6393
```

短测可使用 `--frames 31`，并指定独立输出目录。
下文的转码与评测流程采用冻结的 300 帧协议，不接受 31 帧短测输出。
完整训练耗时较长，建议在服务器上通过 tmux 启动。
训练输出包含用于诊断的完整 PLY，磁盘占用可能远大于实际播放数据包。

闭环适配器将上一帧的解码状态作为训练参考，在提交更新前重新加载已保存的运动模型，
并在新进程中递归回放生成的数据包。上游 `train_frames.py` 未被重写。

## 4. 转码导出与独立回放评测

```bash
python -m flash4dgc encode --source /absolute/RUN/closedloop300 \
  --output /absolute/RUN/flash_plane300

python -m flash4dgc evaluate --source /absolute/RUN/closedloop300 \
  --output /absolute/RUN/flash_plane300 --case closedloop_reference \
  --results-root /absolute/RUN/timing_1
python -m flash4dgc evaluate --source /absolute/RUN/closedloop300 \
  --output /absolute/RUN/flash_plane300 --case flash_state \
  --results-root /absolute/RUN/timing_1
python -m flash4dgc compare --results-root /absolute/RUN/timing_1
```

编码时会验证整数和 GPU FP32 恢复结果的一致性，并输出数据包清单、编解码配置和独立的 Kernel 测速结果。
将两条评测命令的结果目录改为 `timing_2` 后再次运行，即可获得第二轮独立测速。
如需评测 PSNR、SSIM 和 LPIPS，请为两种方法分别添加 `--quality`，
并使用新的结果根目录，例如 `/absolute/RUN/quality`。
质量专用运行不输出测速结果。

**回放需要配合使用两个目录。** Flash 运动数据包位于 `flash_plane300/encoded`，
首帧高斯包、新增高斯包和共享场景配置仍保存在参考模型目录中。
仅复制 Flash 目录并不能得到可独立回放的完整码流。
参考方法回放还需要其原始 MEM 文件。
Flash 回放不需要用于诊断的逐帧完整 PLY，但转码需要源模型的 MEM 文件。
一次已记录实验中的源目录和导出目录路径应保持不变。

## 5. 核心文件与指标口径

|文件|作用|
|---|---|
|`flash4dgc/cli.py`|仓库内统一命令入口|
|`flash4dgc/runtime/run_4dgc_closedloop30.py`|闭环训练与参考模型回放|
|`flash4dgc/runtime/evaluate_flash_closedloop.py`|转码导出、配对回放与指标评测|
|`flash4dgc/runtime/plane_codec.py`, `plane_unpack.cu`|细网格无损分页编码与 CUDA 解码|
|`flash4dgc/runtime/sparse_codec.py`, `sparse_unpack.cu`|粗网格稀疏编码与解码|
|`flash4dgc/runtime/raw_motion_state.py`|FP32 网络权重与空间边界状态加载|

质量评测覆盖第 0–299 帧。回放测速覆盖第 1–299 帧，包含文件读取、状态恢复、
运动推理、新增高斯处理、光栅化和同步。
测速不包含网络传输、初始化/JIT 编译、真值图像加载和质量指标计算。
压缩数据已驻留 GPU 时的纯 CUDA Kernel 解包耗时是另一个独立指标。
合并吞吐 FPS 等于计时帧数除以这些帧的总耗时，不是单独光栅化的 FPS。


