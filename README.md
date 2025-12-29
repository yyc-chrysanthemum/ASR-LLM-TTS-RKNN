# 语音助手项目

这是一个基于开源模型构建的，基于 RK3588 的语音助手系统，支持 ASR（SenseVoice）、LLM（Qwen2.5-0.5B/1.5B）、TTS（MeloTTS）功能。


## 环境要求

```
python=3.10
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

## 功能特性

- **ASR**: 基于 SenseVoice 的语音识别
- **LLM**: 支持流式输出的语言模型 Qwen2.5-0.5B/1.5B
- **TTS**: 基于 MeloTTS 的语音合成
- **实时交互**: 支持流式 LLM 输出和实时 TTS 合成
- **内存监控**: 实时监控系统内存使用情况

## 配置说明

所有配置项位于 `config/config.py`，包括：
- 模型路径配置
- 音频设备配置
- 录音参数配置
- LLM 超时配置
- TTS 采样率配置

## 注意事项

1. 确保所有模型文件已放置在 `models/` 对应目录下
2. 确保音频设备配置正确
3. 确保 LLM 启动脚本 `run_llm.sh` 可执行
4. 确保 ASR 和 TTS 的 Python 模块文件已就位

## 项目结构

```
voice_assistant_project/
├── config/                    # 全局配置模块
│   ├── __init__.py
│   └── config.py             # 所有路径、设备、参数配置
├── services/                  # 服务封装模块
│   ├── __init__.py
│   ├── asr_service.py        # ASR服务类
│   ├── llm_service.py        # LLM服务类
│   ├── tts_service.py        # TTS服务类
│   └── memory_monitor.py     # 内存监控类
├── models/                    # 模型和数据文件夹
│   ├── asr/                  # ASR模型文件
│   ├── llm/                  # LLM相关
│   └── tts/                  # TTS模型文件
├── main.py                    # 主执行入口
├── requirements.txt           # 依赖清单
└── README.md                  # 项目说明
```

## 测试(板子上的路径)
#### 1.LLM模型转换
```bash
cd ～/rknn-llm/rknn-llm-release-v1.2.2/examples/multimodal_model_demo/export/

python export_rkllm.py
```
#### 2.ASR模型
```bash
python ./models/asr/sensevoice_rknn.py --audio_file data/output.wav 
```
如果使用自己的音频文件测试发现识别不正常，你可能需要提前将它转换为16kHz, 16bit, 单声道的wav格式。
```bash
ffmpeg -i input.mp3 -f wav -acodec pcm_s16le -ac 1 -ar 16000 output.wav
```
#### 3.TTS模型
```bash
python ./models/tts/melotts_rknn.py -s "The text you want to generate."
```
#### 4.RKNN模型转换
需要提前安装rknn-toolkit2, 测试可用的版本为2.3.3a25，可从 https://console.zbox.filez.com/l/I00fc3 下载(密码为"rknn")

## 参考链接
**ASR**:
- https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main
- https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2
- 下载或转换onnx模型,可以从 https://huggingface.co/lovemefan/SenseVoice-onnx 下载到onnx模型.
- https://huggingface.co/ThomasTheMaker/SenseVoiceSmall-RKNN2

**TTS**:
- https://huggingface.co/happyme531/MeloTTS-RKNN2

**LLM**:
- https://huggingface.co/3ib0n/Qwen2.5-14B-Instruct-rkllm

## 使用设备：RK3588/3576

**🍊 Orange Pi 5 Plus**
- 设备: Orange Pi 5 Plus (RK3588)
- 系统总内存: ~8 GB (7934.67 MB)

**Purple Pi OH2**
- 设备: Purple Pi OH2 (RK3576)
- 系统总内存: ~4 GB (3895.01 MB)/~8 GB (7934.67 MB)

