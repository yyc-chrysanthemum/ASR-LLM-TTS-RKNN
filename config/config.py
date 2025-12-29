#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置模块
集中管理所有路径、设备、参数配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 工作目录
WORKDIR = os.path.join(os.path.expanduser("~"), "asr-llm-tts")
os.makedirs(WORKDIR, exist_ok=True)

# 模型目录配置
MODELS_DIR = PROJECT_ROOT / "models"
ASR_DIR = MODELS_DIR / "asr"
LLM_DIR = MODELS_DIR / "llm"
TTS_DIR = MODELS_DIR / "tts"

# ASR 模型路径
ASR_RKNN_PATH = str(ASR_DIR / "sense-voice-encoder.rknn")
ASR_EMBED_PATH = str(ASR_DIR / "embedding.npy")
ASR_BPE_PATH = str(ASR_DIR / "chn_jpn_yue_eng_ko_spectok.bpe.model")
ASR_VAD_ONNX_PATH = str(ASR_DIR / "fsmnvad-offline.onnx")
ASR_VAD_CONFIG_YAML = str(ASR_DIR / "fsmn-config.yaml")
ASR_MVN_PATH = str(ASR_DIR / "am.mvn")

# LLM 配置
LLM_SCRIPT_PATH = str(LLM_DIR / "run_llm.sh")

# TTS 模型路径
TTS_ENCODER_PATH = str(TTS_DIR / "encoder.onnx")
TTS_DECODER_PATH = str(TTS_DIR / "decoder.rknn")
TTS_LEXICON_PATH = str(TTS_DIR / "lexicon.txt")
TTS_TOKEN_PATH = str(TTS_DIR / "tokens.txt")
TTS_G_BIN_PATH = str(TTS_DIR / "g.bin")

# 音频配置
RATE = 16000
PLAY_DEVICE = "hw:0,0" #板子麦克风端口 aplay -l 查询
TARGET_PLAY_SR = 16000
TARGET_PLAY_CH = 2

# 录音配置
RMS_THRESHOLD = 500
SILENCE_TIMEOUT_SEC = 2.0
SILENCE_MAX_SEC = 5.0
MAX_RECORD_SEC = 10.0

# LLM 配置
LLM_IDLE_TIMEOUT = 1.2
LLM_INIT_TIMEOUT = 120.0

# TTS 配置
TTS_SAMPLE_RATE = 44100
TTS_SPEED = 1.0