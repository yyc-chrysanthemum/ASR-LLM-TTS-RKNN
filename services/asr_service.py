#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 服务模块
"""
import logging
import re
import time
import sys
from pathlib import Path
from typing import Tuple

# 添加 ASR 目录到路径
ASR_DIR = str(Path(__file__).parent.parent / "models" / "asr")
sys.path.append(ASR_DIR)

from sensevoice_rknn import *
from config.config import *

class AsrService:
    def __init__(self, mvn_path, embed_path, rknn_path, bpe_path, asr_dir):
        self.logger = logging.getLogger("AsrService")
        self.logger.info("加载 ASR 模型...")
        start_time = time.time()
        self.front = WavFrontend(cmvn_file=mvn_path) 
        self.vad = FSMNVad(asr_dir)
        self.model = SenseVoiceInferenceSession(
            embed_path, rknn_path, bpe_path, device_id=-1, intra_op_num_threads=4
        )
        self.languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.logger.info(f"ASR模型加载完毕，耗时 {time.time() - start_time:.2f} 秒。")

    def transcribe(self, waveform_16k_f32, language="zh", use_itn=True) -> Tuple[str, float]:
        self.logger.info("开始 ASR 推理...")
        start_time = time.time()
        
        segments = self.vad.segments_offline(waveform_16k_f32) 
        
        if not segments:
            self.logger.warning("VAD 未检测到语音片段。")
            return "", 0.0
            
        self.logger.info(f"VAD 检测到 {len(segments)} 个片段。")
        full_text = []

        for i, part in enumerate(segments):
            start_ms, end_ms = part[0], part[1]
            start_frame = int(start_ms * 16) 
            end_frame = int(end_ms * 16)
            segment_audio = waveform_16k_f32[start_frame:end_frame]
            
            if len(segment_audio) < 160: 
                continue 

            audio_feats = self.front.get_features(segment_audio)
            asr_result = self.model(
                audio_feats[None, ...], 
                language=self.languages.get(language, 0), 
                use_itn=use_itn
            )
            
            self.logger.info(f"[片段 {i}] [{start_ms/1000:.2f}s - {end_ms/1000:.2f}s] {asr_result}")
            full_text.append(asr_result)
        
        final_text = "".join(full_text)
        cleaned_text = re.sub(r'<\|[^>]*\|>', '', final_text)
        cleaned_text = cleaned_text.strip(' \n\r\t,。!?:;"\'。')
        
        if cleaned_text:
            final_text = cleaned_text
            
        elapsed = time.time() - start_time
        return final_text, elapsed

    def close(self):
        self.model.release()