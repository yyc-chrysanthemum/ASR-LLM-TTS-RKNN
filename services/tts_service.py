#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 服务模块
"""
import logging
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from rknnlite.api.rknn_lite import RKNNLite

# 添加 TTS 目录到路径
TTS_DIR = str(Path(__file__).parent.parent / "models" / "tts")
sys.path.append(TTS_DIR)

from melotts_rknn import *
from config.config import *

class TtsService:
    def __init__(self, encoder_path, decoder_path, lexicon_path, token_path, g_bin_path, 
                 sample_rate=TTS_SAMPLE_RATE, speed=TTS_SPEED):
        self.logger = logging.getLogger("TtsService")
        self.sample_rate = sample_rate
        self.speed = speed
        self.dec_len = 65536 // 512
        self.logger.info("正在加载 TTS 模型...")
        start_time = time.time()

        self.lexicon = Lexicon(lexicon_path, token_path)
        sess_opt = SessionOptions()
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_enc = InferenceSession(encoder_path, sess_opt, providers=["CPUExecutionProvider"])
        self.decoder = RKNNLite()
        ret = self.decoder.load_rknn(decoder_path)
        if ret != 0:
            raise RuntimeError("Load decoder.rknn failed")
        self.decoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.g = np.fromfile(g_bin_path, dtype=np.float32).reshape(1, 256, 1)
        self.logger.info(f"TTS 模型加载完成，耗时 {time.time() - start_time:.2f}s")

    def synthesize_sentence(self, text: str, output_path: str) -> Tuple[bool, float, float, float]:
        start_time = time.time()
        enc_time = dec_time = 0.0

        text = text
        if not text:
            return False, 0, 0, 0

        audio_segments = []

        phone_str, yinjie_num, phones, tones = self.lexicon.convert(text)

        phone_str = intersperse(phone_str, 0)
        phones_np = np.array(intersperse(phones, 0), dtype=np.int32)
        tones_np = np.array(intersperse(tones, 0), dtype=np.int32)
        yinjie_num = np.array(yinjie_num, dtype=np.int32) * 2
        if yinjie_num.size > 0:
            yinjie_num[0] += 1

        pron_slices = generate_pronounce_slice(yinjie_num)
        phone_len = phones_np.shape[0]
        language = np.array([3] * phone_len, dtype=np.int32)

        enc_start = time.time()
        z_p, pronoun_lens, audio_len_scalar = self.sess_enc.run(None, {
            'phone': phones_np,
            'g': self.g,
            'tone': tones_np,
            'language': language,
            'noise_scale': np.array([0.0], dtype=np.float32),
            'length_scale': np.array([1.0 / self.speed], dtype=np.float32),
            'noise_scale_w': np.array([0.0], dtype=np.float32),
            'sdp_ratio': np.array([0.0], dtype=np.float32),
        })
        enc_time += time.time() - enc_start

        audio_len = int(audio_len_scalar)
        pronoun_lens = np.array(pronoun_lens).flatten()
        pron_num = generate_word_pron_num(pronoun_lens, pron_slices)

        actual_size = z_p.shape[-1]
        need_pad = self.dec_len * ((actual_size + self.dec_len - 1) // self.dec_len) - actual_size
        if need_pad > 0:
            z_p = np.pad(z_p, ((0,0),(0,0),(0, need_pad)), 'constant')

        pron_num_slices, zp_slices, strip_flags, _, is_long_list = generate_decode_slices(pron_num, self.dec_len)

        sub_audio_list = []
        for i in range(len(pron_num_slices)):
            p_start, p_end = pron_num_slices[i]
            z_start, z_end = zp_slices[i]
            strip_head, strip_tail = strip_flags[i]

            if is_long_list[i]:
                sub_audio_list.extend(decode_long_word(self.decoder, z_p[..., z_start:z_end], self.g, self.dec_len))
            else:
                zp_slice = z_p[..., z_start:z_end]
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.pad(zp_slice, ((0,0),(0,0),(0, self.dec_len - zp_slice.shape[-1])), 'constant')

                dec_start = time.time()
                audio_raw = self.decoder.inference(inputs=[zp_slice, self.g])[0].flatten()
                dec_time += time.time() - dec_start

                audio_raw = audio_raw[:512 * (z_end - z_start)]

                if strip_head and p_start > 0:
                    audio_raw = audio_raw[512 * pron_num[p_start]:]
                if strip_tail and p_end < len(pron_num):
                    audio_raw = audio_raw[:-512 * pron_num[p_end - 1]]

                sub_audio_list.append(audio_raw)

        merged_audio = merge_sub_audio(sub_audio_list, pad_size=0, audio_len=audio_len)
        audio_segments.append(merged_audio)

        final_audio = audio_numpy_concat(audio_segments, sr=self.sample_rate, speed=self.speed)
        sf.write(output_path, final_audio, self.sample_rate)
        total_time = time.time() - start_time
        return True, total_time, enc_time, dec_time

    def close(self):
        self.decoder.release()