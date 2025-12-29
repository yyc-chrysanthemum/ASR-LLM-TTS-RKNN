#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音助手主程序入口
"""
import logging
import os
import time
import threading
import audioop
import subprocess
import gc
import numpy as np
import pyaudio
import soundfile as sf
from queue import Queue
from enum import Enum
from pathlib import Path

from config.config import *
from services.memory_monitor import MemoryMonitor
from services.asr_service import AsrService
from services.llm_service import LlmService
from services.tts_service import TtsService

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("pyaudio").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# 全局队列
audio_queue = Queue()
play_queue = Queue()

# 内存监控
memory_monitor = MemoryMonitor()

class RecorderState(Enum):
    STOPPED = 0
    LISTENING = 1
    RECORDING = 2

class AudioRecorder:
    logger = logging.getLogger("AudioRecorder")
    p = None
    stream = None
    state = RecorderState.STOPPED
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    SAVE_DIR = WORKDIR

    @classmethod
    def start_stream(cls):
        if cls.p is None:
            cls.p = pyaudio.PyAudio()
        if cls.stream is None:
            cls.stream = cls.p.open(
                format=cls.FORMAT, channels=cls.CHANNELS, rate=RATE, input=True,
                frames_per_buffer=cls.CHUNK, start=False
            )
        cls.logger.info("麦克风流已初始化。")

    @classmethod
    def stop_stream(cls):
        if cls.stream:
            cls.stream.stop_stream()
            cls.stream.close()
            cls.stream = None
        if cls.p:
            cls.p.terminate()
            cls.p = None
        cls.logger.info("麦克风流已关闭。")

    @classmethod
    def record_loop(cls):
        cls.start_stream()
        cls.stream.start_stream() 

        cls.state = RecorderState.LISTENING
        cls.logger.info("录音线程启动，进入监听模式。")

        chunks_per_sec = RATE / cls.CHUNK
        silence_limit_chunks = int(chunks_per_sec * SILENCE_TIMEOUT_SEC)
        max_record_chunks = int(chunks_per_sec * MAX_RECORD_SEC)

        while cls.state != RecorderState.STOPPED:
            cls.logger.info("\n--- 请开始说话 (正在监听麦克风) ---")
            
            frames = []
            silent_chunks = 0
            is_recording = False
            LISTENING_timeout_start = time.time()

            while cls.state != RecorderState.STOPPED:
                if not is_recording and (time.time() - LISTENING_timeout_start > SILENCE_MAX_SEC):
                    cls.logger.debug(f"🕓 {SILENCE_MAX_SEC}秒未检测到语音，继续监听...")
                    LISTENING_timeout_start = time.time()
                
                data = cls.stream.read(cls.CHUNK, exception_on_overflow=False)
                rms = audioop.rms(data, 2)

                if not is_recording:
                    if rms > RMS_THRESHOLD:
                        cls.logger.info("🎯 检测到语音，开始录制...")
                        is_recording = True
                        frames.append(data)
                        silent_chunks = 0
                
                elif is_recording:
                    frames.append(data)
                    if rms < RMS_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    
                    current_chunks = len(frames)
                    
                    if silent_chunks > silence_limit_chunks:
                        cls.logger.info(f"🔇 检测到 {SILENCE_TIMEOUT_SEC}s 静音，停止录制。")
                        break
                    
                    if current_chunks > max_record_chunks:
                        cls.logger.info(f"🎤 达到最大录制时长 ({MAX_RECORD_SEC}秒)，停止录制。")
                        break

            if is_recording and frames:
                audio_data_bytes = b"".join(frames)
                audio_data_int16 = np.frombuffer(audio_data_bytes, dtype=np.int16)
                audio_data_f32 = audio_data_int16.astype(np.float32) / 32768.0
                
                duration = len(audio_data_f32) / RATE
                if duration < 0.5:
                    cls.logger.info(f"录音太短 ({duration:.2f}s)，忽略。")
                else:
                    cls.logger.info(f"录音完成，总时长 {duration:.2f} 秒。将数据放入队列。")
                    audio_queue.put(audio_data_f32)
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(cls.SAVE_DIR, f"record_{timestamp}.wav")
                    sf.write(save_path, audio_data_f32, RATE, subtype='PCM_16')
                    cls.logger.info(f"录音已保存到: {save_path}")
            
        cls.stop_stream()

def convert_to_target_format(src_file, dst_file, target_sr=TARGET_PLAY_SR, target_ch=TARGET_PLAY_CH):
    logger = logging.getLogger("AudioConverter")
    cmd = [
        "ffmpeg", "-y", "-i", src_file,
        "-ar", str(target_sr),
        "-ac", str(target_ch),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        dst_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.warning(f"ffmpeg 转换警告: {result.stderr.decode()}")
        return False
    return True

def play_audio_file(play_src_file, tts_gen_time, play_device=PLAY_DEVICE):
    logger = logging.getLogger("AudioPlayer")
    play_file = os.path.join("/dev/shm", f"tts_out_play_{int(time.time()*1000)}.wav")
   
    if not convert_to_target_format(play_src_file, play_file):
        logger.warning("⚠️ 音频转换失败，将直接播放原文件")
        play_file = play_src_file
    
    duration = sf.info(play_file).duration
    start_play = time.time()
    played = False
    
    cmd = ["aplay", "-D", play_device, play_file]
    logger.debug(f"播放命令: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"🔉 TTS 播放耗时: {time.time() - start_play:.2f}s")
    
    if os.path.exists(play_file) and play_file != play_src_file:
        os.remove(play_file)
    return played

def playback_worker():
    logger = logging.getLogger("PlaybackWorker")
    while True:
        wav_path = play_queue.get()
        if wav_path is None:
            break
        success = play_audio_file(wav_path, 0, PLAY_DEVICE)
        try:
            os.remove(wav_path)
        except Exception as e:
            logger.debug(f"删除临时文件失败: {e}")

def main():
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()
    logger = logging.getLogger("MainPipeline")
    memory_monitor.log_memory("程序启动")
    logger.info("=== 智能助手启动 ===")
   
    logger.info("--- 正在加载 ASR 服务 ---")
    asr_service = AsrService(
        mvn_path=ASR_MVN_PATH, embed_path=ASR_EMBED_PATH, rknn_path=ASR_RKNN_PATH,
        bpe_path=ASR_BPE_PATH, asr_dir=str(ASR_DIR)
    )
   
    logger.info("--- 正在加载 LLM 服务 ---")
    llm_service = LlmService(script_path=LLM_SCRIPT_PATH, idle_timeout=LLM_IDLE_TIMEOUT)
    
    logger.info("--- 正在加载 TTS 服务 ---")
    tts_service = TtsService(
        encoder_path=TTS_ENCODER_PATH, decoder_path=TTS_DECODER_PATH, lexicon_path=TTS_LEXICON_PATH,
        token_path=TTS_TOKEN_PATH, g_bin_path=TTS_G_BIN_PATH
    )
   
    recorder_thread = threading.Thread(target=AudioRecorder.record_loop, daemon=True)
    recorder_thread.start()
   
    while True:
        try:
            mem_pipeline_start = memory_monitor.get_memory_info()
            audio_data_f32 = audio_queue.get()
           
            logger.info(f"\n--- 从队列获取到新的语音 (时长 {len(audio_data_f32) / RATE:.2f}s) ---")
            pipeline_start_time = time.time()
            memory_monitor.log_memory("新一轮推理开始")
           
            user_text, asr_time = asr_service.transcribe(audio_data_f32, language="zh", use_itn=True)
            if not user_text:
                logger.warning("⚠️ ASR 未返回有效结果，跳过本轮")
                continue
            logger.info(f"?? 听写结果: {user_text}")
           
            tts_total_time = 0.0
            sentence_count = 0
            first_sentence_time = None
           
            def on_sentence_generated(sentence: str, sentence_time: float, is_first: bool):
                nonlocal tts_total_time, sentence_count, first_sentence_time
                sentence_clean = sentence.strip()
                for prefix in ["[ASR错误]", "[CMD]", "robot:", "assistant:"]:
                    if sentence_clean.lower().startswith(prefix.lower()):
                        sentence_clean = sentence_clean[len(prefix):].strip()
                if not sentence_clean:
                    return
                sentence_count += 1
                if is_first:
                    first_sentence_time = sentence_time
                
                wav_path = f"/dev/shm/tts_stream_{sentence_count}_{int(time.time()*1000)}.wav"
               
                success, tts_time, enc_time, dec_time = tts_service.synthesize_sentence(sentence_clean, wav_path)
                if success:
                    play_queue.put(wav_path)
                    logger.info(f"✅ 第 {sentence_count} 句合成完成 ({tts_time:.3f}s)")
                    tts_total_time += tts_time
           
            full_reply, llm_time, llm_first_sentence_time = llm_service.chat_stream(user_text, on_sentence_generated)
           
            logger.info(f"💬 LLM 完整回复: {full_reply}")
            logger.info(f"🧠 LLM 总耗时: {llm_time:.2f}s")
            logger.info(f"🗣️ TTS 总合成耗时: {tts_total_time:.2f}s (共 {sentence_count} 个句子)")
           
            pipeline_end_time = time.time()
            total_pipeline_time = pipeline_end_time - pipeline_start_time
           
            logger.info("\n" + "~"*50)
            logger.info("--- 计时结果 ---")
            logger.info(f"🎤 ASR 耗时: {asr_time:.2f}s")
            logger.info(f"⚡ 首句生成时间: {first_sentence_time if first_sentence_time else 0:.2f}s")
            logger.info(f"🧠 LLM 总耗时: {llm_time:.2f}s")
            logger.info(f"??️ TTS 总耗时: {tts_total_time:.2f}s")
            logger.info(f"🔥 整体推理总用时: {total_pipeline_time:.2f}s")
            logger.info(f"?? 首次响应延迟: {asr_time + (first_sentence_time if first_sentence_time else 0):.2f}s")
            logger.info("~"*50)
           
            mem_pipeline_end = memory_monitor.get_memory_info()
            pipeline_delta = memory_monitor.get_memory_delta(mem_pipeline_start)
            logger.info(f"--- 本轮推理内存变化: {memory_monitor.format_delta(pipeline_delta)}")
            memory_monitor.log_memory("本轮推理完成")
           
            gc.collect()
            logger.info("--- 流程结束,返回待命状态 ---\n")
           
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            break
        except Exception as e:
            logger.error(f"❌ 本轮推理发生错误: {e}", exc_info=True)
            logger.info("⚠️ 跳过本轮，继续下一轮...")
            gc.collect()
            continue

if __name__ == "__main__":
    main()
