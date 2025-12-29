#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 服务模块（支持流式输出）
"""
import logging
import re
import os
import signal
import subprocess
import threading
import time
import select
from typing import Optional, Callable

from config.config import LLM_SCRIPT_PATH, LLM_IDLE_TIMEOUT, LLM_INIT_TIMEOUT

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

class LlmService:
    def __init__(self, script_path: str, cwd_dir: Optional[str] = None, 
                 idle_timeout: float = LLM_IDLE_TIMEOUT, init_timeout: float = LLM_INIT_TIMEOUT):
        self.logger = logging.getLogger("LlmService")
        self.script_path = script_path
        self.idle_timeout = float(idle_timeout)
        self.init_timeout = float(init_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stdout_fd: Optional[int] = None
        self._start_and_wait_ready()

    def _start_and_wait_ready(self):
        self.logger.info(f"启动 LLM 守护进程: {self.script_path}")
        
        self._proc = subprocess.Popen(
            [self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            close_fds=True
        )

        self._stdout_fd = self._proc.stdout.fileno()

        ready_buf = ""
        start = time.time()
        while True:
            elapsed = time.time() - start
            remaining = max(0.1, self.init_timeout - elapsed)
            rlist, _, _ = select.select([self._stdout_fd], [], [], remaining)
            if rlist:
                chunk = os.read(self._stdout_fd, 4096)
                if not chunk:
                    break
                s = chunk.decode("utf-8", errors="ignore")
                ready_buf += s
                if "rkllm init success" in ready_buf:
                    self.logger.info("LLM 已初始化完成。")
                    return
            else:
                if time.time() - start >= self.init_timeout:
                    self.logger.error("等待 'rkllm init success' 超时。")
                    raise TimeoutError("LLM init timeout")

    def chat_stream(self, prompt_text: str, sentence_callback: Callable):
        if not prompt_text:
            return "", 0.0, 0.0

        with self._lock:
            start_time = time.time()
            first_sentence_time = None
            
            self._proc.stdin.write((prompt_text + "\n").encode("utf-8"))
            self._proc.stdin.flush()

            collected = ""
            fd = self._stdout_fd
            sentence_delimiters = ['。', '！', '？', '.', '!', '?', '，', '；']
            buffer = ""
            sentence_count = 0

            while True:
                timeout = self.idle_timeout
                rlist, _, _ = select.select([fd], [], [], timeout)

                if rlist:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    s = chunk.decode("utf-8", errors="ignore")
                    collected += s
                    
                    s_clean = ANSI_RE.sub("", s)
                    for line in s_clean.split('\n'):
                        line = line
                        if not line:
                            continue
                        if line.startswith("I rkllm:") or line.startswith("rkllm init") or \
                           line.startswith("Input:") or line.startswith("user:") or \
                           "time_used=" in line or line == prompt_text:
                            continue
                        
                        if line.lower().startswith("robot:"):
                            line = line[len("robot:"):]
                        
                        buffer += line
                    
                    for delimiter in sentence_delimiters:
                        if delimiter in buffer:
                            parts = buffer.split(delimiter)
                            for i in range(len(parts) - 1):
                                sentence = parts[i] + delimiter
                                if sentence.strip(delimiter):
                                    sentence_count += 1
                                    current_time = time.time()
                                    is_first = (sentence_count == 1)
                                    if is_first:
                                        first_sentence_time = current_time - start_time
                                        self.logger.info(f"⚡ LLM 首句生成时间: {first_sentence_time:.2f}s")
                                    
                                    sentence_time = current_time - start_time
                                    self.logger.info(f"📝 LLM 句子 [{sentence_count}] (累计 {sentence_time:.2f}s): {sentence}")
                                    sentence_callback(sentence, sentence_time, is_first)
                            
                            buffer = parts[-1]
                            break
                    
                    continue
                else:
                    break
            
            if buffer:
                sentence_count += 1
                current_time = time.time()
                sentence_time = current_time - start_time
                is_first = (sentence_count == 1)
                if is_first:
                    first_sentence_time = current_time - start_time
                    self.logger.info(f"⚡ LLM 首句生成时间: {first_sentence_time:.2f}s")
                self.logger.info(f"📝 LLM 最后片段 [{sentence_count}] (累计 {sentence_time:.2f}s): {buffer}")
                sentence_callback(buffer, sentence_time, is_first)

            raw_output = ANSI_RE.sub("", collected)
            lines = [ln for ln in raw_output.splitlines() if ln]

            llm_report_sec = 0.0
            for ln in reversed(lines):
                if "time_used=" in ln:
                    m = re.search(r"time_used\s*=\s*(\d+)\s*ms", ln)
                    if m:
                        llm_report_sec = float(m.group(1)) / 1000.0
                    break

            robot_idx = None
            for i, ln in enumerate(lines):
                if ln.lower().startswith("robot:"):
                    robot_idx = i

            answer = ""
            if robot_idx is not None:
                captured = []
                for ln in lines[robot_idx:]:
                    if "time_used=" in ln:
                        break
                    if ln.lower().startswith("robot:"):
                        captured.append(ln[len("robot:"):])
                    else:
                        captured.append(ln)
                answer = " ".join([c for c in captured if c])
            else:
                filtered = []
                for ln in lines:
                    if ln.startswith("I rkllm:") or ln.startswith("rkllm init") or ln.startswith("Input:"):
                        continue
                    if "time_used=" in ln:
                        continue
                    if ln.lower().startswith("user:"):
                        continue
                    if ln == prompt_text or ln.startswith(prompt_text):
                        continue
                    filtered.append(ln)
                answer = " ".join(filtered)

            if prompt_text and answer.lower().startswith(prompt_text.lower()):
                answer = answer[len(prompt_text):]
            answer = re.sub(r'(?i)^user:\s*', '', answer)
            answer = re.sub(r'\s+', ' ', answer)

            elapsed = time.time() - start_time
            report_time = llm_report_sec if llm_report_sec > 0 else elapsed

            self.logger.info(f"💬 LLM 完整回答: {answer!r}，总耗时: {report_time:.3f}s")
            return answer, report_time, first_sentence_time if first_sentence_time else 0.0

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=3)
        self.logger.info("LLM 守护进程已关闭。")