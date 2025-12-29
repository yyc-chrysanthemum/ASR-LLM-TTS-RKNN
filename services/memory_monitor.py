#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存监控服务
"""
import logging
import psutil
from typing import Dict

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.logger = logging.getLogger("MemoryMonitor")
        self.baseline_memory = self.get_memory_info()
        vm = psutil.virtual_memory()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔧 RK3588 板子内存配置:")
        self.logger.info(f"  总内存: {vm.total / 1024 / 1024:.2f} MB ({vm.total / 1024 / 1024 / 1024:.2f} GB)")
        self.logger.info(f"  初始可用: {vm.available / 1024 / 1024:.2f} MB")
        self.logger.info(f"  初始使用率: {vm.percent:.2f}%")
        self.logger.info(f"{'='*60}\n")
        
    def get_memory_info(self) -> Dict[str, float]:
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'process_rss': mem_info.rss / 1024 / 1024,
            'process_vms': mem_info.vms / 1024 / 1024,
            'process_percent': self.process.memory_percent(),
            'system_total': virtual_mem.total / 1024 / 1024,
            'system_available': virtual_mem.available / 1024 / 1024,
            'system_used': virtual_mem.used / 1024 / 1024,
            'system_percent': virtual_mem.percent,
            'system_free': virtual_mem.free / 1024 / 1024,
            'system_buffers': getattr(virtual_mem, 'buffers', 0) / 1024 / 1024,
            'system_cached': getattr(virtual_mem, 'cached', 0) / 1024 / 1024,
        }
    
    def log_memory(self, stage: str, details: str = ""):
        mem = self.get_memory_info()
        delta_process = mem['process_rss'] - self.baseline_memory['process_rss']
        delta_system = mem['system_used'] - self.baseline_memory['system_used']
        
        log_msg = (
            f"\n{'='*70}\n"
            f"?? [{stage}] RK3588 板子内存状态 {details}\n"
            f"{'='*70}\n"
            f"🖥️  板子系统内存:\n"
            f"  ├─ 总内存: {mem['system_total']:.2f} MB\n"
            f"  ├─ 已使用: {mem['system_used']:.2f} MB ({mem['system_percent']:.2f}%)\n"
            f"  ├─ 可用内存: {mem['system_available']:.2f} MB\n"
            f"  └─ 系统内存变化: {delta_system:+.2f} MB\n"
            f"\n"
            f"📱 本进程内存:\n"
            f"  ├─ 物理内存: {mem['process_rss']:.2f} MB\n"
            f"  ├─ 占板子总内存: {mem['process_percent']:.2f}%\n"
            f"  └─ 进程内存变化: {delta_process:+.2f} MB\n"
            f"{'='*70}"
        )
        self.logger.info(log_msg)
        return mem
    
    def get_memory_delta(self, start_mem: Dict[str, float]) -> Dict[str, float]:
        current_mem = self.get_memory_info()
        return {
            'process_rss_delta': current_mem['process_rss'] - start_mem['process_rss'],
            'system_used_delta': current_mem['system_used'] - start_mem['system_used'],
            'system_available_delta': current_mem['system_available'] - start_mem['system_available'],
            'system_percent_delta': current_mem['system_percent'] - start_mem['system_percent'],
        }
    
    def format_delta(self, delta: Dict[str, float]) -> str:
        return (
            f"[板子] 已用: {delta['system_used_delta']:+.2f} MB, "
            f"使用率: {delta['system_percent_delta']:+.2f}% | "
            f"[进程] RSS: {delta['process_rss_delta']:+.2f} MB"
        )