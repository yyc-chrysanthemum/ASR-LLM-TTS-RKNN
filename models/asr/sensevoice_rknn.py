# File: onnx/fsmn_vad_ort_session.py
# ```py

# -*- coding:utf-8 -*-
# @FileName  :fsmn_vad_ort_session.py.py
# @Time      :2024/8/31 16:45
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

import argparse
import logging
import math
import os
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import kaldi_native_fbank as knf
import numpy as np
import sentencepiece as spm
import soundfile as sf
import yaml
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)
from rknnlite.api.rknn_lite import RKNNLite

RKNN_INPUT_LEN = 171

SPEECH_SCALE = 1/2  # 因为是fp16推理，如果中间结果太大可能会溢出变inf，所以需要缩放一下

class VadOrtInferRuntimeSession:
    def __init__(self, config, root_dir: Path):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            config["use_cuda"]
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, config[cuda_ep])]
        EP_list.append((cpu_ep, cpu_provider_options))

        config["model_path"] = root_dir / str(config["model_path"])
        self._verify_model(config["model_path"])
        logging.info(f"Loading onnx model at {str(config['model_path'])}")
        self.session = InferenceSession(
            str(config["model_path"]), sess_options=sess_opt, providers=EP_list
        )

        if config["use_cuda"] and cuda_ep not in self.session.get_providers():
            logging.warning(
                f"{cuda_ep} is not available for current env, "
                f"the inference part is automatically shifted to be "
                f"executed under {cpu_ep}.\n "
                "Please ensure the installed onnxruntime-gpu version"
                " matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(
        self, input_content: List[Union[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        if isinstance(input_content, list):
            input_dict = {
                "speech": input_content[0],
                "in_cache0": input_content[1],
                "in_cache1": input_content[2],
                "in_cache2": input_content[3],
                "in_cache3": input_content[4],
            }
        else:
            input_dict = {"speech": input_content}

        return self.session.run(None, input_dict)

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

# ```

# File: onnx/sense_voice_ort_session.py
# ```py
# -*- coding:utf-8 -*-
# @FileName  :sense_voice_onnxruntime.py
# @Time      :2024/7/17 20:53
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com


formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


class OrtInferRuntimeSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            device_id != "-1"
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        self._verify_model(model_file)

        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            result = self.session.run(self.get_output_names(), input_dict)
            return result
        except Exception as e:
            print(e)
            raise RuntimeError(f"ONNXRuntime inferece failed. ") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


def log_softmax(x: np.ndarray) -> np.ndarray:
    # Subtract the maximum value in each row for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    # Calculate the softmax of x
    softmax = np.exp(x - x_max)
    softmax_sum = np.sum(softmax, axis=-1, keepdims=True)
    softmax = softmax / softmax_sum
    # Calculate the log of the softmax values
    return np.log(softmax)


class SenseVoiceInferenceSession:
    def __init__(
        self,
        embedding_model_file,
        encoder_model_file,
        bpe_model_file,
        device_id=-1,
        intra_op_num_threads=4,
    ):
        logging.info(f"Loading model from {embedding_model_file}")

        self.embedding = np.load(embedding_model_file)
        logging.info(f"Loading model {encoder_model_file}")
        start = time.time()
        self.encoder = RKNNLite(verbose=False)
        self.encoder.load_rknn(encoder_model_file)
        self.encoder.init_runtime()

        logging.info(
            f"Loading {encoder_model_file} takes {time.time() - start:.2f} seconds"
        )
        self.blank_id = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model_file)

    def __call__(self, speech, language: int, use_itn: bool) -> np.ndarray:
        language_query = self.embedding[np.array([[language]])]

        # 14 means with itn, 15 means without itn
        text_norm_query = self.embedding[np.array([[14 if use_itn else 15]])]
        event_emo_query = self.embedding[np.array([[1, 2]])]

        # scale the speech
        speech = speech * SPEECH_SCALE
        
        input_content = np.concatenate(
            [
                language_query,
                event_emo_query,
                text_norm_query,
                speech,
            ],
            axis=1,
        ).astype(np.float32)
        print(input_content.shape)
        # pad [1, len, ...] to [1, RKNN_INPUT_LEN, ... ]
        input_content = np.pad(input_content, ((0, 0), (0, RKNN_INPUT_LEN - input_content.shape[1]), (0, 0)))
        print("padded shape:", input_content.shape)
        start_time = time.time()
        encoder_out = self.encoder.inference(inputs=[input_content])[0]
        end_time = time.time()
        print(f"encoder inference time: {end_time - start_time:.2f} seconds")
        # print(encoder_out)
        def unique_consecutive(arr):
            if len(arr) == 0:
                return arr
            # Create a boolean mask where True indicates the element is different from the previous one
            mask = np.append([True], arr[1:] != arr[:-1])
            out = arr[mask]
            out = out[out != self.blank_id]
            return out.tolist()
        
        #现在shape变成了1, n_vocab, n_seq. 这里axis需要改一下
        # hypos = unique_consecutive(encoder_out[0].argmax(axis=-1))
        hypos = unique_consecutive(encoder_out[0].argmax(axis=0))
        text = self.sp.DecodeIds(hypos)
        return text

# ```

# File: utils/frontend.py
# ```py
# -*- coding:utf-8 -*-
# @FileName  :frontend.py
# @Time      :2024/7/18 09:39
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0,
        **kwargs,
    ) -> None:
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.mel_opts.num_bins = n_mels
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file

        if self.cmvn_file:
            self.cmvn = self.load_cmvn()
        self.fbank_fn = None
        self.fbank_beg_idx = 0
        self.reset_status()

    def reset_status(self):
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_beg_idx = 0

    def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform * (1 << 15)
        self.fbank_fn = knf.OnlineFbank(self.opts)
        self.fbank_fn.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frames = self.fbank_fn.num_frames_ready
        mat = np.empty([frames, self.opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = self.fbank_fn.get_frame(i)
        feat = mat.astype(np.float32)
        feat_len = np.array(mat.shape[0]).astype(np.int32)
        return feat, feat_len

    def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lfr_m != 1 or self.lfr_n != 1:
            feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

        if self.cmvn_file:
            feat = self.apply_cmvn(feat)

        feat_len = np.array(feat.shape[0]).astype(np.int32)
        return feat, feat_len

    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(
            filename,
            always_2d=True,
            dtype="float32",
        )
        assert (
            sample_rate == 16000
        ), f"Only 16000 Hz is supported, but got {sample_rate}Hz"
        self.sample_rate = sample_rate
        data = data[:, 0]  # use only the first channel
        samples = np.ascontiguousarray(data)

        return samples, sample_rate

    @staticmethod
    def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        LFR_inputs = []

        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append(
                    (inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1)
                )
            else:
                # process last LFR frame
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))

                LFR_inputs.append(frame)
        LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
        return LFR_outputs

    def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
        inputs = (inputs + means) * vars
        return inputs

    def get_features(self, inputs: Union[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        if isinstance(inputs, str):
            inputs, _ = self.load_audio(inputs)

        fbank, _ = self.fbank(inputs)
        feats = self.apply_cmvn(self.apply_lfr(fbank, self.lfr_m, self.lfr_n))
        return feats

    def load_cmvn(
        self,
    ) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn

# ```

# File: utils/fsmn_vad.py
# ```py
# -*- coding:utf-8 -*-
# @FileName  :fsmn_vad.py
# @Time      :2024/8/31 16:50
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com



def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    if not Path(yaml_path).exists():
        raise FileExistsError(f"The {yaml_path} does not exist.")

    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


class VadStateMachine(Enum):
    kVadInStateStartPointNotDetected = 1
    kVadInStateInSpeechSegment = 2
    kVadInStateEndPointDetected = 3


class FrameState(Enum):
    kFrameStateInvalid = -1
    kFrameStateSpeech = 1
    kFrameStateSil = 0


# final voice/unvoice state per frame
class AudioChangeState(Enum):
    kChangeStateSpeech2Speech = 0
    kChangeStateSpeech2Sil = 1
    kChangeStateSil2Sil = 2
    kChangeStateSil2Speech = 3
    kChangeStateNoBegin = 4
    kChangeStateInvalid = 5


class VadDetectMode(Enum):
    kVadSingleUtteranceDetectMode = 0
    kVadMutipleUtteranceDetectMode = 1


class VADXOptions:
    def __init__(
        self,
        sample_rate: int = 16000,
        detect_mode: int = VadDetectMode.kVadMutipleUtteranceDetectMode.value,
        snr_mode: int = 0,
        max_end_silence_time: int = 800,
        max_start_silence_time: int = 3000,
        do_start_point_detection: bool = True,
        do_end_point_detection: bool = True,
        window_size_ms: int = 200,
        sil_to_speech_time_thres: int = 150,
        speech_to_sil_time_thres: int = 150,
        speech_2_noise_ratio: float = 1.0,
        do_extend: int = 1,
        lookback_time_start_point: int = 200,
        lookahead_time_end_point: int = 100,
        max_single_segment_time: int = 60000,
        nn_eval_block_size: int = 8,
        dcd_block_size: int = 4,
        snr_thres: int = -100.0,
        noise_frame_num_used_for_snr: int = 100,
        decibel_thres: int = -100.0,
        speech_noise_thres: float = 0.6,
        fe_prior_thres: float = 1e-4,
        silence_pdf_num: int = 1,
        sil_pdf_ids: List[int] = [0],
        speech_noise_thresh_low: float = -0.1,
        speech_noise_thresh_high: float = 0.3,
        output_frame_probs: bool = False,
        frame_in_ms: int = 10,
        frame_length_ms: int = 25,
    ):
        self.sample_rate = sample_rate
        self.detect_mode = detect_mode
        self.snr_mode = snr_mode
        self.max_end_silence_time = max_end_silence_time
        self.max_start_silence_time = max_start_silence_time
        self.do_start_point_detection = do_start_point_detection
        self.do_end_point_detection = do_end_point_detection
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time_thres = sil_to_speech_time_thres
        self.speech_to_sil_time_thres = speech_to_sil_time_thres
        self.speech_2_noise_ratio = speech_2_noise_ratio
        self.do_extend = do_extend
        self.lookback_time_start_point = lookback_time_start_point
        self.lookahead_time_end_point = lookahead_time_end_point
        self.max_single_segment_time = max_single_segment_time
        self.nn_eval_block_size = nn_eval_block_size
        self.dcd_block_size = dcd_block_size
        self.snr_thres = snr_thres
        self.noise_frame_num_used_for_snr = noise_frame_num_used_for_snr
        self.decibel_thres = decibel_thres
        self.speech_noise_thres = speech_noise_thres
        self.fe_prior_thres = fe_prior_thres
        self.silence_pdf_num = silence_pdf_num
        self.sil_pdf_ids = sil_pdf_ids
        self.speech_noise_thresh_low = speech_noise_thresh_low
        self.speech_noise_thresh_high = speech_noise_thresh_high
        self.output_frame_probs = output_frame_probs
        self.frame_in_ms = frame_in_ms
        self.frame_length_ms = frame_length_ms


class E2EVadSpeechBufWithDoa(object):
    def __init__(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0

    def reset(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0


class E2EVadFrameProb(object):
    def __init__(self):
        self.noise_prob = 0.0
        self.speech_prob = 0.0
        self.score = 0.0
        self.frame_id = 0
        self.frm_state = 0


class WindowDetector(object):
    def __init__(
        self,
        window_size_ms: int,
        sil_to_speech_time: int,
        speech_to_sil_time: int,
        frame_size_ms: int,
    ):
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time = sil_to_speech_time
        self.speech_to_sil_time = speech_to_sil_time
        self.frame_size_ms = frame_size_ms

        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame  # 初始化窗

        self.cur_win_pos = 0
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)

        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def reset(self) -> None:
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def get_win_size(self) -> int:
        return int(self.win_size_frame)

    def detect_one_frame(
        self, frameState: FrameState, frame_count: int
    ) -> AudioChangeState:
        cur_frame_state = FrameState.kFrameStateSil
        if frameState == FrameState.kFrameStateSpeech:
            cur_frame_state = 1
        elif frameState == FrameState.kFrameStateSil:
            cur_frame_state = 0
        else:
            return AudioChangeState.kChangeStateInvalid
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
            self.pre_frame_state == FrameState.kFrameStateSil
            and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSpeech
            return AudioChangeState.kChangeStateSil2Speech

        if (
            self.pre_frame_state == FrameState.kFrameStateSpeech
            and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSil
            return AudioChangeState.kChangeStateSpeech2Sil

        if self.pre_frame_state == FrameState.kFrameStateSil:
            return AudioChangeState.kChangeStateSil2Sil
        if self.pre_frame_state == FrameState.kFrameStateSpeech:
            return AudioChangeState.kChangeStateSpeech2Speech
        return AudioChangeState.kChangeStateInvalid

    def frame_size_ms(self) -> int:
        return int(self.frame_size_ms)


class E2EVadModel:
    def __init__(self, config, vad_post_args: Dict[str, Any], root_dir: Path):
        super(E2EVadModel, self).__init__()
        self.vad_opts = VADXOptions(**vad_post_args)
        self.windows_detector = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        self.model = VadOrtInferRuntimeSession(config, root_dir)
        self.all_reset_detection()

    def all_reset_detection(self):
        # init variables
        self.is_final = False
        self.data_buf_start_frame = 0
        self.frm_cnt = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.continous_silence_frame_count = 0
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.number_end_time_detected = 0
        self.sil_frame = 0
        self.sil_pdf_ids = self.vad_opts.sil_pdf_ids
        self.noise_average_decibel = -100.0
        self.pre_end_silence_detected = False
        self.next_seg = True

        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.frame_probs = []
        self.max_end_sil_frame_cnt_thresh = (
            self.vad_opts.max_end_silence_time - self.vad_opts.speech_to_sil_time_thres
        )
        self.speech_noise_thres = self.vad_opts.speech_noise_thres
        self.scores = None
        self.scores_offset = 0
        self.max_time_out = False
        self.decibel = []
        self.decibel_offset = 0
        self.data_buf_size = 0
        self.data_buf_all_size = 0
        self.waveform = None
        self.reset_detection()

    def reset_detection(self):
        self.continous_silence_frame_count = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.windows_detector.reset()
        self.sil_frame = 0
        self.frame_probs = []

    def compute_decibel(self) -> None:
        frame_sample_length = int(
            self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
        )
        frame_shift_length = int(
            self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
        )
        if self.data_buf_all_size == 0:
            self.data_buf_all_size = len(self.waveform[0])
            self.data_buf_size = self.data_buf_all_size
        else:
            self.data_buf_all_size += len(self.waveform[0])

        for offset in range(
            0, self.waveform.shape[1] - frame_sample_length + 1, frame_shift_length
        ):
            self.decibel.append(
                10
                * np.log10(
                    np.square(
                        self.waveform[0][offset : offset + frame_sample_length]
                    ).sum()
                    + 1e-6
                )
            )

    def compute_scores(self, feats: np.ndarray) -> None:
        scores = self.model(feats)
        self.vad_opts.nn_eval_block_size = scores[0].shape[1]
        self.frm_cnt += scores[0].shape[1]  # count total frames
        if isinstance(feats, list):
            # return B * T * D
            feats = feats[0]

        assert (
            scores[0].shape[1] == feats.shape[1]
        ), "The shape between feats and scores does not match"

        self.scores = scores[0]  # the first calculation
        self.scores_offset += self.scores.shape[1]

        return scores[1:]

    def pop_data_buf_till_frame(self, frame_idx: int) -> None:  # need check again
        while self.data_buf_start_frame < frame_idx:
            if self.data_buf_size >= int(
                self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                self.data_buf_start_frame += 1
                self.data_buf_size = (
                    self.data_buf_all_size
                    - self.data_buf_start_frame
                    * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
                )

    def pop_data_to_output_buf(
        self,
        start_frm: int,
        frm_cnt: int,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
    ) -> None:
        self.pop_data_buf_till_frame(start_frm)
        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(
                    self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
                ),
            )
            expected_sample_number += int(extra_sample)
        if end_point_is_sent_end:
            expected_sample_number = max(expected_sample_number, self.data_buf_size)
        if self.data_buf_size < expected_sample_number:
            logging.error("error in calling pop data_buf\n")

        if len(self.output_data_buf) == 0 or first_frm_is_start_point:
            self.output_data_buf.append(E2EVadSpeechBufWithDoa())
            self.output_data_buf[-1].reset()
            self.output_data_buf[-1].start_ms = start_frm * self.vad_opts.frame_in_ms
            self.output_data_buf[-1].end_ms = self.output_data_buf[-1].start_ms
            self.output_data_buf[-1].doa = 0
        cur_seg = self.output_data_buf[-1]
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            logging.error("warning\n")
        out_pos = len(cur_seg.buffer)  # cur_seg.buff现在没做任何操作
        data_to_pop = 0
        if end_point_is_sent_end:
            data_to_pop = expected_sample_number
        else:
            data_to_pop = int(
                frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            )
        if data_to_pop > self.data_buf_size:
            logging.error("VAD data_to_pop is bigger than self.data_buf.size()!!!\n")
            data_to_pop = self.data_buf_size
            expected_sample_number = self.data_buf_size

        cur_seg.doa = 0
        for sample_cpy_out in range(0, data_to_pop):
            # cur_seg.buffer[out_pos ++] = data_buf_.back();
            out_pos += 1
        for sample_cpy_out in range(data_to_pop, expected_sample_number):
            # cur_seg.buffer[out_pos++] = data_buf_.back()
            out_pos += 1
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            logging.error("Something wrong with the VAD algorithm\n")
        self.data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def on_silence_detected(self, valid_frame: int):
        self.lastest_confirmed_silence_frame = valid_frame
        if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            self.pop_data_buf_till_frame(valid_frame)
        # silence_detected_callback_
        # pass

    def on_voice_detected(self, valid_frame: int) -> None:
        self.latest_confirmed_speech_frame = valid_frame
        self.pop_data_to_output_buf(valid_frame, 1, False, False, False)

    def on_voice_start(self, start_frame: int, fake_result: bool = False) -> None:
        if self.vad_opts.do_start_point_detection:
            pass
        if self.confirmed_start_frame != -1:
            logging.error("not reset vad properly\n")
        else:
            self.confirmed_start_frame = start_frame

        if (
            not fake_result
            and self.vad_state_machine
            == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.pop_data_to_output_buf(
                self.confirmed_start_frame, 1, True, False, False
            )

    def on_voice_end(
        self, end_frame: int, fake_result: bool, is_last_frame: bool
    ) -> None:
        for t in range(self.latest_confirmed_speech_frame + 1, end_frame):
            self.on_voice_detected(t)
        if self.vad_opts.do_end_point_detection:
            pass
        if self.confirmed_end_frame != -1:
            logging.error("not reset vad properly\n")
        else:
            self.confirmed_end_frame = end_frame
        if not fake_result:
            self.sil_frame = 0
            self.pop_data_to_output_buf(
                self.confirmed_end_frame, 1, False, True, is_last_frame
            )
        self.number_end_time_detected += 1

    def maybe_on_voice_end_last_frame(
        self, is_final_frame: bool, cur_frm_idx: int
    ) -> None:
        if is_final_frame:
            self.on_voice_end(cur_frm_idx, False, True)
            self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected

    def get_latency(self) -> int:
        return int(self.latency_frm_num_at_start_point() * self.vad_opts.frame_in_ms)

    def latency_frm_num_at_start_point(self) -> int:
        vad_latency = self.windows_detector.get_win_size()
        if self.vad_opts.do_extend:
            vad_latency += int(
                self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms
            )
        return vad_latency

    def get_frame_state(self, t: int) -> FrameState:
        frame_state = FrameState.kFrameStateInvalid
        cur_decibel = self.decibel[t - self.decibel_offset]
        cur_snr = cur_decibel - self.noise_average_decibel
        # for each frame, calc log posterior probability of each state
        if cur_decibel < self.vad_opts.decibel_thres:
            frame_state = FrameState.kFrameStateSil
            self.detect_one_frame(frame_state, t, False)
            return frame_state

        sum_score = 0.0
        noise_prob = 0.0
        assert len(self.sil_pdf_ids) == self.vad_opts.silence_pdf_num
        if len(self.sil_pdf_ids) > 0:
            assert len(self.scores) == 1  # 只支持batch_size = 1的测试
            sil_pdf_scores = [
                self.scores[0][t - self.scores_offset][sil_pdf_id]
                for sil_pdf_id in self.sil_pdf_ids
            ]
            sum_score = sum(sil_pdf_scores)
            noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
            total_score = 1.0
            sum_score = total_score - sum_score
        speech_prob = math.log(sum_score)
        if self.vad_opts.output_frame_probs:
            frame_prob = E2EVadFrameProb()
            frame_prob.noise_prob = noise_prob
            frame_prob.speech_prob = speech_prob
            frame_prob.score = sum_score
            frame_prob.frame_id = t
            self.frame_probs.append(frame_prob)
        if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
            if (
                cur_snr >= self.vad_opts.snr_thres
                and cur_decibel >= self.vad_opts.decibel_thres
            ):
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if self.noise_average_decibel < -99.9:
                self.noise_average_decibel = cur_decibel
            else:
                self.noise_average_decibel = (
                    cur_decibel
                    + self.noise_average_decibel
                    * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def infer_offline(
        self,
        feats: np.ndarray,
        waveform: np.ndarray,
        in_cache: Dict[str, np.ndarray] = dict(),
        is_final: bool = False,
    ) -> Tuple[List[List[List[int]]], Dict[str, np.ndarray]]:
        self.waveform = waveform
        self.compute_decibel()

        self.compute_scores(feats)
        if not is_final:
            self.detect_common_frames()
        else:
            self.detect_last_frames()
        segments = []
        for batch_num in range(0, feats.shape[0]):  # only support batch_size = 1 now
            segment_batch = []
            if len(self.output_data_buf) > 0:
                for i in range(self.output_data_buf_offset, len(self.output_data_buf)):
                    if (
                        not self.output_data_buf[i].contain_seg_start_point
                        or not self.output_data_buf[i].contain_seg_end_point
                    ):
                        continue
                    segment = [
                        self.output_data_buf[i].start_ms,
                        self.output_data_buf[i].end_ms,
                    ]
                    segment_batch.append(segment)
                    self.output_data_buf_offset += 1  # need update this parameter
            if segment_batch:
                segments.append(segment_batch)

        if is_final:
            # reset class variables and clear the dict for the next query
            self.all_reset_detection()
        return segments, in_cache

    def infer_online(
        self,
        feats: np.ndarray,
        waveform: np.ndarray,
        in_cache: list = None,
        is_final: bool = False,
        max_end_sil: int = 800,
    ) -> Tuple[List[List[List[int]]], Dict[str, np.ndarray]]:
        feats = [feats]
        if in_cache is None:
            in_cache = []

        self.max_end_sil_frame_cnt_thresh = (
            max_end_sil - self.vad_opts.speech_to_sil_time_thres
        )
        self.waveform = waveform  # compute decibel for each frame
        feats.extend(in_cache)
        in_cache = self.compute_scores(feats)
        self.compute_decibel()

        if is_final:
            self.detect_last_frames()
        else:
            self.detect_common_frames()

        segments = []
        # only support batch_size = 1 now
        for batch_num in range(0, feats[0].shape[0]):
            if len(self.output_data_buf) > 0:
                for i in range(self.output_data_buf_offset, len(self.output_data_buf)):
                    if not self.output_data_buf[i].contain_seg_start_point:
                        continue
                    if (
                        not self.next_seg
                        and not self.output_data_buf[i].contain_seg_end_point
                    ):
                        continue
                    start_ms = self.output_data_buf[i].start_ms if self.next_seg else -1
                    if self.output_data_buf[i].contain_seg_end_point:
                        end_ms = self.output_data_buf[i].end_ms
                        self.next_seg = True
                        self.output_data_buf_offset += 1
                    else:
                        end_ms = -1
                        self.next_seg = False
                    segments.append([start_ms, end_ms])

        return segments, in_cache

    def get_frames_state(
        self,
        feats: np.ndarray,
        waveform: np.ndarray,
        in_cache: list = None,
        is_final: bool = False,
        max_end_sil: int = 800,
    ):
        feats = [feats]
        states = []
        if in_cache is None:
            in_cache = []

        self.max_end_sil_frame_cnt_thresh = (
            max_end_sil - self.vad_opts.speech_to_sil_time_thres
        )
        self.waveform = waveform  # compute decibel for each frame
        feats.extend(in_cache)
        in_cache = self.compute_scores(feats)
        self.compute_decibel()

        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return states

        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.get_frame_state(self.frm_cnt - 1 - i)
            states.append(frame_state)
            if i == 0 and is_final:
                logging.info("last frame detected")
                self.detect_one_frame(frame_state, self.frm_cnt - 1, True)
            else:
                self.detect_one_frame(frame_state, self.frm_cnt - 1 - i, False)

        return states

    def detect_common_frames(self) -> int:
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.get_frame_state(self.frm_cnt - 1 - i)
            # print(f"cur frame: {self.frm_cnt - 1 - i}, state is {frame_state}")
            self.detect_one_frame(frame_state, self.frm_cnt - 1 - i, False)

        self.decibel = self.decibel[self.vad_opts.nn_eval_block_size - 1 :]
        self.decibel_offset = self.frm_cnt - 1 - i
        return 0

    def detect_last_frames(self) -> int:
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.get_frame_state(self.frm_cnt - 1 - i)
            if i != 0:
                self.detect_one_frame(frame_state, self.frm_cnt - 1 - i, False)
            else:
                self.detect_one_frame(frame_state, self.frm_cnt - 1, True)

        return 0

    def detect_one_frame(
        self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool
    ) -> None:
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if math.fabs(1.0) > float(self.vad_opts.fe_prior_thres):
                tmp_cur_frm_state = FrameState.kFrameStateSpeech
            else:
                tmp_cur_frm_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil
        state_change = self.windows_detector.detect_one_frame(
            tmp_cur_frm_state, cur_frm_idx
        )
        frm_shift_in_ms = self.vad_opts.frame_in_ms
        if AudioChangeState.kChangeStateSil2Speech == state_change:
            self.continous_silence_frame_count = 0
            self.pre_end_silence_detected = False

            if (
                self.vad_state_machine
                == VadStateMachine.kVadInStateStartPointNotDetected
            ):
                start_frame = max(
                    self.data_buf_start_frame,
                    cur_frm_idx - self.latency_frm_num_at_start_point(),
                )
                self.on_voice_start(start_frame)
                self.vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.on_voice_detected(t)
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                for t in range(self.latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.on_voice_detected(t)
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.maybe_on_voice_end_last_frame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Sil == state_change:
            self.continous_silence_frame_count = 0
            if (
                self.vad_state_machine
                == VadStateMachine.kVadInStateStartPointNotDetected
            ):
                pass
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.maybe_on_voice_end_last_frame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Speech == state_change:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.max_time_out = True
                    self.on_voice_end(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx)
                else:
                    self.maybe_on_voice_end_last_frame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSil2Sil == state_change:
            self.continous_silence_frame_count += 1
            if (
                self.vad_state_machine
                == VadStateMachine.kVadInStateStartPointNotDetected
            ):
                # silence timeout, return zero length decision
                if (
                    (
                        self.vad_opts.detect_mode
                        == VadDetectMode.kVadSingleUtteranceDetectMode.value
                    )
                    and (
                        self.continous_silence_frame_count * frm_shift_in_ms
                        > self.vad_opts.max_start_silence_time
                    )
                ) or (is_final_frame and self.number_end_time_detected == 0):
                    for t in range(
                        self.lastest_confirmed_silence_frame + 1, cur_frm_idx
                    ):
                        self.on_silence_detected(t)
                    self.on_voice_start(0, True)
                    self.on_voice_end(0, True, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    if cur_frm_idx >= self.latency_frm_num_at_start_point():
                        self.on_silence_detected(
                            cur_frm_idx - self.latency_frm_num_at_start_point()
                        )
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    self.continous_silence_frame_count * frm_shift_in_ms
                    >= self.max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(
                        self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms
                    )
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.on_voice_end(cur_frm_idx - lookback_frame, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif self.vad_opts.do_extend and not is_final_frame:
                    if self.continous_silence_frame_count <= int(
                        self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.on_voice_detected(cur_frm_idx)
                else:
                    self.maybe_on_voice_end_last_frame(is_final_frame, cur_frm_idx)
            else:
                pass

        if (
            self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
            and self.vad_opts.detect_mode
            == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.reset_detection()


class FSMNVad(object):
    def __init__(self, config_dir: str):
        config_dir = Path(config_dir)
        self.config = read_yaml(config_dir / "fsmn-config.yaml")
        self.frontend = WavFrontend(
            cmvn_file=config_dir / "fsmn-am.mvn",
            **self.config["WavFrontend"]["frontend_conf"],
        )
        self.config["FSMN"]["model_path"] = config_dir / "fsmnvad-offline.onnx"

        self.vad = E2EVadModel(
            self.config["FSMN"], self.config["vadPostArgs"], config_dir
        )

    def set_parameters(self, mode):
        pass

    def extract_feature(self, waveform):
        fbank, _ = self.frontend.fbank(waveform)
        feats, feats_len = self.frontend.lfr_cmvn(fbank)
        return feats.astype(np.float32), feats_len

    def is_speech(self, buf, sample_rate=16000):
        assert sample_rate == 16000, "only support 16k sample rate"

    def segments_offline(self, waveform_path: Union[str, Path, np.ndarray]):
        """get sements of audio"""

        if isinstance(waveform_path, np.ndarray):
            waveform = waveform_path
        else:
            if not os.path.exists(waveform_path):
                raise FileExistsError(f"{waveform_path} is not exist.")
            if os.path.isfile(waveform_path):
                logging.info(f"load audio {waveform_path}")
                waveform, _sample_rate = sf.read(
                    waveform_path,
                    dtype="float32",
                )
            else:
                raise FileNotFoundError(str(Path))
            assert (
                _sample_rate == 16000
            ), f"only support 16k sample rate, current sample rate is {_sample_rate}"

        feats, feats_len = self.extract_feature(waveform)
        waveform = waveform[None, ...]
        segments_part, in_cache = self.vad.infer_offline(
            feats[None, ...], waveform, is_final=True
        )
        return segments_part[0]

# ```

# File: sense_voice.py
# ```py
# -*- coding:utf-8 -*-
# @FileName  :sense_voice.py.py
# @Time      :2024/7/18 15:40
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

def main():
    arg_parser = argparse.ArgumentParser(description="Sense Voice")
    arg_parser.add_argument("-a", "--audio_file", required=True, type=str, help="Model")
    download_model_path = os.path.dirname(__file__)
    arg_parser.add_argument(
        "-dp",
        "--download_path",
        default=download_model_path,
        type=str,
        help="dir path of resource downloaded",
    )
    arg_parser.add_argument("-d", "--device", default=-1, type=int, help="Device")
    arg_parser.add_argument(
        "-n", "--num_threads", default=4, type=int, help="Num threads"
    )
    arg_parser.add_argument(
        "-l",
        "--language",
        choices=languages.keys(),
        default="auto",
        type=str,
        help="Language",
    )
    arg_parser.add_argument("--use_itn", action="store_true", help="Use ITN")
    args = arg_parser.parse_args()

    front = WavFrontend(os.path.join(download_model_path, "am.mvn"))

    model = SenseVoiceInferenceSession(
        os.path.join(download_model_path, "embedding.npy"),
        os.path.join(
            download_model_path,
            "sense-voice-encoder.rknn",
        ),
        os.path.join(download_model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
        args.device,
        args.num_threads,
    )
    waveform, _sample_rate = sf.read(
        args.audio_file,
        dtype="float32",
        always_2d=True
    )

    logging.info(f"Audio {args.audio_file} is {len(waveform) / _sample_rate} seconds, {waveform.shape[1]} channel")
    # load vad model
    start = time.time()
    vad = FSMNVad(download_model_path)
    for channel_id, channel_data in enumerate(waveform.T):
        segments = vad.segments_offline(channel_data)
        results = ""
        for part in segments:
            audio_feats = front.get_features(channel_data[part[0] * 16 : part[1] * 16])
            asr_result = model(
                audio_feats[None, ...],
                language=languages[args.language],
                use_itn=args.use_itn,
            )
            logging.info(f"[Channel {channel_id}] [{part[0] / 1000}s - {part[1] / 1000}s] {asr_result}")
        vad.vad.all_reset_detection()
    decoding_time = time.time() - start
    logging.info(f"Decoder audio takes {decoding_time} seconds")
    logging.info(f"The RTF is {decoding_time/(waveform.shape[1] * len(waveform) / _sample_rate)}.")


if __name__ == "__main__":
    main()

