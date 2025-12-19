import os
import tempfile
import torch
import numpy as np
from pydub import AudioSegment
from audiostretchy.stretch import AudioStretch

class AudioStretchNodeXuhuan1024:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed_rate": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.05
                    }
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "stretch"
    CATEGORY = "SmallToolXuhuan1024"

    def stretch(self, audio, speed_rate):
        # 1. 提取并规范化输入 [B, C, N]
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # 确保输入是 3D: [B, C, N]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)

        # 取第一个 Batch 进行处理
        # waveform[0] -> [Channels, Samples]
        curr_waveform = waveform[0]
        
        # 转为单声道用于处理 (audiostretchy 对单声道支持更稳)
        if curr_waveform.shape[0] > 1:
            curr_waveform = curr_waveform.mean(dim=0, keepdim=True)

        # 2. 转换为 numpy int16 用于 pydub/wav
        samples = curr_waveform.squeeze(0).cpu().numpy()
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)

        # 3. 使用临时文件进行 audiostretchy 处理
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.wav")
            out_path = os.path.join(tmpdir, "out.wav")

            # 写入临时文件
            audio_seg = AudioSegment(
                samples.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # 统一采样率防止算法出错
            audio_seg = audio_seg.set_frame_rate(44100)
            audio_seg.export(in_path, format="wav")

            # 执行拉伸
            stretcher = AudioStretch()
            stretcher.open(in_path)
            stretcher.stretch(ratio=1.0 / speed_rate)
            stretcher.save(out_path)

            # 读回处理后的音频
            processed = AudioSegment.from_wav(out_path)
            new_sample_rate = processed.frame_rate
            
            # 4. 转换回 Tensor [B, C, N]
            p_samples = np.array(processed.get_array_of_samples(), dtype=np.float32)
            p_samples /= 32768.0  # 归一化到 [-1, 1]
            
            # 重新构造维度 [Channels, Samples]
            # processed.get_array_of_samples() 得到的是平铺的一维数组
            p_waveform = torch.from_numpy(p_samples).reshape(processed.channels, -1)
            
            # 关键修复：升维至 3D [Batch=1, Channels, Samples]
            # 这是 ComfyUI 预览/保存节点不报错的核心要求
            final_waveform = p_waveform.unsqueeze(0)

        # 5. 防御性处理
        final_waveform = torch.nan_to_num(final_waveform, nan=0.0).clamp(-1.0, 1.0).float()

        return ({
            "waveform": final_waveform,
            "sample_rate": new_sample_rate
        },)