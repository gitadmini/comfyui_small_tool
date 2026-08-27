from .nodes.change_vocal_speed import AudioStretchNodeXuhuan1024
from .nodes.ollama_path_chat import OllamaPathChat
from .nodes.ollama_path_chat import OllamaStopModel

NODE_CLASS_MAPPINGS = { 
    "ChangeVocalSpeedXuhuan1024" : AudioStretchNodeXuhuan1024,
    "OllamaPathChat": OllamaPathChat,
    "OllamaStopModel": OllamaStopModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChangeVocalSpeedXuhuan1024" : "修改语速",
    "OllamaPathChat": "Ollama Qwen3.8 Path Chat",
    "OllamaStopModel": "Ollama Stop Model",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

