from .nodes.change_vocal_speed import AudioStretchNodeXuhuan1024

NODE_CLASS_MAPPINGS = { 
    "ChangeVocalSpeedXuhuan1024" : AudioStretchNodeXuhuan1024
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "ChangeVocalSpeedXuhuan1024" : "修改语速"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']