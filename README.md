# comfyui_small_tool

这是一个小工具项目，将集成自己平时所需的小工具

## ollama qwen3.8

调用ollama qwen3.8的chat，可以在prompt中传文件地址，模型能够读取文件中内容。从而只输入一个字符串，就能让模型分析图片或者视频。

缓存选择是，调用模型，完成后下次直接读取缓存。选择否，则调用ollama stop来停掉模型，达到释放缓存的目的，此时只是释放缓存，不再调用模型。

需要事先本地安装ollama，并安装qwen38: /root/ollama run qwen3.8:latest


## 说话声音变速

使用audiostretchy，相比于一般的变速，能够较好地保留原声音色、音调、音质


## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/gitadmini/comfyui_small_tool.git
cd comfyui_small_tool
pip install -r requirements.txt
# 如果是便携版 linux
../../../python_embeded/python -m pip install -r requirements.txt
# 如果是便携版 windows
..\..\..\python_embeded\python -m pip install -r requirements.txt
```

