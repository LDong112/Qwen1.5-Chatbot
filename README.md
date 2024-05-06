# Qwen1.5 Chatbot Web Application using Streamlit



## 简介

本项目展示了一个基于[Streamlit](https://streamlit.io/)构建的简洁而交互式的Web应用，旨在利用ModelScope上的Qwen1.5-0.5B-Chat-GPTQ-Int4模型（也可以采用qwen1.5系列的其他模型）与用户进行对话。应用集成了用户注册、对话历史管理及动态聊天展示等功能，提供了一个直观的界面，实现了人机交流体验。项目参考链接：[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm/tree/master/Qwen1.5)。

## 环境准备

- python版本是3.10
- cuda版本是12.1
- torch版本是2.1
-  pip install modelscope==1.9.5
- pip install "transformers>=4.37.0"
- pip install streamlit==1.24.0
- pip install sentencepiece==0.1.99
- pip install accelerate==0.24.1
- pip install transformers_stream_generator==0.0.4

## 快速上手

1. 克隆本仓库到本地。

```
git clone https://github.com/[你的用户名]/Qwen1.5-Chatbot.git && cd Qwen1.5-Chatbot
```

2. 运行 download.py 下载模型。

```python download.py
python download.py 
```

3. 使用Streamlit启动应用。

```
streamlit run chatbot.py
```

4. 在浏览器中打开由Streamlit提供的地址（通常为[http://localhost:8501](http://localhost:8501)）。

## 使用指南

- 启动后，输入用户名或选择注册新账号。
- 在聊天输入框中键入问题。
- 可从侧边栏菜单中选择删除当前账号。

