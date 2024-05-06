import os
from modelscope import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import json

st.title("💬 Qwen1.5 Chatbot")

if not os.path.exists("user_histories"):
    os.makedirs("user_histories")

model_name_or_path = './model/qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4'

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
    return tokenizer, model

tokenizer, model = get_model()

# 加载或创建用户数据
def load_users():
    try:
        with open("user_histories/users.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open("user_histories/users.json", "w") as file:
        json.dump(users, file)

users = load_users()

# 登录或注册用户
def login_or_register_user():
    if users:
        user_id_options = list(users.keys()) + ["Register New User"]
        user_id = st.sidebar.selectbox("Select user:", options=user_id_options)
        if user_id == "Register New User":
            user_id = st.sidebar.text_input("Enter your username:")
            if user_id:
                users[user_id] = {"conversations": {"default": []}}  # 创建一个名为"default"的默认对话记录键
                save_users(users)
    else:
        user_id = st.sidebar.text_input("Enter your username:")
        if user_id:
            users[user_id] = {"conversations": {"default": []}}  # 创建一个名为"default"的默认对话记录键
            save_users(users)

    # 添加删除用户选项
    if user_id:
        if st.sidebar.button("Delete Current Account"):
            del users[user_id]
            save_users(users)
            return None  # 返回None以确保用户被注销
    return user_id


user_id = login_or_register_user()

# 如果用户未登录，则退出程序
if not user_id:
    st.warning("Please enter a username to continue.")
    st.stop()

# 如果session_state中没有"user_id"，则将用户ID存储到session_state中
if "user_id" not in st.session_state:
    st.session_state["user_id"] = user_id

# 显示所有用户的下拉菜单
selected_user_id = user_id
# 获取用户所选的对话记录键
selected_conversation_key = "default"  # 默认使用名为"default"的对话记录键

# 创建一个UI，显示所选用户的对话记录
if selected_user_id in users:
    # 如果所选用户存在，则获取其对话记录
    selected_user_conversations = users[selected_user_id]["conversations"].get(selected_conversation_key, [])
    # 遍历对话记录中的所有消息，并显示在聊天界面上
    for msg in selected_user_conversations:
        st.chat_message(msg["role"]).write(msg["content"])
else:
    st.write("No user selected.")

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到所选对话记录中的消息列表中
    users[selected_user_id]["conversations"].setdefault(selected_conversation_key, []).append({"role": "user", "content": prompt})
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 构建输入
    input_ids = tokenizer.apply_chat_template(users[selected_user_id]["conversations"][selected_conversation_key], tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 将模型的输出添加到对话记录中的消息列表中
    users[selected_user_id]["conversations"][selected_conversation_key].append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)

    # 保存对话记录
    save_users(users)
