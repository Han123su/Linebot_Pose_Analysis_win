from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, VideoMessage, TextSendMessage, FollowEvent,
    QuickReply, QuickReplyButton, MessageAction
)
import os
import uuid
import subprocess
import shutil
import tempfile
import side_video_handler  # 側面影片分析模組

# 初始化 Flask 與 LineBot
app = Flask(__name__)
line_bot_api = LineBotApi('ZKFSSh5O1UScyOoIOVZHPuSSQISeQgjzZIanIPQADT8iKXPzhUHn+0IPcUklijOKeChIcYemYwnrvzorDZ/J5nhQCSJxJ1Y5s0keI2sTBxuV8dO6T9Qs4w8ye0B5rNLR5VXlyziOYLWRvP40ZCg2UgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('ebddbbcefa93f0e69889881adf816763')

# 使用者狀態管理
user_choices = {}  # 使用者選擇：back / side
user_states = {}   # 狀態：waiting, ready
user_uploaded_side = {}  # 若為 side，記錄該使用者已上傳哪一側

# 快速選單介面
def send_video_type_selection(user_id, reply_token, welcome_text="請選擇你要分析的影片類型："):
    user_choices[user_id] = None
    user_states[user_id] = 'waiting'
    message = TextSendMessage(
        text=welcome_text,
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(action=MessageAction(label="背面影片", text="選擇背面影片")),
                QuickReplyButton(action=MessageAction(label="側面影片", text="選擇側面影片")),
            ]
        )
    )
    line_bot_api.reply_message(reply_token, message)

# Webhook 接收入口
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 加入好友事件
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    send_video_type_selection(user_id, event.reply_token, "歡迎使用步態分析系統！請選擇你要分析的影片角度")

# 文字訊息處理
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if text == "選擇背面影片":
        user_choices[user_id] = "back"
        user_states[user_id] = "ready"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已選擇背面影片，請上傳影片"))

    elif text == "選擇側面影片":
        user_choices[user_id] = "side"
        user_states[user_id] = "ready"
        user_uploaded_side[user_id] = None
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你已選擇側面影片，請上傳左或右側任一影片"))

    else:
        state = user_states.get(user_id, 'waiting')
        if state == "ready":
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請上傳影片進行分析"))
        else:
            send_video_type_selection(user_id, event.reply_token, "請先選擇你要分析的影片類型：")

# 影片訊息處理
@handler.add(MessageEvent, message=VideoMessage)
def handle_video(event):
    user_id = event.source.user_id
    choice = user_choices.get(user_id, None)

    line_bot_api.push_message(
    user_id,
    TextSendMessage(text="處理中請耐心等待⏳...")
    )

    # 暫存影片
    message_id = event.message.id
    ext = '.mp4'
    # tmp_dir = tempfile.mkdtemp()
    tmp_dir = os.path.join("static")
    os.makedirs(tmp_dir, exist_ok=True)
    video_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")
    with open(video_path, 'wb') as f:
        for chunk in line_bot_api.get_message_content(message_id).iter_content():
            f.write(chunk)

    if choice == "back":
        # 背面影片分析流程
        xlsx_path = video_path.replace('.mp4', '.xlsx')
        subprocess.run([
            "python", "Pose_tracking_back_withBall.py",
            "--video", video_path,
            "--output", xlsx_path
        ])

        result = subprocess.run([
            "python", "analyze_back_main.py",
            "--input", xlsx_path,
            "--image_folder", "result_images"
        ], capture_output=True, text=True)

        reply_text = result.stdout[-5000:] if result.returncode == 0 else f"分析失敗：{result.stderr}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

         # 分析完成後重置使用者狀態
        user_choices.pop(user_id, None)
        user_states.pop(user_id, None)
        user_uploaded_side.pop(user_id, None) 

    elif choice == "side":
        base_dir = os.path.dirname(video_path)
        reply_text = side_video_handler.handle_side_video(video_path, base_dir)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

        # 只有分析完成才重置使用者狀態
        if reply_text.startswith("分析結果："):
            user_choices.pop(user_id, None)
            user_states.pop(user_id, None)
            user_uploaded_side.pop(user_id, None)
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⚠️請先選擇影片角度"))

    # 清理分析產生的暫存資料夾
    for folder_name in ["FRAMES", "FRAMES_MP", "FRAMES_MODIFY", "FRAMES_TRACKING"]:
        folder_path = os.path.join(tmp_dir, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
# 啟動 Flask
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
