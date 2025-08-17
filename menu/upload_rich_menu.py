from linebot import LineBotApi
from linebot.models import RichMenu
import json

# Channel Access Token
line_bot_api = LineBotApi('ZKFSSh5O1UScyOoIOVZHPuSSQISeQgjzZIanIPQADT8iKXPzhUHn+0IPcUklijOKeChIcYemYwnrvzorDZ/J5nhQCSJxJ1Y5s0keI2sTBxuV8dO6T9Qs4w8ye0B5rNLR5VXlyziOYLWRvP40ZCg2UgdB04t89/1O/w1cDnyilFU=')

# 讀取 rich_menu.json
with open("rich_menu.json", "r", encoding="utf-8") as f:
    rich_menu_data = json.load(f)

# 建立 Rich Menu
rich_menu = RichMenu.new_from_json_dict(rich_menu_data)
rich_menu_id = line_bot_api.create_rich_menu(rich_menu)

# 上傳選單圖片（你必須提供 rich_menu_image.png）
with open("rich_menu_image.png", "rb") as f:
    line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", f)

# 設為所有使用者的預設選單
line_bot_api.set_default_rich_menu(rich_menu_id)

print("Rich Menu 建立完成，ID：", rich_menu_id)
