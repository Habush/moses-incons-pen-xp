__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import requests


tg_token = "XXXXXXXXX"
tg_chat_id = "XXXXXXXX"


def log_msg(msg):
    print(msg)
    send_update_to_tg(msg)

# small function to notify me on telegram about the status of long-running cells    
def send_update_to_tg(msg):
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                  params=dict(chat_id=tg_chat_id, text=msg))