__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import requests


tg_token = "5211852780:AAHy5u4Hxgw5zpaKkkiWf_GCAXkYrFOTDz8"
tg_chat_id = "303606146"


def log_msg(msg):
    print(msg)
    send_update_to_tg(msg)

# small function to notify me on telegram about the status of long-running cells
def send_update_to_tg(msg):
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                  params=dict(chat_id=tg_chat_id, text=msg))