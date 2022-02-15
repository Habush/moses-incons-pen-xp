__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import requests


tg_token = "5211852780:AAFsZT_368EssQKvVPOcS7sE8TjONSCbXas"
tg_chat_id = "303606146"


def log_msg(msg):
    print(msg)
    send_update_to_tg(msg)

def send_update_to_tg(msg):
    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                  params=dict(chat_id=tg_chat_id, text=msg))