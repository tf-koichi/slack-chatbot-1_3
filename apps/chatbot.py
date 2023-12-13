import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from utils import ChatEngine

chatbot_app_token = os.environ["CHATBOT_APP_TOKEN"]
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]

app = App(token=slack_bot_token)

@app.message()
def handle(message, say):
    global chat_engine_dict
    if message["user"] not in chat_engine_dict.keys():
        chat_engine_dict[message["user"]] = ChatEngine()
    
    for reply in chat_engine_dict[message["user"]].reply_message(message['text']):
        say(reply)

@app.command("/verbose")
def verbose_function(ack, body, respond):
    ack()
    global chat_engine_dict
    user_id = body["user_id"]
    if user_id not in chat_engine_dict.keys():
        chat_engine_dict[user_id] = ChatEngine()
    
    switch = body["text"].lower().strip()
    if not switch:
        respond("Verbose mode." if chat_engine_dict[user_id].verbose else "Quiet mode.")
    elif switch == "on":
        chat_engine_dict[user_id].verbose = True
        respond("Verbose mode.")
    elif switch == "off":
        chat_engine_dict[user_id].verbose = False
        respond("Quiet mode.")
    else:
        respond("usage: /verbose [on|off]")

@app.command("/style")
def style_function(ack, body, respond):
    ack()
    global chat_engine_dict
    user_id = body["user_id"]
    switch = body["text"].lower().strip()
    if switch:
        chat_engine_dict[user_id] = ChatEngine(style=switch)
        respond(f"Style: {chat_engine_dict[user_id].style}")
    elif user_id in chat_engine_dict.keys():
        respond(f"Style: {chat_engine_dict[user_id].style}")
    else:
        respond("まだ会話が始まっていません。")
 
model = "gpt-4-0613"
def quotify(s: str) -> str:
    """Adds quotes to a string.
    Args:
        s (str): The string to add quotes to.
    Returns:
        (str) The string with quotes added.
    """
    return "\n".join([f"> {l}" for l in s.split("\n")])

ChatEngine.setup(model, quotify_fn=quotify)
chat_engine_dict = dict()
SocketModeHandler(app, chatbot_app_token).start()
