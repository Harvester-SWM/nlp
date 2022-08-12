import telegram
import argparse
import os

parser = argparse.ArgumentParser(description="usage")

parser.add_argument('--command', type=str, default='beomi/kcbert-large', help='type of model')
parser.add_argument('--now_number', type=int, default='1', help='now test number')
parser.add_argument('--total_number', type=int, default='1', help='total test number')
parser.add_argument('--time_elapsed', type=int, default='1', help='total test number')


user_input = parser.parse_args()

command = user_input.command
test_number = user_input.now_number
total_number = user_input.total_number
time_elapsed = user_input.time_elapsed

message_to_send = f'''
command : {command}
now_test_number : {test_number}
total_test_number : {total_number}
time_elapsed : {time_elapsed} 
'''
print(message_to_send)

token = os.environ['TELEGRAM_TOKEN']# 절대 push 하지 말자! env 써서해야함;;
user_id = os.environ['TELEGRAM_USER_ID'] # 절대 push 하지 말자! env 써서해야함;;
chat_id = os.environ['TELEGRAM_CHAT_ID'] # 절대 push 하지 말자! env 써서해야함;;

bot = telegram.Bot(token)
bot.sendMessage(chat_id=chat_id, text=message_to_send)