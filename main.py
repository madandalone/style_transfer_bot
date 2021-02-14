import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup
from aiogram.types import InlineKeyboardButton
from aiogram.types import KeyboardButton
from io import BytesIO
from copy import deepcopy
from style_transfer import *
from gan import *

LOGGING = True
CONNECTION_TYPE = 'POLLING'

logging.basicConfig(level=logging.INFO)

bot = Bot(token='1636425876:AAEKdSB_c9mOF9Ry_2xC8iBkz7CjbmYRtjk')
stbot = Dispatcher(bot)

input_photo = {}

start = InlineKeyboardMarkup().add(InlineKeyboardButton('Перенос выбранного стиля',
                               callback_data='st')).add(InlineKeyboardButton('Стилизация под картину Клода Моне',
                               callback_data='monet'))
cancel = InlineKeyboardMarkup().add(InlineKeyboardButton('Отмена', callback_data='main_menu'))

menu = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton('Меню')).add('Показать пример')

class User_settings:
    def __init__(self):
        self.settings = {'num_epochs': 50,
                         'imsize': 256}
        self.photos = []


@stbot.message_handler(commands=['start', 'help'])
async def process_start_command(message: types.Message):
    sticker = open('hello.jpg', 'rb')
    await bot.send_sticker(message.chat.id, sticker)
    await message.reply(f"Добрый вечер, {message.from_user.first_name}!\n", reply_markup=menu)
    await bot.send_message(message.chat.id,
                           "Я твой персональный раб по переносу стиля.  " +
                           "Я могу клево обработать твою фоточку.\n", reply_markup=start)


@stbot.callback_query_handler(lambda c: c.data == 'main_menu')
async def main_menu(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Смотри, что могу:")
    await callback_query.message.edit_reply_markup(reply_markup=start)

@stbot.callback_query_handler(lambda c: c.data == 'st')
async def style_transfer(callback_query):
    await bot.answer_callback_query(callback_query.id)

    if callback_query.from_user.id not in input_photo:
        input_photo[callback_query.from_user.id] = User_settings()

    input_photo[callback_query.from_user.id].st_type = 1

    if input_photo[callback_query.from_user.id].st_type == 1:
        await callback_query.message.edit_text(
                                               "Пришли мне фоточку, стиль с которой нужно перенести.")

        input_photo[callback_query.from_user.id].need_photos = 2

    await callback_query.message.edit_reply_markup(reply_markup=cancel)

    # input_photo[callback_query.from_user.id].set_default_settings()

# моне
@stbot.callback_query_handler(lambda c: c.data == 'monet')
async def monet(callback_query):
    await bot.answer_callback_query(callback_query.id)

    await callback_query.message.edit_text(
        "Твоя фоточка будет обработана в стиле картины Клода Моне. " +
        "Выбери желаемый режим работы:")

    if callback_query.from_user.id not in input_photo:
        input_photo[callback_query.from_user.id] = User_settings()

    input_photo[callback_query.from_user.id].st_type = 'monet'

    if input_photo[callback_query.from_user.id].st_type == 'monet':
        await callback_query.message.edit_text(
                                               "Пришли мне фоточку (документом!), и я добавлю на нее стиль Клода Моне.")

        input_photo[callback_query.from_user.id].need_photos = 1

    await callback_query.message.edit_reply_markup(reply_markup=cancel)

    # input_photo[callback_query.from_user.id].set_default_settings()

@stbot.message_handler(content_types=['text'])
async def get_example(message):
    print(message["text"])
    sticker = open('hello.jpg', 'rb')
    if str(message["text"])=='Показать пример':
        sticker1 = open('./1.jpg', 'rb')
        sticker2 = open('./2.jpg', 'rb')
        await bot.send_sticker(message.chat.id, sticker1)
        await bot.send_sticker(message.chat.id, sticker2)
    elif str(message["text"])=='Меню':
        await bot.send_sticker(message.chat.id, sticker)
        await message.reply(f"Добрый вечер, {message.from_user.first_name}!\n", reply_markup=menu)
        await bot.send_message(message.chat.id,
                               "Я твой персональный раб по переносу стиля. " +
                               "Я могу клево обработать твою фоточку.\n", reply_markup=start)
# load images
@stbot.callback_query_handler(lambda c: c.data == 'next')
async def load_images(callback_query):

    if input_photo[callback_query.from_user.id].st_type == 1:
        await callback_query.message.edit_text(
                                               "Пришли мне фоточку, стиль с которой нужно перенести.")

        input_photo[callback_query.from_user.id].need_photos = 2

    elif input_photo[callback_query.from_user.id].st_type == 'monet':
        await callback_query.message.edit_text(
                                               "Пришли мне фоточку, и я добавлю на нее стиль Клода Моне.")

        input_photo[callback_query.from_user.id].need_photos = 1


    await callback_query.message.edit_reply_markup(reply_markup=cancel)

@stbot.message_handler(content_types=['photo', 'document'])
async def get_image(message):
    if message.content_type == 'photo':
        img = message.photo[-1]

    else:
        img = message.document
        if img.mime_type[:5] != 'image':
            await bot.send_message(message.chat.id,
                "Это разве фотка? Пришли ФОТКУ.",
                reply_markup=start)
            return

    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)

    if message.chat.id not in input_photo:
        await bot.send_message(message.chat.id,
            "Какие настройки ты хочешь выбрать?", reply_markup=start)
        return

    if not hasattr(input_photo[message.chat.id], 'need_photos'):
        await bot.send_message(message.chat.id,
            "Какие настройки ты хочешь выбрать?", reply_markup=start)
        return

    input_photo[message.chat.id].photos.append(photo)

    if input_photo[message.chat.id].st_type == 1:
        if input_photo[message.chat.id].need_photos == 2:
            input_photo[message.chat.id].need_photos = 1

            await bot.send_message(message.chat.id,
                                   "А теперь пришли мне фоточку НА которую мы перенесем выбранный стиль." ,
                                   reply_markup=cancel)

        elif input_photo[message.chat.id].need_photos == 1:
            await bot.send_message(message.chat.id, "Идет обработка. Это может занять несколько минут.")

            # for debug
            log(input_photo[message.chat.id])

            output = await style_transfer(Style_transfer, input_photo[message.chat.id],
                                          *input_photo[message.chat.id].photos)

            await bot.send_document(message.chat.id, deepcopy(output))
            await bot.send_photo(message.chat.id, output)
            await bot.send_message(message.chat.id,
                                   "Еще разок обработаем фоточку?", reply_markup=start)

            del input_photo[message.chat.id]


    elif input_photo[message.chat.id].st_type in ['monet'] and \
            input_photo[message.chat.id].need_photos == 1:
        await bot.send_message(message.chat.id, "Идет обработка. Это может занять несколько минут.")

        log(input_photo[message.chat.id])

        output = gan_transfer(input_photo[message.chat.id],
                              input_photo[message.chat.id].photos[0])

        await bot.send_document(message.chat.id, deepcopy(output))
        await bot.send_photo(message.chat.id, output)
        await bot.send_message(message.chat.id,
                               "Еще разок обработаем фоточку?", reply_markup=start)

        del input_photo[message.chat.id]
async def style_transfer(st_class, user, *imgs):
    st = st_class(*imgs,
                  imsize=user.settings['imsize'],
                  num_steps=user.settings['num_epochs'],
                  style_weight=100000, content_weight=1)

    output = await st.transfer()

    return tensor2img(output)


def gan_transfer(user, img):
    output = transfer(img,
                      style=user.st_type,
                      imsize=user.settings['imsize'])

    return tensor2img(output.add(1).div(2))


def tensor2img(t):
    output = np.rollaxis(t.cpu().detach().numpy()[0], 0, 3)
    output = Image.fromarray(np.uint8(output * 255))

    bio = BytesIO()
    bio.name = 'result.jpeg'
    output.save(bio, 'JPEG')
    bio.seek(0)

    return bio


def log(user):
    if LOGGING:
        print()
        print('type:', user.st_type)
        if user.st_type == 1 or user.st_type == 2:
            print('settings:', user.settings)
            print('Epochs:')
        else:
            print('settings: imsize:', user.settings['imsize'])


def draw_img(img):
    plt.imshow(np.rollaxis(img.cpu().detach()[0].numpy(), 0, 3))
    plt.show()


def draw_photo(*photos):
    for photo in photos:
        img = np.array(Image.open(photo))
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    if CONNECTION_TYPE == 'POLLING':
        executor.start_polling(dp, skip_updates=True)



    else:
        print("Invalid 'CONNECTION_TYPE'")
