import logging
import os
import re

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

import rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
WEBHOOK_PATH = "/webhook"
PORT = int(os.environ.get("PORT", 8080))

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


def format_latex_for_telegram(text: str) -> str:
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: '\n<pre>' + m.group(1).strip() + '</pre>\n', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'<code>\1</code>', text)
    
    text = text.replace('\\langle', '⟨')
    text = text.replace('\\rangle', '⟩')
    text = text.replace('\\mathbb{R}', 'ℝ')
    text = text.replace('\\mathbb{N}', 'ℕ')
    text = text.replace('\\mathbb{Z}', 'ℤ')
    text = text.replace('\\mathbb{Q}', 'ℚ')
    text = text.replace('\\mathbb{C}', 'ℂ')
    text = text.replace('\\in', '∈')
    text = text.replace('\\to', '→')
    text = text.replace('\\rightarrow', '→')
    text = text.replace('\\leftarrow', '←')
    text = text.replace('\\Rightarrow', '⇒')
    text = text.replace('\\Leftarrow', '⇐')
    text = text.replace('\\Leftrightarrow', '⇔')
    text = text.replace('\\forall', '∀')
    text = text.replace('\\exists', '∃')
    text = text.replace('\\leq', '≤')
    text = text.replace('\\geq', '≥')
    text = text.replace('\\neq', '≠')
    text = text.replace('\\approx', '≈')
    text = text.replace('\\infty', '∞')
    text = text.replace('\\alpha', 'α')
    text = text.replace('\\beta', 'β')
    text = text.replace('\\gamma', 'γ')
    text = text.replace('\\Gamma', 'Γ')
    text = text.replace('\\delta', 'δ')
    text = text.replace('\\Delta', 'Δ')
    text = text.replace('\\epsilon', 'ε')
    text = text.replace('\\varepsilon', 'ε')
    text = text.replace('\\theta', 'θ')
    text = text.replace('\\Theta', 'Θ')
    text = text.replace('\\lambda', 'λ')
    text = text.replace('\\Lambda', 'Λ')
    text = text.replace('\\mu', 'μ')
    text = text.replace('\\nu', 'ν')
    text = text.replace('\\rho', 'ρ')
    text = text.replace('\\sigma', 'σ')
    text = text.replace('\\Sigma', 'Σ')
    text = text.replace('\\tau', 'τ')
    text = text.replace('\\pi', 'π')
    text = text.replace('\\Pi', 'Π')
    text = text.replace('\\omega', 'ω')
    text = text.replace('\\Omega', 'Ω')
    text = text.replace('\\phi', 'φ')
    text = text.replace('\\varphi', 'φ')
    text = text.replace('\\Phi', 'Φ')
    text = text.replace('\\psi', 'ψ')
    text = text.replace('\\Psi', 'Ψ')
    text = text.replace('\\xi', 'ξ')
    text = text.replace('\\Xi', 'Ξ')
    text = text.replace('\\chi', 'χ')
    text = text.replace('\\eta', 'η')
    text = text.replace('\\zeta', 'ζ')
    text = text.replace('\\partial', '∂')
    text = text.replace('\\nabla', '∇')
    text = text.replace('\\sum', 'Σ')
    text = text.replace('\\prod', '∏')
    text = text.replace('\\int', '∫')
    text = text.replace('\\times', '×')
    text = text.replace('\\cdot', '·')
    text = text.replace('\\pm', '±')
    text = text.replace('\\sqrt', '√')
    text = text.replace('\\cap', '∩')
    text = text.replace('\\cup', '∪')
    text = text.replace('\\subset', '⊂')
    text = text.replace('\\subseteq', '⊆')
    text = text.replace('\\supset', '⊃')
    text = text.replace('\\supseteq', '⊇')
    text = text.replace('\\emptyset', '∅')
    text = text.replace('\\wedge', '∧')
    text = text.replace('\\vee', '∨')
    text = text.replace('\\neg', '¬')
    text = text.replace('\\oplus', '⊕')
    text = text.replace('\\otimes', '⊗')
    
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathcal\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'<b>\1</b>', text)
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)
    
    text = re.sub(r'###\s+(.+)', r'\n<b>━━━ \1 ━━━</b>\n', text)
    text = re.sub(r'##\s+(.+)', r'\n<b>▸ \1</b>\n', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^- (.+)$', r'  ▪ \1', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+(.+?):', r'\n<b>\1:</b>', text, flags=re.MULTILINE)
    
    return text


@dp.message(Command("start"))
async def cmd_start(message: Message):
    welcome_text = (
        "<b>Привет! Я твой личный гид по учебнику Яндекса по машинному обучению</b>\n"
        "(от Школы анализа данных).\n\n"
        "<b>Спрашивай что угодно:</b>\n"
        "▪ Почему градиентный спуск сходится?\n"
        "▪ Как работает attention?\n"
        "▪ Что такое conformal prediction?\n"
        "▪ В чём подвох в этой задаче на CatBoost?\n\n"
        "Отвечаю точно по тексту handbook → https://education.yandex.ru/handbook/ml\n"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@dp.message(F.text)
async def handle_question(message: Message):
    try:
        status_msg = await message.answer("Обрабатываю ваш вопрос...")
        
        question = message.text
        answer = rag.run(question=question, k=5)
        
        formatted_answer = format_latex_for_telegram(answer)
        
        await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
        await message.answer(formatted_answer, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        await message.answer("Произошла ошибка при обработке вопроса. Попробуйте еще раз.")


@dp.message()
async def handle_other_messages(message: Message):
    await message.answer(
        "Я могу отвечать только на текстовые вопросы.\n"
        "Пожалуйста, отправьте ваш вопрос текстом."
    )


async def on_startup(app: web.Application):
    webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
    await bot.set_webhook(webhook_url)
    logger.info(f"Webhook set to {webhook_url}")


async def on_shutdown(app: web.Application):
    await bot.delete_webhook()
    await bot.session.close()


def main():
    app = web.Application()
    
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)
    
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    
    setup_application(app, dp, bot=bot)
    
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

