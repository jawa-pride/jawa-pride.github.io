"""
============================================
LIBER ANIMUS — Telegram Bot (Human Mode)
============================================
Setup:
1. pip install python-telegram-bot
2. Isi BOT_TOKEN dan ADMIN_CHAT_ID
3. python bot.py

Cara kerja:
- User kirim pesan ke bot + kode anonim mereka
- Bot forward ke kamu (admin) dengan kode anon
- Kamu reply di Telegram → dikirim balik ke user
============================================
"""

import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# ============================================
# ⚠️ KONFIGURASI — WAJIB DIISI
# ============================================
BOT_TOKEN = "MASUKKAN_BOT_TOKEN_DARI_BOTFATHER"
ADMIN_CHAT_ID = 123456789  # Ganti dengan Telegram ID lo (cek via @userinfobot)
BOT_USERNAME = "liberanimus_bot"  # Ganti dengan username bot lo

# ============================================
# STORAGE SESI (in-memory — production pakai DB)
# ============================================
# Format: { anon_code: telegram_chat_id }
user_sessions = {}
# Format: { admin_reply_context: anon_code }
reply_context = {}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================
# HANDLERS
# ============================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /start"""
    chat_id = update.effective_chat.id
    args = context.args

    # Jika ada kode anonim di deep link (/start KodeAnon123)
    anon_code = args[0] if args else None

    if anon_code:
        user_sessions[anon_code] = chat_id
        await update.message.reply_text(
            f"✅ *Sesi Human Mode aktif!*\n\n"
            f"Kode anonimmu: `{anon_code}`\n\n"
            f"Ceritakan apa yang ingin kamu sampaikan. "
            f"Admin akan membalas secepatnya 💙\n\n"
            f"_Ingat: identitasmu tetap anonim — admin hanya tahu kode-mu._",
            parse_mode='Markdown'
        )
        logger.info(f"New user session: {anon_code} -> {chat_id}")
    else:
        await update.message.reply_text(
            "👋 *Selamat datang di Liber Animus Human Mode*\n\n"
            "Untuk memulai, buka website Liber Animus dan aktifkan Human Mode "
            "dari halaman chat. Kamu akan mendapat link langsung ke bot ini 💙\n\n"
            "Website: liberanimus.vercel.app",
            parse_mode='Markdown'
        )

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Terima pesan dari user → forward ke admin"""
    chat_id = update.effective_chat.id
    message = update.message.text

    # Cek apakah ini admin yang reply
    if chat_id == ADMIN_CHAT_ID:
        await handle_admin_reply(update, context)
        return

    # Cari kode anon user ini
    anon_code = None
    for code, cid in user_sessions.items():
        if cid == chat_id:
            anon_code = code
            break

    if not anon_code:
        await update.message.reply_text(
            "⚠️ Sesi kamu belum aktif. Buka Liber Animus dan aktifkan Human Mode terlebih dahulu.",
        )
        return

    # Forward ke admin dengan keyboard reply
    keyboard = [[
        InlineKeyboardButton(f"💬 Balas {anon_code}", callback_data=f"reply:{anon_code}")
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=f"📨 *Pesan dari {anon_code}*\n\n{message}",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

    # Konfirmasi ke user
    await update.message.reply_text(
        "✅ Pesanmu terkirim. Admin akan membalas secepatnya 💙\n"
        "_Biasanya dalam 1-3 jam sesuai jam operasional._",
        parse_mode='Markdown'
    )

async def handle_admin_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin reply ke user"""
    message = update.message.text

    # Cek format reply: /reply KODEANON pesan
    if message.startswith('/reply'):
        parts = message.split(' ', 2)
        if len(parts) < 3:
            await update.message.reply_text("Format: /reply KodeAnon pesan kamu")
            return
        anon_code = parts[1]
        reply_msg = parts[2]
    elif update.message.reply_to_message:
        # Reply ke pesan forwarded
        original = update.message.reply_to_message.text
        # Ekstrak kode dari "📨 Pesan dari #Biru1234"
        lines = original.split('\n')
        if lines and 'dari' in lines[0]:
            anon_code = lines[0].split('dari ')[-1].strip().rstrip('*')
            reply_msg = message
        else:
            await update.message.reply_text("Tidak bisa mendeteksi kode anonim. Gunakan /reply KODEANON pesan")
            return
    else:
        await update.message.reply_text(
            "Untuk reply ke user, gunakan:\n"
            "`/reply KodeAnon pesanmu`\n\n"
            "Atau reply langsung ke pesan yang di-forward.",
            parse_mode='Markdown'
        )
        return

    # Kirim ke user
    target_chat_id = user_sessions.get(anon_code)
    if not target_chat_id:
        await update.message.reply_text(f"❌ User {anon_code} tidak ditemukan atau sesi sudah habis.")
        return

    await context.bot.send_message(
        chat_id=target_chat_id,
        text=f"💙 *Balasan dari Admin:*\n\n{reply_msg}",
        parse_mode='Markdown'
    )

    await update.message.reply_text(f"✅ Pesan terkirim ke {anon_code}")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()

    if query.data.startswith('reply:'):
        anon_code = query.data.split(':')[1]
        reply_context[update.effective_chat.id] = anon_code
        await query.message.reply_text(
            f"Membalas ke *{anon_code}*\nKirim pesanmu sekarang:",
            parse_mode='Markdown'
        )

async def sessions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: lihat semua sesi aktif"""
    if update.effective_chat.id != ADMIN_CHAT_ID:
        return

    if not user_sessions:
        await update.message.reply_text("Belum ada sesi aktif.")
        return

    text = f"📊 *Sesi Aktif ({len(user_sessions)}):*\n\n"
    for code, cid in user_sessions.items():
        text += f"• `{code}`\n"

    await update.message.reply_text(text, parse_mode='Markdown')

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin: broadcast ke semua user aktif"""
    if update.effective_chat.id != ADMIN_CHAT_ID:
        return

    msg = ' '.join(context.args)
    if not msg:
        await update.message.reply_text("Format: /broadcast pesanmu")
        return

    sent = 0
    for code, cid in user_sessions.items():
        try:
            await context.bot.send_message(
                chat_id=cid,
                text=f"📢 *Pesan dari Liber Animus:*\n\n{msg}",
                parse_mode='Markdown'
            )
            sent += 1
        except Exception as e:
            logger.error(f"Failed to send to {code}: {e}")

    await update.message.reply_text(f"✅ Broadcast terkirim ke {sent} user.")

# ============================================
# MAIN
# ============================================

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("sessions", sessions_command))
    app.add_handler(CommandHandler("broadcast", broadcast_command))
    app.add_handler(CommandHandler("reply", handle_admin_reply))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    print(f"""
╔══════════════════════════════════╗
║   LIBER ANIMUS — Bot Running     ║
║   Bot: @{BOT_USERNAME:<24}║
║   Admin ID: {ADMIN_CHAT_ID:<20}║
╚══════════════════════════════════╝
    """)

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
