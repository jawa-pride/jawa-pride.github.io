# LIBER ANIMUS — Setup Guide

## Struktur File
```
liber-animus/
├── index.html          → Landing page
├── app.html            → Chat app utama
├── css/
│   └── style.css       → Global styles (shared)
├── js/
│   ├── gemini.js       → Gemini AI API logic
│   ├── chat.js         → Chat UI & history
│   ├── mood.js         → Mood tracker
│   └── payment.js      → Trakteer payment
└── telegram-bot/
    └── bot.py          → Telegram bot (Human Mode)
```

---

## STEP 1 — Gemini API Key
1. Buka https://aistudio.google.com
2. Klik "Get API Key" → Create API Key
3. Buka `js/gemini.js`
4. Ganti `MASUKKAN_API_KEY_LO_DISINI` dengan key baru lo

---

## STEP 2 — Trakteer
1. Daftar di https://trakteer.id
2. Setup halaman donasi lo
3. Buka `js/payment.js`
4. Ganti `USERNAME_TRAKTEER_LO` dengan username lo

---

## STEP 3 — Telegram Bot
1. Chat @BotFather di Telegram → /newbot
2. Ikuti instruksi → dapat BOT_TOKEN
3. Dapat ADMIN_CHAT_ID lo via @userinfobot
4. Buka `telegram-bot/bot.py`
5. Isi BOT_TOKEN dan ADMIN_CHAT_ID
6. Install: `pip install python-telegram-bot`
7. Jalanin: `python bot.py`
8. Update link bot di `app.html` (cari `YOUR_BOT_USERNAME`)

---

## STEP 4 — Deploy ke Vercel
1. Buka https://vercel.com → login
2. Klik "Add New Project" → "Import"
3. Drag & drop folder `liber-animus` ini
4. Deploy! Dapat URL gratis: `liberanimus.vercel.app`

---

## STEP 5 — Bot Hosting (agar bot jalan 24/7)
- **Railway.app** → gratis $5 credit/bulan
- Upload folder `telegram-bot/`
- Add `requirements.txt`: `python-telegram-bot==20.7`
- Set environment variables: BOT_TOKEN, ADMIN_CHAT_ID

---

## Catatan Penting
- API key Gemini JANGAN di-commit ke GitHub (gunakan env variable di Vercel)
- Bot Telegram perlu server tersendiri (bukan Vercel)
- Mood tracker pakai localStorage — data tersimpan di device user
- Premium verification masih manual — upgrade nanti ke DB + auto-verify
