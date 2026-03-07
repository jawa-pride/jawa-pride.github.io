/* ============================================
   LIBER ANIMUS — gemini.js
   Semua logic Gemini AI ada di sini.
   Ganti GEMINI_API_KEY dengan key baru lo.
   ============================================ */

const GeminiAPI = (() => {

  // ⚠️ GANTI DENGAN API KEY BARU LO
  const API_KEY = 'AIzaSyDAbyHB9kx6QCmP6x3hMqIi9H__HpLMBkE';
  const BASE_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${API_KEY}`;

  // System prompts per kategori
  const PROMPTS = {
    depresi: `Kamu adalah Liber AI, teman curhat empatik dari platform Liber Animus Indonesia.
Tugasmu: membantu orang yang sedang merasa berat, depresi, atau butuh didengarkan.
Aturan WAJIB:
- Selalu validasi perasaan user DULU sebelum apapun
- Bahasa Indonesia hangat, santai, seperti teman dekat — tidak formal
- Jangan buru-buru kasih solusi, dengarkan dulu
- Respons SINGKAT: maksimal 4 kalimat
- Selalu akhiri dengan 1 pertanyaan terbuka yang mengajak cerita
- Jika ada tanda krisis (ingin menyakiti diri/bunuh diri), langsung arahkan ke 119 ext 8`,

    bullying: `Kamu adalah Liber AI dari Liber Animus, pendamping korban perundungan.
Aturan WAJIB:
- Tegaskan bahwa apa yang mereka alami SALAH dan BUKAN kesalahan mereka
- Dukung secara emosional dulu sebelum informasi apapun
- Jika ditanya soal hukum, sebut referensi umum (UU Perlindungan Anak pasal 76C, UU ITE pasal 45) + disclaimer: "ini hanya referensi, bukan konsultasi hukum profesional — untuk kasus serius hubungi LBH terdekat"
- Bahasa Indonesia hangat dan kuat (empowering)
- Respons 3-4 kalimat, akhiri dengan pertanyaan`,

    putus: `Kamu adalah Liber AI dari Liber Animus, teman menemani patah hati.
Aturan WAJIB:
- Akui bahwa patah hati itu nyata dan sangat menyakitkan
- JANGAN langsung bilang "move on" atau ceramahi — temani prosesnya
- Tidak menghakimi mantan atau user
- Bantu user memproses perasaan perlahan
- Bahasa santai, seperti sahabat dekat
- Respons 3-4 kalimat hangat, akhiri dengan pertanyaan terbuka`,

    keluarga: `Kamu adalah Liber AI dari Liber Animus, untuk masalah keluarga.
Aturan WAJIB:
- Jika belum tahu posisi user, tanya dulu (anak/orang tua/pasangan)
- Netral — tidak memihak siapapun dalam keluarga
- Validasi bahwa dinamika keluarga itu kompleks
- Bahasa hangat dan tidak menghakimi
- Respons 3-4 kalimat, akhiri dengan pertanyaan`,

    umum: `Kamu adalah Liber AI dari Liber Animus — platform safe space anonim Indonesia untuk bercerita.
Aturan WAJIB:
- Dengarkan dengan penuh empati
- Validasi perasaan user sebelum apapun
- Bahasa Indonesia hangat, santai, seperti teman
- Jika ada tanda krisis, sarankan 119 ext 8
- Respons singkat 3-4 kalimat, akhiri dengan pertanyaan terbuka`
  };

  const CRISIS_KEYWORDS = [
    'bunuh diri', 'mati aja', 'ingin mati', 'pengen mati',
    'tidak mau hidup', 'gak mau hidup', 'sudah tidak kuat',
    'menyakiti diri', 'nyakitin diri', 'self harm', 'overdosis'
  ];

  function isCrisis(text) {
    return CRISIS_KEYWORDS.some(k => text.toLowerCase().includes(k));
  }

  async function chat(userMessage, history = [], topic = 'umum') {
    // Crisis detection — override AI
    if (isCrisis(userMessage)) {
      return {
        text: `Aku dengar kamu... dan aku sangat peduli dengan keselamatanmu. 💙 Tolong hubungi hotline kesehatan jiwa sekarang ya: 📞 **119 ext 8** — mereka siap membantu 24 jam, gratis, dan rahasia. Kamu tidak harus menanggung ini sendirian.`,
        crisis: true
      };
    }

    const systemPrompt = PROMPTS[topic] || PROMPTS.umum;

    // Build conversation dengan system prompt di awal
    const contents = [
      {
        role: 'user',
        parts: [{ text: systemPrompt }]
      },
      {
        role: 'model',
        parts: [{ text: 'Siap. Aku akan menjadi Liber AI yang empatik dan supportif untuk user ini.' }]
      },
      ...history,
      {
        role: 'user',
        parts: [{ text: userMessage }]
      }
    ];

    const response = await fetch(BASE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.85,
          maxOutputTokens: 350,
          topP: 0.9
        },
        safetySettings: [
          { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
          { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' }
        ]
      })
    });

    if (!response.ok) {
      const err = await response.json();
      const status = response.status;
      if (status === 429) throw new Error('RATE_LIMIT');
      if (status === 400) throw new Error('BAD_REQUEST');
      if (status === 403) throw new Error('INVALID_KEY');
      throw new Error('API_ERROR');
    }

    const data = await response.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!text) throw new Error('EMPTY_RESPONSE');

    return { text, crisis: false };
  }

  function getErrorMessage(code) {
    const messages = {
      RATE_LIMIT: 'Terlalu banyak pesan. Tunggu sebentar ya... 🌙',
      INVALID_KEY: 'API key bermasalah. Hubungi admin.',
      BAD_REQUEST: 'Ada yang salah dengan permintaan. Coba lagi.',
      EMPTY_RESPONSE: 'Liber AI tidak merespons. Coba ulangi ya...',
      API_ERROR: 'Koneksi bermasalah. Coba beberapa saat lagi.'
    };
    return messages[code] || 'Terjadi kesalahan. Coba lagi ya... 🌙';
  }

  // Public API
  return { chat, isCrisis, getErrorMessage };

})();
