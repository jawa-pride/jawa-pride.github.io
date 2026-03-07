/* ============================================
   LIBER ANIMUS — chat.js
   UI logic untuk halaman chat (app.html)
   ============================================ */

const LiberChat = (() => {

  let history = [];
  let currentTopic = 'umum';
  let isSending = false;
  let anonCode = '';

  // Generate kode anonim unik
  function generateAnonCode() {
    const colors = ['Biru','Hijau','Ungu','Merah','Kuning','Perak','Emas','Putih'];
    const num = Math.floor(1000 + Math.random() * 9000);
    return `#${colors[Math.floor(Math.random() * colors.length)]}${num}`;
  }

  function init() {
    // Ambil atau buat kode anonim
    anonCode = sessionStorage.getItem('la_anon') || generateAnonCode();
    sessionStorage.setItem('la_anon', anonCode);

    // Ambil topic dari URL param
    const params = new URLSearchParams(window.location.search);
    currentTopic = params.get('topic') || 'umum';

    // Set active topic button
    document.querySelectorAll('.topic-btn').forEach(btn => {
      if (btn.dataset.topic === currentTopic) btn.classList.add('active');
    });

    // Update anon code display
    const codeEl = document.getElementById('anonCode');
    if (codeEl) codeEl.textContent = anonCode;

    // Pesan pembuka sesuai topik
    const openers = {
      depresi: 'Hei... senang kamu ada di sini. Aku mendengarkan, tidak ada yang perlu disembunyikan. Cerita aja perlahan 🌙',
      bullying: 'Hei, aku di sini bersamamu. Apa yang kamu alami itu tidak benar, dan kamu tidak sendirian. Mau cerita? 💙',
      putus: 'Hei... patah hati itu berat banget. Aku di sini, dan kita bisa ngobrol seperlunya. Gak ada buru-buru 💜',
      keluarga: 'Hei, masalah keluarga bisa sangat kompleks dan melelahkan. Aku siap dengerin cerita kamu 🏠',
      umum: 'Hei... senang kamu mau cerita. Di sini tidak ada penghakiman, hanya ruang aman untukmu 🌙'
    };

    addMessage('ai', openers[currentTopic] || openers.umum);

    // Event listeners
    document.getElementById('sendBtn')?.addEventListener('click', sendMessage);
    document.getElementById('chatInput')?.addEventListener('keypress', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  }

  function addMessage(role, text, isLoading = false) {
    const container = document.getElementById('chatMessages');
    if (!container) return null;

    const div = document.createElement('div');
    div.className = `msg ${role === 'user' ? 'msg-user' : 'msg-ai'}`;
    if (isLoading) div.id = 'loadingMsg';

    // Parse **bold** markdown sederhana
    const formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    div.innerHTML = `
      <div class="msg-name">${role === 'user' ? `Kamu ${anonCode}` : 'Liber AI 🌙'}</div>
      <div class="msg-content">${isLoading ? '<span class="typing-indicator"><span></span><span></span><span></span></span>' : formatted}</div>
    `;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  async function sendMessage() {
    if (isSending) return;

    const input = document.getElementById('chatInput');
    const text = input?.value.trim();
    if (!text) return;

    isSending = true;
    input.value = '';
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;

    addMessage('user', text);
    const loadingEl = addMessage('ai', '', true);

    try {
      const result = await GeminiAPI.chat(text, history, currentTopic);

      loadingEl?.remove();
      addMessage('ai', result.text);

      if (result.crisis) {
        // Tampilkan crisis banner
        showCrisisBanner();
      } else {
        // Tambah ke history
        history.push({ role: 'user', parts: [{ text }] });
        history.push({ role: 'model', parts: [{ text: result.text }] });

        // Batasi history 20 pesan (hemat token)
        if (history.length > 20) history = history.slice(-20);
      }

    } catch (err) {
      loadingEl?.remove();
      const msg = GeminiAPI.getErrorMessage(err.message);
      addMessage('ai', msg);
    }

    isSending = false;
    input.disabled = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }

  function setTopic(topic) {
    currentTopic = topic;
    history = [];

    // Reset UI
    const container = document.getElementById('chatMessages');
    if (container) container.innerHTML = '';

    const openers = {
      depresi: 'Oke, aku di sini untuk dengerin kamu. Cerita aja, tidak ada yang salah di sini 🌙',
      bullying: 'Aku siap dengerin. Apa yang kamu alami tidak seharusnya terjadi — kamu layak mendapat rasa aman 💙',
      putus: 'Patah hati itu memang sakit banget. Mau cerita dari mana dulu? 💜',
      keluarga: 'Masalah keluarga bisa sangat berat. Aku di sini, cerita aja perlahan 🏠',
      umum: 'Hei, aku siap dengerin apapun yang mau kamu ceritakan 🌙'
    };

    addMessage('ai', openers[topic] || openers.umum);

    // Update URL tanpa reload
    const url = new URL(window.location);
    url.searchParams.set('topic', topic);
    window.history.pushState({}, '', url);
  }

  function showCrisisBanner() {
    const existing = document.getElementById('crisisBanner');
    if (existing) return;

    const banner = document.createElement('div');
    banner.id = 'crisisBanner';
    banner.className = 'crisis-float';
    banner.innerHTML = `
      <div class="crisis-float-inner">
        <span>🆘</span>
        <div>
          <strong>Butuh bantuan segera?</strong>
          <p>Hubungi hotline kesehatan jiwa 24 jam</p>
        </div>
        <a href="tel:119" class="crisis-call">📞 119 ext 8</a>
        <button onclick="document.getElementById('crisisBanner').remove()">✕</button>
      </div>
    `;
    document.body.appendChild(banner);
  }

  function showToast(msg, type = 'success') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
  }

  return { init, setTopic, showToast };

})();

// Auto-init saat DOM ready
document.addEventListener('DOMContentLoaded', LiberChat.init);
