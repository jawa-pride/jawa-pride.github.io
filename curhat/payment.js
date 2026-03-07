/* ============================================
   LIBER ANIMUS — payment.js
   Integrasi Trakteer untuk Premium 15k
   ============================================ */

const LiberPayment = (() => {

  // ⚠️ GANTI dengan username Trakteer lo
  const TRAKTEER_USERNAME = 'https://trakteer.id/charis6';
  const TRAKTEER_URL = `https://trakteer.id/${TRAKTEER_USERNAME}/tip`;

  const PLANS = {
    free: {
      name: 'Gratis',
      price: 0,
      features: [
        'AI Chat (10 pesan/hari)',
        'Guided Journaling (3 prompt/minggu)',
        'Mood Tracker dasar',
        'Anonim penuh'
      ]
    },
    premium: {
      name: 'Premium',
      price: 15000,
      priceLabel: 'Rp 15.000/bulan',
      features: [
        'AI Chat tanpa batas',
        'Guided Journaling tanpa batas',
        'Mood Analytics lengkap + grafik',
        'Jurnal tersimpan permanen',
        '⚡ Human Mode via Telegram (prioritas)',
        'Referensi hukum perundungan'
      ]
    }
  };

  // Cek status premium dari sessionStorage (manual verify setelah bayar)
  function isPremium() {
    return sessionStorage.getItem('la_premium') === 'true';
  }

  function setPremium(val) {
    sessionStorage.setItem('la_premium', val ? 'true' : 'false');
  }

  // Redirect ke Trakteer
  function openTrakteer(amount = 15000) {
    const url = `${TRAKTEER_URL}?quantity=1&price=${amount}&message=Premium+Liber+Animus`;
    window.open(url, '_blank');

    // Tampilkan instruksi verifikasi
    showVerifyModal();
  }

  function showVerifyModal() {
    const existing = document.getElementById('verifyModal');
    if (existing) { existing.classList.add('active'); return; }

    const modal = document.createElement('div');
    modal.id = 'verifyModal';
    modal.className = 'modal-overlay active';
    modal.innerHTML = `
      <div class="modal">
        <button class="modal-close" onclick="document.getElementById('verifyModal').classList.remove('active')">✕</button>
        <h3>Verifikasi <em style="color:var(--accent)">Premium</em></h3>
        <p>Setelah selesai bayar di Trakteer, masukkan email yang kamu pakai saat transaksi untuk aktivasi akses premium.</p>
        <div class="form-group">
          <label class="form-label">Email Trakteer kamu</label>
          <input type="email" class="form-input" id="verifyEmail" placeholder="email@kamu.com">
        </div>
        <div style="background:rgba(126,184,247,0.05);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:20px;">
          <p style="font-size:12px;color:var(--text-muted);line-height:1.7;">
            💡 Setelah verifikasi manual oleh admin (biasanya dalam 1-2 jam), akses premium kamu akan aktif. 
            Kamu akan dapat notifikasi via Telegram.
          </p>
        </div>
        <button class="btn-primary btn-full" onclick="LiberPayment.submitVerify()">Kirim Verifikasi →</button>
      </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('active'); });
  }

  function submitVerify() {
    const email = document.getElementById('verifyEmail')?.value?.trim();
    if (!email || !email.includes('@')) {
      alert('Masukkan email yang valid ya!');
      return;
    }

    // Di production: kirim ke backend/Telegram bot lo untuk verifikasi manual
    // Untuk sekarang: simulasi
    console.log('Verification request:', email);

    document.getElementById('verifyModal').innerHTML = `
      <div class="modal">
        <div style="text-align:center;padding:20px">
          <div style="font-size:56px;margin-bottom:24px">✉️</div>
          <h3>Permintaan <em style="color:var(--success)">Terkirim!</em></h3>
          <p style="margin-top:16px">Verifikasi kamu akan diproses dalam 1-2 jam. Pantau Telegram kamu ya!</p>
          <button class="btn-primary" style="margin-top:32px" onclick="document.getElementById('verifyModal').classList.remove('active')">Oke, Mengerti!</button>
        </div>
      </div>
    `;
  }

  // Render pricing cards ke target element
  function renderPricing(targetId) {
    const container = document.getElementById(targetId);
    if (!container) return;

    const premium = isPremium();

    container.innerHTML = `
      <div class="pricing-grid">
        <div class="pricing-card">
          <div class="pricing-tier">Gratis Selamanya</div>
          <div class="pricing-price">
            <span class="price-currency">Rp</span>
            <span class="price-amount">0</span>
            <span class="price-period">/bulan</span>
          </div>
          <p class="price-note">Tidak perlu kartu kredit</p>
          <ul class="price-features">
            ${PLANS.free.features.map(f => `<li>${f}</li>`).join('')}
          </ul>
          <button class="btn-ghost btn-full" onclick="window.location.href='app.html'">
            ${premium ? 'Kamu sudah Premium ✓' : 'Mulai Gratis'}
          </button>
        </div>

        <div class="pricing-card featured">
          <div class="pricing-tier">Premium</div>
          <div class="pricing-price">
            <span class="price-currency">Rp</span>
            <span class="price-amount">15</span>
            <span class="price-period">rb/bulan</span>
          </div>
          <p class="price-note">Harga secangkir kopi ☕</p>
          <ul class="price-features">
            ${PLANS.premium.features.map(f => `<li>${f}</li>`).join('')}
          </ul>
          ${premium
            ? `<button class="btn-primary btn-full" disabled style="opacity:0.7">✓ Premium Aktif</button>`
            : `<button class="btn-primary btn-full" onclick="LiberPayment.openTrakteer()">Upgrade Premium →</button>`
          }
        </div>
      </div>
      <p style="text-align:center;margin-top:24px;font-size:13px;color:var(--text-muted)">
        💙 Bayar lebih jika mampu — kelebihannya membantu user lain yang tidak mampu.
      </p>
    `;
  }

  // Cek limit chat harian
  function checkChatLimit() {
    if (isPremium()) return { allowed: true, remaining: Infinity };

    const today = new Date().toISOString().split('T')[0];
    const key = `la_chat_${today}`;
    const count = parseInt(localStorage.getItem(key) || '0');
    const limit = 10;

    return {
      allowed: count < limit,
      remaining: limit - count,
      count
    };
  }

  function incrementChatCount() {
    if (isPremium()) return;
    const today = new Date().toISOString().split('T')[0];
    const key = `la_chat_${today}`;
    const count = parseInt(localStorage.getItem(key) || '0');
    localStorage.setItem(key, count + 1);
  }

  return {
    openTrakteer, showVerifyModal, submitVerify,
    renderPricing, isPremium, setPremium,
    checkChatLimit, incrementChatCount
  };

})();
