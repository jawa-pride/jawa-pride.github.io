/* ============================================
   LIBER ANIMUS — mood.js
   Mood tracker logic — simpan di localStorage
   ============================================ */

const MoodTracker = (() => {

  const STORAGE_KEY = 'la_moods';

  const MOOD_LEVELS = [
    { value: 1, emoji: '😭', label: 'Sangat Berat', color: '#f87171' },
    { value: 2, emoji: '😔', label: 'Berat', color: '#fb923c' },
    { value: 3, emoji: '😐', label: 'Biasa', color: '#fbbf24' },
    { value: 4, emoji: '🙂', label: 'Lumayan', color: '#a3e635' },
    { value: 5, emoji: '😊', label: 'Baik', color: '#6ee7b7' }
  ];

  function getMoods() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch { return []; }
  }

  function saveMood(value, note = '') {
    const moods = getMoods();
    const today = new Date().toISOString().split('T')[0];

    // Hapus entry hari ini kalau ada
    const filtered = moods.filter(m => m.date !== today);
    filtered.push({ date: today, value, note, ts: Date.now() });

    // Simpan max 90 hari
    const trimmed = filtered.slice(-90);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    return true;
  }

  function getLast7Days() {
    const moods = getMoods();
    const result = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const dateStr = d.toISOString().split('T')[0];
      const entry = moods.find(m => m.date === dateStr);
      result.push({
        date: dateStr,
        day: ['Min','Sen','Sel','Rab','Kam','Jum','Sab'][d.getDay()],
        value: entry?.value || null,
        note: entry?.note || ''
      });
    }
    return result;
  }

  function getAverage() {
    const moods = getMoods().filter(m => m.value);
    if (!moods.length) return null;
    return (moods.reduce((s, m) => s + m.value, 0) / moods.length).toFixed(1);
  }

  function getStreak() {
    const moods = getMoods();
    let streak = 0;
    let d = new Date();
    while (true) {
      const dateStr = d.toISOString().split('T')[0];
      if (moods.find(m => m.date === dateStr)) {
        streak++;
        d.setDate(d.getDate() - 1);
      } else break;
    }
    return streak;
  }

  // Render chart di element dengan id targetId
  function renderChart(targetId) {
    const container = document.getElementById(targetId);
    if (!container) return;

    const data = getLast7Days();

    container.innerHTML = `
      <div class="mood-chart">
        <div class="mood-bars">
          ${data.map(d => {
            const mood = MOOD_LEVELS.find(m => m.value === d.value);
            const height = d.value ? `${(d.value / 5) * 100}%` : '8px';
            const color = mood?.color || 'rgba(126,184,247,0.15)';
            const isToday = d.date === new Date().toISOString().split('T')[0];
            return `
              <div class="mood-bar-wrap ${isToday ? 'today' : ''}">
                <div class="mood-bar-fill" style="height:${height};background:${color}" title="${d.value ? mood.label : 'Belum dicatat'} — ${d.date}">
                  ${d.value ? `<span class="mood-emoji">${mood.emoji}</span>` : ''}
                </div>
                <div class="mood-day">${d.day}</div>
              </div>
            `;
          }).join('')}
        </div>
        <div class="mood-stats">
          <div class="mood-stat">
            <span class="mood-stat-val">${getAverage() || '—'}</span>
            <span class="mood-stat-label">Rata-rata</span>
          </div>
          <div class="mood-stat">
            <span class="mood-stat-val">${getStreak()}</span>
            <span class="mood-stat-label">Hari berturut</span>
          </div>
          <div class="mood-stat">
            <span class="mood-stat-val">${getMoods().length}</span>
            <span class="mood-stat-label">Total catatan</span>
          </div>
        </div>
      </div>
    `;
  }

  // Render mood picker
  function renderPicker(targetId, onSave) {
    const container = document.getElementById(targetId);
    if (!container) return;

    const today = new Date().toISOString().split('T')[0];
    const existing = getMoods().find(m => m.date === today);

    container.innerHTML = `
      <div class="mood-picker">
        <p class="mood-picker-title">Bagaimana perasaanmu hari ini?</p>
        <div class="mood-options">
          ${MOOD_LEVELS.map(m => `
            <button class="mood-opt ${existing?.value === m.value ? 'selected' : ''}"
              data-value="${m.value}" style="--mood-color:${m.color}"
              onclick="MoodTracker.selectMood(${m.value}, '${targetId}')">
              <span class="mood-opt-emoji">${m.emoji}</span>
              <span class="mood-opt-label">${m.label}</span>
            </button>
          `).join('')}
        </div>
        <div class="mood-note-wrap" id="moodNoteWrap" style="display:${existing ? 'block' : 'none'}">
          <textarea class="mood-note" id="moodNote" placeholder="Catatan singkat (opsional)...">${existing?.note || ''}</textarea>
          <button class="btn-primary" onclick="MoodTracker.confirmSave('${targetId}')">Simpan Mood Hari Ini ✓</button>
        </div>
        ${existing ? `<p class="mood-saved">✓ Mood hari ini sudah dicatat — ${MOOD_LEVELS.find(m=>m.value===existing.value)?.emoji} ${MOOD_LEVELS.find(m=>m.value===existing.value)?.label}</p>` : ''}
      </div>
    `;
  }

  function selectMood(value, pickerId) {
    document.querySelectorAll('.mood-opt').forEach(b => b.classList.remove('selected'));
    document.querySelector(`[data-value="${value}"]`)?.classList.add('selected');
    const noteWrap = document.getElementById('moodNoteWrap');
    if (noteWrap) noteWrap.style.display = 'block';
    // Store selected value temporarily
    document.getElementById('moodNoteWrap')?.setAttribute('data-selected', value);
  }

  function confirmSave(pickerId) {
    const val = parseInt(document.getElementById('moodNoteWrap')?.getAttribute('data-selected') || '0');
    const note = document.getElementById('moodNote')?.value || '';
    if (!val) return;
    saveMood(val, note);
    renderPicker(pickerId);
    renderChart('moodChart');
    // Tampilkan toast
    const mood = MOOD_LEVELS.find(m => m.value === val);
    if (window.LiberChat) LiberChat.showToast(`Mood dicatat: ${mood.emoji} ${mood.label}`, 'success');
  }

  return { renderChart, renderPicker, selectMood, confirmSave, getLast7Days, getAverage, getStreak, saveMood };

})();
