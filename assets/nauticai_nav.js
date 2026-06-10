(function () {
  function $(s) { return document.querySelector(s); }
  var RISKN = { wreck: 'Critical', uxo: 'Critical', mine: 'Critical', pipeline: 'High', cable: 'High', cable_crossing: 'High', obstruction: 'High', boulder: 'Medium', anchor: 'Medium', debris: 'Medium', scour: 'Medium', rock: 'Low' };
  function riskOf(n) { return RISKN[('' + n).toLowerCase().replace(/\s+/g, '_')] || 'Medium'; }

  function applyTheme(t) {
    document.body.classList.toggle('light', t === 'light');
    var el = $('#themeToggle');
    if (el) el.innerHTML = t === 'light' ? '&#9728;' : '&#9789;';
  }
  var theme = 'dark';
  try { theme = localStorage.getItem('nauticai_theme') || 'dark'; } catch (e) {}
  applyTheme(theme);
  var tt = $('#themeToggle');
  if (tt) {
    tt.addEventListener('click', function (e) {
      e.stopPropagation();
      theme = document.body.classList.contains('light') ? 'dark' : 'light';
      try { localStorage.setItem('nauticai_theme', theme); } catch (e) {}
      applyTheme(theme);
    });
  }

  function readSession() {
    try {
      var s = JSON.parse(localStorage.getItem('nauticai_session'));
      if (s && s.hasData && (s.detections || []).length) return s;
    } catch (e) {}
    return null;
  }
  function sessionStats() {
    var s = readSession();
    if (!s) return { all: 0, hi: 0, frames: 0 };
    var hi = 0;
    (s.detections || []).forEach(function (d) {
      var r = riskOf(d.cls);
      if (r === 'Critical' || r === 'High') hi++;
    });
    return { all: s.detections.length, hi: hi, frames: s.frames || 0 };
  }
  function buildNotifs() {
    var a = sessionStats(), items = [];
    if (a.all) {
      items.push(['#3fae6a', a.all + ' detections in this session']);
      if (a.hi) items.push(['#e0564f', a.hi + ' high-risk contact' + (a.hi > 1 ? 's' : '') + ' flagged']);
      items.push(['#2bb6c4', a.frames + ' frame' + (a.frames > 1 ? 's' : '') + ' loaded']);
    } else {
      items.push(['#6f8593', 'No detections yet']);
      items.push(['#6f8593', 'Load frames in Data Review']);
    }
    items.push(['#3fae6a', 'All systems operational']);
    var list = $('#bellList');
    if (!list) return;
    list.innerHTML = items.map(function (it) {
      return '<div class="dditem" style="cursor:default;display:flex;align-items:center;gap:9px"><span class="dot" style="background:' + it[0] + '"></span>' + it[1] + '</div>';
    }).join('');
  }
  function updateBell() {
    var a = sessionStats(), b = $('#bellBadge');
    if (!b) return;
    if (a.hi > 0) { b.style.display = ''; b.textContent = a.hi; }
    else b.style.display = 'none';
  }
  function closeMenus() {
    ['bellMenu', 'userMenu'].forEach(function (id) {
      var m = $('#' + id);
      if (m) m.classList.remove('open');
    });
  }
  function toggleMenu(id, before) {
    var m = $('#' + id);
    if (!m) return;
    var open = m.classList.contains('open');
    closeMenus();
    if (!open) { if (before) before(); m.classList.add('open'); }
  }
  var bell = $('#bell');
  if (bell) bell.addEventListener('click', function (e) { e.stopPropagation(); toggleMenu('bellMenu', buildNotifs); });
  var uw = $('#userWrap');
  if (uw) uw.addEventListener('click', function (e) { e.stopPropagation(); toggleMenu('userMenu'); });
  ['bellMenu', 'userMenu'].forEach(function (id) {
    var m = $('#' + id);
    if (m) m.addEventListener('click', function (e) { e.stopPropagation(); });
  });
  document.addEventListener('click', closeMenus);
  var cs = $('#clearSession');
  if (cs) {
    cs.addEventListener('click', function (e) {
      e.preventDefault();
      try {
        localStorage.removeItem('nauticai_session');
        localStorage.removeItem('nauticai_review_frames');
        localStorage.removeItem('nauticai_survey_info');
        localStorage.removeItem('nauticai_survey');
        sessionStorage.removeItem('nauticai_twin');
      } catch (e) {}
      closeMenus();
      setTimeout(function () { location.href = '/review'; }, 400);
    });
  }
  updateBell();
  window.NautiCAI_updateBell = updateBell;
  window.NautiCAI_buildNotifs = buildNotifs;
})();
