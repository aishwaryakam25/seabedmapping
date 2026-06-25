/* NautiCAI shared nav — theme, notifications, account menu */
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
    if (!s) return { all: 0, hi: 0, frames: 0, rov: 0 };
    var hi = 0;
    (s.detections || []).forEach(function (d) {
      var r = riskOf(d.cls);
      if (r === 'Critical' || r === 'High') hi++;
    });
    var rov = 0;
    try {
      if (window.SeabedSurveyOps) {
        var c = SeabedSurveyOps.loadContacts();
        var w = SeabedSurveyOps.loadWorkflow();
        rov = SeabedSurveyOps.openHighRiskCount(c, w);
      }
    } catch (e) {}
    return { all: s.detections.length, hi: hi, frames: s.frames || 0, rov: rov };
  }
  function buildNotifs() {
    var a = sessionStats(), items = [];
    if (a.all) {
      items.push(['#3fae6a', a.all + ' detections in this session']);
      if (a.hi) items.push(['#e0564f', a.hi + ' high-risk contact' + (a.hi > 1 ? 's' : '') + ' flagged']);
      if (a.rov) items.push(['#d2691e', a.rov + ' open ROV follow-up' + (a.rov > 1 ? 's' : '')]);
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
    var n = Math.max(a.hi, a.rov || 0);
    if (n > 0) { b.style.display = ''; b.textContent = n; }
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
        localStorage.removeItem('seabed_rov_workflow');
        localStorage.removeItem('seabed_rov_workflow_export');
        ['seabed_site_fingerprints','seabed_survey_snapshots','seabed_install_schedule','seabed_corridor','seabed_lifecycle','seabed_compare_survey_a','seabed_compare_survey_b'].forEach(function(k){localStorage.removeItem(k);});
        sessionStorage.removeItem('nauticai_twin');
      } catch (e) {}
      closeMenus();
      setTimeout(function () { location.href = '/review'; }, 400);
    });
  }
  updateBell();
  window.NautiCAI_updateBell = updateBell;
  window.NautiCAI_buildNotifs = buildNotifs;

  document.querySelectorAll('.navlinks').forEach(function (nav) {
    if (!nav.querySelector('a[href="/briefing"]')) {
      var bf = document.createElement('a');
      bf.href = '/briefing';
      bf.textContent = 'Briefing';
      var dr = nav.querySelector('a[href="/defects-report"]');
      if (dr) nav.insertBefore(bf, dr);
      else nav.appendChild(bf);
    }
    if (!nav.querySelector('a[href="/defects-report"]')) {
      var dr = document.createElement('a');
      dr.href = '/defects-report';
      dr.textContent = 'Defects Report';
      var reports = nav.querySelector('a[href="/reports"]');
      if (reports) nav.insertBefore(dr, reports);
      else nav.appendChild(dr);
    }
    if (!nav.querySelector('a[href="/copilot"]')) {
      var cp = document.createElement('a');
      cp.href = '/copilot';
      cp.textContent = 'NautiCAI Chatbot';
      var rov = nav.querySelector('a[href="/rov"]');
      if (rov) nav.insertBefore(cp, rov);
      else nav.appendChild(cp);
    }
    if (!nav.querySelector('a[href="/triage"]')) {
      var tr = document.createElement('a');
      tr.href = '/triage';
      tr.textContent = 'Triage';
      var cp2 = nav.querySelector('a[href="/copilot"]');
      if (cp2) cp2.parentNode.insertBefore(tr, cp2.nextSibling);
      else nav.appendChild(tr);
    }
    if (!nav.querySelector('a[href="/dive-plan"]')) {
      var dp = document.createElement('a');
      dp.href = '/dive-plan';
      dp.textContent = 'Dive Plan';
      var rov = nav.querySelector('a[href="/rov"]');
      if (rov) rov.parentNode.insertBefore(dp, rov);
      else nav.appendChild(dp);
    }
    if (!nav.querySelector('a[href="/intelligence"]')) {
      var a = document.createElement('a');
      a.href = '/intelligence';
      a.textContent = 'Intelligence';
      var reports = nav.querySelector('a[href="/reports"]');
      if (reports) nav.insertBefore(a, reports);
      else nav.appendChild(a);
    }
    if (location.pathname === '/briefing') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var ba = nav.querySelector('a[href="/briefing"]');
      if (ba) ba.classList.add('on');
    }
    if (location.pathname === '/defects-report') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var da = nav.querySelector('a[href="/defects-report"]');
      if (da) da.classList.add('on');
    }
    if (location.pathname === '/copilot') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var ca = nav.querySelector('a[href="/copilot"]');
      if (ca) ca.classList.add('on');
    }
    if (location.pathname === '/triage') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var ta = nav.querySelector('a[href="/triage"]');
      if (ta) ta.classList.add('on');
    }
    if (location.pathname === '/dive-plan') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var da = nav.querySelector('a[href="/dive-plan"]');
      if (da) da.classList.add('on');
    }
    if (location.pathname === '/intelligence') {
      nav.querySelectorAll('a').forEach(function (x) { x.classList.remove('on'); });
      var ia = nav.querySelector('a[href="/intelligence"]');
      if (ia) ia.classList.add('on');
    }
  });
})();
