/**
 * Seabed Defects Executive Report — aggregates Data Review session into
 * Azolla-style marine service report tables (rock, boulder, wreck, etc.).
 */
(function (global) {
  var DEFECT_ORDER = [
    'rock', 'boulder', 'wreck', 'pipeline', 'cable', 'cable_crossing',
    'debris', 'uxo', 'mine', 'anchor', 'obstruction', 'scour'
  ];

  var LEGEND = [
    { key: 'rock', label: 'Rock', color: '#b5791f' },
    { key: 'boulder', label: 'Boulder', color: '#c2410c' },
    { key: 'wreck', label: 'Wreck', color: '#be185d' },
    { key: 'pipeline', label: 'Pipeline', color: '#2f7fce' },
    { key: 'cable', label: 'Cable', color: '#0aa0b4' },
    { key: 'debris', label: 'Debris', color: '#8b5cf6' },
    { key: 'uxo', label: 'UXO', color: '#b3261e' },
    { key: 'mine', label: 'Mine', color: '#b3261e' },
    { key: 'obstruction', label: 'Obstruction', color: '#566270' },
    { key: 'scour', label: 'Scour', color: '#15803d' }
  ];

  var RISK = {
    wreck: 'Critical', uxo: 'Critical', mine: 'Critical',
    pipeline: 'High', cable: 'High', cable_crossing: 'High', obstruction: 'High',
    boulder: 'Medium', anchor: 'Medium', debris: 'Medium', scour: 'Medium', rock: 'Low'
  };

  function norm(k) {
    return ('' + k).toLowerCase().replace(/\s+/g, '_');
  }

  function cap(s) {
    return ('' + s).replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function riskOf(cls) {
    return RISK[norm(cls)] || 'Medium';
  }

  function loadSession() {
    var s = null, info = null, frames = [];
    try { s = JSON.parse(localStorage.getItem('nauticai_session')); } catch (e) {}
    try { info = JSON.parse(localStorage.getItem('nauticai_survey_info')); } catch (e) {}
    try { frames = JSON.parse(localStorage.getItem('nauticai_review_frames') || '[]'); } catch (e) { frames = []; }
    if (!s) s = { hasData: false, detections: [], frames: 0 };
    if (info) {
      s.project = s.project || info.project;
      s.survey_line = s.survey_line || info.survey_line;
      s.vessel = s.vessel || info.vessel;
      s.sensor = s.sensor || info.sensor;
      s.date = s.date || info.date;
      s.location = s.location || info.location;
      s.range = s.range || info.range;
    }
    var dets = Array.isArray(s.detections) ? s.detections.slice() : [];
    if (frames.length) {
      var FW = 60, FH = 60, SP = 72, rs = 150;
      if (s.range) { var m = parseInt(s.range, 10); if (!isNaN(m)) rs = m; }
      var rebuilt = [];
      frames.forEach(function (fr, fi) {
        var ox = fi * SP;
        (fr.dets || []).forEach(function (det) {
          if (!det || !det.cls) return;
          rebuilt.push({
            cls: det.cls, conf: det.conf, frame: fi + 1,
            x: ox + (det.cxn != null ? det.cxn : 0.5) * FW,
            y: (det.cyn != null ? det.cyn : 0.5) * FH,
            range: Math.abs((det.cxn != null ? det.cxn : 0.5) - 0.5) * 2 * rs,
            status: det.status || 'Review', thumb: fr.thumb || fr.pdfSrc || ''
          });
        });
      });
      if (rebuilt.length) dets = rebuilt;
      s.frames = frames.length;
      s.hasData = true;
      s._frames = frames;
    }
    s.detections = dets;
    return s;
  }

  function demoSession() {
    var cls = ['rock', 'rock', 'boulder', 'wreck', 'pipeline', 'cable', 'debris', 'rock', 'boulder', 'cable_crossing'];
    return {
      demo: true, frames: 4, hasData: true,
      project: 'North Sea Wind Farm', survey_line: 'L12_20240515_1045',
      vessel: 'CRV Seabed Explorer', sensor: 'EdgeTech 4200 (SSS)',
      location: 'North Sea Sector A', date: '15 May 2024 10:45',
      detections: cls.map(function (c, i) {
        return {
          cls: c, conf: 0.62 + ((i * 37) % 35) / 100, frame: 1 + (i % 4),
          status: i % 4 === 0 ? 'Verified' : 'Review'
        };
      })
    };
  }

  function severityGrade(cls, count, pct, avgConf) {
    if (!count) return { grade: 'D', label: 'Clean' };
    var risk = riskOf(cls);
    if (risk === 'Critical' || (risk === 'High' && pct >= 8)) return { grade: 'C', label: 'Heavy' };
    if (risk === 'High' || risk === 'Medium' || pct >= 5) return { grade: 'B', label: 'Moderate' };
    if (avgConf >= 0.85 && count <= 2) return { grade: 'A', label: 'Light' };
    return { grade: 'B', label: 'Moderate' };
  }

  function executiveText(cls, stats) {
    var n = stats.count, label = cap(cls.replace(/_/g, ' '));
    if (!n) {
      return 'No ' + label.toLowerCase() + ' contacts identified on the surveyed lines. Seabed acoustic character in this category is consistent with clear corridor conditions at desk-review stage.';
    }
    var risk = riskOf(cls);
    var confPct = Math.round((stats.avgConf || 0) * 100);
    var frames = stats.frames.join(', ');
    var parts = [
      n + ' ' + label.toLowerCase() + ' contact' + (n > 1 ? 's' : '') + ' detected across frame' + (stats.frames.length > 1 ? 's' : '') + ' ' + frames + '.',
      'Mean model confidence ' + confPct + '%.',
      risk === 'Critical' || risk === 'High'
        ? 'High-risk class — ROV verification recommended prior to cable or turbine installation.'
        : risk === 'Medium'
          ? 'Moderate-risk class — desk review and selective verification advised.'
          : 'Low-risk class — monitor during installation window; no immediate ROV conflict indicated.'
    ];
    if (stats.verified) parts.push(stats.verified + ' contact(s) analyst-verified.');
    return parts.join(' ');
  }

  function remarksCode(cls, stats) {
    if (!stats.count) return 'CLR';
    var risk = riskOf(cls);
    var r = risk === 'Critical' ? 'CR' : risk === 'High' ? 'HI' : risk === 'Medium' ? 'MD' : 'LO';
    return r + '/' + stats.count + '/' + Math.round((stats.avgConf || 0) * 100) + '%';
  }

  function aggregateByClass(detections) {
    var total = (detections || []).length;
    var by = {};
  (detections || []).forEach(function (d) {
      if (!d || !d.cls) return;
      var k = norm(d.cls);
      if (!by[k]) by[k] = { count: 0, conf: 0, frames: {}, verified: 0, rov: 0 };
      by[k].count++;
      by[k].conf += d.conf || 0;
      by[k].frames[d.frame || 1] = 1;
      if (d.status === 'Verified') by[k].verified++;
      if (d.status !== 'Verified' && d.status !== 'Rejected' && (riskOf(k) === 'Critical' || riskOf(k) === 'High')) {
        by[k].rov++;
      }
    });
    Object.keys(by).forEach(function (k) {
      var b = by[k];
      b.avgConf = b.count ? b.conf / b.count : 0;
      b.pct = total ? Math.round((b.count / total) * 1000) / 10 : 0;
      b.frames = Object.keys(b.frames).map(Number).sort(function (a, b) { return a - b; });
    });
    return { total: total, by: by };
  }

  function buildRows(detections) {
    var agg = aggregateByClass(detections);
    var rows = [];
    var seen = {};
    DEFECT_ORDER.forEach(function (k) {
      seen[k] = 1;
      var stats = agg.by[k] || { count: 0, pct: 0, avgConf: 0, frames: [], verified: 0, rov: 0 };
      var sev = severityGrade(k, stats.count, stats.pct, stats.avgConf);
      rows.push({
        cls: k,
        label: cap(k.replace(/_/g, ' ')),
        executive: executiveText(k, stats),
        pct: stats.count ? stats.pct : 0,
        severity: sev.grade,
        severityLabel: sev.label,
        verified: stats.verified > 0,
        rovRequired: stats.rov > 0 || (stats.count > 0 && (riskOf(k) === 'Critical' || riskOf(k) === 'High')),
        remarks: remarksCode(k, stats),
        count: stats.count
      });
    });
    Object.keys(agg.by).sort().forEach(function (k) {
      if (seen[k]) return;
      var stats = agg.by[k];
      var sev = severityGrade(k, stats.count, stats.pct, stats.avgConf);
      rows.push({
        cls: k,
        label: cap(k.replace(/_/g, ' ')),
        executive: executiveText(k, stats),
        pct: stats.pct,
        severity: sev.grade,
        severityLabel: sev.label,
        verified: stats.verified > 0,
        rovRequired: stats.rov > 0,
        remarks: remarksCode(k, stats),
        count: stats.count
      });
    });
    return { rows: rows, total: agg.total };
  }

  function photoSamples(detections, frames, limit) {
    limit = limit || 6;
    var out = [];
    var dets = detections || [];
    for (var i = 0; i < dets.length && out.length < limit; i++) {
      var d = dets[i];
      var thumb = d.thumb || '';
      if (!thumb && frames && frames.length) {
        var fr = frames[(d.frame || 1) - 1];
        if (fr) thumb = fr.thumb || fr.pdfSrc || '';
      }
      out.push({
        cls: d.cls,
        frame: d.frame || 1,
        conf: d.conf,
        thumb: thumb,
        caption: cap(d.cls) + ' · Frame ' + (d.frame || 1) + ' · ' + Math.round((d.conf || 0) * 100) + '%'
      });
    }
    return out;
  }

  function buildPayload(session) {
    session = session || loadSession();
    if (!session || !session.detections || !session.detections.length) session = demoSession();
    var built = buildRows(session.detections);
    return {
      meta: {
        project: session.project || 'Survey project',
        survey_line: session.survey_line || '—',
        vessel: session.vessel || '—',
        sensor: session.sensor || 'Side-scan sonar',
        location: session.location || '—',
        date: session.date || new Date().toLocaleDateString(),
        frames: session.frames || 0,
        demo: !!session.demo
      },
      legend: LEGEND,
      rows: built.rows,
      total: built.total,
      photos: photoSamples(session.detections, session._frames, 8)
    };
  }

  global.SeabedDefectReport = {
    LEGEND: LEGEND,
    DEFECT_ORDER: DEFECT_ORDER,
    loadSession: loadSession,
    demoSession: demoSession,
    buildPayload: buildPayload,
    buildRows: buildRows,
    riskOf: riskOf,
    cap: cap
  };
})(window);
