/**
 * Seabed Survey Ops — shared CONFIG and data helpers.
 * Wire points: localStorage keys, field mapping, theme, nav hrefs, risk weights.
 */
(function (global) {
  var CONFIG = {
    dataKeys: ['nauticai_session', 'seabed_session'],
    framesKey: 'nauticai_review_frames',
    surveyKey: 'nauticai_survey_info',
    workflowKey: 'seabed_rov_workflow',
    themeKeys: ['nauticai_theme'],
    nav: {
      workflow: '/workflow',
      rov: '/rov',
      dashboard: '/dashboard',
      map: '/map',
      review: '/review',
      detections: '/detections',
      twin: '/twin',
      reports: '/reports',
      settings: '/settings'
    },
    fieldMap: {
      id: ['id', 'contactId'],
      cls: ['cls', 'class', 'label'],
      conf: ['conf', 'confidence', 'score'],
      frame: ['frame', 'frameIndex', 'fi'],
      x: ['x', 'east', 'e'],
      y: ['y', 'north', 'n'],
      range: ['range', 'range_m', 'slantRange']
    },
    classRisk: {
      wreck: 1.0, uxo: 1.0, mine: 1.0,
      pipeline: 0.88, cable: 0.85, cable_crossing: 0.85, obstruction: 0.82,
      boulder: 0.62, anchor: 0.58, debris: 0.55, scour: 0.48, rock: 0.35
    },
    classRiskLabel: {
      wreck: 'Critical', uxo: 'Critical', mine: 'Critical',
      pipeline: 'High', cable: 'High', cable_crossing: 'High', obstruction: 'High',
      boulder: 'Medium', anchor: 'Medium', debris: 'Medium', scour: 'Medium', rock: 'Low'
    },
    highT: 0.65,
    medT: 0.35,
    riskColors: {
      Critical: '#e0564f', High: '#d2691e', Medium: '#d9b13b', Low: '#3fae6a'
    },
    classColors: {
      rock: '#d9b13b', boulder: '#d2691e', wreck: '#e0564f', pipeline: '#36b3c4',
      cable: '#3fae6a', cable_crossing: '#36b3c4', debris: '#9b6dd6', uxo: '#e0564f',
      mine: '#e0564f', anchor: '#2bb6c4', obstruction: '#9aa7b2', scour: '#7aa05a'
    },
    states: ['flagged', 'scheduled', 'in_progress', 'closed'],
    stateLabels: {
      none: 'Unreviewed', flagged: 'Flagged', scheduled: 'Scheduled',
      in_progress: 'In progress', closed: 'Closed'
    },
    exportFilename: 'rov_deployment_queue.json'
  };

  function normKey(s) {
    return ('' + (s || '')).toLowerCase().replace(/\s+/g, '_');
  }

  function pick(obj, keys, fallback) {
    if (!obj) return fallback;
    for (var i = 0; i < keys.length; i++) {
      if (obj[keys[i]] !== undefined && obj[keys[i]] !== null && obj[keys[i]] !== '') return obj[keys[i]];
    }
    return fallback;
  }

  function normConf(v) {
    var n = parseFloat(v);
    if (isNaN(n)) return 0;
    return n > 1 ? n / 100 : n;
  }

  function classWeight(cls) {
    var k = normKey(cls);
    return CONFIG.classRisk[k] != null ? CONFIG.classRisk[k] : 0.5;
  }

  function riskLabel(cls) {
    var k = normKey(cls);
    return CONFIG.classRiskLabel[k] || 'Medium';
  }

  function riskScore(cls, conf) {
    return classWeight(cls) * normConf(conf);
  }

  function riskTier(score) {
    if (score >= CONFIG.highT) return 'High';
    if (score >= CONFIG.medT) return 'Medium';
    return 'Low';
  }

  function colorForClass(cls) {
    return CONFIG.classColors[normKey(cls)] || '#2bb6c4';
  }

  function readJSON(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      if (!raw) return fallback;
      return JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  }

  function writeJSON(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); return true; } catch (e) { return false; }
  }

  function readSession() {
    for (var i = 0; i < CONFIG.dataKeys.length; i++) {
      var s = readJSON(CONFIG.dataKeys[i], null);
      if (s && (s.detections || []).length) return s;
      if (s && s.hasData && (s.detections || []).length) return s;
    }
    return null;
  }

  function mapDetection(d, i) {
    var cls = pick(d, CONFIG.fieldMap.cls, 'object');
    var conf = normConf(pick(d, CONFIG.fieldMap.conf, 0));
    return {
      id: pick(d, CONFIG.fieldMap.id, 'CT-' + String(i + 1).padStart(3, '0')),
      cls: cls,
      conf: conf,
      frame: parseInt(pick(d, CONFIG.fieldMap.frame, 1), 10) || 1,
      x: pick(d, CONFIG.fieldMap.x, 0),
      y: pick(d, CONFIG.fieldMap.y, 0),
      range: pick(d, CONFIG.fieldMap.range, null),
      risk: riskLabel(cls),
      score: riskScore(cls, conf),
      tier: riskTier(riskScore(cls, conf))
    };
  }

  function sampleContacts() {
    var rows = [
      ['wreck', 0.91, 3, 142, 28, 48],
      ['pipeline', 0.84, 2, 88, 41, 62],
      ['boulder', 0.76, 1, 201, 55, 31],
      ['cable', 0.72, 4, 56, 22, 74],
      ['debris', 0.68, 2, 167, 38, 19],
      ['rock', 0.61, 1, 94, 47, 55],
      ['obstruction', 0.79, 3, 118, 33, 44],
      ['scour', 0.58, 4, 73, 61, 28]
    ];
    return rows.map(function (r, i) {
      return mapDetection({
        id: 'DEMO-' + String(i + 1).padStart(3, '0'),
        cls: r[0], conf: r[1], frame: r[2], x: r[3], y: r[4], range: r[5]
      }, i);
    });
  }

  function loadContacts() {
    var sess = readSession();
    if (sess && sess.detections && sess.detections.length) {
      return sess.detections.map(function (d, i) {
        return mapDetection(d, i);
      });
    }
    return sampleContacts();
  }

  function defaultWorkflow() {
    return { version: 1, updated: new Date().toISOString(), items: {} };
  }

  function loadWorkflow() {
    var w = readJSON(CONFIG.workflowKey, null);
    if (!w || typeof w !== 'object') return defaultWorkflow();
    if (!w.items) w.items = {};
    return w;
  }

  function saveWorkflow(w) {
    w.updated = new Date().toISOString();
    writeJSON(CONFIG.workflowKey, w);
    if (global.NautiCAI_updateBell) global.NautiCAI_updateBell();
  }

  function itemState(w, id) {
    var it = w.items[id];
    return (it && it.state) || 'none';
  }

  function setItemState(w, id, state, patch) {
    var cur = w.items[id] || { id: id };
    cur.state = state;
    if (patch) {
      Object.keys(patch).forEach(function (k) { cur[k] = patch[k]; });
    }
    cur.updated = new Date().toISOString();
    w.items[id] = cur;
    return cur;
  }

  function openHighRiskCount(contacts, w) {
    var n = 0;
    contacts.forEach(function (c) {
      var st = itemState(w, c.id);
      if (st === 'closed') return;
      if (c.tier === 'High' || c.risk === 'Critical' || c.risk === 'High') {
        if (st === 'none' || st === 'flagged' || st === 'scheduled' || st === 'in_progress') n++;
      }
    });
    return n;
  }

  function statusCounts(contacts, w) {
    var counts = { none: 0, flagged: 0, scheduled: 0, in_progress: 0, closed: 0 };
    contacts.forEach(function (c) {
      var st = itemState(w, c.id);
      counts[st] = (counts[st] || 0) + 1;
    });
    return counts;
  }

  function buildExportQueue(contacts, w, meta) {
    var sess = readSession() || {};
    var survey = readJSON(CONFIG.surveyKey, {}) || {};
    var queued = contacts.filter(function (c) {
      var st = itemState(w, c.id);
      return st === 'flagged' || st === 'scheduled' || st === 'in_progress';
    }).map(function (c) {
      var it = w.items[c.id] || {};
      return {
        contact_id: c.id,
        class: c.cls,
        confidence: c.conf,
        risk_score: +c.score.toFixed(3),
        risk_tier: c.tier,
        survey_risk: c.risk,
        frame: c.frame,
        position: { x: c.x, y: c.y, range_m: c.range },
        workflow_state: itemState(w, c.id),
        scheduled_at: it.scheduled_at || null,
        rov_notes: it.notes || '',
        finding: it.finding || null
      };
    });
    return {
      generated: new Date().toISOString(),
      project: survey.project || sess.project || 'Offshore survey',
      survey_line: survey.survey_line || sess.survey_line || '',
      vessel: survey.vessel || sess.vessel || '',
      queue: queued,
      summary: statusCounts(contacts, w)
    };
  }

  function applyThemeFromStorage() {
    var t = 'dark';
    try {
      for (var i = 0; i < CONFIG.themeKeys.length; i++) {
        var v = localStorage.getItem(CONFIG.themeKeys[i]);
        if (v) { t = v; break; }
      }
    } catch (e) {}
    document.body.classList.toggle('light', t === 'light');
    var el = document.querySelector('#themeToggle');
    if (el) el.innerHTML = t === 'light' ? '&#9728;' : '&#9789;';
    return t;
  }

  function cap(s) {
    return ('' + s).replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  global.SeabedSurveyOps = {
    CONFIG: CONFIG,
    normKey: normKey,
    normConf: normConf,
    classWeight: classWeight,
    riskLabel: riskLabel,
    riskScore: riskScore,
    riskTier: riskTier,
    colorForClass: colorForClass,
    readSession: readSession,
    loadContacts: loadContacts,
    loadWorkflow: loadWorkflow,
    saveWorkflow: saveWorkflow,
    itemState: itemState,
    setItemState: setItemState,
    openHighRiskCount: openHighRiskCount,
    statusCounts: statusCounts,
    buildExportQueue: buildExportQueue,
    applyThemeFromStorage: applyThemeFromStorage,
    cap: cap
  };
})(window);
