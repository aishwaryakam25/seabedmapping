/**
 * NautiCAI Survey Intelligence — differentiated seabed mapping features.
 */
(function (global) {
  var KEYS = {
    fingerprints: 'seabed_site_fingerprints',
    snapshots: 'seabed_survey_snapshots',
    install: 'seabed_install_schedule',
    corridor: 'seabed_corridor',
    lifecycle: 'seabed_lifecycle',
    compareA: 'seabed_compare_survey_a',
    compareB: 'seabed_compare_survey_b'
  };

  var LIFECYCLE = ['ai_detected', 'desk_review', 'rov_scheduled', 'rov_confirmed', 'clearance_issued'];
  var LIFECYCLE_LABEL = {
    ai_detected: 'AI detected',
    desk_review: 'Desk review',
    rov_scheduled: 'ROV scheduled',
    rov_confirmed: 'ROV confirmed',
    clearance_issued: 'Clearance issued'
  };

  function readJSON(k, fb) {
    try {
      var r = localStorage.getItem(k);
      return r ? JSON.parse(r) : fb;
    } catch (e) {
      return fb;
    }
  }
  function writeJSON(k, v) {
    try {
      localStorage.setItem(k, JSON.stringify(v));
      return true;
    } catch (e) {
      return false;
    }
  }

  /** 1. Shadow height + burial from box geometry (illustrative physics). */
  function shadowMetrics(det, natW, natH, rangeScale) {
    rangeScale = rangeScale || 150;
    var b = det.box || [0, 0, 0, 0];
    var w = Math.max(1, b[2] - b[0]);
    var h = Math.max(1, b[3] - b[1]);
    var cxn = det.cxn != null ? det.cxn : (b[0] + b[2]) / 2 / Math.max(1, natW);
    var aspect = h / w;
    var shadowLenNorm = Math.min(0.45, aspect * 0.35 + (det.wnorm || w / Math.max(1, natW)) * 0.25);
    var heightM = shadowLenNorm * rangeScale * 0.55;
    var burial;
    if (aspect < 0.55 && heightM < 0.35) burial = 'buried';
    else if (heightM < 0.8) burial = 'partially_buried';
    else burial = 'proud';
    return {
      height_m: +heightM.toFixed(2),
      shadow_len_m: +(shadowLenNorm * rangeScale).toFixed(2),
      burial: burial,
      burial_label: burial === 'buried' ? 'Likely buried' : burial === 'partially_buried' ? 'Partially buried' : 'Proud on seabed'
    };
  }

  /** 2. Cross-line / cross-frame fusion by normalized position. */
  function fuseContacts(framesData, tol) {
    tol = tol || 0.06;
    var fused = [];
    (framesData || []).forEach(function (fr, fi) {
      (fr.dets || []).forEach(function (d, di) {
        var cxn = d.cxn != null ? d.cxn : 0.5;
        var cyn = d.cyn != null ? d.cyn : 0.5;
        var hit = null;
        for (var i = 0; i < fused.length; i++) {
          var f = fused[i];
          if (Math.abs(f.cxn - cxn) < tol && Math.abs(f.cyn - cyn) < tol && f.cls === d.cls) {
            hit = f;
            break;
          }
        }
        if (hit) {
          hit.lines = hit.lines || [hit.frame];
          if (hit.lines.indexOf(fi + 1) < 0) hit.lines.push(fi + 1);
          hit.conf = Math.max(hit.conf, d.conf);
          hit.corroboration = hit.lines.length;
        } else {
          fused.push({
            id: 'FUS-' + String(fused.length + 1).padStart(3, '0'),
            cls: d.cls,
            conf: d.conf,
            cxn: cxn,
            cyn: cyn,
            frame: fi + 1,
            lines: [fi + 1],
            corroboration: 1
          });
        }
      });
    });
    return fused.sort(function (a, b) {
      return b.corroboration - a.corroboration;
    });
  }

  /** 3. Contact lifecycle */
  function loadLifecycle() {
    return readJSON(KEYS.lifecycle, { items: {} });
  }
  function saveLifecycle(lc) {
    lc.updated = new Date().toISOString();
    writeJSON(KEYS.lifecycle, lc);
  }
  function getLifecycleState(contactId) {
    var lc = loadLifecycle();
    return (lc.items[contactId] && lc.items[contactId].state) || 'ai_detected';
  }
  function setLifecycleState(contactId, state, patch) {
    var lc = loadLifecycle();
    var cur = lc.items[contactId] || { id: contactId };
    cur.state = state;
    if (patch) Object.keys(patch).forEach(function (k) { cur[k] = patch[k]; });
    cur.updated = new Date().toISOString();
    lc.items[contactId] = cur;
    saveLifecycle(lc);
    return cur;
  }
  function advanceLifecycle(contactId, target) {
    return setLifecycleState(contactId, target, { advanced: new Date().toISOString() });
  }

  /** 4. Site fingerprint on analyst rejection */
  function recordFingerprint(det, frameIdx) {
    var fp = readJSON(KEYS.fingerprints, { rejects: [] });
    fp.rejects.push({
      cls: det.cls,
      cxn: det.cxn,
      cyn: det.cyn,
      conf: det.conf,
      frame: frameIdx,
      hash: [det.cls, (det.cxn || 0).toFixed(3), (det.cyn || 0).toFixed(3), (det.conf || 0).toFixed(2)].join('|'),
      at: new Date().toISOString()
    });
    if (fp.rejects.length > 200) fp.rejects = fp.rejects.slice(-200);
    writeJSON(KEYS.fingerprints, fp);
  }
  function fingerprintWarning(det) {
    var fp = readJSON(KEYS.fingerprints, { rejects: [] });
    var best = 0;
    fp.rejects.forEach(function (r) {
      if (r.cls !== det.cls) return;
      var dx = Math.abs((r.cxn || 0) - (det.cxn || 0));
      var dy = Math.abs((r.cyn || 0) - (det.cyn || 0));
      var sim = Math.max(0, 1 - (dx + dy) * 4);
      if (sim > best) best = sim;
    });
    return best >= 0.55 ? { warn: true, similarity: +best.toFixed(2) } : { warn: false, similarity: best };
  }

  /** 5. Corridor risk along chainage */
  function scoreCorridor(detections, chainKm, rangeScale) {
    rangeScale = rangeScale || 150;
    var segments = 20;
    var segRisk = new Array(segments).fill(0);
    (detections || []).forEach(function (d) {
      var risk = (global.SeabedSurveyOps && global.SeabedSurveyOps.riskScore)
        ? global.SeabedSurveyOps.riskScore(d.cls, d.conf)
        : d.conf || 0.5;
      var along = d.frame != null ? (d.frame - 1) / Math.max(1, chainKm || 4) : (d.x || 0) / 260;
      var si = Math.min(segments - 1, Math.max(0, Math.floor(along * segments)));
      segRisk[si] += risk;
    });
    var max = Math.max.apply(null, segRisk.concat([0.001]));
    return segRisk.map(function (v, i) {
      return { segment: i + 1, chainage_pct: Math.round(((i + 0.5) / segments) * 100), risk: +(v / max).toFixed(3), raw: +v.toFixed(3) };
    });
  }

  /** 6. Contractor / survey diff */
  function diffSurveys(a, b, tol) {
    tol = tol || 0.08;
    function key(d) {
      return (d.cls || '') + '@' + Math.round((d.cxn || d.x / 260 || 0.5) * 20) + ',' + Math.round((d.cyn || d.y / 60 || 0.5) * 20);
    }
    function normList(sess) {
      return (sess && sess.detections) || [];
    }
    var la = normList(a);
    var lb = normList(b);
    var mapB = {};
    lb.forEach(function (d) { mapB[key(d)] = d; });
    var onlyA = [];
    var onlyB = [];
    var matched = [];
    la.forEach(function (d) {
      var k = key(d);
      if (mapB[k]) matched.push({ a: d, b: mapB[k] });
      else onlyA.push(d);
    });
    var mapA = {};
    la.forEach(function (d) { mapA[key(d)] = d; });
    lb.forEach(function (d) {
      if (!mapA[key(d)]) onlyB.push(d);
    });
    return { matched: matched.length, only_a: onlyA, only_b: onlyB, moved: [] };
  }

  /** 7. Explainable attribution */
  function explainDetection(det, natW, natH) {
    natW = natW || 1024;
    natH = natH || 1257;
    var b = det.box || [0, 0, 0, 0];
    var w = b[2] - b[0];
    var h = b[3] - b[1];
    var aspect = h / Math.max(1, w);
    var cues = [];
    if (aspect > 1.2) cues.push('elongated acoustic shadow');
    if (aspect < 0.7) cues.push('compact bright return');
    if ((det.conf || 0) > 0.85) cues.push('strong hyperbolic apex');
    if (w / natW > 0.08) cues.push('broad seabed contact');
    else cues.push('isolated point target');
    var sm = shadowMetrics(det, natW, natH, 150);
    cues.push(sm.burial_label.toLowerCase());
    return {
      summary: cues.slice(0, 3).join(' · '),
      cues: cues,
      physics: sm
    };
  }

  /** 8. ROV ROI / carbon */
  function roiMetrics(detections, opts) {
    opts = opts || {};
    var costPerDive = opts.costPerDive || 18500;
    var hoursPerDive = opts.hoursPerDive || 6;
    var co2PerHour = opts.co2PerHour || 42;
    var autoAccept = opts.autoAccept || 0.9;
    var n = (detections || []).length;
    var auto = 0;
    var review = 0;
    (detections || []).forEach(function (d) {
      if ((d.conf || 0) >= autoAccept && d.status !== 'Rejected') auto++;
      else if (d.status !== 'Rejected') review++;
    });
    var divesSaved = Math.floor(auto * 0.35);
    return {
      total: n,
      auto_accept: auto,
      manual_review: review,
      dives_avoided: divesSaved,
      cost_saved_usd: divesSaved * costPerDive,
      co2_saved_kg: Math.round(divesSaved * hoursPerDive * co2PerHour * 0.15)
    };
  }

  /** 9. Temporal 4D change */
  function saveSnapshot(label, session) {
    var snaps = readJSON(KEYS.snapshots, { list: [] });
    snaps.list.push({
      label: label || 'Survey ' + (snaps.list.length + 1),
      at: new Date().toISOString(),
      session: session
    });
    if (snaps.list.length > 10) snaps.list = snaps.list.slice(-10);
    writeJSON(KEYS.snapshots, snaps);
    return snaps;
  }
  function temporalDiff(idxA, idxB) {
    var snaps = readJSON(KEYS.snapshots, { list: [] });
    var a = snaps.list[idxA];
    var b = snaps.list[idxB];
    if (!a || !b) return null;
    var d = diffSurveys(a.session, b.session);
    return {
      survey_a: a.label,
      survey_b: b.label,
      new_contacts: d.only_b.length,
      removed_contacts: d.only_a.length,
      stable: d.matched,
      only_a: d.only_a,
      only_b: d.only_b
    };
  }

  /** 10. Installation urgency */
  function setInstallDate(isoDate) {
    writeJSON(KEYS.install, { install_date: isoDate });
  }
  function getInstallDate() {
    return (readJSON(KEYS.install, {}) || {}).install_date || '';
  }
  function urgencyScore(det, installDate) {
    if (!installDate) return { days: null, urgency: 'unknown', label: 'No install date set' };
    var now = new Date();
    var inst = new Date(installDate);
    var days = Math.ceil((inst - now) / 86400000);
    var base = (global.SeabedSurveyOps && global.SeabedSurveyOps.riskScore)
      ? global.SeabedSurveyOps.riskScore(det.cls, det.conf)
      : det.conf || 0.5;
    var urgency = base * (days <= 7 ? 1.4 : days <= 30 ? 1.15 : 1.0);
    var tier = urgency >= 0.75 ? 'critical' : urgency >= 0.45 ? 'elevated' : 'routine';
    return {
      days: days,
      urgency: tier,
      score: +urgency.toFixed(3),
      label: days <= 0 ? 'Install activity due now' : days + ' days to install — ' + tier
    };
  }

  function enrichDetection(det, natW, natH, rangeScale, frameIdx) {
    var sm = shadowMetrics(det, natW, natH, rangeScale);
    var ex = explainDetection(det, natW, natH);
    var fp = fingerprintWarning(det);
    var urg = urgencyScore(det, getInstallDate());
  var cid = det.contact_id || det.id || ('CT-' + Math.round((det.cxn || 0.5) * 1000));
    return Object.assign({}, det, {
      contact_id: cid,
      height_m: sm.height_m,
      burial: sm.burial,
      burial_label: sm.burial_label,
      shadow_len_m: sm.shadow_len_m,
      explain: ex.summary,
      explain_cues: ex.cues,
      site_warn: fp.warn,
      site_similarity: fp.similarity,
      lifecycle: getLifecycleState(cid),
      urgency: urg.urgency,
      urgency_label: urg.label,
      urgency_score: urg.score
    });
  }

  function exportContactsGeoJSON(detections, meta) {
    meta = meta || {};
    var feats = (detections || []).map(function (d, i) {
      return {
        type: 'Feature',
        properties: {
          id: d.contact_id || 'CT-' + (i + 1),
          class: d.cls,
          confidence: d.conf,
          frame: d.frame,
          status: d.status,
          height_m: d.height_m,
          burial: d.burial
        },
        geometry: {
          type: 'Point',
          coordinates: [d.east || 512345 + (d.cxn - 0.5) * 200, d.north || 6123456 - (d.cyn - 0.5) * 200]
        }
      };
    });
    return { type: 'FeatureCollection', name: meta.project || 'NautiCAI survey', features: feats };
  }

  function exportContactsCSV(detections) {
    var hdr = ['contact_id', 'class', 'confidence', 'frame', 'status', 'height_m', 'burial', 'range_m', 'urgency', 'lifecycle'];
    var rows = [hdr.join(',')];
    (detections || []).forEach(function (d, i) {
      rows.push([
        d.contact_id || 'CT-' + (i + 1),
        d.cls,
        d.conf,
        d.frame || '',
        d.status || 'Review',
        d.height_m || '',
        d.burial || '',
        d.range || '',
        d.urgency || '',
        d.lifecycle || ''
      ].join(','));
    });
    return rows.join('\n');
  }

  /** Clearance-readiness summary used by the client Briefing page. */
  function surveyReadiness(detections) {
    var dets = detections || [];
    var R = { wreck: 'Critical', uxo: 'Critical', mine: 'Critical', pipeline: 'High', cable: 'High', cable_crossing: 'High', obstruction: 'High', rock: 'Low' };
    function rk(c) { var k = ('' + c).toLowerCase().replace(/\s+/g, '_'); return R[k] || (k === 'rock' ? 'Low' : 'Medium'); }
    var total = dets.length, hi = 0, hiOpen = 0, hiResolved = 0;
    dets.forEach(function (d) {
      var r = rk(d.cls);
      if (r === 'Critical' || r === 'High') {
        hi++;
        if (d.status === 'Verified' || d.status === 'Rejected') hiResolved++; else hiOpen++;
      }
    });
    if (!total) return { score: 0, color: '#6f8593', tier: 'pending', tier_label: 'Pending analysis', dives_avoided: 0, high_risk_open: 0, recommendation: 'Load and analyse side-scan frames in Data Review to assess clearance readiness.' };
    var score = Math.round(100 * (total - hiOpen) / total);
    var color = score >= 80 ? '#3fae6a' : (score >= 50 ? '#d9b13b' : '#e0564f');
    var tier, tier_label, recommendation;
    if (hi === 0 || hiOpen === 0) { tier = 'ready'; tier_label = 'Ready for clearance review'; recommendation = 'No open high-significance contacts. Proceed to clearance review with a qualified surveyor.'; }
    else if (hiResolved > 0) { tier = 'progress'; tier_label = 'Review in progress'; recommendation = 'Resolve the remaining open high-significance contacts before installation.'; }
    else { tier = 'blocked'; tier_label = 'High-risk verification required'; recommendation = 'Escalate open high-significance contacts to ROV verification before installation.'; }
    return { score: score, color: color, tier: tier, tier_label: tier_label, dives_avoided: Math.max(0, total - hi), high_risk_open: hiOpen, recommendation: recommendation };
  }

  global.SeabedIntelligence = {
    KEYS: KEYS,
    surveyReadiness: surveyReadiness,
    LIFECYCLE: LIFECYCLE,
    LIFECYCLE_LABEL: LIFECYCLE_LABEL,
    shadowMetrics: shadowMetrics,
    fuseContacts: fuseContacts,
    loadLifecycle: loadLifecycle,
    getLifecycleState: getLifecycleState,
    setLifecycleState: setLifecycleState,
    advanceLifecycle: advanceLifecycle,
    recordFingerprint: recordFingerprint,
    fingerprintWarning: fingerprintWarning,
    scoreCorridor: scoreCorridor,
    diffSurveys: diffSurveys,
    explainDetection: explainDetection,
    roiMetrics: roiMetrics,
    saveSnapshot: saveSnapshot,
    temporalDiff: temporalDiff,
    getInstallDate: getInstallDate,
    setInstallDate: setInstallDate,
    urgencyScore: urgencyScore,
    enrichDetection: enrichDetection,
    exportContactsGeoJSON: exportContactsGeoJSON,
    exportContactsCSV: exportContactsCSV,
    readJSON: readJSON,
    writeJSON: writeJSON
  };
})(window);
