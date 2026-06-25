/**
 * Shared lifecycle apply helpers — used by Copilot and Triage bulk approval.
 * Writes seabed_lifecycle (map shape) and syncs per-det lifecycle on review frames.
 */
(function (global) {
  var LK = 'seabed_lifecycle';
  var FK = 'nauticai_review_frames';

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

  function normalizeLifecycle(raw) {
    if (!raw) return { items: {} };
    if (Array.isArray(raw)) {
      var items = {};
      raw.forEach(function (x) {
        if (x && x.id) items[x.id] = x;
      });
      return { items: items };
    }
    if (!raw.items) raw.items = {};
    return raw;
  }

  function loadLifecycle() {
    return normalizeLifecycle(readJSON(LK, { items: {} }));
  }

  function saveLifecycle(lc) {
    lc.updated = new Date().toISOString();
    localStorage.setItem(LK, JSON.stringify(lc));
    return lc;
  }

  function advanceLifecycle(contactId, state, patch) {
    var lc = loadLifecycle();
    var cur = lc.items[contactId] || { id: contactId };
    cur.state = state;
    if (patch) {
      Object.keys(patch).forEach(function (k) { cur[k] = patch[k]; });
    }
    cur.updated = new Date().toISOString();
    lc.items[contactId] = cur;
    saveLifecycle(lc);
    return cur;
  }

  function syncDetectionLifecycle(contactId, state) {
    var frames = readJSON(FK, []);
    var touched = false;
    frames.forEach(function (fr) {
      (fr.dets || []).forEach(function (d) {
        var id = d.contact_id || d.id;
        if (id === contactId) {
          touched = true;
          d.lifecycle = state;
          if (state === 'rov_confirmed') d.status = 'Verified';
          else d.status = 'Review';
        }
      });
    });
    if (touched) localStorage.setItem(FK, JSON.stringify(frames));
    return touched;
  }

  /** Apply one approved proposal (contact_id + proposed_state). */
  function applyTransition(prop) {
    if (!prop || !prop.contact_id || !prop.proposed_state) return false;
    advanceLifecycle(prop.contact_id, prop.proposed_state, { source: 'agent_approval' });
    syncDetectionLifecycle(prop.contact_id, prop.proposed_state);
    return true;
  }

  function applyMany(proposals) {
    var n = 0;
    (proposals || []).forEach(function (p) {
      if (applyTransition(p)) n += 1;
    });
    return n;
  }

  global.SeabedLifecycle = {
    KEY: LK,
    FRAMES_KEY: FK,
    LIFECYCLE_LABEL: LIFECYCLE_LABEL,
    load: loadLifecycle,
    save: saveLifecycle,
    advance: advanceLifecycle,
    applyTransition: applyTransition,
    applyMany: applyMany
  };
})(typeof window !== 'undefined' ? window : globalThis);
