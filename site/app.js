"use strict";

// The catalog lookup from src/lookup_satellite.py, in the browser: the same
// search order, labels and notes, plus the live forecasts from
// src/live_forecasts.py. data/*.json are written by src/build_site.py.

const R_EARTH = 6378.137; // km
const MAX_LISTED = 20;
const PREDICT_BELOW = 1000; // km perigee: higher objects stay up for centuries
const DAY_MS = 86400000;
const HOUR_MS = 3600000;
const SOON_DAYS = 30; // forecasts listed before "show all"
const TIME_OF_DAY_BELOW = 3; // days: closer than this, the hour matters
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LEAD_LABELS = { 30: "1 month before", 7: "1 week before", 3: "3 days before", 1: "1 day before" };

const OBJECT_TYPES = { PAY: "payload (satellite)", "R/B": "rocket body", DEB: "debris", UNK: "unknown" };
const OPS_STATUS = {
  "+": "operational", "-": "not operational", P: "partially operational",
  B: "backup", S: "spare", X: "extended mission", D: "decayed", "?": "unknown",
};
const ORBIT_TYPES = { ORB: "in orbit", LAN: "landed", IMP: "impacted", DOC: "docked", "R/T": "round trip" };
const CENTRES = { SU: "the Sun", MO: "the Moon", MA: "Mars", VE: "Venus", JU: "Jupiter", ME: "Mercury" };

const input = document.getElementById("q");
const form = document.getElementById("search-form");
const statusLine = document.getElementById("status");
const list = document.getElementById("matches");
const detail = document.getElementById("detail");

let catalog = null; // { updated, count, columns: { NAME: [...] } }
let upperNames = [];
let flux = null; // { start, values }: 81-day average solar flux for every day
let readyMessage = "";
let forecasts = null; // data/forecasts.json, or null if there are none yet
let forecastById = new Map();

function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(text).replace(/[&<>"']/g, (c) => map[c]);
}

function getRow(i) {
  const row = {};
  for (const [name, values] of Object.entries(catalog.columns)) row[name] = values[i];
  return row;
}

function parseDate(text) {
  if (!text) return null;
  const [y, m, d] = text.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
}

function daysBetween(start, end) {
  return Math.floor((end - start) / DAY_MS);
}

function compareText(a, b) {
  if (a === b) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return a < b ? -1 : 1;
}

/** Matching rows, best first: exact name, then starts with, then contains; oldest launch first within each. */
function find(query) {
  query = query.trim().toUpperCase();
  if (!query) return [];
  if (/^\d+$/.test(query)) {
    const i = catalog.columns.NORAD_CAT_ID.indexOf(Number(query));
    return i < 0 ? [] : [{ i, rank: 0 }];
  }
  const found = [];
  for (let i = 0; i < upperNames.length; i++) {
    const name = upperNames[i];
    if (!name.includes(query)) continue;
    let rank = 2;
    if (name === query) rank = 0;
    else if (name.startsWith(query)) rank = 1;
    found.push({ i, rank });
  }
  const launches = catalog.columns.LAUNCH_DATE;
  found.sort((a, b) => a.rank - b.rank || compareText(launches[a.i], launches[b.i]));
  return found;
}

function formatNumber(value, unit = "", decimals = 0) {
  if (value === null || value === undefined) return "unknown";
  const text = value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return text + unit;
}

function describeDuration(days) {
  if (days < 1) return "less than a day";
  const count = days.toLocaleString("en-US");
  if (days < 60) return `${count} days`;
  if (days < 730) return `${count} days (~${Math.round(days / 30.44)} months)`;
  return `${count} days (~${(days / 365.25).toFixed(1)} years)`;
}

/** The 81-day average solar flux known on a given day, or null before the record starts. */
function fluxOn(dayMs) {
  const index = daysBetween(parseDate(flux.start), dayMs);
  if (index < 0) return null;
  return flux.values[Math.min(index, flux.values.length - 1)];
}

/** "Nov 29", or with the time to the hour "Nov 29, 14:00 UTC" (or the minute); the year only if it isn't this one. */
function formatWhen(ms, withTime = false, toMinute = false) {
  if (withTime && !toMinute) ms = Math.round(ms / HOUR_MS) * HOUR_MS;
  const d = new Date(ms);
  let text = `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
  if (d.getUTCFullYear() !== new Date().getUTCFullYear()) text += `, ${d.getUTCFullYear()}`;
  const pad = (n) => String(n).padStart(2, "0");
  if (withTime) text += `, ${pad(d.getUTCHours())}:${pad(toMinute ? d.getUTCMinutes() : 0)} UTC`;
  return text;
}

function formatPercent(value) {
  return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

/** A forecast error in days as "+5 h" or "−1.5 d". */
function formatError(days) {
  const sign = days < 0 ? "−" : "+";
  const size = Math.abs(days);
  return size < 2 ? `${sign}${Math.round(size * 24)} h` : `${sign}${size.toFixed(1)} d`;
}

/** Predicted reentry and its 90% range as text. */
function describeForecast(f) {
  const withTime = f.days < TIME_OF_DAY_BELOW;
  const when = formatWhen(Date.parse(f.reentry), withTime);
  const range = f.low && f.high
    ? `${formatWhen(Date.parse(f.low), withTime)} – ${formatWhen(Date.parse(f.high), withTime)}` : "—";
  return { when, range };
}

function fitText(f) {
  return f.fit === "8-day" ? "the last 8 days" : `${Math.max(1, Math.round(f.tracked)).toLocaleString("en-US")} days`;
}

function canPredict(row) {
  return row.DECAY_DATE === null && row.ORBIT_CENTER === "EA" && row.PERIGEE !== null && row.PERIGEE < PREDICT_BELOW;
}

function factBlock(title, rows) {
  const items = rows.map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
  return `<div><h4>${escapeHtml(title)}</h4><dl>${items}</dl></div>`;
}

function showObject(i) {
  const row = getRow(i);
  const reentered = row.DECAY_DATE !== null;
  const launch = parseDate(row.LAUNCH_DATE);
  const where = ORBIT_TYPES[row.ORBIT_TYPE] || row.ORBIT_TYPE;

  const launchRows = [["Date", row.LAUNCH_DATE || "unknown"], ["Site", row.LAUNCH_SITE]];
  if (launch !== null) {
    const solar = fluxOn(launch);
    if (solar !== null) launchRows.push(["Solar flux", `${solar} sfu (81-day avg)`]);
  }

  let lifeRows;
  if (reentered) {
    lifeRows = [["Reentered", `${row.DECAY_DATE} (${where})`]];
    if (launch !== null) lifeRows.push(["In orbit", describeDuration(daysBetween(launch, parseDate(row.DECAY_DATE)))]);
  } else {
    lifeRows = [["Status", `still up (${where})`]];
    if (launch !== null) lifeRows.push(["So far", describeDuration(daysBetween(launch, Date.now()))]);
  }

  const orbitRows = [
    ["Perigee", formatNumber(row.PERIGEE, " km")],
    ["Apogee", formatNumber(row.APOGEE, " km")],
    ["Inclination", formatNumber(row.INCLINATION, "°", 2)],
    ["Period", formatNumber(row.PERIOD, " min", 1)],
  ];
  if (row.PERIGEE !== null && row.APOGEE !== null) {
    const ra = row.APOGEE + R_EARTH;
    const rp = row.PERIGEE + R_EARTH;
    orbitRows.push(["Eccentricity", ((ra - rp) / (ra + rp)).toFixed(4)]);
  }
  orbitRows.push(["Radar size", formatNumber(row.RCS, " m²", 3)]);
  if (row.ORBIT_CENTER !== "EA") {
    orbitRows.unshift(["Orbits", `${CENTRES[row.ORBIT_CENTER] || row.ORBIT_CENTER} (not Earth)`]);
  }

  const chips = [
    { text: reentered ? `reentered ${row.DECAY_DATE.slice(0, 4)}` : "still up", strong: true },
    { text: OBJECT_TYPES[row.OBJECT_TYPE] || row.OBJECT_TYPE },
    { text: `owner ${row.OWNER}` },
  ];
  if (OPS_STATUS[row.OPS_STATUS_CODE] && row.OPS_STATUS_CODE !== "D") chips.push({ text: OPS_STATUS[row.OPS_STATUS_CODE] });
  chips.push({ text: `NORAD ${row.NORAD_CAT_ID}` }, { text: row.OBJECT_ID });

  let note = "";
  const live = forecastById.get(row.NORAD_CAT_ID);
  if (live && !reentered) {
    const { when, range } = describeForecast(live);
    note = `<p><strong>Live forecast: reentry around ${escapeHtml(when)}</strong> `
      + `(90% range ${escapeHtml(range)}), from its orbit of ${escapeHtml(formatWhen(Date.parse(live.epoch), true))}, `
      + `${live.perigee} × ${live.apogee} km. Physics model only; drag fitted over ${escapeHtml(fitText(live))} of tracking.`
      + (live.group === 2 ? " Starlink satellites are steered down by their operator, so treat this as rough." : "")
      + ' <a href="#forecasts">All upcoming reentries</a></p>';
  } else if (reentered && row.PERIGEE !== null && row.PERIGEE < 250) {
    note = "<p>The perigee is under 250 km, so the catalog orbit is probably its final one before reentry.</p>";
  } else if (canPredict(row)) {
    note = "<p><strong>Still in low orbit, so drag will bring it down eventually.</strong> "
      + "It has no live forecast right now: it may not be expected within a year, or be holding its altitude "
      + "with thrusters, or have too few recent orbits. The physics + ML hybrid runs in the lookup tool "
      + 'on your own computer. <a href="#run">How to run it</a></p>';
  } else if (!reentered && row.ORBIT_CENTER === "EA" && row.PERIGEE !== null) {
    note = `<p>Its perigee is above ${PREDICT_BELOW.toLocaleString("en-US")} km, where drag is so weak `
      + "that it will stay up for centuries, so there's no reentry to predict.</p>";
  }

  detail.innerHTML = `
    <h3>${escapeHtml(row.OBJECT_NAME)}</h3>
    <ul class="chips">${chips.map((c) => `<li${c.strong ? ' class="strong"' : ""}>${escapeHtml(c.text)}</li>`).join("")}</ul>
    <div class="facts">
      ${factBlock("Launch", launchRows)}
      ${factBlock("Lifetime", lifeRows)}
      ${factBlock("Orbit (catalog)", orbitRows)}
    </div>
    ${note ? `<div class="callout">${note}</div>` : ""}`;
  detail.hidden = false;
}

function selectObject(i, scroll = true) {
  showObject(i);
  list.replaceChildren();
  input.value = catalog.columns.OBJECT_NAME[i];
  statusLine.textContent = "";
  history.replaceState(null, "", `#${catalog.columns.NORAD_CAT_ID[i]}`);
  if (scroll) detail.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function showMatches(query) {
  detail.hidden = true;
  list.replaceChildren();
  if (!query.trim()) {
    statusLine.textContent = readyMessage;
    return;
  }
  const found = find(query);
  if (found.length === 0) {
    statusLine.textContent = `No objects match "${query.trim()}".`;
    return;
  }
  const total = found.length.toLocaleString("en-US");
  if (found.length === 1) statusLine.textContent = "1 match";
  else if (found.length > MAX_LISTED) statusLine.textContent = `${total} matches, showing the first ${MAX_LISTED}`;
  else statusLine.textContent = `${total} matches`;

  for (const { i } of found.slice(0, MAX_LISTED)) {
    const c = catalog.columns;
    const reentered = c.DECAY_DATE[i] !== null;
    const state = reentered ? `reentered ${c.DECAY_DATE[i].slice(0, 4)}` : "still up";
    const type = OBJECT_TYPES[c.OBJECT_TYPE[i]] || c.OBJECT_TYPE[i];
    const item = document.createElement("li");
    item.innerHTML = `
      <button type="button" data-index="${i}">
        <span class="match-name">${escapeHtml(c.OBJECT_NAME[i])}</span>
        <span class="match-state${reentered ? "" : " up"}">${state}</span>
        <span class="match-meta">NORAD ${c.NORAD_CAT_ID[i]} · ${escapeHtml(type)} · launched ${c.LAUNCH_DATE[i] || "?"}</span>
      </button>`;
    list.append(item);
  }
}

function showFromHash() {
  const id = location.hash.slice(1);
  if (!/^\d+$/.test(id)) return;
  const i = catalog.columns.NORAD_CAT_ID.indexOf(Number(id));
  if (i >= 0) selectObject(i);
}

function forecastRow(f) {
  const { when, range } = describeForecast(f);
  const type = OBJECT_TYPES[f.type] || f.type || "unknown type";
  const tag = f.group === 2 ? ' · <span class="tag">operator-steered</span>' : "";
  return `<tr>
    <th scope="row"><a href="#${f.id}">${escapeHtml(f.name)}</a><span class="sub">${escapeHtml(type)} · NORAD ${f.id}${tag}</span></th>
    <td>${f.perigee.toLocaleString("en-US")} × ${f.apogee.toLocaleString("en-US")} km</td>
    <td class="strong">${escapeHtml(when)}</td>
    <td>${escapeHtml(range)}</td>
    <td>${escapeHtml(fitText(f))}</td>
  </tr>`;
}

function showForecasts(showAll = false) {
  const status = document.getElementById("forecast-status");
  const box = document.getElementById("forecast-list");
  if (!forecasts) {
    status.textContent = "No forecasts yet. They appear after the first forecast run.";
    return;
  }
  const all = forecasts.objects;
  const soon = all.filter((f) => Date.parse(f.reentry) - Date.now() < SOON_DAYS * DAY_MS);
  const shown = showAll ? all : soon;
  const counts = forecasts.counts || {};
  const others = [["not sinking", "holding altitude"], ["not drag", "changing faster than drag explains"],
    ["too few orbits", "with too few orbits so far"], ["over a year", "more than a year away"]]
    .filter(([key]) => counts[key]).map(([key, label]) => `${counts[key].toLocaleString("en-US")} ${label}`);
  status.textContent = `Updated ${formatWhen(Date.parse(forecasts.made), true, true)}. `
    + `${all.length.toLocaleString("en-US")} objects forecast to reenter within a year, ${soon.length.toLocaleString("en-US")} `
    + `within ${SOON_DAYS} days.` + (others.length ? ` Also tracking ${others.join(", ")}.` : "");
  if (shown.length === 0) {
    box.innerHTML = all.length ? `<p class="note">None expected within ${SOON_DAYS} days.</p>` : "";
  } else {
    box.innerHTML = `<div class="table-scroll"><table class="forecast-table">
      <thead><tr><th scope="col">Object</th><th scope="col">Orbit</th><th scope="col">Predicted reentry</th>
        <th scope="col">90% range</th><th scope="col">Drag fitted over</th></tr></thead>
      <tbody>${shown.map(forecastRow).join("")}</tbody></table></div>`;
  }
  if (all.length > soon.length) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "more";
    button.textContent = showAll ? `Show only the next ${SOON_DAYS} days`
      : `Show all ${all.length.toLocaleString("en-US")} forecasts within a year`;
    button.addEventListener("click", () => showForecasts(!showAll));
    box.append(button);
  }
}

function showTrackRecord(scores) {
  const box = document.getElementById("track-record");
  if (!scores || !scores.overall.forecasts) {
    const since = scores && scores.since ? ` Forecasting started ${formatWhen(Date.parse(scores.since))}.` : "";
    box.innerHTML = `<p class="status">No forecasts scored yet.${since} Scores appear once objects we forecast `
      + "come down and the catalog lists them, usually within a few days.</p>";
    return;
  }
  const row = (label, s, cls = "") => `<tr${cls}><th scope="row">${escapeHtml(label)}</th>
    <td>${s.forecasts.toLocaleString("en-US")}</td><td>${s.objects.toLocaleString("en-US")}</td>
    <td>${formatPercent(s.median_error)}</td><td>${formatPercent(s.within_25)}</td><td>${formatPercent(s.in_range)}</td></tr>`;
  const bands = scores.bands.filter((b) => b.forecasts > 0);
  const leads = Object.keys(LEAD_LABELS).map(Number).sort((a, b) => b - a);
  const recent = scores.recent.map((r) => `<tr>
      <th scope="row"><a href="#${r.id}">${escapeHtml(r.name || r.id)}</a></th>
      <td>${escapeHtml(formatWhen(Date.parse(r.decay_date)))}</td>
      ${leads.map((lead) => {
        const f = r.leads[String(lead)];
        return f ? `<td>${escapeHtml(formatWhen(Date.parse(f.reentry), lead <= 1))} <span class="sub">${formatError(f.error_days)}</span></td>` : "<td>—</td>";
      }).join("")}
    </tr>`).join("");
  box.innerHTML = `
    <p>Since ${escapeHtml(formatWhen(Date.parse(scores.since)))}, by the time that was actually left (median error relative to it, as in the results below):</p>
    <div class="table-scroll"><table>
      <thead><tr><th scope="col">Time left</th><th scope="col">Forecasts</th><th scope="col">Objects</th>
        <th scope="col">Median error</th><th scope="col">Within 25%</th><th scope="col">90% range held</th></tr></thead>
      <tbody>${bands.map((b) => row(b.label[0].toUpperCase() + b.label.slice(1).replace("-", "–"), b)).join("")}${row("Overall", scores.overall, ' class="highlight"')}</tbody>
    </table></div>
    ${recent ? `<h4 class="table-title">Recent reentries: what we forecast beforehand</h4>
    <div class="table-scroll"><table class="recent-table">
      <thead><tr><th scope="col">Object</th><th scope="col">Reentered</th>${leads.map((l) => `<th scope="col">${LEAD_LABELS[l]}</th>`).join("")}</tr></thead>
      <tbody>${recent}</tbody></table></div>
    <p class="note">Each cell is the reentry we predicted that long before, and how far off it was from midday on the catalog's reentry date.</p>` : ""}`;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function start() {
  // The forecasts are optional: the site works without them.
  const live = Promise.allSettled([fetchJson("data/forecasts.json"), fetchJson("data/scores.json")]);
  try {
    [catalog, flux] = await Promise.all([fetchJson("data/catalog.json"), fetchJson("data/solar_flux.json")]);
  } catch (error) {
    statusLine.textContent = "Couldn't load the catalog. Try reloading the page.";
    console.error(error);
    return;
  }
  const [forecastResult, scoresResult] = await live;
  if (forecastResult.status === "fulfilled") {
    forecasts = forecastResult.value;
    forecastById = new Map(forecasts.objects.map((f) => [f.id, f]));
  }
  showForecasts();
  showTrackRecord(scoresResult.status === "fulfilled" ? scoresResult.value : null);
  upperNames = catalog.columns.OBJECT_NAME.map((name) => String(name).toUpperCase());
  readyMessage = `${catalog.count.toLocaleString("en-US")} objects. Catalog current to ${catalog.updated}.`;
  statusLine.textContent = readyMessage;
  document.getElementById("catalog-date").textContent = `Catalog current to ${catalog.updated}.`;
  input.disabled = false;

  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => showMatches(input.value), 120);
  });

  // Enter picks the best match, like typing a full name or NORAD ID in the terminal lookup.
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    clearTimeout(timer);
    const found = find(input.value);
    if (found.length) selectObject(found[0].i);
    else showMatches(input.value);
  });

  list.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-index]");
    if (button) selectObject(Number(button.dataset.index));
  });

  window.addEventListener("hashchange", showFromHash);
  showFromHash();
}

start();
