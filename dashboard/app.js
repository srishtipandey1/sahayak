const scenarioSelect = document.getElementById("scenario");
const languageSelect = document.getElementById("language");
const layerContainer = document.getElementById("layers");
const map = L.map("map", { zoomControl: false, zoomSnap: 0.1 });
L.control.zoom({ position: "bottomleft" }).addTo(map);
const overlayLayers = {};
const layerVisibility = {};
let facts = {};
let geoImage;
let imageBounds;
let tileLayer;
const layerData = {};
let reviewEnabled = false;
let reviewHistory = {};
const focusedLayerNames = new Set(["streams", "blocked_edges"]);
const defaultHiddenLayers = new Set(["buildings_hit"]);
let showWideContext = false;

const palette = {
  strict: "#d66b42",
  default: "#df941c",
  lenient: "#8060aa",
  flowpath: "#7b4baa",
  cut_off: "#bd3e3a",
  degraded: "#d88912",
  unaffected: "#27805d",
  not_assessable: "#697b77"
};
const translations = {
  en: {
    scenario: "Hazard scenario", strict: "Strict", default: "Default", lenient: "Lenient",
    language: "Language", basemap: "OpenStreetMap tiles (online)",
    opacity: "Post-event image opacity", headline: "Headline numbers",
    ask: "Ask", question: "Ask about this run…", askButton: "Ask", reports: "Reports",
    layers: "Map layers", reliability: "Optical reliability", valid: "Valid fraction",
    notGroundTruth: "Detection footprints are not ground truth."
  },
  hi: {
    scenario: "खतरा परिदृश्य", strict: "सख्त", default: "डिफ़ॉल्ट", lenient: "उदार",
    language: "भाषा", basemap: "OpenStreetMap टाइलें (ऑनलाइन)",
    opacity: "घटना-पश्चात छवि अपारदर्शिता", headline: "मुख्य जानकारी",
    ask: "पूछें", question: "इस रन के बारे में पूछें…", askButton: "पूछें", reports: "रिपोर्ट",
    layers: "मानचित्र परतें", reliability: "ऑप्टिकल विश्वसनीयता", valid: "मान्य अंश",
    notGroundTruth: "पहचान क्षेत्र ज़मीनी सत्यापन नहीं हैं।"
  },
  ne: {
    scenario: "जोखिम परिदृश्य", strict: "कडा", default: "पूर्वनिर्धारित", lenient: "उदार",
    language: "भाषा", basemap: "OpenStreetMap टाइलहरू (अनलाइन)",
    opacity: "घटनापछिको छविको अस्पष्टता", headline: "मुख्य तथ्य",
    ask: "सोध्नुहोस्", question: "यस रनबारे सोध्नुहोस्…", askButton: "सोध्नुहोस्",
    reports: "प्रतिवेदन", layers: "नक्साका तहहरू", reliability: "अप्टिकल विश्वसनीयता",
    valid: "मान्य अंश", notGroundTruth: "पहिचान गरिएका क्षेत्रहरू स्थलगत सत्यापन होइनन्।"
  }
};
const measureLabels = {
  en: {
    hazard_area: "Hazard area", buildings_inside: "Buildings inside",
    buildings_near_30m: "Nearby buildings", road_km_hit: "Road length hit",
    bridges_hit_within_20m: "Bridges hit", settlements_affected: "Settlements affected",
    cut_off: "Cut off", degraded: "Degraded", not_assessable: "Not assessable"
  },
  hi: {
    hazard_area: "खतरा क्षेत्र", buildings_inside: "भीतर की इमारतें",
    buildings_near_30m: "आस-पास की इमारतें", road_km_hit: "प्रभावित सड़क लंबाई",
    bridges_hit_within_20m: "प्रभावित पुल", settlements_affected: "प्रभावित बस्तियाँ",
    cut_off: "संपर्क-विच्छेद", degraded: "मार्ग प्रभावित", not_assessable: "आकलन संभव नहीं"
  },
  ne: {
    hazard_area: "जोखिम क्षेत्र", buildings_inside: "भित्रका भवनहरू",
    buildings_near_30m: "नजिकका भवनहरू", road_km_hit: "प्रभावित सडक लम्बाइ",
    bridges_hit_within_20m: "प्रभावित पुलहरू", settlements_affected: "प्रभावित बस्तीहरू",
    cut_off: "सम्पर्क विच्छेद", degraded: "मार्ग प्रभावित", not_assessable: "मूल्याङ्कन गर्न नसकिने"
  }
};
const layerLabels = {
  hazard_strict: "Strict hazard detections",
  hazard_default: "Default hazard detections",
  hazard_lenient: "Lenient hazard detections",
  buildings_hit: "Buildings in hazard area",
  roads_hit: "Roads intersecting hazard",
  bridges_hit: "Bridges near hazard",
  blocked_edges: "Blocked road edges",
  streams: "Stream network",
  flowpath: "Flowpath trace"
};
const layerCategories = [
  { label: "Hazard", matches: name => name.startsWith("hazard_") },
  {
    label: "Exposure",
    matches: name => ["buildings_hit", "roads_hit", "bridges_hit"].includes(name)
  },
  {
    label: "Access",
    matches: name => name.startsWith("settlements_") || name === "blocked_edges"
  },
  { label: "Terrain", matches: name => ["streams", "flowpath"].includes(name) }
];
const propertyLabels = {
  name: "Name",
  place: "Settlement type",
  building: "Building",
  amenity: "Facility",
  height: "Height",
  building_levels: "Building levels",
  highway: "Road type",
  road_class: "Road class",
  surface: "Surface",
  bridge: "Bridge",
  tunnel: "Tunnel",
  layer: "Map layer",
  ref: "Route reference",
  scenario: "Scenario",
  inside: "Inside detection area",
  near_30m: "Within 30 m",
  is_hit: "Intersects detection area",
  is_hit_20m: "Within 20 m",
  is_bridge: "Bridge",
  is_tunnel: "Tunnel",
  road_name: "Road name",
  settlement: "Settlement",
  feature_type: "Feature type",
  default_class: "Default access",
  strict_class: "Strict access",
  lenient_class: "Lenient access",
  default_hospital_before_name: "Nearest health facility before hazard",
  strict_hospital_before_name: "Nearest health facility before hazard",
  lenient_hospital_before_name: "Nearest health facility before hazard"
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  })[char]);
}

function humanize(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function layerStyle(feature, layerName) {
  const properties = feature.properties || {};
  const accessClass = properties[`${scenarioSelect.value}_class`];
  if (layerName === "streams") {
    return { color: "#2685ae", weight: 1.35, opacity: 0.78 };
  }
  if (layerName === "flowpath" || properties.feature_type === "flowpath") {
    return { color: "#7b4baa", weight: 4, opacity: 0.95, dashArray: "8 6" };
  }
  if (layerName === "roads_hit" || layerName === "blocked_edges") {
    return {
      color: layerName === "blocked_edges" ? "#bd3e3a" : "#bd6337",
      weight: layerName === "blocked_edges" ? 4 : 3,
      opacity: 0.9
    };
  }
  if (layerName === "buildings_hit") {
    return { color: "#743a2c", weight: 1.2, fillColor: "#a95b3d", fillOpacity: 0.72 };
  }
  if (layerName === "bridges_hit") {
    return { color: "#68448f", weight: 3, fillColor: "#8964aa", fillOpacity: 0.78 };
  }
  if (layerName.startsWith("settlements_")) {
    return { color: palette[accessClass] || palette.not_assessable, weight: 2 };
  }
  if (layerName.startsWith("hazard_")) {
    const color = palette[properties.scenario] || "#df941c";
    return { color, weight: 1.5, fillColor: color, fillOpacity: 0.28 };
  }
  return { color: "#46615b", weight: 1.5, fillOpacity: 0.38 };
}

function markerIcon(layerName, properties) {
  let markerClass = "";
  if (layerName === "buildings_hit") {
    markerClass = "map-marker--building";
  } else if (layerName === "bridges_hit") {
    markerClass = "map-marker--bridge";
  } else if (layerName.startsWith("settlements_")) {
    const accessClass = properties[`${scenarioSelect.value}_class`] || "not_assessable";
    markerClass = `map-marker--settlement map-marker--${accessClass.replaceAll("_", "-")}`;
  }
  if (!markerClass) return null;
  return L.divIcon({
    className: "map-div-icon",
    html: `<span class="map-marker ${markerClass}"></span>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8]
  });
}

function popupValue(value) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return "";
  return String(value);
}

function popupContent(feature, layerName) {
  const properties = feature.properties || {};
  const title = properties.name || properties.settlement || layerLabels[layerName] || layerName;
  const rows = Object.entries(properties)
    .filter(([key, value]) =>
      key !== "tags_json" && key !== "review_id" && value !== null && value !== undefined
    )
    .map(([key, value]) => [key, popupValue(value)])
    .filter(([, value]) => value.trim() !== "")
    .map(([key, value]) =>
      `<div class="popup-row"><strong>${escapeHtml(propertyLabels[key] || humanize(key))}:` +
      `</strong> ${escapeHtml(value)}</div>`
    )
    .join("");
  const review = reviewEnabled && layerName.startsWith("hazard_")
    ? reviewPopupContent(layerName, properties.review_id)
    : "";
  const overlappingReviews = reviewEnabled && layerName.startsWith("settlements_")
    ? overlappingHazardReviews(feature)
    : "";
  return `<div class="popup-title">${escapeHtml(title)}</div>${rows}${review}${overlappingReviews}`;
}

function reviewPopupContent(layerName, featureId) {
  if (!featureId) return "";
  const key = `${layerName}:${featureId}`;
  const history = reviewHistory[key] || [];
  const latest = history.length ? history[history.length - 1] : null;
  const latestLabel = latest
    ? `Latest: ${humanize(latest.status)} by ${latest.reviewer} · ` +
      `${new Date(latest.timestamp_utc).toLocaleString()}` +
      `${latest.reason ? ` · ${latest.reason}` : ""}`
    : "No review recorded yet.";
  return `<section class="review-panel" data-review-key="${escapeHtml(key)}"
      data-review-layer="${escapeHtml(layerName)}" data-review-id="${escapeHtml(featureId)}">
    <strong>Human review</strong>
    <small>Reviewer label is self-reported; no account authentication is used.</small>
    <label>Reviewer label<input data-reviewer maxlength="80" required
      value="${escapeHtml(latest?.reviewer || "")}"></label>
    <label>Optional reason<textarea data-review-reason maxlength="500"></textarea></label>
    <p class="review-latest" aria-live="polite">${escapeHtml(latestLabel)}</p>
    <div class="review-actions">
      <button type="button" data-review-status="confirmed">Confirm</button>
      <button type="button" data-review-status="rejected">Reject</button>
      <button type="button" data-review-status="skipped">Skip</button>
    </div>
    <p class="review-feedback" aria-live="polite"></p>
  </section>`;
}

function pointInRing(point, ring) {
  if (
    !Array.isArray(ring) ||
    ring.length < 3 ||
    ring.some(coordinate =>
      !Array.isArray(coordinate) ||
      coordinate.length < 2 ||
      !Number.isFinite(coordinate[0]) ||
      !Number.isFinite(coordinate[1])
    )
  ) {
    return false;
  }
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [x1, y1] = ring[previous];
    const [x2, y2] = ring[index];
    const cross = (point[0] - x1) * (y2 - y1) - (point[1] - y1) * (x2 - x1);
    if (
      Math.abs(cross) < 1e-10 &&
      point[0] >= Math.min(x1, x2) - 1e-10 &&
      point[0] <= Math.max(x1, x2) + 1e-10 &&
      point[1] >= Math.min(y1, y2) - 1e-10 &&
      point[1] <= Math.max(y1, y2) + 1e-10
    ) {
      return true;
    }
    if ((y1 > point[1]) !== (y2 > point[1])) {
      const crossingX = x1 + ((point[1] - y1) * (x2 - x1)) / (y2 - y1);
      if (point[0] < crossingX) inside = !inside;
    }
  }
  return inside;
}

function geometryContainsPoint(geometry, point) {
  const polygonContainsPoint = polygon =>
    Array.isArray(polygon) &&
    polygon.length > 0 &&
    pointInRing(point, polygon[0]) &&
    !polygon.slice(1).some(ring => pointInRing(point, ring));
  if (geometry?.type === "Polygon") return polygonContainsPoint(geometry.coordinates);
  if (geometry?.type === "MultiPolygon" && Array.isArray(geometry.coordinates)) {
    return geometry.coordinates.some(polygonContainsPoint);
  }
  return false;
}

function overlappingHazardReviews(feature) {
  const [longitude, latitude] = feature.geometry?.coordinates || [];
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return "";
  const point = [longitude, latitude];
  return Object.entries(layerData)
    .filter(([name]) =>
      name.startsWith("hazard_") &&
      overlayLayers[name] &&
      map.hasLayer(overlayLayers[name])
    )
    .flatMap(([name, data]) => (data.features || [])
      .filter(detection => geometryContainsPoint(detection.geometry, point))
      .map(detection => reviewPopupContent(name, detection.properties?.review_id)))
    .join("");
}

function bindReviewControls(popupElement) {
  for (const panel of popupElement?.querySelectorAll(".review-panel") || []) {
    const reviewerInput = panel.querySelector("[data-reviewer]");
    const reasonInput = panel.querySelector("[data-review-reason]");
    const feedback = panel.querySelector(".review-feedback");
    const latestLabel = panel.querySelector(".review-latest");
    const layerName = panel.dataset.reviewLayer;
    const featureId = panel.dataset.reviewId;
    for (const button of panel.querySelectorAll("[data-review-status]")) {
      button.addEventListener("click", async () => {
        const reviewer = reviewerInput.value.trim();
        if (!reviewer) {
          feedback.textContent = "Enter a reviewer label before saving.";
          reviewerInput.focus();
          return;
        }
        const buttons = [...panel.querySelectorAll("[data-review-status]")];
        buttons.forEach(item => { item.disabled = true; });
        feedback.textContent = "Saving review…";
        try {
          const response = await fetch("/api/reviews", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              layer_name: layerName,
              feature_id: featureId,
              status: button.dataset.reviewStatus,
              reviewer,
              reason: reasonInput.value.trim() || null
            })
          });
          const result = await readJson(response);
          const key = `${layerName}:${featureId}`;
          reviewHistory[key] ||= [];
          reviewHistory[key].push(result.review);
          const saved = result.review;
          latestLabel.textContent =
            `Latest: ${humanize(saved.status)} by ${saved.reviewer} · ` +
            new Date(saved.timestamp_utc).toLocaleString() +
            `${saved.reason ? ` · ${saved.reason}` : ""}`;
          showMessage(feedback, `Saved: ${humanize(saved.status)}.`);
        } catch (error) {
          showError(feedback, error, "Review error: ");
        } finally {
          buttons.forEach(item => { item.disabled = false; });
        }
      });
    }
  }
}

function layerVisible(name) {
  return Object.prototype.hasOwnProperty.call(layerVisibility, name)
    ? layerVisibility[name]
    : !defaultHiddenLayers.has(name);
}

function coordinateBounds(geometry) {
  const coordinates = [];
  const visit = value => {
    if (!Array.isArray(value)) return;
    if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") {
      coordinates.push([value[1], value[0]]);
      return;
    }
    for (const child of value) visit(child);
  };
  visit(geometry?.coordinates);
  return coordinates.length ? L.latLngBounds(coordinates) : null;
}

function includeFeature(feature, layerName) {
  if (
    showWideContext ||
    (!focusedLayerNames.has(layerName) && !layerName.startsWith("settlements_"))
  ) {
    return true;
  }
  const geometry = feature.geometry;
  if (!geometry) return false;
  if (geometry.type === "Point" && layerName.startsWith("settlements_")) {
    const [longitude, latitude] = geometry.coordinates;
    return imageBounds?.contains([latitude, longitude]) ?? true;
  }
  const bounds = coordinateBounds(geometry);
  return bounds ? (imageBounds?.intersects(bounds) ?? true) : false;
}

function addGeoJson(name, data) {
  if (overlayLayers[name]) map.removeLayer(overlayLayers[name]);
  layerData[name] = data;
  const layer = L.geoJSON(data, {
    filter: feature => includeFeature(feature, name),
    style: feature => layerStyle(feature, name),
    pointToLayer: (feature, latlng) => {
      const properties = feature.properties || {};
      const accessClass = properties[`${scenarioSelect.value}_class`];
      const icon = markerIcon(name, properties);
      if (icon) return L.marker(latlng, { icon });
      const color = palette[accessClass] || (name === "bridges_hit" ? "#7652a3" : "#277da1");
      return L.circleMarker(latlng, {
        radius: 5,
        color,
        weight: 2,
        fillColor: color,
        fillOpacity: 0.88
      });
    },
    onEachFeature: (feature, item) => {
      item.bindPopup(() => popupContent(feature, name));
      if (reviewEnabled) {
        item.on("popupopen", event => {
          bindReviewControls(event.popup.getElement());
        });
      }
    }
  });
  overlayLayers[name] = layer;
  if (layerVisible(name)) layer.addTo(map);
  return layer;
}

function headlineValue(id) {
  return (facts.headline || []).find(item => item.id === id)?.value ?? null;
}

function formatHeadlineValue(entry) {
  if (entry?.value === null || entry?.value === undefined) return "Not available";
  const value = typeof entry.value === "number"
    ? new Intl.NumberFormat("en", { maximumFractionDigits: 2 }).format(entry.value)
    : String(entry.value);
  let unit = entry.unit || "";
  if (entry.value === 1) {
    unit = {
      buildings: "building",
      bridges: "bridge",
      settlements: "settlement"
    }[unit] || unit;
  }
  return [value, unit].filter(Boolean).join(" ");
}

function renderHeadline() {
  if (!facts.headline) return;
  const section = document.getElementById("headline");
  section.replaceChildren();
  const scenario = scenarioSelect.value;
  const labels = measureLabels[languageSelect.value] || measureLabels.en;
  const keys = [
    "hazard_area", "buildings_inside", "buildings_near_30m", "road_km_hit",
    "bridges_hit_within_20m", "settlements_affected", "cut_off", "degraded", "not_assessable"
  ];
  for (const key of keys) {
    const entry = facts.headline.find(item => item.id === `${key}_${scenario}`);
    const card = document.createElement("div");
    card.className = "stat-card";
    const value = document.createElement("strong");
    value.className = "stat-value";
    value.textContent = formatHeadlineValue(entry);
    const label = document.createElement("span");
    label.className = "stat-label";
    label.textContent = labels[key];
    card.append(value, label);
    section.append(card);
  }

  const reliability = headlineValue("optical_reliability") ?? "not available";
  const valid = headlineValue("optical_valid_fraction");
  const validText = typeof valid === "number" ? `${(valid * 100).toFixed(1)}%` : "not available";
  const banner = document.getElementById("reliability");
  banner.className = "reliability-banner";
  if (reliability === "adequate") {
    banner.classList.add("state-adequate");
  } else if (reliability === "unverified") {
    banner.classList.add("state-unverified");
  } else {
    banner.classList.add("state-low");
  }
  const content = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = `Image reliability: ${reliability}`;
  const detail = document.createElement("small");
  detail.textContent = `${validText} valid pixels · detections are not ground truth`;
  content.append(title, detail);
  const dot = document.createElement("span");
  dot.className = "reliability-dot";
  banner.replaceChildren(dot, content);
}

function translateInterface() {
  const language = translations[languageSelect.value] ? languageSelect.value : "en";
  const words = translations[language];
  document.documentElement.lang = language;
  document.querySelector('label[for="scenario"] span').textContent = words.scenario;
  document.querySelector('label[for="language"] span').textContent = words.language;
  document.querySelector("#layers-summary").textContent = words.layers;
  document.querySelector("#question").placeholder = words.question;
  document.querySelector("#question").setAttribute("aria-label", words.question);
  document.querySelector("#ask-form button").textContent = words.askButton;
  for (const option of scenarioSelect.options) option.textContent = words[option.value];
  renderHeadline();
}

function errorDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map(item => {
      if (typeof item === "string") return item;
      const location = Array.isArray(item?.loc) ? `${item.loc.join(".")}: ` : "";
      return `${location}${item?.msg || JSON.stringify(item)}`;
    }).join("; ");
  }
  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }
  return "";
}

async function readJson(response) {
  const body = await response.text();
  let result;
  try {
    result = body ? JSON.parse(body) : {};
  } catch {
    throw new Error(`Request failed (${response.status}): ${body || "empty response"}`);
  }
  if (!response.ok) {
    throw new Error(errorDetail(result.detail) || `Request failed (${response.status})`);
  }
  return result;
}

function showError(element, error, prefix = "") {
  element.classList.add("error");
  const message = error instanceof Error ? error.message : String(error);
  element.textContent = `${prefix}${message}`;
}

function showMessage(element, message) {
  element.classList.remove("error");
  element.textContent = message;
}

function appendLayerCheckbox(group, name) {
  const label = document.createElement("label");
  label.className = "layer-item";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = layerVisible(name);
  checkbox.addEventListener("change", () => {
    layerVisibility[name] = checkbox.checked;
    if (checkbox.checked) {
      if (overlayLayers[name]) map.addLayer(overlayLayers[name]);
    } else if (overlayLayers[name]) {
      map.removeLayer(overlayLayers[name]);
    }
  });
  const text = document.createElement("span");
  text.textContent = layerLabels[name] || humanize(name);
  label.append(checkbox, text);
  group.append(label);
}

async function loadLayers() {
  const scenario = scenarioSelect.value;
  const response = await fetch(`/api/layers?scenario=${encodeURIComponent(scenario)}`);
  const result = await readJson(response);
  const names = new Set(result.layers);
  for (const name of Object.keys(overlayLayers)) {
    if ((name.startsWith("hazard_") || name.startsWith("settlements_")) && !names.has(name)) {
      map.removeLayer(overlayLayers[name]);
      delete overlayLayers[name];
    }
  }
  layerContainer.replaceChildren();
  for (const category of layerCategories) {
    const namesInCategory = result.layers.filter(category.matches);
    if (!namesInCategory.length) continue;
    const group = document.createElement("section");
    group.className = "layer-group";
    const title = document.createElement("h3");
    title.className = "layer-group-title";
    title.textContent = category.label;
    group.append(title);
    layerContainer.append(group);
    for (const name of namesInCategory) {
      const label = document.createElement("label");
      label.className = "layer-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = layerVisible(name);
      checkbox.addEventListener("change", () => {
        layerVisibility[name] = checkbox.checked;
        if (checkbox.checked) {
          if (overlayLayers[name]) map.addLayer(overlayLayers[name]);
        } else if (overlayLayers[name]) {
          map.removeLayer(overlayLayers[name]);
        }
      });
      const text = document.createElement("span");
      text.textContent = layerLabels[name] || humanize(name);
      label.append(checkbox, text);
      group.append(label);
      if (!layerData[name]) {
        const endpoint = name.startsWith("hazard_") ? "reviewable" : "layers";
        const layerResponse = await fetch(
          `/api/${endpoint}/${encodeURIComponent(name)}.geojson`
        );
        addGeoJson(name, await readJson(layerResponse));
      } else if (overlayLayers[name]) {
        if (checkbox.checked) map.addLayer(overlayLayers[name]);
        else map.removeLayer(overlayLayers[name]);
      } else {
        addGeoJson(name, layerData[name]);
      }
    }
  }
  for (const name of result.layers) {
    if (!layerCategories.some(category => category.matches(name))) {
      appendLayerCheckbox(layerContainer, name);
    }
  }
}

document.getElementById("ask-form").addEventListener("submit", async event => {
  event.preventDefault();
  const answer = document.getElementById("answer");
  const button = document.querySelector("#ask-form button");
  button.disabled = true;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: document.getElementById("question").value,
        lang: languageSelect.value
      })
    });
    const result = await readJson(response);
    showMessage(answer, result.answer);
  } catch (error) {
    showError(answer, error, "Ask error: ");
  } finally {
    button.disabled = false;
  }
});

scenarioSelect.addEventListener("change", () => {
  renderHeadline();
  loadLayers().catch(error => {
    showError(document.getElementById("answer"), error, "Layer error: ");
  });
});
languageSelect.addEventListener("change", translateInterface);
document.getElementById("opacity").addEventListener("input", event => {
  const opacity = Number(event.target.value);
  if (geoImage) geoImage.setOpacity(opacity);
  document.getElementById("opacity-value").textContent = `${Math.round(opacity * 100)}%`;
});
document.getElementById("basemap").addEventListener("change", event => {
  if (event.target.checked) {
    tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors"
    }).addTo(map);
  } else if (tileLayer) {
    map.removeLayer(tileLayer);
    tileLayer = undefined;
  }
});

async function initialize() {
  const [factsResponse, headlineResponse] = await Promise.all([
    fetch("/api/facts"),
    fetch("/api/headline")
  ]);
  facts = { ...(await readJson(factsResponse)), headline: await readJson(headlineResponse) };
  try {
    const reviewsResponse = await fetch("/api/reviews");
    const reviews = await readJson(reviewsResponse);
    reviewEnabled = reviews.enabled === true;
    reviewHistory = reviews.reviews || {};
  } catch (error) {
    showError(document.getElementById("answer"), error, "Review data error: ");
  }
  translateInterface();
  const bounds = facts.dashboard?.post_event_image?.bounds;
  if (!bounds) throw new Error("Dashboard image bounds are missing from facts.json.");
  imageBounds = L.latLngBounds(bounds);
  const image = facts.dashboard.post_event_image;
  geoImage = L.imageOverlay(image.url, image.bounds, { opacity: 0.82 }).addTo(map);
  await loadLayers();
  const reports = document.getElementById("reports");
  for (const language of ["en", "hi", "ne"]) {
    const link = document.createElement("a");
    link.className = "report-link";
    link.href = `/report/situation-${language}.pdf`;
    link.textContent = `${language.toUpperCase()} report`;
    reports.append(link);
  }
  const fitFocus = () => {
    map.invalidateSize({ pan: false });
    map.fitBounds(imageBounds, {
      paddingTopLeft: [Math.min(390, document.querySelector(".briefing-panel").offsetWidth + 28), 18],
      paddingBottomRight: [20, 18],
      maxZoom: 16
    });
  };
  document.getElementById("fit-map").addEventListener("click", fitFocus);
  document.getElementById("wide-context").addEventListener("change", event => {
    showWideContext = event.target.checked;
    for (const name of Object.keys(layerData)) {
      if (focusedLayerNames.has(name) || name.startsWith("settlements_")) {
        addGeoJson(name, layerData[name]);
      }
    }
  });
  window.addEventListener("resize", () => map.invalidateSize({ pan: false }));
  window.setTimeout(fitFocus, 80);
  map.on("click", async event => {
    const answer = document.getElementById("answer");
    if (!imageBounds.contains(event.latlng)) {
      showMessage(answer, "Click inside the satellite image to start a flowpath.");
      return;
    }
    try {
      const response = await fetch("/api/flowpath", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lon: event.latlng.lng,
          lat: event.latlng.lat,
          distance_km: 30
        })
      });
      const result = await readJson(response);
      const path = addGeoJson("flowpath", result.geojson);
      const pathBounds = path.getBounds();
      if (pathBounds.isValid()) map.fitBounds(pathBounds.pad(0.12));
      showMessage(answer, `Flowpath traced: ${result.facts.output}`);
    } catch (error) {
      showError(answer, error, "Flowpath error: ");
    }
  });
}

initialize().catch(error => {
  showError(document.getElementById("headline"), error, "Dashboard failed to initialize: ");
});
