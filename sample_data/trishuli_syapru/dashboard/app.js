const scenarioSelect = document.getElementById("scenario");
const languageSelect = document.getElementById("language");
const layerContainer = document.getElementById("layers");
const map = L.map("map", { zoomControl: true });
const overlayLayers = {};
const layerVisibility = {};
let facts = {};
let geoImage;
let tileLayer;
const palette = {
  strict: "#e64b35",
  default: "#f0a202",
  lenient: "#7b2cbf",
  flowpath: "#0757c7",
  cut_off: "#d62728",
  degraded: "#ff7f0e",
  unaffected: "#2ca02c",
  not_assessable: "#777777"
};
const translations = {
  en: {
    scenario: "Hazard scenario", strict: "Strict", default: "Default", lenient: "Lenient",
    language: "Language", basemap: "OpenStreetMap tiles (online)",
    opacity: "Post-event image opacity", headline: "Headline",
    ask: "Ask about this run", question: "Question", askButton: "Ask", reports: "Reports",
    layers: "Map layers", reliability: "Optical reliability", valid: "valid fraction",
    notGroundTruth: "Detection footprints are not ground truth."
  },
  hi: {
    scenario: "खतरा परिदृश्य", strict: "सख्त", default: "डिफ़ॉल्ट", lenient: "उदार",
    language: "भाषा", basemap: "OpenStreetMap टाइलें (ऑनलाइन)",
    opacity: "घटना-पश्चात छवि अपारदर्शिता", headline: "मुख्य जानकारी",
    ask: "इस रन के बारे में पूछें", question: "प्रश्न", askButton: "पूछें", reports: "रिपोर्ट",
    layers: "मानचित्र परतें", reliability: "ऑप्टिकल विश्वसनीयता", valid: "मान्य अंश",
    notGroundTruth: "पहचान क्षेत्र ज़मीनी सत्यापन नहीं हैं।"
  },
  ne: {
    scenario: "जोखिम परिदृश्य", strict: "कडा", default: "पूर्वनिर्धारित", lenient: "उदार",
    language: "भाषा", basemap: "OpenStreetMap टाइलहरू (अनलाइन)",
    opacity: "घटनापछिको छविको अस्पष्टता", headline: "मुख्य तथ्य",
    ask: "यस रनबारे सोध्नुहोस्", question: "प्रश्न", askButton: "सोध्नुहोस्", reports: "प्रतिवेदन",
    layers: "नक्साका तहहरू", reliability: "अप्टिकल विश्वसनीयता", valid: "मान्य अंश",
    notGroundTruth: "पहिचान गरिएका क्षेत्रहरू स्थलगत सत्यापन होइनन्।"
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  })[char]);
}

function layerStyle(feature) {
  const properties = feature.properties || {};
  const accessClass = properties[`${scenarioSelect.value}_class`];
  return {
    color: palette[properties.scenario] || palette[accessClass] || "#333",
    weight: ["flowpath", "road_crossing"].includes(properties.feature_type) ? 3 : 1,
    fillOpacity: 0.25
  };
}

function addGeoJson(name, data) {
  if (overlayLayers[name]) map.removeLayer(overlayLayers[name]);
  const layer = L.geoJSON(data, {
    style: layerStyle,
    pointToLayer: (feature, latlng) => {
      const properties = feature.properties || {};
      const accessClass = properties[`${scenarioSelect.value}_class`];
      return L.circleMarker(latlng, {
        radius: name.startsWith("settlements_") ? 5 : 7,
        color: palette[accessClass] || palette[properties.feature_type] || "#1d4ed8",
        fillOpacity: 0.8
      });
    },
    onEachFeature: (feature, item) => {
      const rows = Object.entries(feature.properties || {})
        .filter(([key]) => key !== "tags_json")
        .map(([key, value]) => `<strong>${escapeHtml(key)}</strong>: ${escapeHtml(value)}`)
        .join("<br>");
      item.bindPopup(rows || escapeHtml(name));
    }
  });
  overlayLayers[name] = layer;
  if (layerVisibility[name] !== false) layer.addTo(map);
}

function translateInterface() {
  const language = translations[languageSelect.value] ? languageSelect.value : "en";
  const words = translations[language];
  document.documentElement.lang = language;
  document.querySelector('label[for="scenario"]').textContent = words.scenario;
  document.querySelector('label[for="language"]').textContent = words.language;
  document.querySelector('label[for="basemap"]').textContent = words.basemap;
  document.querySelector('label[for="opacity"]').textContent = words.opacity;
  document.querySelector("#headline-section h2").textContent = words.headline;
  document.querySelector("#ask-section h2").textContent = words.ask;
  document.querySelector("#reports-section h2").textContent = words.reports;
  document.querySelector("#layers-summary").textContent = words.layers;
  document.querySelector("#question").placeholder = words.question;
  document.querySelector("#question").setAttribute("aria-label", words.question);
  document.querySelector("#ask-form button").textContent = words.askButton;
  for (const option of scenarioSelect.options) {
    option.textContent = words[option.value];
  }
  renderHeadline();
}

function headlineValue(id) {
  return (facts.headline || []).find(item => item.id === id)?.value ?? null;
}

function renderHeadline() {
  if (!facts.headline) return;
  const words = translations[languageSelect.value] || translations.en;
  const section = document.getElementById("headline");
  section.replaceChildren();
  const scenario = scenarioSelect.value;
  const labels = measureLabels[languageSelect.value] || measureLabels.en;
  for (const key of ["hazard_area", "buildings_inside", "buildings_near_30m", "road_km_hit",
    "bridges_hit_within_20m", "settlements_affected", "cut_off", "degraded", "not_assessable"]) {
    const entry = facts.headline.find(item => item.id === `${key}_${scenario}`);
    const row = document.createElement("div");
    row.className = "fact";
    row.textContent = `${labels[key]}: ${entry?.value ?? "not available"} ${entry?.unit || ""}`;
    section.append(row);
  }
  const reliability = headlineValue("optical_reliability") ?? "not available";
  const valid = headlineValue("optical_valid_fraction");
  const validText = typeof valid === "number" ? `${(valid * 100).toFixed(1)}%` : "not available";
  const banner = document.getElementById("reliability");
  banner.textContent = `${words.reliability}: ${reliability}; ${words.valid}: ${validText}. ${
    words.notGroundTruth
  }`;
  banner.classList.toggle("low", ["low", "unverified", "not available"].includes(reliability));
}

async function readJson(response) {
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || `Request failed (${response.status})`);
  return result;
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
  for (const name of result.layers) {
    const label = document.createElement("label");
    label.className = "layer-item";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = layerVisibility[name] !== false;
    checkbox.addEventListener("change", () => {
      layerVisibility[name] = checkbox.checked;
      if (checkbox.checked) map.addLayer(overlayLayers[name]);
      else map.removeLayer(overlayLayers[name]);
    });
    label.append(checkbox, ` ${name}`);
    layerContainer.append(label);
    if (!overlayLayers[name]) {
      const layerResponse = await fetch(`/api/layers/${encodeURIComponent(name)}.geojson`);
      addGeoJson(name, await readJson(layerResponse));
    } else if (checkbox.checked) {
      map.addLayer(overlayLayers[name]);
    }
  }
}

document.getElementById("ask-form").addEventListener("submit", async event => {
  event.preventDefault();
  const answer = document.getElementById("answer");
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
    answer.textContent = result.answer;
  } catch (error) {
    answer.textContent = String(error);
  }
});

scenarioSelect.addEventListener("change", () => {
  renderHeadline();
  loadLayers().catch(error => {
    document.getElementById("answer").textContent = String(error);
  });
});
languageSelect.addEventListener("change", translateInterface);
document.getElementById("opacity").addEventListener("input", event => {
  if (geoImage) geoImage.setOpacity(Number(event.target.value));
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
  translateInterface();
  const bounds = facts.dashboard?.post_event_image?.bounds;
  if (!bounds) throw new Error("Dashboard image bounds are missing from facts.json.");
  map.fitBounds(bounds);
  const image = facts.dashboard.post_event_image;
  geoImage = L.imageOverlay(image.url, image.bounds, { opacity: 0.75 }).addTo(map);
  await loadLayers();
  const reports = document.getElementById("reports");
  for (const language of ["en", "hi", "ne"]) {
    const link = document.createElement("a");
    link.href = `/report/situation-${language}.pdf`;
    link.textContent = ` ${language.toUpperCase()} PDF `;
    reports.append(link);
  }
  map.on("click", async event => {
    try {
      const response = await fetch("/api/flowpath", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lon: event.latlng.lng, lat: event.latlng.lat })
      });
      const result = await readJson(response);
      addGeoJson("flowpath", result.geojson);
      document.getElementById("answer").textContent =
        `Flowpath saved: ${result.facts.output}`;
    } catch (error) {
      document.getElementById("answer").textContent = String(error);
    }
  });
}

initialize().catch(error => {
  document.getElementById("headline").textContent = `Dashboard failed to initialize: ${error}`;
});
