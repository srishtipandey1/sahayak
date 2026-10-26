"""Localized, number-free labels for reports and grounded answers."""

from typing import Final

MESSAGES: Final[dict[str, dict[str, str]]] = {
    "en": {
        "title": "Flood and debris situation report",
        "area": "Area and event date",
        "data_used": "Data used",
        "headline": "Mapped findings",
        "access": "Settlement access",
        "robust": "Robust cut-off settlements",
        "cut_off": "Cut off",
        "degraded": "Degraded access",
        "unaffected": "Unaffected",
        "not_assessable": "Not assessable",
        "limitations": "Reliability and limitations",
        "detections": "Mapped footprints are detections, not ground truth.",
        "prototype": "Educational prototype, not an operational tool.",
        "available": "not available",
        "answer_hit": "The mapped detection area is {area} ha within the focus extent {bbox}.",
        "answer_damage": (
            "The default scenario records {buildings} buildings inside, {near} buildings near, "
            "{roads} km of roads intersecting detections, and {bridges} bridges near detections."
        ),
        "answer_counts": (
            "In the default scenario: {cut_off} cut off, {degraded} degraded, "
            "{unaffected} unaffected, and {not_assessable} not assessable."
        ),
        "answer_villages": "Default-scenario cut-off settlements: {names}.",
        "answer_hospital": (
            "Nearest pre-event mapped hospital to {settlement}: {hospital}; "
            "route {distance} km, estimated {minutes} minutes."
        ),
        "answer_reliability": (
            "Optical reliability is {reliability}; valid-pixel fraction is {valid}."
        ),
        "answer_limitations": (
            "These are detection-derived estimates, not ground-truth damage or verified road "
            "status. OSM completeness, cloud and terrain effects limit interpretation."
        ),
        "answer_data": "Recorded data sources: {sources}. Acquisition dates: {dates}.",
        "unsupported": (
            "Supported questions: where did it hit; what was damaged; counts; which villages "
            "are cut off; nearest hospital to a settlement; reliability; limitations; "
            "and data used."
        ),
        "none": "not available",
        "road": "road",
        "settlement": "settlement",
        "hospital_missing": "not available",
        "scenario_strict": "Strict",
        "scenario_default": "Default",
        "scenario_lenient": "Lenient",
        "row_hazard": "Detection area (ha)",
        "row_buildings_inside": "Buildings inside",
        "row_buildings_near": "Buildings near",
        "row_road_km": "Road length intersecting detections (km)",
        "row_bridges": "Bridges near detections",
        "row_settlements": "Settlements affected",
        "row_cut_off": "Settlements cut off",
        "row_degraded": "Settlements with degraded access",
        "row_not_assessable": "Settlements not assessable",
        "row_reliability": "Optical reliability",
        "row_valid": "Valid optical pixels",
        "row_radar": "Radar classification finding",
        "row_dates": "Acquisition dates",
    },
    "hi": {
        "title": "बाढ़ और मलबा स्थिति रिपोर्ट",
        "area": "क्षेत्र और घटना की तारीख",
        "data_used": "उपयोग किए गए डेटा",
        "headline": "मानचित्र से प्राप्त निष्कर्ष",
        "access": "बस्तियों तक पहुँच",
        "robust": "सभी परिदृश्यों में कटे हुए बस्तियाँ",
        "cut_off": "संपर्क कटा",
        "degraded": "पहुँच प्रभावित",
        "unaffected": "अप्रभावित",
        "not_assessable": "आकलन संभव नहीं",
        "limitations": "विश्वसनीयता और सीमाएँ",
        "detections": "मानचित्रित क्षेत्र पहचान हैं, जमीनी सत्य नहीं।",
        "prototype": "शैक्षिक प्रोटोटाइप, परिचालन उपकरण नहीं।",
        "available": "उपलब्ध नहीं",
        "answer_hit": "मानचित्रित पहचान क्षेत्र {area} हेक्टेयर है; फोकस सीमा {bbox} है।",
        "answer_damage": (
            "डिफ़ॉल्ट परिदृश्य में पहचान क्षेत्र के भीतर {buildings} भवन, पास में {near} भवन, "
            "प्रतिच्छेदित सड़कें {roads} किमी, और पास में {bridges} पुल दर्ज हैं।"
        ),
        "answer_counts": (
            "डिफ़ॉल्ट परिदृश्य: {cut_off} संपर्क कटा, {degraded} प्रभावित, "
            "{unaffected} अप्रभावित, और {not_assessable} का आकलन संभव नहीं।"
        ),
        "answer_villages": "डिफ़ॉल्ट परिदृश्य में कटे हुए बस्तियाँ: {names}।",
        "answer_hospital": (
            "{settlement} के निकटतम पूर्व-घटना मानचित्रित अस्पताल: {hospital}; "
            "मार्ग {distance} किमी, अनुमानित समय {minutes} मिनट।"
        ),
        "answer_reliability": "ऑप्टिकल विश्वसनीयता {reliability} है; वैध पिक्सेल अंश {valid} है।",
        "answer_limitations": (
            "ये पहचान-आधारित अनुमान हैं, जमीनी सत्यापित क्षति या सड़क स्थिति नहीं। "
            "OSM की पूर्णता, बादल और भूभाग व्याख्या को सीमित करते हैं।"
        ),
        "answer_data": "दर्ज डेटा स्रोत: {sources}। अधिग्रहण तारीखें: {dates}।",
        "unsupported": (
            "समर्थित प्रश्न: कहाँ असर हुआ; क्या क्षतिग्रस्त हुआ; संख्या; कौन-सी बस्तियाँ "
            "कटीं; निकटतम अस्पताल; विश्वसनीयता; सीमाएँ; और उपयोग किया गया डेटा।"
        ),
        "none": "उपलब्ध नहीं",
        "road": "सड़क",
        "settlement": "बस्ती",
        "hospital_missing": "उपलब्ध नहीं",
        "scenario_strict": "कठोर",
        "scenario_default": "डिफ़ॉल्ट",
        "scenario_lenient": "विस्तृत",
        "row_hazard": "पहचान क्षेत्र (हेक्टेयर)",
        "row_buildings_inside": "भीतर के भवन",
        "row_buildings_near": "पास के भवन",
        "row_road_km": "पहचान से प्रतिच्छेद सड़क (किमी)",
        "row_bridges": "पहचान के पास पुल",
        "row_settlements": "प्रभावित बस्तियाँ",
        "row_cut_off": "संपर्क कटी बस्तियाँ",
        "row_degraded": "प्रभावित पहुँच वाली बस्तियाँ",
        "row_not_assessable": "आकलन संभव नहीं",
        "row_reliability": "ऑप्टिकल विश्वसनीयता",
        "row_valid": "वैध ऑप्टिकल पिक्सेल",
        "row_radar": "रडार वर्गीकरण निष्कर्ष",
        "row_dates": "अधिग्रहण तारीखें",
    },
    "ne": {
        "title": "बाढ तथा पहिरो अवस्था प्रतिवेदन",
        "area": "क्षेत्र र घटनाको मिति",
        "data_used": "प्रयोग गरिएका तथ्याङ्क",
        "headline": "नक्साबाट प्राप्त नतिजा",
        "access": "बस्तीको पहुँच",
        "robust": "सबै परिदृश्यमा पहुँचविहीन बस्ती",
        "cut_off": "सम्पर्कविच्छेद",
        "degraded": "पहुँच कमजोर",
        "unaffected": "अप्रभावित",
        "not_assessable": "मूल्याङ्कन गर्न नसकिने",
        "limitations": "विश्वसनीयता र सीमाहरू",
        "detections": "नक्साका क्षेत्रहरू पहिचान हुन्, स्थल-सत्य होइनन्।",
        "prototype": "शैक्षिक प्रोटोटाइप, सञ्चालनका लागि होइन।",
        "available": "उपलब्ध छैन",
        "answer_hit": "नक्सामा पहिचान क्षेत्र {area} हेक्टर छ; केन्द्रित क्षेत्रको सीमा {bbox} हो।",
        "answer_damage": (
            "डिफल्ट परिदृश्यमा पहिचान क्षेत्रभित्र {buildings} भवन, नजिक {near} भवन, "
            "प्रतिच्छेद भएका सडक {roads} किमी, र नजिक {bridges} पुल छन्।"
        ),
        "answer_counts": (
            "डिफल्ट परिदृश्य: {cut_off} सम्पर्कविच्छेद, {degraded} कमजोर पहुँच, "
            "{unaffected} अप्रभावित, र {not_assessable} मूल्याङ्कन गर्न नसकिने।"
        ),
        "answer_villages": "डिफल्ट परिदृश्यमा सम्पर्कविच्छेद भएका बस्ती: {names}।",
        "answer_hospital": (
            "{settlement} को नजिकको घटना-अघिको नक्साङ्कित अस्पताल: {hospital}; "
            "मार्ग {distance} किमी, अनुमानित {minutes} मिनेट।"
        ),
        "answer_reliability": "अप्टिकल विश्वसनीयता {reliability} छ; वैध पिक्सेल अंश {valid} छ।",
        "answer_limitations": (
            "यी पहिचानमा आधारित अनुमान हुन्, स्थल-सत्यापित क्षति वा सडकको अवस्था होइनन्। "
            "OSM को अपूर्णता, बादल र भूभागले व्याख्यामा सीमा लगाउँछन्।"
        ),
        "answer_data": "दर्ज तथ्याङ्कका स्रोतहरू: {sources}। प्राप्त मितिहरू: {dates}।",
        "unsupported": (
            "समर्थित प्रश्न: कहाँ असर भयो; के क्षति भयो; संख्या; कुन बस्ती सम्पर्कविच्छेद "
            "भयो; नजिकको अस्पताल; विश्वसनीयता; सीमाहरू; र प्रयोग गरिएको तथ्याङ्क।"
        ),
        "none": "उपलब्ध छैन",
        "road": "सडक",
        "settlement": "बस्ती",
        "hospital_missing": "उपलब्ध छैन",
        "scenario_strict": "कडा",
        "scenario_default": "डिफल्ट",
        "scenario_lenient": "विस्तृत",
        "row_hazard": "पहिचान क्षेत्र (हेक्टर)",
        "row_buildings_inside": "भित्रका भवन",
        "row_buildings_near": "नजिकका भवन",
        "row_road_km": "पहिचानसँग प्रतिच्छेद सडक (किमी)",
        "row_bridges": "पहिचान नजिकका पुल",
        "row_settlements": "प्रभावित बस्ती",
        "row_cut_off": "सम्पर्कविच्छेद बस्ती",
        "row_degraded": "कमजोर पहुँच भएका बस्ती",
        "row_not_assessable": "मूल्याङ्कन गर्न नसकिने बस्ती",
        "row_reliability": "अप्टिकल विश्वसनीयता",
        "row_valid": "वैध अप्टिकल पिक्सेल",
        "row_radar": "रडार वर्गीकरणको नतिजा",
        "row_dates": "प्राप्त मितिहरू",
    },
}


def message(language: str, key: str, **values: object) -> str:
    """Format a localized message with named placeholders."""
    try:
        template = MESSAGES[language][key]
    except KeyError as error:
        raise ValueError(f"Unsupported report language or message: {language}.{key}") from error
    return template.format(**values)
