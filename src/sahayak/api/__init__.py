"""FastAPI endpoints for facts, mapped layers, questions, and terrain flowpaths."""

from __future__ import annotations

import json
from typing import Any

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id


def create_app(
    *,
    run_id: str,
    settings: Settings | None = None,
    sample: bool = False,
):
    """Build an API app for a mapped run or the self-contained sample bundle."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel

    validate_run_id(run_id)
    active_settings = settings or Settings()
    if sample:
        root = active_settings.resolved_data_dir.parent / "sample_data" / run_id
    else:
        root = active_settings.resolved_runs_dir / run_id
    dashboard_dir = root / "dashboard"
    app = FastAPI(
        title="Sahayak",
        description="Educational prototype, not an operational tool.",
        docs_url=None,
        redoc_url=None,
    )

    class AskRequest(BaseModel):
        question: str
        lang: str = "en"

    class FlowpathRequest(BaseModel):
        lon: float
        lat: float
        distance_km: float = 30.0

    @app.get("/")
    def index() -> FileResponse:
        page = dashboard_dir / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Dashboard has not been built.")
        return FileResponse(page)

    @app.get("/api/facts")
    def facts_endpoint() -> JSONResponse:
        path = root / "facts.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="facts.json is unavailable.")
        facts = json.loads(path.read_text(encoding="utf-8"))
        return JSONResponse(facts)

    @app.get("/api/headline")
    def headline_endpoint() -> JSONResponse:
        path = root / "headline.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="headline.json is unavailable.")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/layers")
    def layer_list(scenario: str | None = None) -> dict[str, list[str]]:
        if scenario is not None and scenario not in {"strict", "default", "lenient"}:
            raise HTTPException(
                status_code=400,
                detail="Scenario must be strict, default, or lenient.",
            )
        folder = dashboard_dir / "layers"
        layers = sorted(path.stem for path in folder.glob("*.geojson")) if folder.is_dir() else []
        if scenario is not None:
            layers = [
                name
                for name in layers
                if not name.startswith(("hazard_", "settlements_"))
                or name.endswith(f"_{scenario}")
            ]
        return {"layers": layers}

    @app.get("/api/layers/{layer_name}.geojson")
    def layer_endpoint(layer_name: str) -> FileResponse:
        if not layer_name.replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid layer name.")
        path = dashboard_dir / "layers" / f"{layer_name}.geojson"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Layer is unavailable.")
        return FileResponse(path, media_type="application/geo+json")

    @app.post("/api/ask")
    def ask_endpoint(request: AskRequest) -> dict[str, Any]:
        headline_path = root / "headline.json"
        if not headline_path.is_file():
            raise HTTPException(status_code=404, detail="headline.json is unavailable.")
        from sahayak.copilot import answer_from_headline

        try:
            headline = json.loads(headline_path.read_text(encoding="utf-8"))
            return answer_from_headline(
                headline=headline,
                language=request.lang,
                question=request.question,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/flowpath")
    def flowpath_endpoint(request: FlowpathRequest) -> dict[str, Any]:
        if sample:
            raise HTTPException(
                status_code=400,
                detail="Flowpath generation is unavailable in the derived sample bundle.",
            )
        from sahayak.flowpath import run_flowpath

        try:
            result = run_flowpath(
                run_id=run_id,
                longitude=request.lon,
                latitude=request.lat,
                distance_km=request.distance_km,
                settings=active_settings,
            )
            geojson = json.loads(result["path"].read_text(encoding="utf-8"))
            return {"facts": result["facts"], "geojson": geojson}
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/report/{filename}")
    def report_file(filename: str) -> FileResponse:
        if filename not in {
            "situation-en.pdf",
            "situation-hi.pdf",
            "situation-ne.pdf",
        }:
            raise HTTPException(status_code=404, detail="Report is unavailable.")
        path = root / "report" / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Report is unavailable.")
        return FileResponse(path, media_type="application/pdf")

    from fastapi.staticfiles import StaticFiles

    if dashboard_dir.is_dir():
        app.mount("/", StaticFiles(directory=dashboard_dir), name="dashboard-static")
    return app
