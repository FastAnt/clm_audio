from __future__ import annotations
import argparse
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .common import ROOT, sha256, write_json
from .audio import decode_audio
from .infer import Ranker


def audio_window(waveform, position_s, sample_rate=16000):
    position_s = min(max(float(position_s), 0.), len(waveform)/sample_rate)
    center = round(position_s*sample_rate)
    start, end = center-5*sample_rate, center+5*sample_rate
    source_start, source_end = max(0, start), min(len(waveform), end)
    window = np.zeros(10*sample_rate, dtype=np.float32)
    offset = source_start-start
    window[offset:offset+source_end-source_start] = waveform[source_start:source_end]
    details = dict(position_s=position_s, requested_start_s=start/sample_rate, requested_end_s=end/sample_rate,
                   source_start_s=source_start/sample_rate, source_end_s=source_end/sample_rate,
                   silence_before_s=max(0, -start)/sample_rate, silence_after_s=max(0, end-len(waveform))/sample_rate,
                   total_seconds=10.)
    return window, details


class AnalyzeRequest(BaseModel):
    audio_id: str
    position_s: float
    question: str
    choices: list[str]
    model_id: str | None = None


def create_app(checkpoint, device="cuda"):
    checkpoint = Path(checkpoint).absolute()
    models = {}
    for model_id, name, directory in (
        ("common", "General · samsone_clm_common", "samsone_clm_common"),
        ("music", "Music · samsone_clm_music", "samsone_clm_music"),
    ):
        path = ROOT/"models"/directory/"calibrated.pt"
        if path.is_file():
            models[model_id] = dict(id=model_id, name=name, path=path.absolute())
    default_model = next((key for key, model in models.items()
                          if model["path"].resolve() == checkpoint.resolve()), None)
    if default_model is None:
        default_model = "configured"
        models[default_model] = dict(id=default_model, name=checkpoint.parent.name, path=checkpoint)
    active_model = default_model
    ranker = Ranker(checkpoint, device=device)
    checkpoint_sha = sha256(checkpoint)
    checkpoint_stamp = (str(checkpoint.resolve()), checkpoint.stat().st_mtime_ns)
    checksums = {checkpoint_stamp: checkpoint_sha}
    lock = threading.Lock()
    uploads = ROOT/"cache/web_uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    files = {}
    app = FastAPI(title="Samsone–CLM Audio System 1")

    def model_entry(model_id):
        if model_id not in models:
            raise HTTPException(422, "Unknown model_id; choose a model from /api/models")
        return models[model_id]

    def refresh(model_id):
        nonlocal checkpoint, checkpoint_sha, checkpoint_stamp, active_model
        path = model_entry(model_id)["path"]
        stamp = (str(path.resolve()), path.stat().st_mtime_ns)
        if stamp != checkpoint_stamp or model_id != active_model:
            if stamp not in checksums:
                checksums[stamp] = sha256(path)
            ranker.reload(path)
            checkpoint_sha = checksums[stamp]
            checkpoint_stamp = stamp
            checkpoint = path
            active_model = model_id

    def stored_upload(audio_id):
        if not re.fullmatch(r"[0-9a-f]{32}",audio_id):
            raise HTTPException(404,"Audio upload not found")
        if audio_id in files:
            return files[audio_id]
        array_path=uploads/(audio_id+".npy")
        # Restore legacy uploads as well; the original and decoded files already persist.
        originals=[p for p in uploads.glob(audio_id+".*") if p.is_file() and p.suffix not in (".npy",".json")]
        if not array_path.is_file() or len(originals)!=1:
            raise HTTPException(404,"Audio upload not found")
        waveform=np.load(array_path,mmap_mode="r",allow_pickle=False)
        if waveform.ndim!=1 or waveform.dtype!=np.float32:
            raise HTTPException(404,"Audio upload not found")
        files[audio_id]=dict(original=originals[0],decoded=array_path,duration_s=len(waveform)/16000)
        return files[audio_id]

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).parent/"static/index.html")

    @app.get("/training")
    def training_page():
        return FileResponse(Path(__file__).parent/"static/training.html")

    @app.get("/api/training")
    def training_status():
        from .monitor import training_snapshot
        return training_snapshot()

    @app.get("/api/training/plot.png")
    def training_plot(stage: int | None = Query(None, ge=1, le=6), run_id: str | None = Query(None)):
        from .monitor import render_png, resolve_experiment
        try:
            run = resolve_experiment(run_id) if run_id else None
        except ValueError as error:
            raise HTTPException(404,str(error)) from error
        path = render_png(stage,run)
        return FileResponse(path, media_type="image/png", filename=path.name,
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/training/comparison.png")
    def training_comparison():
        path=ROOT/'runs/v3/training_comparison.png'
        if not path.exists():
            raise HTTPException(404,'Comparison plot has not been written yet')
        return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store'})

    @app.get("/api/health")
    def health():
        with lock:
            refresh(active_model)
            return dict(status="ready", checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                        model_id=active_model, model_name=models[active_model]["name"], device=str(device),
                        samsone_frozen=all(not p.requires_grad for p in ranker.encoder.model.parameters()), score="cosine",
                        score_scale=float(ranker.model.scale), head_input_l2=ranker.model.head_input_l2,
                        encoder_recipe=ranker.encoder.provenance["recipe"],adapter_kind=ranker.model.spec["adapter_kind"])

    @app.get("/api/models")
    def available_models():
        with lock:
            entries = []
            for model in models.values():
                path = model["path"]
                stamp = (str(path.resolve()), path.stat().st_mtime_ns)
                if stamp not in checksums:
                    checksums[stamp] = sha256(path)
                entries.append(dict(id=model["id"], name=model["name"], checkpoint=str(path.resolve()),
                                    checkpoint_sha256=checksums[stamp]))
            return dict(models=entries, default_model=default_model, active_model=active_model, device=str(device))

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        audio_id = uuid.uuid4().hex
        path = uploads/(audio_id + Path(file.filename or "audio.wav").suffix)
        with path.open("wb") as target:
            while chunk := await file.read(1024*1024):
                target.write(chunk)
        try:
            waveform = decode_audio(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "Cannot decode this audio file") from error
        array_path = uploads/(audio_id+".npy")
        np.save(array_path, waveform)
        files[audio_id] = dict(original=path, decoded=array_path, duration_s=len(waveform)/16000)
        write_json(uploads/(audio_id+".upload.json"),dict(original=path.name,decoded=array_path.name,duration_s=len(waveform)/16000))
        return dict(audio_id=audio_id, audio_url=f"/api/audio/{audio_id}", duration_s=len(waveform)/16000)

    @app.get("/api/audio/{audio_id}")
    def audio_file(audio_id: str):
        record=stored_upload(audio_id)
        return FileResponse(record["original"])

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest):
        selected_model = request.model_id if request.model_id is not None else default_model
        model_entry(selected_model)
        record=stored_upload(request.audio_id)
        if not np.isfinite(request.position_s):
            raise HTTPException(422, "Playback position must be finite")
        if not request.question.strip() or len(request.choices)<2 or any(not c.strip() for c in request.choices):
            raise HTTPException(422, "Enter a question and at least two nonempty choices")
        waveform = np.load(record["decoded"], mmap_mode="r",allow_pickle=False)
        window, details = audio_window(waveform, request.position_s)
        path = uploads/(uuid.uuid4().hex+"-window.wav")
        sf.write(path, window, 16000, subtype="FLOAT")
        started = time.perf_counter()
        try:
            with lock:
                refresh(selected_model)
                ranking = ranker.rank(path, request.question, request.choices)
                analyzed_checkpoint, analyzed_sha = str(checkpoint.resolve()), checkpoint_sha
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            path.unlink(missing_ok=True)
        return dict(ranking=ranking, window=details, latency_ms=(time.perf_counter()-started)*1000,
                    checkpoint=analyzed_checkpoint, checkpoint_sha256=analyzed_sha,
                    model_id=selected_model, model_name=models[selected_model]["name"], device=str(device))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    import uvicorn
    torch.set_num_threads(4)
    uvicorn.run(create_app(args.checkpoint, args.device), host=args.host, port=args.port)

if __name__ == "__main__":
    main()
