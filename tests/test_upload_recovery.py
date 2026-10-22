import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from samsone_clm import server


class StubRanker:
    def __init__(self,*args,**kwargs):pass
    def rank(self,audio,question,choices):
        return [{'choice':choices[0],'probability':.75},{'choice':choices[1],'probability':.25}]


async def request(app,method,path,body=b'',headers=()):
    """Exercise real ASGI routes without adding a dependency to the live environment."""
    messages=[]
    delivered=False
    async def receive():
        nonlocal delivered
        if not delivered:
            delivered=True
            return dict(type='http.request',body=body,more_body=False)
        await asyncio.Event().wait()
    async def send(message):
        messages.append(message)
    scope=dict(type='http',asgi={'version':'3.0','spec_version':'2.4'},http_version='1.1',
        method=method,scheme='http',path=path,raw_path=path.encode(),query_string=b'',
        root_path='',headers=list(headers),client=('127.0.0.1',1),server=('test',80))
    await app(scope,receive,send)
    start=next(m for m in messages if m['type']=='http.response.start')
    payload=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
    return start['status'],payload


def test_upload_playback_and_analysis_survive_web_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'Ranker',StubRanker)
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture')
    audio=io.BytesIO()
    sf.write(audio,np.zeros(16000,dtype=np.float32),16000,format='WAV',subtype='FLOAT')
    multipart=(b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="test.wav"\r\n'
               b'Content-Type: audio/wav\r\n\r\n'+audio.getvalue()+b'\r\n--boundary--\r\n')
    async def exercise():
        first=server.create_app(checkpoint,device='cpu')
        status,body=await request(first,'POST','/api/upload',multipart,
            [(b'content-type',b'multipart/form-data; boundary=boundary')])
        assert status==200
        record=json.loads(body)
        # Restore the user's legacy format without relying on a new sidecar.
        (tmp_path/'cache/web_uploads'/(record['audio_id']+'.upload.json')).unlink()
        second=server.create_app(checkpoint,device='cpu')
        status,body=await request(second,'GET',record['audio_url'],headers=[(b'range',b'bytes=0-31')])
        assert status==206
        payload=dict(audio_id=record['audio_id'],position_s=.5,question='what instrument is playing?',
                     choices=['piano','guitar'])
        status,body=await request(second,'POST','/api/analyze',json.dumps(payload).encode(),
                                 [(b'content-type',b'application/json')])
        assert status==200
        result=json.loads(body)
        assert result['window']['total_seconds']==10
        assert result['ranking'][0]['choice']=='piano'
        payload['audio_id']='../model'
        status,body=await request(second,'POST','/api/analyze',json.dumps(payload).encode(),
                                 [(b'content-type',b'application/json')])
        assert status==404
    asyncio.run(exercise())


def test_model_selection_uses_requested_weights_and_preserves_upload(tmp_path, monkeypatch):
    instances = []

    class SwitchingRanker:
        def __init__(self, checkpoint, device):
            self.checkpoint = Path(checkpoint)
            self.reloads = []
            self.encoder = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([])),
                                           provenance={"recipe": "fixture"})
            self.model = SimpleNamespace(scale=20., head_input_l2=True, spec={"adapter_kind": "linear"})
            instances.append(self)

        def reload(self, checkpoint):
            self.checkpoint = Path(checkpoint)
            self.reloads.append(self.checkpoint)

        def rank(self, audio, question, choices):
            assert sf.info(audio).frames == 160000
            first = 0 if self.checkpoint.parent.name == "samsone_clm_common" else 1
            return [{"choice": choices[first], "probability": .75},
                    {"choice": choices[1-first], "probability": .25}]

    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "Ranker", SwitchingRanker)
    paths = {}
    for model_id in ("common", "music"):
        path = tmp_path/"models"/f"samsone_clm_{model_id}"/"calibrated.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(model_id.encode())
        paths[model_id] = path
    app = server.create_app(paths["music"], device="cpu")
    audio_id = "a"*32
    uploads = tmp_path/"cache/web_uploads"
    sf.write(uploads/f"{audio_id}.wav", np.zeros(16000, dtype=np.float32), 16000)
    np.save(uploads/f"{audio_id}.npy", np.zeros(16000, dtype=np.float32))

    async def exercise():
        status, body = await request(app, "GET", "/api/models")
        assert status == 200
        catalog = json.loads(body)
        assert catalog["default_model"] == "music"
        assert {m["id"] for m in catalog["models"]} == {"common", "music"}
        payload = dict(audio_id=audio_id, position_s=0., question="what is playing?", choices=["piano", "guitar"])
        for model_id in ("common", "music", None, "common", "common"):
            if model_id is None:
                payload.pop("model_id", None)
            else:
                payload["model_id"] = model_id
            status, body = await request(app, "POST", "/api/analyze", json.dumps(payload).encode(),
                                        [(b"content-type", b"application/json")])
            assert status == 200
            result = json.loads(body)
            selected = model_id or "music"
            assert result["model_id"] == selected
            assert result["device"] == "cpu"
            assert result["checkpoint_sha256"] == server.sha256(paths[selected])
            assert result["ranking"][0]["choice"] == ("piano" if selected == "common" else "guitar")
            assert result["window"]["silence_before_s"] == 5.
            status, body = await request(app, "GET", "/api/health")
            assert status == 200
            assert json.loads(body)["model_id"] == selected  # Health must not reset the model.
        before = list(instances[0].reloads)
        payload["model_id"] = str(paths["music"])
        status, body = await request(app, "POST", "/api/analyze", json.dumps(payload).encode(),
                                    [(b"content-type", b"application/json")])
        assert status == 422
        assert instances[0].reloads == before
        assert len(instances) == 1
        assert instances[0].reloads == [paths["common"], paths["music"], paths["common"]]
        assert (uploads/f"{audio_id}.npy").is_file()

    asyncio.run(exercise())
