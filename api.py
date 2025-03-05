# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os
import re
from enum import Enum
from io import BytesIO
from typing import List
from scipy import signal

import soundfile as sf
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from typing_extensions import Annotated

from model import SenseVoiceSmall
from utils.pri import PriFile
from utils.vec import Wav2Vec2VAD
import logging

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/docs") == -1

# Filter out /endpoint
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(
    model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0")
)
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")],
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        ban_emo_unk=True,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


@app.post("/api/v1/vad")
async def get_vad_from_file(
    file: Annotated[bytes, File(description="wav or mp3 audios")],
):
    try:
        data, sr = sf.read(BytesIO(file))
        vec_extractor = Wav2Vec2VAD()
        # 计算新的采样点数
        number_of_samples = round(len(data) * float(16000) / sr)
        # 对音频数据进行重采样
        resampled_data = signal.resample(data, number_of_samples)
        vad_data = vec_extractor.process(resampled_data, raw=True)

    except Exception as e:
        raise HTTPException(status_code=418, detail=f"ERROR: {e}")

    return {
        "result": {
            "v": vad_data.Valence,
            "a": vad_data.Arousal,
            "d": vad_data.Dominance,
        }
    }


@app.post("/api/v1/pri")
async def get_pri_from_file(
    file: Annotated[bytes, File(description="wav or mp3 audios")],
):
    try:
        data, sr = sf.read(BytesIO(file))
        pri_data = PriFile((data, sr))
    except Exception as e:
        raise HTTPException(status_code=418, detail=f"ERROR: {e}")

    return {
        "result": {
            "mean_pri": pri_data.mean_measure(),
            "max_pri": pri_data.max_measure(),
            "rate": pri_data.rate,
            "loundness": pri_data.loundness,
            "pitches": pri_data.pitches,
        }
    }
