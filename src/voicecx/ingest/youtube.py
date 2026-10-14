"""YouTube 영상 입력.

두 가지 경로가 있다.
  1) 영상 번들 JSON (load_bundle): 메타데이터 + 자막 문장 + 댓글을 미리 모아 둔 파일. 테스트·재현용
  2) URL (fetch_bundle): yt-dlp로 메타데이터와 **사람이 단 자막**을 받고,
     YOUTUBE_API_KEY가 있으면 YouTube Data API로 댓글을 받는다.
     자동 생성 자막은 정확도가 낮아서 쓰지 않는다. 사람 자막이 없으면 음성을 받아 STT를 돌린다(선택).

YouTube 이용약관을 지키고, 공개 영상만 비상업·연구 목적으로 분석한다.
원본 음성은 분석 후 지우고, 파생 데이터(지수, 문장 요약)만 보관한다.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from pydantic import BaseModel, Field

from voicecx.ingest.subtitles import cues_to_sentences, parse_subtitles

_VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")
COMMENTS_ENDPOINT = "https://www.googleapis.com/youtube/v3/commentThreads"


class VideoSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    stt_conf: float | None = None


class VideoComment(BaseModel):
    text: str
    likes: int = 0
    published_at: str | None = None


class VideoBundle(BaseModel):
    video_id: str
    meta: dict = Field(default_factory=dict)       # title, channel, description, tags, published_at, views, likes, brand
    segments: list[VideoSegment]
    comments: list[VideoComment] = Field(default_factory=list)
    audio: str | None = None
    transcript_source: str = "bundle"              # bundle | manual_subtitles | stt
    source_path: str | None = None

    def audio_path(self) -> Path | None:
        if not self.audio:
            return None
        path = Path(self.audio)
        if not path.is_absolute() and self.source_path:
            path = Path(self.source_path).parent / path
        return path


def video_id_from_url(url: str) -> str:
    m = _VIDEO_ID.search(url)
    if not m:
        raise ValueError(f"YouTube 영상 ID를 찾을 수 없습니다: {url}")
    return m.group(1)


def load_bundle(path: str | Path) -> VideoBundle:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("video_id", path.stem)
    data["source_path"] = str(path)
    bundle = VideoBundle.model_validate(data)
    bundle.segments = sorted((s for s in bundle.segments if s.text.strip()), key=lambda s: s.start)
    if not bundle.segments:
        raise ValueError(f"{path}: 자막·전사 문장이 없습니다")
    return bundle


def parse_comment_threads(payload: dict) -> list[VideoComment]:
    """YouTube Data API commentThreads 응답에서 최상위 댓글만 꺼낸다."""
    out = []
    for item in payload.get("items", []):
        snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
        if text.strip():
            out.append(
                VideoComment(text=text.strip(), likes=int(snippet.get("likeCount", 0)), published_at=snippet.get("publishedAt"))
            )
    return out


def fetch_comments(video_id: str, api_key: str | None = None, max_comments: int = 500) -> list[VideoComment]:
    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return []
    comments: list[VideoComment] = []
    page_token = None
    while len(comments) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "maxResults": 100, "order": "relevance", "textFormat": "plainText", "key": key}
        if page_token:
            params["pageToken"] = page_token
        with urllib.request.urlopen(f"{COMMENTS_ENDPOINT}?{urllib.parse.urlencode(params)}", timeout=30) as r:
            payload = json.loads(r.read().decode("utf-8"))
        comments.extend(parse_comment_threads(payload))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return comments[:max_comments]


def _manual_korean_subtitle(info: dict) -> str | None:
    subs = info.get("subtitles") or {}   # 사람이 단 자막만 (automatic_captions는 쓰지 않음)
    for lang in ("ko", "ko-KR", "ko-kr"):
        if lang in subs:
            return lang
    return None


def fetch_bundle(url: str, max_comments: int = 500, allow_stt: bool = False, stt_model: str = "medium") -> VideoBundle:
    """URL에서 영상 번들을 만든다. `pip install -e .[video]` 필요 (STT까지 하려면 [audio]도)."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - 선택 의존성
        raise RuntimeError("YouTube 수집에는 yt-dlp가 필요합니다: pip install -e .[video]") from exc

    video_id = video_id_from_url(url)
    with tempfile.TemporaryDirectory() as tmp:
        base = {"quiet": True, "skip_download": True, "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s")}
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
        meta = {
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "description": (info.get("description") or "")[:2000],
            "tags": info.get("tags") or [],
            "categories": info.get("categories") or [],
            "published_at": info.get("upload_date"),
            "duration": info.get("duration"),
            "views": info.get("view_count"),
            "likes": info.get("like_count"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

        lang = _manual_korean_subtitle(info)
        if lang:
            opts = {**base, "writesubtitles": True, "subtitleslangs": [lang], "subtitlesformat": "vtt"}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            vtt = next(Path(tmp).glob(f"{video_id}*.vtt"))
            sentences = cues_to_sentences(parse_subtitles(vtt))
            segments = [VideoSegment(start=s.start, end=s.end, text=s.text) for s in sentences]
            source = "manual_subtitles"
        elif allow_stt:
            from voicecx.audio.stt import assign_speakers, diarize, transcribe

            opts = {**base, "skip_download": False, "format": "bestaudio/best",
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
                    "postprocessor_args": ["-ar", "16000", "-ac", "1"]}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            wav = next(Path(tmp).glob(f"{video_id}*.wav"))
            turns = assign_speakers(transcribe(wav, model_size=stt_model), diarize(wav, num_speakers=None))
            segments = [VideoSegment(**t) for t in turns]
            source = "stt"
            # 원본 음성은 임시 폴더와 함께 지워진다 (음향 분석이 필요하면 번들로 따로 저장)
        else:
            raise RuntimeError("사람이 단 한국어 자막이 없습니다. --stt 옵션으로 음성 전사를 하거나 번들 JSON을 쓰세요.")

    return VideoBundle(
        video_id=video_id,
        meta=meta,
        segments=segments,
        comments=fetch_comments(video_id, max_comments=max_comments),
        transcript_source=source,
    )
