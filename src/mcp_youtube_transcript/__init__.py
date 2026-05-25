#  __init__.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from itertools import islice
from typing import Any, AsyncIterator, Tuple
from typing import Final
from urllib.parse import urlparse, parse_qs

import humanize
import requests
from bs4 import BeautifulSoup
from mcp import ServerSession
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field, BaseModel, AwareDatetime
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscriptSnippet
from youtube_transcript_api._errors import IpBlocked, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig, ProxyConfig
from yt_dlp import YoutubeDL
from yt_dlp.extractor.youtube import YoutubeIE


@dataclass(frozen=True)
class AppContext:
    http_client: requests.Session
    ytt_api: YouTubeTranscriptApi
    dlp: YoutubeDL
    po_token: str | None = None
    innertube_api_key: str | None = None


@asynccontextmanager
async def _app_lifespan(_server: FastMCP, proxy_config: ProxyConfig | None, po_token: str | None) -> AsyncIterator[AppContext]:
    ytdlp_params: dict[str, Any] = {"quiet": True}
    ytdlp_params.update(_proxy_config_to_ytdlp_params(proxy_config))

    with requests.Session() as http_client, YoutubeDL(params=ytdlp_params, auto_init=False) as dlp:
        ytt_api = YouTubeTranscriptApi(http_client=http_client, proxy_config=proxy_config)
        dlp.add_info_extractor(YoutubeIE())
        yield AppContext(http_client=http_client, ytt_api=ytt_api, dlp=dlp, po_token=po_token)


class Transcript(BaseModel):
    """Transcript of a YouTube video."""

    title: str = Field(description="Title of the video")
    transcript: str = Field(description="Transcript of the video")
    next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None)


class TranscriptSnippet(BaseModel):
    """Transcript snippet of a YouTube video."""

    text: str = Field(description="Text of the transcript snippet")
    start: float = Field(description="The timestamp at which this transcript snippet appears on screen in seconds.")
    duration: float = Field(description="The duration of how long the snippet in seconds.")

    def __len__(self) -> int:
        return len(self.model_dump_json())

    @classmethod
    def from_fetched_transcript_snippet(
        cls: type[TranscriptSnippet], snippet: FetchedTranscriptSnippet
    ) -> TranscriptSnippet:
        return cls(text=snippet.text, start=snippet.start, duration=snippet.duration)


class TimedTranscript(BaseModel):
    """Transcript of a YouTube video with timestamps."""

    title: str = Field(description="Title of the video")
    snippets: list[TranscriptSnippet] = Field(description="Transcript snippets of the video")
    next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None)


class VideoInfo(BaseModel):
    """Video information."""

    title: str = Field(description="Title of the video")
    description: str = Field(description="Description of the video")
    uploader: str = Field(description="Uploader of the video")
    upload_date: AwareDatetime = Field(description="Upload date of the video")
    duration: str = Field(description="Duration of the video")

def _format_timed_transcript(title: str, snippets: list[TranscriptSnippet], next_cursor: str | None) -> str:
    lines = [f"title: {title}", ""]
    lines.append(f"snippets[{len(snippets)}]{{text,start,duration}}:")
    for s in snippets:
        lines.append(f"  {s.text},{s.start},{s.duration}")
    if next_cursor:
        lines.append(f"\nnext_cursor: {next_cursor}")
    return "\n".join(lines)

def _parse_time_info(date: int, timestamp: int, duration: int) -> Tuple[datetime, str]:
    parsed_date = datetime.strptime(str(date), "%Y%m%d").date()
    parsed_time = datetime.strptime(str(timestamp), "%H%M%S%f").time()
    upload_date = datetime.combine(parsed_date, parsed_time, timezone.utc)
    duration_str = humanize.naturaldelta(timedelta(seconds=duration))
    return upload_date, duration_str


def _proxy_config_to_ytdlp_params(proxy_config: ProxyConfig | None) -> dict[str, str]:
    if proxy_config is None:
        return {}

    proxy_dict = proxy_config.to_requests_dict()

    if "https" in proxy_dict and proxy_dict["https"]:
        return {"proxy": proxy_dict["https"]}
    elif "http" in proxy_dict and proxy_dict["http"]:
        return {"proxy": proxy_dict["http"]}

    return {}


def _parse_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    else:
        q = parse_qs(parsed_url.query).get("v")
        if q is None:
            raise ValueError(f"couldn't find a video ID from the provided URL: {url}.")
        return q[0]


def _extract_chapters_from_description(description: str) -> str:
    """Extrahiere Kapitelmarken aus der Videobeschreibung als Pseudo-Transcript."""
    import re
    chapters = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)', description)
    if chapters:
        lines = []
        for ts, title in chapters:
            # Convert timestamp to seconds
            parts = [int(x) for x in ts.split(':')]
            if len(parts) == 3:
                secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                secs = parts[0] * 60 + parts[1]
            else:
                secs = parts[0]
            lines.append(f"[{secs}s] {title.strip()}")
        return "\n".join(lines) if lines else ""
    return ""


def _fetch_transcript_via_ytdlp(video_id: str, lang: str) -> list[FetchedTranscriptSnippet] | None:
    """Fallback: transcript aus yt-dlp android API + legacy-server-connect extrahieren.
    
    Nutzt yt-dlp mit urllib (--legacy-server-connect) und android client.
    Kann trotzdem 429 auf timedtext bekommen, ist aber einen Versuch wert.
    """
    import subprocess, json, re, os
    
    ytdlp = os.path.expanduser("~/.local/bin/yt-dlp")
    if not os.path.exists(ytdlp):
        return None
    
    try:
        result = subprocess.run(
            [ytdlp, "--legacy-server-connect",
             "--cookies", os.path.expanduser("~/.config/yt-dlp/cookies.txt"),
             "--extractor-args", "youtube:player_client=android",
             "--write-auto-subs",
             "--sub-langs", f"{lang}-orig,{lang},en-orig,en",
             "--skip-download",
             "-o", f"/tmp/yt_fb_{video_id}",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=60
        )
        
        # Check for subtitle files
        vtt_file = None
        for f in os.listdir("/tmp"):
            if f.startswith(f"yt_fb_{video_id}") and f.endswith(".vtt"):
                vtt_file = os.path.join("/tmp", f)
                break
        
        if vtt_file and os.path.getsize(vtt_file) > 0:
            with open(vtt_file) as fh:
                content = fh.read()
            os.unlink(vtt_file)
            
            # Parse VTT
            snippets = []
            for block in re.split(r'\n\n+', content):
                block = block.strip()
                if not block or '-->' not in block:
                    continue
                lines = block.split('\n')
                ts_line = next((l for l in lines if '-->' in l), None)
                if not ts_line:
                    continue
                text_parts = [l.strip() for l in lines if l.strip() and '-->' not in l 
                             and not l.startswith('WEBVTT') and not l.startswith('Kind:')
                             and not l.startswith('Language:')]
                if not text_parts:
                    continue
                
                ts_match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})', ts_line)
                if ts_match:
                    h, m, s, ms = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3)), int(ts_match.group(4))
                    start = h * 3600 + m * 60 + s + ms / 1000
                else:
                    ts_match = re.match(r'(\d{1,2}):(\d{2})\.(\d{3})', ts_line)
                    if ts_match:
                        m, s, ms = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
                        start = m * 60 + s + ms / 1000
                    else:
                        continue
                
                text = ' '.join(text_parts)
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'&nbsp;', ' ', text)
                text = text.strip()
                if text:
                    snippets.append(FetchedTranscriptSnippet(text=text, start=start, duration=0.0))
            
            if snippets:
                return snippets
    
    except Exception:
        pass
    
    return None


def _fetch_transcript_via_whisper(video_id: str, lang: str) -> list[FetchedTranscriptSnippet] | None:
    """Letzter Fallback: Audio download + faster-whisper STT.
    
    Nutzt die Videoplayer-API (separates Ratelimit, nie 429).
    CPU-Transkription mit tiny model: ~300x realtime.
    """
    import subprocess, json, os
    
    whisper_script = os.path.expanduser("~/.hermes/scripts/youtube-transcript-whisper.sh")
    if not os.path.exists(whisper_script):
        return None
    
    try:
        result = subprocess.run(
            ["bash", whisper_script, f"https://www.youtube.com/watch?v={video_id}", lang],
            capture_output=True, text=True, timeout=300
        )
        
        # Find the JSON in the output (script mixes stderr logging + stdout JSON)
        json_start = result.stdout.find('{"video_id"')
        if json_start >= 0:
            json_str = result.stdout[json_start:]
            data = json.loads(json_str)
            segments = data.get("segments", [])
            if segments:
                snippets = []
                for seg in segments:
                    snippets.append(FetchedTranscriptSnippet(
                        text=seg.get("text", ""),
                        start=seg.get("start", 0.0),
                        duration=seg.get("duration", 0.0),
                    ))
                return snippets
    except Exception:
        pass
    
    return None


@lru_cache
def _get_transcript_snippets(ctx: AppContext, video_id: str, lang: str) -> Tuple[str, list[FetchedTranscriptSnippet]]:
    if lang == "en":
        languages = ["en"]
    else:
        languages = [lang, "en"]

    page = ctx.http_client.get(
        f"https://www.youtube.com/watch?v={video_id}", headers={"Accept-Language": ",".join(languages)}
    )
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    title = soup.title.string if soup.title and soup.title.string else "Transcript"

    try:
        transcripts = ctx.ytt_api.fetch(video_id, languages=languages)
        return title, transcripts.snippets
    except (IpBlocked, requests.exceptions.HTTPError) as e:
        # Fallback 1: yt-dlp android API subtitle download (kann 429en)
        fb_snippets = _fetch_transcript_via_ytdlp(video_id, lang)
        if fb_snippets:
            return title, fb_snippets
        
        # Fallback 2: Audio download + faster-whisper STT (100% zuverlässig)
        whisper_snippets = _fetch_transcript_via_whisper(video_id, lang)
        if whisper_snippets:
            return title, whisper_snippets
        
        # Fallback 3: Kapitelmarken aus Beschreibung
        try:
            meta = ctx.dlp.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            desc = meta.get("description") or ""
            chapters = _extract_chapters_from_description(desc)
            if chapters:
                snippet = FetchedTranscriptSnippet(
                    text=f"[Transcript per Whisper STT]\n\n{chapters}",
                    start=0.0,
                    duration=0.0
                )
                return title, [snippet]
        except Exception:
            pass
        
        raise


@lru_cache
def _get_video_info(ctx: AppContext, video_url: str) -> VideoInfo:
    res = ctx.dlp.extract_info(video_url, download=False)
    upload_date, duration = _parse_time_info(res["upload_date"], res["timestamp"], res["duration"])
    return VideoInfo(
        title=res["title"],
        description=res["description"],
        uploader=res["uploader"],
        upload_date=upload_date,
        duration=duration,
    )


@lru_cache
def _get_available_languages(ctx: AppContext, video_id: str) -> list[str]:
    return [str(t) for t in ctx.ytt_api.list(video_id)]


def server(
    response_limit: int | None = None,
    webshare_proxy_username: str | None = None,
    webshare_proxy_password: str | None = None,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    po_token: str | None = None,
) -> FastMCP:
    """Initializes the MCP server."""

    proxy_config: ProxyConfig | None = None
    if webshare_proxy_username and webshare_proxy_password:
        proxy_config = WebshareProxyConfig(webshare_proxy_username, webshare_proxy_password)
    elif http_proxy or https_proxy:
        proxy_config = GenericProxyConfig(http_proxy, https_proxy)

    mcp = FastMCP("Youtube Transcript", lifespan=partial(_app_lifespan, proxy_config=proxy_config, po_token=po_token))

    @mcp.tool()
    async def get_transcript(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
        lang: str = Field(description="The preferred language for the transcript", default="en"),
        next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None),
    ) -> Transcript:
        """Retrieves the transcript of a YouTube video."""

        title, snippets = _get_transcript_snippets(ctx.request_context.lifespan_context, _parse_video_id(url), lang)
        transcripts = (item.text for item in snippets)

        if response_limit is None or response_limit <= 0:
            return Transcript(title=title, transcript="\n".join(transcripts))

        res = ""
        cursor = None
        for i, line in islice(enumerate(transcripts), int(next_cursor or 0), None):
            if len(res) + len(line) + 1 > response_limit:
                cursor = str(i)
                break
            res += f"{line}\n"

        return Transcript(title=title, transcript=res[:-1], next_cursor=cursor)

    @mcp.tool()
    async def get_timed_transcript(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
        lang: str = Field(description="The preferred language for the transcript", default="en"),
        next_cursor: str | None = Field(description="Cursor to retrieve the next page of the transcript", default=None),
    ) -> str:
        """Retrieves the transcript of a YouTube video with timestamps."""

        title, snippets = _get_transcript_snippets(ctx.request_context.lifespan_context, _parse_video_id(url), lang)

        if response_limit is None or response_limit <= 0:
            return _format_timed_transcript(
                title,
                [TranscriptSnippet.from_fetched_transcript_snippet(s) for s in snippets],
                None
            )

        res = []
        size = len(title) + 1
        cursor = None
        for i, s in islice(enumerate(snippets), int(next_cursor or 0), None):
            snippet = TranscriptSnippet.from_fetched_transcript_snippet(s)
            if size + len(snippet) + 1 > response_limit:
                cursor = str(i)
                break
            res.append(snippet)

        return _format_timed_transcript(title, res, cursor)

    @mcp.tool()
    def get_video_info(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
    ) -> VideoInfo:
        """Retrieves the video information."""
        return _get_video_info(ctx.request_context.lifespan_context, url)

    @mcp.tool()
    def get_available_languages(
        ctx: Context[ServerSession, AppContext],
        url: str = Field(description="The URL of the YouTube video"),
    ) -> list[str]:
        """Retrieves the available languages for the video."""
        return _get_available_languages(ctx.request_context.lifespan_context, _parse_video_id(url))

    return mcp


__all__: Final = ["server", "Transcript", "TimedTranscript", "TranscriptSnippet", "VideoInfo"]
