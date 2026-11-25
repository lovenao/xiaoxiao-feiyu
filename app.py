#!/usr/bin/env python3
"""小小肥鱼 - 一个单文件、本地运行的 OpenAI 兼容多轮问答 Agent。

特性：
  * 零第三方依赖（仅 Python 标准库）
  * 多供应商配置、自动拉取模型、选择思考强度
  * SSE 流式回答、历史对话持久化、上下文缓存友好的提示词结构

运行：
    python app.py

数据文件（均位于本文件同目录）：
    config.json          供应商与 API Key（仅保存在本机）
    conversations.json   历史对话
"""

from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
import ssl
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request
from urllib.parse import parse_qs, unquote, urlparse

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
CONVERSATIONS_PATH = APP_DIR / "conversations.json"
CONVERSATION_INDEX_PATH = APP_DIR / "conversation_index.json"
CONVERSATION_CACHE_DIR = APP_DIR / "conversation_cache"
VERSION = "1.2.0"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 24 * 1024 * 1024
MAX_IMAGE_DATA_URL = 6_000_000
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 100
FILE_LOCK = threading.RLock()

# 固定、无时间戳、无随机内容的系统提示词，是提高前缀缓存命中率的关键。
SYSTEM_PROMPT = (
    "你是一个严谨、直接、可靠的 AI 助手。回答优先保证准确、清晰、可执行；"
    "不确定时明确说明，不编造事实。默认使用简体中文回答，除非用户使用其他语言。"
    "代码必须使用带语言标注的 Markdown 代码块。"
)

TEACHING_PROMPT = (
    "你是一位耐心、严谨的教学辅助老师，负责帮助用户真正学会知识，而不是只给答案。"
    "默认使用简体中文。回答时先判断用户当前卡在哪里，再用循序渐进的方式解释。"
    "优先给出直觉、关键概念和为什么，再给严谨定义、推导与例题。"
    "数学和科学公式使用 LaTeX；代码使用带语言标注的 Markdown 代码块。"
    "涉及作业或题目时，先引导思路，再给出完整步骤与最终答案，并指出常见错误。"
    "不确定时明确说明，不编造事实，不跳过关键推理。"
)

DEFAULT_PROMPTS = [
    {
        "id": "p_ai_assistant",
        "name": "AI 助手",
        "content": SYSTEM_PROMPT,
        "builtin": True,
    },
    {
        "id": "p_teaching",
        "name": "教学辅助",
        "content": TEACHING_PROMPT,
        "builtin": True,
    },
    {
        "id": "p_blank",
        "name": "空白",
        "content": "",
        "builtin": True,
    },
]
DEFAULT_PROMPT_ID = DEFAULT_PROMPTS[0]["id"]
MAX_PROMPT_CHARS = 30_000

# 供应商没有声明思考强度时，允许透传这些通用值；不支持时 stream_chat 会自动降级。
STANDARD_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra", "none"}

# 上下文用量：供应商没声明窗口时按 1M 处理；超过该值或触发上游超长错误则提示压缩。
CONTEXT_LIMIT_DEFAULT = 1_000_000
CONTEXT_WARN_RATIO = 0.85
IMAGE_TOKEN_ESTIMATE = 900
MAX_DIGEST_CHARS = 24_000
MIN_COMPRESS_TURNS = 3

COMPRESS_SYSTEM_PROMPT = (
    "你是一个对话压缩器。把给定的对话历史压缩成简洁但信息完整的摘要，供后续继续对话使用。"
    "必须保留：用户的目标与偏好、已经确认的事实、结论与关键数据、重要公式或代码要点、"
    "尚未完成的事项与待办。删除寒暄、重复表述和无关内容。"
    "不要编造，不要输出与摘要无关的话，不要输出 Markdown 代码块围栏。"
    "用简体中文分条输出。"
)

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
WEB_SEARCH_MAX_RESULTS = 50
WEB_SEARCH_DEFAULT_RESULTS = 10
WEB_RESULT_MARKDOWN_CHARS = 12_000
WEB_TOTAL_CONTEXT_CHARS = 36_000
TOOL_MAX_ROUNDS = 3
DEFAULT_TOOL_INSTRUCTIONS = (
    "你可以使用工具获取外部信息。只有当问题需要实时、网络上才有的信息，或用户明确要求搜索时，才调用 web_search。"
    "普通问答、写作、代码和数学推导不要搜索。调用工具前不要让用户确认，直接使用；需要搜索时必须在本轮立即调用 web_search，不要只回复“我来查一下”然后结束。不要编造工具结果。"
    "回答时如果使用了搜索结果，必须在相关陈述后附 Markdown 链接来源。"
    "如果提供了 memory_search/memory_read_conversation/memory_save 工具：需要历史细节时先搜索索引，再读取指定对话；"
    "只有明确、长期有价值且有原文依据的信息才调用 memory_save。不要保存密钥、密码、令牌，也不要把一次性请求写成长期偏好。"
)


def default_settings() -> dict:
    return {
        "memory": {"enabled": False, "provider_id": "", "model_id": "", "review_prompt": ""},
        "web": {"enabled": False, "api_key": "", "max_results": WEB_SEARCH_DEFAULT_RESULTS},
    }

CONTEXT_ERROR_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "max context",
    "context window",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "input is too long",
    "reduce the length",
    "exceeds the maximum",
    "maximum number of tokens",
    "上下文",
    "超出",
    "过长",
)
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
MIN_TIMEOUT = 3
MAX_TIMEOUT = 600
MAX_RETRIES = 10


class ApiError(Exception):
    """可以直接返回给前端的业务错误。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class UpstreamError(Exception):
    """模型服务返回的错误。"""

    def __init__(self, message: str, status: int | None = None, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.detail = detail


class ClientDisconnected(Exception):
    pass


class StreamBusinessError(UpstreamError):
    """上游在流内返回的业务错误，不应重试。"""

    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


def _read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(f"无法读取数据文件 {path.name}：{exc}", 500) from exc


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _normalize_prompts(raw) -> list[dict]:
    prompts: list[dict] = []
    seen_ids: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            content = str(item.get("content") or "").strip()
            if not name:
                continue
            prompt_id = str(item.get("id") or "").strip()
            if not prompt_id or prompt_id in seen_ids:
                prompt_id = new_id("p_")
            seen_ids.add(prompt_id)
            prompts.append(
                {
                    "id": prompt_id,
                    "name": name[:80],
                    "content": content[:MAX_PROMPT_CHARS],
                    "builtin": bool(item.get("builtin")),
                }
            )
    if not prompts:
        prompts = [dict(item) for item in DEFAULT_PROMPTS]
    if not any(str(item.get("id")) == "p_blank" for item in prompts):
        prompts.append({
            "id": "p_blank",
            "name": "空白",
            "content": "",
            "builtin": True,
        })
    return prompts


def load_config() -> dict:
    data = _read_json(CONFIG_PATH, {"providers": []})
    if not isinstance(data, dict):
        data = {"providers": []}
    if not isinstance(data.get("providers"), list):
        data["providers"] = []
    defaults = default_settings()
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    merged = {}
    for section, values in defaults.items():
        current = settings.get(section) if isinstance(settings.get(section), dict) else {}
        merged[section] = {**values, **current}
    data["settings"] = merged
    data["prompts"] = _normalize_prompts(data.get("prompts"))
    prompt_ids = {str(item.get("id")) for item in data["prompts"]}
    active = str(data.get("active_prompt_id") or "")
    if active not in prompt_ids:
        active = str(data["prompts"][0].get("id") or DEFAULT_PROMPT_ID)
    data["active_prompt_id"] = active
    return data


def save_config(config: dict) -> None:
    defaults = default_settings()
    settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
    config["settings"] = {}
    for section, values in defaults.items():
        current = settings.get(section) if isinstance(settings.get(section), dict) else {}
        config["settings"][section] = {**values, **current}
    config["prompts"] = _normalize_prompts(config.get("prompts"))
    prompt_ids = {str(item.get("id")) for item in config["prompts"]}
    if str(config.get("active_prompt_id") or "") not in prompt_ids:
        config["active_prompt_id"] = str(config["prompts"][0].get("id") or DEFAULT_PROMPT_ID)
    _atomic_write_json(CONFIG_PATH, config)


def prompt_by_id(config: dict, prompt_id: str) -> dict | None:
    for prompt in config.get("prompts") or []:
        if str(prompt.get("id")) == prompt_id:
            return prompt
    return None


def load_conversations() -> list[dict]:
    data = _read_json(CONVERSATIONS_PATH, [])
    if isinstance(data, dict):
        data = data.get("conversations", [])
    return data if isinstance(data, list) else []


def save_conversations(conversations: list[dict]) -> None:
    _atomic_write_json(CONVERSATIONS_PATH, conversations)
    _refresh_conversation_sidecars(conversations)


def normalize_base_url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if not re.match(r"^https?://[^\s]+$", value, re.IGNORECASE):
        raise ApiError("Base URL 必须以 http:// 或 https:// 开头")
    return value


def api_endpoint(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(suffix):
        return base
    return base + suffix


def _decode_body(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def _extract_error_message(raw: bytes) -> str:
    text = _decode_body(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("detail")
                if msg:
                    return str(msg)
            if isinstance(err, str) and err:
                return err
            for key in ("message", "detail", "error_description"):
                if obj.get(key):
                    return str(obj[key])
    except json.JSONDecodeError:
        pass
    if text.startswith("<"):
        return "服务返回了 HTML 页面，请检查 Base URL 是否指向 OpenAI 兼容 API"
    return text[:1200] or "未知错误"


def friendly_http_error(status: int, detail: str) -> str:
    if status == 401:
        return "认证失败（401）：请检查 API Key"
    if status == 403:
        return "没有访问权限（403）：请检查 API Key 或账户权限"
    if status == 404:
        return "接口不存在（404）：请检查 Base URL，通常应填写到 /v1 或服务根地址"
    if status == 429:
        return "请求过于频繁或额度不足（429）：请稍后再试"
    if status >= 500:
        return f"模型服务暂时不可用（{status}）：{detail}"
    return f"模型服务返回错误（{status}）：{detail}"


def _request_json(url: str, api_key: str, timeout: int = 30) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": f"XiaoXiaoFatFish/{VERSION}",
    }
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            raw = resp.read()
    except error.HTTPError as exc:
        detail = _extract_error_message(exc.read())
        raise UpstreamError(friendly_http_error(exc.code, detail), exc.code, detail) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpstreamError(f"无法连接模型服务：{reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError("连接模型服务超时，请检查网络或 Base URL") from exc
    try:
        data = json.loads(_decode_body(raw))
    except json.JSONDecodeError as exc:
        raise UpstreamError("模型服务返回的不是有效 JSON，请检查 Base URL") from exc
    if not isinstance(data, dict):
        raise UpstreamError("模型服务返回了无法识别的数据")
    if data.get("error"):
        raise UpstreamError(_extract_error_message(raw))
    return data


def _parse_effort_values(raw) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        parts = [part.strip() for part in re.split(r"\s*[-/,|]\s*", text) if part.strip()]
        if len(parts) > 1 and all(part.lower() in STANDARD_EFFORTS for part in parts):
            return parts
        return [text]
    if isinstance(raw, list):
        return [
            str(item).strip()
            for item in raw
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
    return []


def _effort_levels(model: dict) -> tuple[list[str], str]:
    if not isinstance(model, dict):
        return [], ""
    infos: list[dict] = []
    for key in ("effort", "reasoning", "thinking"):
        value = model.get(key)
        if isinstance(value, dict):
            infos.append(value)
    capabilities = model.get("capabilities")
    if isinstance(capabilities, dict):
        for key in ("reasoning", "thinking", "effort"):
            value = capabilities.get(key)
            if isinstance(value, dict):
                infos.append(value)

    levels: list[str] = []
    default = ""
    for info in infos:
        raw = (
            info.get("supported_levels")
            or info.get("supported_efforts")
            or info.get("reasoning_efforts")
            or info.get("efforts")
            or info.get("levels")
        )
        levels = _parse_effort_values(raw)
        default = str(
            info.get("default_level")
            or info.get("default_effort")
            or info.get("default")
            or default
            or ""
        )
        if levels:
            break

    if not levels:
        for key in ("reasoning_efforts", "supported_efforts", "efforts", "effort_levels", "effort"):
            levels = _parse_effort_values(model.get(key))
            if levels:
                break

    def normalize(value: str) -> str:
        text = str(value).strip()
        return text.lower() if text.lower() in STANDARD_EFFORTS else text

    normalized: list[str] = []
    for level in levels:
        level = normalize(level)
        if level and level not in normalized:
            normalized.append(level)
    if default:
        default = normalize(default)
    if default not in normalized:
        default = normalized[0] if normalized else ""
    return normalized, default


def normalize_models(data: dict) -> list[dict]:
    raw_models = data.get("data")
    if raw_models is None:
        raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raise UpstreamError("模型接口没有返回 data 或 models 列表")
    models: list[dict] = []
    for item in raw_models:
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or item.get("name") or "").strip()
        if not model_id:
            continue
        display = str(item.get("name") or item.get("display_name") or model_id)
        levels, default = _effort_levels(item)
        models.append(
            {
                "id": model_id,
                "name": display,
                "efforts": levels,
                "default_effort": default,
                "context_window": item.get("context_window") or item.get("context_length"),
                "max_output_tokens": item.get("max_output_tokens"),
            }
        )
    models.sort(key=lambda m: (m["id"].lower(), m["name"].lower()))
    if not models:
        raise UpstreamError("模型列表为空，请确认该 Base URL 支持 /models 接口")
    return models


def fetch_provider_models(
    base_url: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> list[dict]:
    url = api_endpoint(base_url, "/models")
    last_error: UpstreamError | None = None
    for attempt in range(retries + 1):
        try:
            return normalize_models(_request_json(url, api_key, timeout=timeout))
        except UpstreamError as exc:
            last_error = exc
            if attempt < retries and _retryable_error(exc):
                time.sleep(_retry_delay(attempt + 1))
                continue
            raise
    raise last_error or UpstreamError("无法获取模型列表")


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def provider_timeout(provider: dict) -> int:
    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, _coerce_int(provider.get("timeout"), DEFAULT_TIMEOUT)))


def provider_retries(provider: dict) -> int:
    return max(0, min(MAX_RETRIES, _coerce_int(provider.get("retries"), DEFAULT_RETRIES)))


def _retryable_error(exc: UpstreamError) -> bool:
    if exc.status is None:
        return True
    return exc.status in (408, 409, 425, 429, 500, 502, 503, 504)


def _retry_delay(attempt: int) -> float:
    return float(min(2 ** (attempt - 1), 5))


def public_provider(provider: dict) -> dict:
    key = str(provider.get("api_key") or "")
    return {
        "id": provider.get("id"),
        "name": provider.get("name"),
        "base_url": provider.get("base_url"),
        "key_hint": ("••••" + key[-4:]) if key else "",
        "models": provider.get("models") or [],
        "timeout": provider_timeout(provider),
        "retries": provider_retries(provider),
        "created_at": provider.get("created_at"),
        "updated_at": provider.get("updated_at"),
    }


def provider_by_id(config: dict, provider_id: str) -> dict:
    for provider in config.get("providers", []):
        if provider.get("id") == provider_id:
            return provider
    raise ApiError("供应商不存在，可能已被删除", 404)


def find_model(provider: dict, model_id: str) -> dict | None:
    for model in provider.get("models") or []:
        if model.get("id") == model_id:
            return model
    return None


def make_title(text: str, limit: int = 30) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    clean = clean.strip("#*`>- ")
    if not clean:
        return "新对话"
    return clean[:limit] + ("…" if len(clean) > limit else "")


def normalize_images(raw) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ApiError("图片参数格式错误")
    images: list[str] = []
    for item in raw:
        value = item.get("data_url") if isinstance(item, dict) else item
        if not isinstance(value, str) or not value.startswith("data:image/"):
            raise ApiError("图片格式无效，只支持 data:image/... 格式")
        if len(value) > MAX_IMAGE_DATA_URL:
            raise ApiError("单张图片过大，请压缩后再发送")
        images.append(value)
    return images


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _normalize_extracted_text(text: str) -> tuple[str, bool]:
    cleaned = re.sub(r"\n{3,}", "\n\n", str(text)).strip()
    return cleaned, False


def _extract_docx_text(raw: bytes) -> str:
    try:
        from docx import Document  # type: ignore

        doc = Document(io.BytesIO(raw))
        parts = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                line = " | ".join(cell for cell in cells if cell)
                if line:
                    parts.append(line)
        return "\n".join(parts)
    except ImportError:
        pass
    except Exception:
        # 损坏或特殊 DOCX 时退回标准库解析正文 XML。
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ApiError("DOCX 文件无法读取，可能不是有效的 Word 文档") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ApiError("DOCX 正文解析失败") from exc
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    parts: list[str] = []
    for paragraph in root.iter(ns + "p"):
        texts = [node.text or "" for node in paragraph.iter(ns + "t")]
        line = "".join(texts).strip()
        if line:
            parts.append(line)
    return "\n".join(parts)


def _render_pdf_page(page, fitz_module) -> dict | None:
    try:
        rect = page.rect
        max_edge = max(float(rect.width), float(rect.height), 1.0)
        scale = min(1.8, 1600.0 / max_edge)
        if scale <= 0:
            return None
        pix = page.get_pixmap(matrix=fitz_module.Matrix(scale, scale), alpha=False)
        jpeg = pix.tobytes("jpeg", jpg_quality=80)
        return {
            "data_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            "width": pix.width,
            "height": pix.height,
        }
    except Exception:
        return None


def _extract_pdf_image(doc, xref: int, fitz_module) -> dict | None:
    try:
        pix = fitz_module.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:
            pix = fitz_module.Pixmap(fitz_module.csRGB, pix)
        if pix.width > 1600 or pix.height > 1600:
            factor = max(1, (max(pix.width, pix.height) + 1599) // 1600)
            pix.shrink(factor)
        quality = 80
        jpeg = pix.tobytes("jpeg", jpg_quality=quality)
        while len(jpeg) > 2_000_000 and quality > 50:
            quality -= 10
            jpeg = pix.tobytes("jpeg", jpg_quality=quality)
        if len(jpeg) > 3_000_000:
            pix.shrink(2)
            jpeg = pix.tobytes("jpeg", jpg_quality=70)
        if len(jpeg) > 4_000_000:
            return None
        return {
            "data_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            "width": pix.width,
            "height": pix.height,
        }
    except Exception:
        return None


def _page_image_infos(page) -> list[dict]:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return []
    return [info for info in infos if isinstance(info, dict)]


def _is_significant_page_image(info: dict, page_area: float) -> bool:
    try:
        width = float(info.get("width") or 0)
        height = float(info.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width * height >= 250_000:
        return True
    bbox = info.get("bbox") or ()
    if len(bbox) == 4 and page_area > 0:
        try:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
            if area / page_area >= 0.02:
                return True
        except (TypeError, ValueError):
            pass
    return width >= 240 and height >= 160


def _extract_pdf_content(raw: bytes) -> tuple[str, int, list[dict], bool, bool]:
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ApiError("服务器未安装 PyMuPDF，无法解析 PDF（可运行 pip install pymupdf）") from exc
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ApiError("PDF 文件无法打开，可能已损坏或加密") from exc
    page_count = int(getattr(doc, "page_count", 0) or 0)
    text_parts: list[str] = []
    page_images: list[dict] = []
    seen_xrefs: set[int] = set()
    total_text = 0
    images_truncated = False
    pages_truncated = False
    try:
        for index, page in enumerate(doc):
            if index >= MAX_PDF_PAGES:
                pages_truncated = True
                break
            page_text = (page.get_text("text") or "").strip()
            if page_text:
                text_parts.append(f"--- 第 {index + 1} 页 ---\n{page_text}")
                total_text += len(page_text)

            page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
            infos = _page_image_infos(page)
            significant = [
                info for info in infos if _is_significant_page_image(info, page_area)
            ]
            added_on_page = False
            for info in significant:
                try:
                    xref = int(info.get("xref") or 0)
                except (TypeError, ValueError):
                    xref = 0
                if not xref or xref in seen_xrefs:
                    continue
                image = _extract_pdf_image(doc, xref, fitz)
                if image:
                    image["page"] = index + 1
                    image["has_text"] = bool(page_text)
                    page_images.append(image)
                    seen_xrefs.add(xref)
                    added_on_page = True

            # 没有文字、也没有可用嵌入图片的页面，或者图片提取失败但页面明显有图时，
            # 才退回整页渲染。
            needs_render = (not page_text and not added_on_page) or (
                bool(page_text) and bool(significant) and not added_on_page
            )
            if needs_render:
                rendered = _render_pdf_page(page, fitz)
                if rendered:
                    rendered["page"] = index + 1
                    rendered["has_text"] = bool(page_text)
                    rendered["fallback_render"] = True
                    page_images.append(rendered)
                    added_on_page = True

            if not page_text and added_on_page:
                text_parts.append(f"--- 第 {index + 1} 页（图片内容已附）---")
    finally:
        doc.close()

    if page_images:
        page_numbers: list[str] = []
        for image in page_images:
            page_number = str(image.get("page") or "?")
            if page_number not in page_numbers:
                page_numbers.append(page_number)
        text_parts.insert(0, f"[PDF 已附页面图片：第 {', '.join(page_numbers)} 页]")
    if pages_truncated:
        text_parts.append(f"[PDF 共 {page_count} 页，仅处理前 {MAX_PDF_PAGES} 页]")
    return "\n\n".join(text_parts), page_count, page_images, images_truncated, pages_truncated


def extract_file_text(name: str, raw: bytes) -> dict:
    safe_name = Path(str(name or "file")).name[:160]
    suffix = Path(safe_name).suffix.lower()
    if not raw:
        raise ApiError("文件内容为空")
    pages = 0
    page_images: list[dict] = []
    images_truncated = False
    pages_truncated = False
    if suffix == ".pdf":
        text, pages, page_images, images_truncated, pages_truncated = _extract_pdf_content(raw)
        if not text.strip() and not page_images:
            raise ApiError("PDF 没有可提取的文字或图片")
        if not text.strip():
            text = "[PDF 为扫描版，页面已作为图片附上]"
    elif suffix == ".docx":
        text = _extract_docx_text(raw)
    elif suffix == ".doc":
        raise ApiError("旧版 .doc 暂不支持，请另存为 .docx 或 PDF")
    elif suffix in (".txt", ".md", ".markdown", ".csv", ".json", ".log", ".xml"):
        text = _decode_text_bytes(raw)
    elif suffix in (".html", ".htm"):
        text = re.sub(r"<[^>]+>", " ", _decode_text_bytes(raw))
    else:
        text = _decode_text_bytes(raw)
        if "\x00" in text[:4000]:
            raise ApiError("不支持这种二进制文件，请上传 PDF、DOCX 或文本文件")
    text, truncated = _normalize_extracted_text(text)
    if not text.strip() and not page_images:
        raise ApiError("文件没有可提取的文字或图片")
    result = {
        "name": safe_name,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
        "pages": pages or None,
    }
    if page_images:
        result["page_images"] = page_images
    if images_truncated:
        result["images_truncated"] = True
    if pages_truncated:
        result["pages_truncated"] = True
    return result


def normalize_files(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ApiError("文件参数格式错误")
    files: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ApiError("文件参数格式错误")
        name = Path(str(item.get("name") or "文件")).name[:160]
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(f"附件 {name} 没有可用的文字内容")
        text = text.strip()
        entry = {"name": name, "text": text, "chars": len(text)}
        if item.get("truncated"):
            entry["truncated"] = True
        if item.get("pages"):
            try:
                entry["pages"] = int(item.get("pages"))
            except (TypeError, ValueError):
                pass
        if item.get("pages_truncated"):
            entry["pages_truncated"] = True
        raw_page_images = item.get("page_images")
        page_images: list[dict] = []
        if isinstance(raw_page_images, list):
            for image in raw_page_images:
                if isinstance(image, str):
                    data_url = image
                    page_number = 0
                elif isinstance(image, dict):
                    data_url = image.get("data_url")
                    try:
                        page_number = int(image.get("page") or 0)
                    except (TypeError, ValueError):
                        page_number = 0
                else:
                    continue
                if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
                    continue
                if len(data_url) > MAX_IMAGE_DATA_URL:
                    entry["images_truncated"] = True
                    continue
                page_image = {"data_url": data_url}
                if page_number:
                    page_image["page"] = page_number
                for key in ("width", "height"):
                    try:
                        if isinstance(image, dict) and image.get(key):
                            page_image[key] = int(image.get(key))
                    except (TypeError, ValueError):
                        pass
                page_images.append(page_image)
        if page_images:
            entry["page_images"] = page_images
        if item.get("images_truncated"):
            entry["images_truncated"] = True
        files.append(entry)
    return files


def file_text_block(file: dict) -> str:
    name = str(file.get("name") or "附件")
    text = str(file.get("text") or "").strip()
    if not text:
        return ""
    return f"【附件：{name}】\n{text}\n【附件结束】"


def split_turns(messages: list[dict]) -> list[tuple[dict, list[dict]]]:
    turns: list[tuple[dict, list[dict]]] = []
    current_user = None
    assistants: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            if current_user is not None:
                turns.append((current_user, assistants))
            current_user = message
            assistants = []
        elif role == "assistant" and current_user is not None:
            assistants.append(message)
    if current_user is not None:
        turns.append((current_user, assistants))
    for user, items in turns:
        try:
            user["active_variant"] = int(user.get("active_variant") or 0)
        except (TypeError, ValueError):
            user["active_variant"] = 0
        for index, assistant in enumerate(items):
            try:
                assistant["variant"] = int(assistant.get("variant") or index)
            except (TypeError, ValueError):
                assistant["variant"] = index
        if user["active_variant"] < 0:
            user["active_variant"] = 0
    return turns


def clean_generated_title(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            text = str(obj.get("title") or obj.get("conversation_title") or "").strip()
    except (json.JSONDecodeError, TypeError):
        pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    title = lines[0]
    title = re.sub(r"^(标题|对话标题)\s*[:：]\s*", "", title)
    title = title.strip(" #*`'\"“”‘’《》")
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"[。.!！?？;；:：]+$", "", title).strip()
    return make_title(title, 30)


def generate_first_turn_title(config: dict, conv: dict) -> str:
    settings = (config.get("settings") or {}).get("memory") or {}
    provider_id = str(settings.get("provider_id") or "").strip()
    model_id = str(settings.get("model_id") or "").strip()
    if not provider_id or not model_id:
        return ""
    turns = split_turns(conv.get("messages") or [])
    if len(turns) != 1:
        return ""
    user, assistants = turns[0]
    assistant, _index = active_assistant(user, assistants)
    if assistant is None or not str(assistant.get("content") or "").strip():
        return ""
    try:
        provider = provider_by_id(config, provider_id)
        if not find_model(provider, model_id):
            return ""
    except ApiError:
        return ""
    user_text = str(user.get("content") or "").strip()
    extras = []
    if user.get("images"):
        extras.append("[用户附带了图片]")
    for file in user.get("files") or []:
        extras.append("[附件：" + str(file.get("name") or "未命名")[:80] + "]")
    if extras:
        user_text = (user_text + "\n" + "\n".join(extras)).strip()
    answer_text = str(assistant.get("content") or "").strip()
    prompt = f"【用户第一轮提问】\n{user_text[:6000]}\n\n【助手第一轮回答】\n{answer_text[:10000]}"
    messages = [
        {"role": "system", "content": AUTO_TITLE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    chunks = []
    try:
        for event_type, data in stream_chat(provider, model_id, messages, ""):
            if event_type == "content":
                chunks.append(str(data.get("text") or ""))
    except Exception:
        return ""
    return clean_generated_title("".join(chunks))


def active_assistant(user: dict, assistants: list[dict]) -> tuple[dict | None, int]:
    try:
        index = int(user.get("active_variant") or 0)
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= len(assistants):
        return None, 0
    return assistants[index], index


def estimate_tokens_for_text(text: str) -> int:
    """粗略估算 token 数：中日韩字符按 0.75，其他字符按 1/3.2。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        code = ord(ch)
        if (
            0x2E80 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
        else:
            other += 1
    return int(cjk * 0.75 + other / 3.2) + 1


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += 4
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens_for_text(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    total += IMAGE_TOKEN_ESTIMATE
                else:
                    total += estimate_tokens_for_text(str(part.get("text") or ""))
    return total


def conversation_context(conv: dict) -> dict:
    """估算下一次请求会发送的上下文规模，用于界面提示。"""
    try:
        messages = build_upstream_messages(conv)
        tokens = estimate_messages_tokens(messages)
    except Exception:  # noqa: BLE001 - 估算失败不应影响主流程
        tokens = 0
    try:
        limit = int(conv.get("context_limit") or CONTEXT_LIMIT_DEFAULT)
    except (TypeError, ValueError):
        limit = CONTEXT_LIMIT_DEFAULT
    if limit <= 0:
        limit = CONTEXT_LIMIT_DEFAULT
    digest = conv.get("digest") or {}
    return {
        "tokens": tokens,
        "limit": limit,
        "ratio": round(tokens / limit, 4) if limit else 0,
        "warn": tokens >= limit * CONTEXT_WARN_RATIO,
        "digest_turns": int(digest.get("turns") or 0),
        "estimated": True,
    }


def is_context_error(message: str, detail: str = "") -> bool:
    text = f"{message} {detail}".lower()
    return any(marker in text for marker in CONTEXT_ERROR_MARKERS)


def conversation_summary(conv: dict, with_context: bool = False) -> dict:
    messages = conv.get("messages") or []
    turns = split_turns(messages)
    last_text = ""
    if turns:
        user, assistants = turns[-1]
        assistant, _index = active_assistant(user, assistants)
        if assistant and assistant.get("content"):
            last_text = assistant.get("content", "")
        else:
            last_text = user.get("content") or ("[图片]" if user.get("images") else ("[附件]" if user.get("files") else ""))
    preview = re.sub(r"\s+", " ", last_text).strip()[:90]
    return {
        "id": conv.get("id"),
        "title": conv.get("title") or "新对话",
        "provider_id": conv.get("provider_id"),
        "provider_name": conv.get("provider_name"),
        "model": conv.get("model"),
        "model_name": conv.get("model_name"),
        "reasoning_effort": conv.get("reasoning_effort"),
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at"),
        "message_count": len(messages),
        "preview": preview,
        "archived": bool(conv.get("archived")),
        "prompt_id": conv.get("prompt_id"),
        "prompt_name": conv.get("prompt_name"),
        **({"context": conversation_context(conv)} if with_context else {}),
    }


def public_conversation(conv: dict) -> dict:
    result = conversation_summary(conv, with_context=True)
    result["messages"] = conv.get("messages") or []
    return result


def _conversation_sidecar_paths() -> tuple[Path, Path]:
    parent = CONVERSATIONS_PATH.parent.resolve()
    if parent == APP_DIR.resolve():
        return CONVERSATION_INDEX_PATH, CONVERSATION_CACHE_DIR
    return parent / "conversation_index.json", parent / "conversation_cache"


def _conversation_cache_path(conversation_id: str) -> Path:
    _index_path, cache_dir = _conversation_sidecar_paths()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(conversation_id or "conversation"))
    return cache_dir / (safe + ".json")


def _conversation_search_text(conv: dict) -> str:
    parts = []
    for message in conv.get("messages") or []:
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(content)
        for file in message.get("files") or []:
            if not isinstance(file, dict):
                continue
            name = str(file.get("name") or "").strip()
            text = str(file.get("text") or "").strip()
            if name:
                parts.append(name)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _conversation_index_record(conv: dict) -> dict:
    summary = conversation_summary(conv, with_context=False)
    summary["search_text"] = _conversation_search_text(conv)
    summary["cache_file"] = _conversation_cache_path(str(conv.get("id") or "")).name
    summary["fingerprint"] = hashlib.sha256(
        json.dumps(conv, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return summary


def _conversations_source_stamp() -> dict:
    try:
        stat = CONVERSATIONS_PATH.stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    except OSError:
        return {}


def _refresh_conversation_sidecars(conversations: list[dict]) -> None:
    index_path, cache_dir = _conversation_sidecar_paths()
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_data = _read_json(index_path, {})
    old_items = old_data.get("items") if isinstance(old_data, dict) and isinstance(old_data.get("items"), list) else []
    old_by_id = {str(item.get("id") or ""): item for item in old_items if isinstance(item, dict)}
    records = []
    keep_files = set()
    for conv in conversations:
        record = _conversation_index_record(conv)
        conv_id = str(record.get("id") or "")
        cache_name = str(record.get("cache_file") or "")
        cache_path = cache_dir / cache_name
        old = old_by_id.get(conv_id) or {}
        changed = (
            not cache_path.exists()
            or old.get("fingerprint") != record.get("fingerprint")
        )
        if changed:
            _atomic_write_json(cache_path, conv)
        keep_files.add(cache_name)
        records.append(record)
    for cache_path in cache_dir.glob("*.json"):
        if cache_path.name not in keep_files:
            try:
                cache_path.unlink()
            except OSError:
                pass
    _atomic_write_json(index_path, {
        "version": 1,
        "source": _conversations_source_stamp(),
        "items": records,
    })


def load_conversation_index() -> list[dict]:
    index_path, _cache_dir = _conversation_sidecar_paths()
    data = _read_json(index_path, {})
    source = data.get("source") if isinstance(data, dict) else {}
    if isinstance(data, dict) and isinstance(data.get("items"), list) and source == _conversations_source_stamp():
        return data["items"]
    conversations = load_conversations()
    _refresh_conversation_sidecars(conversations)
    refreshed = _read_json(index_path, {})
    return refreshed.get("items", []) if isinstance(refreshed, dict) else []


def load_conversation_by_id(conversation_id: str) -> dict | None:
    wanted = str(conversation_id or "")
    index_items = load_conversation_index()
    record = next((item for item in index_items if str(item.get("id") or "") == wanted), None)
    if record:
        _index_path, cache_dir = _conversation_sidecar_paths()
        cache_path = cache_dir / str(record.get("cache_file") or "")
        if cache_path.exists():
            data = _read_json(cache_path, None)
            if isinstance(data, dict):
                return data
    conversations = load_conversations()
    _refresh_conversation_sidecars(conversations)
    conv = next((item for item in conversations if item.get("id") == wanted), None)
    return conv


def search_conversation_index(query: str, limit: int = 80) -> list[dict]:
    needle = str(query or "").strip()
    if not needle:
        return []
    needle_lower = needle.casefold()
    results = []
    for item in load_conversation_index():
        title = str(item.get("title") or "对话")
        match_type = ""
        source = ""
        if needle_lower in title.casefold():
            match_type = "标题"
            source = title
        else:
            search_text = str(item.get("search_text") or "")
            if needle_lower in search_text.casefold():
                match_type = "原文"
                source = search_text
        if not match_type:
            continue
        result = {key: value for key, value in item.items() if key not in ("search_text", "cache_file", "fingerprint")}
        result["match_type"] = match_type
        result["snippet"] = _search_snippet(source, needle)
        results.append(result)
    results.sort(key=lambda item: (1 if item.get("match_type") == "标题" else 0, item.get("updated_at") or ""), reverse=True)
    return results[:max(1, min(200, int(limit or 80)))]


def _search_snippet(text: str, query: str, radius: int = 80) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    position = clean.casefold().find(query.casefold())
    if position < 0:
        return clean[:radius * 2] + ("…" if len(clean) > radius * 2 else "")
    start = max(0, position - radius)
    end = min(len(clean), position + len(query) + radius)
    return ("…" if start else "") + clean[start:end] + ("…" if end < len(clean) else "")


def search_conversations_text(conversations: list[dict], query: str, limit: int = 80) -> list[dict]:
    needle = str(query or "").strip()
    if not needle:
        return []
    needle_lower = needle.casefold()
    results = []
    for conv in conversations:
        title = str(conv.get("title") or "对话")
        best = None
        if needle_lower in title.casefold():
            best = {"match_type": "标题", "snippet": _search_snippet(title, needle)}
        for message in conv.get("messages") or []:
            content = str(message.get("content") or "")
            candidates = [content]
            for file in message.get("files") or []:
                if isinstance(file, dict):
                    candidates.append(str(file.get("name") or ""))
                    candidates.append(str(file.get("text") or ""))
            for candidate in candidates:
                if needle_lower in candidate.casefold():
                    best = {
                        "match_type": "原文",
                        "snippet": _search_snippet(candidate, needle),
                    }
                    break
            if best and best.get("match_type") == "原文":
                break
        if best is None:
            continue
        summary = conversation_summary(conv, with_context=True)
        summary.update(best)
        results.append(summary)
    results.sort(key=lambda item: (0 if item.get("match_type") == "标题" else 1, item.get("updated_at") or ""), reverse=False)
    results.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return results[:max(1, min(200, int(limit or 80)))]


def normalize_usage(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    total = raw.get("total_tokens") or (prompt + completion)
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cached = raw.get("prompt_cache_hit_tokens")
    if cached is None:
        cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else 0
    miss = raw.get("prompt_cache_miss_tokens")
    if miss is None and prompt:
        try:
            miss = max(0, int(prompt) - int(cached or 0))
        except (TypeError, ValueError):
            miss = 0
    reasoning = completion_details.get("reasoning_tokens", 0) if isinstance(completion_details, dict) else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached or 0,
        "cache_miss_tokens": miss or 0,
        "reasoning_tokens": reasoning or 0,
    }


def build_environment_note() -> str:
    local = datetime.now().astimezone()
    offset = local.strftime("%z")
    tz = f"UTC{offset[:3]}:{offset[3:]}" if offset else (local.tzname() or "local")
    return f"当前日期：{local:%Y-%m-%d}\n当前时间：{local:%H:%M}（{tz}）"


MEMORY_PATH = APP_DIR / "memory.json"
PROFILE_MAX_CHARS = 2500
MEMORY_INDEX_SUMMARY_MAX_CHARS = 300
MEMORY_NOTE_MAX_CHARS = 600
MEMORY_NOTE_MAX_ITEMS = 20
MEMORY_SAVE_MAX_CHARS = MEMORY_NOTE_MAX_CHARS * MEMORY_NOTE_MAX_ITEMS
MEMORY_SNAPSHOT_MAX_CONVERSATIONS = 40
MEMORY_SNAPSHOT_EDIT_MAX_CHARS = 50000
MEMORY_SCAN_MINUTES = 30
MEMORY_REVIEW_MAX_CONVERSATIONS = 20
MEMORY_REVIEW_CONTEXT_CHARS = 48000
MEMORY_SCAN_LOCK = threading.Lock()

AUTO_TITLE_SYSTEM = (
    "你是对话标题生成器。阅读用户的第一轮提问和助手回答，概括这次对话的具体主题。"
    "生成一个简洁、可检索的中文标题，最多 20 个汉字。"
    "只输出标题本身，不要引号、Markdown、序号、解释或结尾标点。"
    "不要使用“关于”“对话”“问题”等空泛开头，也不要照抄整句问题。"
)


MEMORY_SUMMARY_COMPRESS_SYSTEM = (
    "你是记忆索引摘要压缩器。把给定摘要压缩到 300 个中文字符以内。"
    "保留主题、结论、关键事实和必要关键词，不添加新信息，不输出解释、标题或 Markdown。"
    "只输出压缩后的摘要正文。"
)


MEMORY_REVIEW_SYSTEM = (
    "你是长期记忆整理器。只根据提供的对话证据更新记忆，不能猜测用户性格、水平、身份或偏好。"
    "只保存有明确依据、以后仍然有用的信息；一次性请求、临时任务状态和已完成工作的流水账不要写成长期偏好。"
    "用户明确说“以后都这样/记住/别再”时才可升级为稳定偏好。后来的纠正要覆盖旧结论。"
    "先判断整段对话是否值得长期记忆：寒暄、客套、单纯闲聊、一次性短问答、临时操作和以后不需要复用的内容都应标记为 false。"
    "只有对话对以后仍有复用价值，或用户明确要求记住时才标记为 true。已有摘要本身有意义时，不能因为最近一轮闲聊就改成 false。"
    "不得保存 API Key、密码、令牌、身份证号等敏感信息。"
    "用户画像必须写成陈述句而非命令句，总长度不能超过 2500 个中文字符。"
    "profile 必须是有条理的 Markdown，禁止写成一整段。固定使用以下二级标题分组："
    "## 基本信息、## 偏好与习惯、## 学习与工作方式、## 当前长期目标、## 重要约束。"
    "每个分类下每条信息单独一行并以 '- ' 开头；只保留有可靠证据且以后仍有用的内容。"
    "没有内容的分类直接省略。合并旧画像时要去重、消除矛盾，不要写流水账，也不要把对话索引摘要塞进画像。"
    "conversation_summary 必须压缩到 300 个中文字符以内，保留主题、结论和关键事实。"
    "只输出一个 JSON 对象，不要 Markdown 代码围栏，格式为："
    '{"worth_remembering":true,"profile":"完整的新用户画像","conversation_summary":"当前会话摘要","keywords":["关键词1","关键词2"]}'
)


def _clean_profile(value: str) -> str:
    return str(value or "").strip()


def _parse_iso_ts(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def load_memory() -> dict:
    data = _read_json(MEMORY_PATH, {"version": 1, "profile": "", "conversations": []})
    if not isinstance(data, dict):
        data = {"version": 1, "profile": "", "conversations": []}
    data["profile"] = _clean_profile(data.get("profile"))
    raw_reviews = data.get("reviews") if isinstance(data.get("reviews"), dict) else {}
    reviews = {}
    for conv_id, state in raw_reviews.items():
        if not isinstance(state, dict):
            continue
        key = str(conv_id or "")
        if not key:
            continue
        try:
            reviewed_count = max(0, int(state.get("reviewed_message_count") or 0))
        except (TypeError, ValueError):
            reviewed_count = 0
        try:
            failures = max(0, int(state.get("failures") or 0))
        except (TypeError, ValueError):
            failures = 0
        reviews[key] = {
            "reviewed_message_count": reviewed_count,
            "reviewed_at": state.get("reviewed_at"),
            "worth_remembering": bool(state.get("worth_remembering", True)),
            "failures": failures,
            "last_error": str(state.get("last_error") or "")[:500],
            "last_failed_at": state.get("last_failed_at"),
            "pending_message_count": int(state.get("pending_message_count") or 0),
            "last_activity_at": state.get("last_activity_at"),
        }
    items = data.get("conversations")
    if not isinstance(items, list):
        items = []
    seen = set()
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        conv_id = str(item.get("id") or "")
        if not conv_id or conv_id in seen:
            continue
        seen.add(conv_id)
        summary = str(item.get("summary") or "").strip()
        keywords = [str(x)[:40] for x in (item.get("keywords") or []) if str(x).strip()][:10]
        notes = [str(x).strip()[:MEMORY_NOTE_MAX_CHARS] for x in (item.get("notes") or []) if str(x).strip()][:MEMORY_NOTE_MAX_ITEMS]
        worth_remembering = bool(item.get("worth_remembering", True))
        legacy_count = int(item.get("last_memory_message_count") or 0)
        review = reviews.setdefault(conv_id, {
            "reviewed_message_count": 0,
            "reviewed_at": None,
            "worth_remembering": worth_remembering,
            "failures": 0,
            "last_error": "",
            "last_failed_at": None,
            "pending_message_count": 0,
            "last_activity_at": item.get("last_activity_at"),
        })
        review["reviewed_message_count"] = max(int(review.get("reviewed_message_count") or 0), legacy_count)
        if item.get("last_memory_at") and not review.get("reviewed_at"):
            review["reviewed_at"] = item.get("last_memory_at")
        if not worth_remembering:
            review["worth_remembering"] = False
            continue
        if not (summary or keywords or notes):
            continue
        review["worth_remembering"] = True
        normalized.append({
            "id": conv_id,
            "title": str(item.get("title") or "对话")[:120],
            "summary": summary,
            "keywords": keywords,
            "notes": notes,
            "turns": int(item.get("turns") or 0),
            "archived": bool(item.get("archived")),
            "worth_remembering": worth_remembering,
            "updated_at": item.get("updated_at"),
            "last_activity_at": item.get("last_activity_at"),
            "last_memory_at": item.get("last_memory_at"),
            "last_memory_message_count": legacy_count,
        })
    data["reviews"] = reviews
    data["conversations"] = normalized
    return data


def save_memory(memory: dict) -> None:
    memory["profile"] = _clean_profile(memory.get("profile"))
    memory["updated_at"] = utc_now()
    _atomic_write_json(MEMORY_PATH, memory)


def _conversation_last_activity(conv: dict) -> str:
    latest = ""
    latest_ts = 0.0
    for message in conv.get("messages") or []:
        stamp = str(message.get("created_at") or "")
        ts = _parse_iso_ts(stamp)
        if stamp and ts >= latest_ts:
            latest = stamp
            latest_ts = ts
    return latest or str(conv.get("updated_at") or "")


def _memory_entry(memory: dict, conversation_id: str) -> dict | None:
    for item in memory.get("conversations") or []:
        if item.get("id") == conversation_id:
            return item
    return None


def _review_state(memory: dict, conversation_id: str) -> dict:
    reviews = memory.setdefault("reviews", {})
    state = reviews.get(conversation_id)
    if not isinstance(state, dict):
        state = {}
        reviews[conversation_id] = state
    state.setdefault("reviewed_message_count", 0)
    state.setdefault("reviewed_at", None)
    state.setdefault("worth_remembering", True)
    state.setdefault("failures", 0)
    state.setdefault("last_error", "")
    state.setdefault("last_failed_at", None)
    state.setdefault("pending_message_count", 0)
    return state


def _mark_review_success(memory: dict, conv: dict, worth_remembering: bool) -> dict:
    state = _review_state(memory, str(conv.get("id") or ""))
    state["reviewed_message_count"] = len(conv.get("messages") or [])
    state["reviewed_at"] = utc_now()
    state["worth_remembering"] = bool(worth_remembering)
    state["failures"] = 0
    state["last_error"] = ""
    state["last_failed_at"] = None
    state["pending_message_count"] = 0
    state["last_activity_at"] = _conversation_last_activity(conv)
    return state


def _mark_review_failure(memory: dict, conv: dict, error: str) -> dict:
    state = _review_state(memory, str(conv.get("id") or ""))
    state["failures"] = int(state.get("failures") or 0) + 1
    state["last_error"] = str(error or "unknown error")[:500]
    state["last_failed_at"] = utc_now()
    state["pending_message_count"] = len(conv.get("messages") or [])
    state["last_activity_at"] = _conversation_last_activity(conv)
    return state


def _remove_memory_entry(memory: dict, conversation_id: str) -> None:
    memory["conversations"] = [
        item for item in (memory.get("conversations") or [])
        if str(item.get("id") or "") != conversation_id
    ]


def _entry_is_meaningful(entry: dict) -> bool:
    if not isinstance(entry, dict) or entry.get("worth_remembering") is False:
        return False
    return bool(
        str(entry.get("summary") or "").strip()
        or [x for x in (entry.get("notes") or []) if str(x).strip()]
        or [x for x in (entry.get("keywords") or []) if str(x).strip()]
    )


def _touch_memory_entry(memory: dict, conv: dict) -> dict | None:
    entry = _memory_entry(memory, str(conv.get("id") or ""))
    if entry is None:
        return None
    entry["title"] = str(conv.get("title") or "对话")[:120]
    entry["turns"] = len(split_turns(conv.get("messages") or []))
    entry["updated_at"] = utc_now()
    entry["last_activity_at"] = _conversation_last_activity(conv)
    entry["archived"] = bool(conv.get("archived"))
    return entry


def _upsert_memory_entry(memory: dict, conv: dict, summary: str | None = None, keywords: list | None = None, mark_reviewed: bool = False, worth_remembering: bool = True) -> dict:
    entry = _memory_entry(memory, str(conv.get("id") or ""))
    if entry is None:
        entry = {"id": conv.get("id"), "notes": [], "keywords": [], "last_memory_message_count": 0}
        memory.setdefault("conversations", []).append(entry)
    _touch_memory_entry(memory, conv)
    entry["worth_remembering"] = bool(worth_remembering)
    if summary is not None:
        entry["summary"] = str(summary).strip()
    if keywords is not None:
        entry["keywords"] = [str(x)[:40] for x in keywords if str(x).strip()][:10]
    if mark_reviewed:
        entry["last_memory_message_count"] = len(conv.get("messages") or [])
        entry["last_memory_at"] = utc_now()
    return entry


def sync_memory_entries(memory: dict, conversations: list[dict]) -> dict:
    by_id = {str(conv.get("id") or ""): conv for conv in conversations}
    synced = []
    for entry in memory.get("conversations") or []:
        conv = by_id.get(str(entry.get("id") or ""))
        if conv is None:
            continue
        entry["title"] = str(conv.get("title") or "对话")[:120]
        entry["archived"] = bool(conv.get("archived"))
        entry["turns"] = len(split_turns(conv.get("messages") or []))
        entry["last_activity_at"] = _conversation_last_activity(conv)
        synced.append(entry)
    memory["conversations"] = synced
    reviews = memory.get("reviews") if isinstance(memory.get("reviews"), dict) else {}
    memory["reviews"] = {key: value for key, value in reviews.items() if key in by_id}
    return memory


def memory_snapshot_summary(conv: dict) -> dict:
    snapshot = str(conv.get("memory_snapshot") or "")
    manual = bool(conv.get("memory_snapshot_manual"))
    return {
        "id": conv.get("id"),
        "title": conv.get("title") or "对话",
        "archived": bool(conv.get("archived")),
        "chars": len(snapshot),
        "deleted": bool(manual and not snapshot),
        "manual": manual,
        "updated_at": conv.get("memory_snapshot_updated_at") or conv.get("updated_at"),
    }


def build_memory_snapshot(memory: dict, conversations: list[dict] | None = None) -> str:
    excluded_ids = set()
    if conversations is not None:
        sync_memory_entries(memory, conversations)
        excluded_ids = {
            str(conv.get("id") or "")
            for conv in conversations
            if conv.get("memory_excluded")
        }
    profile = _clean_profile(memory.get("profile"))
    entries = sorted(
        (
            item for item in (memory.get("conversations") or [])
            if _entry_is_meaningful(item) and str(item.get("id") or "") not in excluded_ids
        ),
        key=lambda x: x.get("last_activity_at") or x.get("updated_at") or "",
        reverse=True,
    )
    lines = ["【用户长期画像】", profile or "（暂无）", "", "【历史对话索引】"]
    for item in entries[:MEMORY_SNAPSHOT_MAX_CONVERSATIONS]:
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        tag = "已归档" if item.get("archived") else "可继续"
        lines.append(f"- {item.get('title')}｜{tag}｜ID: {item.get('id')}｜{summary}")
    return "\n".join(lines)


def memory_search_text(query: str, limit: int = 5) -> str:
    memory = load_memory()
    all_conversations = load_conversations()
    sync_memory_entries(memory, all_conversations)
    excluded_ids = {
        str(conv.get("id") or "")
        for conv in all_conversations
        if conv.get("memory_excluded")
    }
    query_l = str(query or "").lower().strip()
    tokens = [t for t in re.split(r"\s+", query_l) if t]
    scored = []
    for item in memory.get("conversations") or []:
        if str(item.get("id") or "") in excluded_ids or not _entry_is_meaningful(item):
            continue
        notes = item.get("notes") or []
        hay = " ".join([str(item.get("title") or ""), str(item.get("summary") or ""), " ".join(item.get("keywords") or []), " ".join(notes)]).lower()
        score = sum(3 if token in str(item.get("title") or "").lower() else 1 for token in tokens if token in hay)
        if query_l and query_l in hay:
            score += 5
        if score or not query_l:
            scored.append((score, item))
    scored.sort(key=lambda x: (x[0], x[1].get("updated_at") or ""), reverse=True)
    results = []
    for _score, item in scored[:max(1, min(10, limit))]:
        results.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "archived": bool(item.get("archived")),
            "summary": item.get("summary"),
            "keywords": item.get("keywords"),
            "notes_preview": " ".join(item.get("notes") or [])[:500],
            "updated_at": item.get("updated_at"),
        })
    return json.dumps({"results": results}, ensure_ascii=False, indent=2)


def memory_read_text(conversation_id: str) -> str:
    conversations = load_conversations()
    conv = next((x for x in conversations if x.get("id") == conversation_id), None)
    if conv is None:
        return json.dumps({"error": "conversation_not_found", "message": "找不到这个历史对话"}, ensure_ascii=False)
    memory = load_memory()
    entry = _memory_entry(memory, conversation_id)
    turns = split_turns(conv.get("messages") or [])
    transcript = _turns_transcript(turns[-8:], limit_chars=20000)
    return json.dumps({
        "id": conversation_id,
        "title": conv.get("title") or "对话",
        "archived": bool(conv.get("archived")),
        "saved_notes": (entry.get("notes") or []) if entry else [],
        "transcript": transcript,
    }, ensure_ascii=False)


def _split_memory_text(text: str, limit: int) -> list[str]:
    remaining = str(text or "").strip()
    chunks = []
    markers = ("\n\n", "\n", "。", "！", "？", "；", ";", ". ", "，", ",")
    while len(remaining) > limit:
        window = remaining[:limit]
        candidates = []
        for marker in markers:
            pos = window.rfind(marker)
            if pos >= limit // 2:
                candidates.append(pos + len(marker))
        cut = max(candidates) if candidates else limit
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def memory_save_text(target: str, content: str, mode: str = "append", conversation_id: str = "") -> str:
    text = str(content or "").strip()
    target = str(target or "profile").strip()
    mode = str(mode or "append").strip()
    if not text:
        return json.dumps({"error": "内容不能为空"}, ensure_ascii=False)
    if target not in ("profile", "conversation"):
        return json.dumps({"error": "unknown_target"}, ensure_ascii=False)
    if mode not in ("append", "replace"):
        return json.dumps({"error": "unknown_mode"}, ensure_ascii=False)
    with FILE_LOCK:
        memory = load_memory()
        if target == "profile":
            old = str(memory.get("profile") or "").strip()
            if mode == "replace":
                new_profile = text
            elif text in old:
                return json.dumps({"ok": True, "message": "这条记忆已经存在", "profile_chars": len(old)}, ensure_ascii=False)
            else:
                new_profile = (old + "\n- " + text).strip() if old else "- " + text
            if len(new_profile) > PROFILE_MAX_CHARS:
                return json.dumps({"error": "profile_full", "message": f"画像超过 {PROFILE_MAX_CHARS} 字。请用 mode=replace 提交压缩后的完整画像。", "profile": old, "profile_chars": len(old)}, ensure_ascii=False)
            memory["profile"] = new_profile
            save_memory(memory)
            return json.dumps({"ok": True, "profile_chars": len(new_profile)}, ensure_ascii=False)
        if len(text) > MEMORY_SAVE_MAX_CHARS:
            return json.dumps({"error": "memory_too_large", "message": f"单次对话记忆不能超过 {MEMORY_SAVE_MAX_CHARS} 字，请分成多次写入。"}, ensure_ascii=False)
        conversations = load_conversations()
        conv = next((x for x in conversations if x.get("id") == conversation_id), None)
        if conv is None:
            return json.dumps({"error": "conversation_not_found"}, ensure_ascii=False)
        if conv.get("memory_excluded"):
            return json.dumps({"error": "conversation_excluded", "message": "这个对话已被排除出长期记忆"}, ensure_ascii=False)
        entry = _upsert_memory_entry(memory, conv)
        notes = list(entry.get("notes") or [])
        chunks = _split_memory_text(text, MEMORY_NOTE_MAX_CHARS)
        if mode == "append":
            new_chunks = [chunk for chunk in chunks if chunk not in notes]
            if not new_chunks:
                return json.dumps({"ok": True, "message": "这条记忆已经存在", "notes": len(notes)}, ensure_ascii=False)
            if len(notes) + len(new_chunks) > MEMORY_NOTE_MAX_ITEMS:
                return json.dumps({"error": "conversation_memory_full", "message": "当前对话记忆条目已满，请先整理已有记忆或改用画像。", "notes": len(notes), "limit": MEMORY_NOTE_MAX_ITEMS}, ensure_ascii=False)
            notes.extend(new_chunks)
        else:
            if len(chunks) > MEMORY_NOTE_MAX_ITEMS:
                return json.dumps({"error": "conversation_memory_full", "message": "替换内容超过当前对话记忆容量，请缩短后重试。", "limit": MEMORY_NOTE_MAX_ITEMS}, ensure_ascii=False)
            notes = chunks
        entry["notes"] = notes
        save_memory(memory)
        return json.dumps({"ok": True, "message": "已写入当前对话记忆", "notes": len(notes), "chunks": len(chunks)}, ensure_ascii=False)


def execute_memory_tool(name: str, args: dict, current_conversation_id: str) -> tuple[str, list[dict]]:
    if name == "memory_search":
        query = str(args.get("query") or "").strip()
        try:
            limit = int(args.get("limit") or 5)
        except (TypeError, ValueError):
            limit = 5
        return memory_search_text(query, limit), []
    if name == "memory_read_conversation":
        return memory_read_text(str(args.get("conversation_id") or "").strip()), []
    if name == "memory_save":
        return memory_save_text(
            str(args.get("target") or "profile"),
            str(args.get("content") or "").strip(),
            str(args.get("mode") or "append"),
            current_conversation_id,
        ), []
    return json.dumps({"error": "unknown_tool", "message": f"未知工具：{name}"}, ensure_ascii=False), []


def memory_search_tool_schema() -> dict:
    return {"type": "function", "function": {"name": "memory_search", "description": "搜索过去对话的索引。需要引用以前做过的项目、决定或用户偏好时使用；需要细节时再用 memory_read_conversation 读取。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"]}}}


def memory_read_tool_schema() -> dict:
    return {"type": "function", "function": {"name": "memory_read_conversation", "description": "读取一个历史对话的文字正文。归档对话也可以读取。", "parameters": {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]}}}


def memory_save_tool_schema() -> dict:
    return {"type": "function", "function": {"name": "memory_save", "description": "主动保存有长期价值、且有对话原文依据的信息。较长内容会自动分块；不要保存一次性请求、临时状态、寒暄或密钥。", "parameters": {"type": "object", "properties": {"target": {"type": "string", "enum": ["profile", "conversation"]}, "content": {"type": "string", "description": "陈述式的事实或任务要点；长内容会自动分块保存"}, "mode": {"type": "string", "enum": ["append", "replace"], "description": "append 追加；replace 用新内容替换画像或当前对话的笔记"}}, "required": ["target", "content"]}}}


def _memory_review_transcript(turns: list[tuple[dict, list[dict]]], limit_chars: int) -> str:
    selected: list[str] = []
    total = 0
    for turn in reversed(turns):
        block = _turns_transcript([turn], limit_chars=limit_chars)
        if not block:
            continue
        extra = 2 if selected else 0
        if total + len(block) + extra > limit_chars:
            if not selected:
                selected.append(block[:limit_chars])
            break
        selected.append(block)
        total += len(block) + extra
    if len(selected) < len(turns):
        selected.append("……（更早内容已省略）")
    return "\n\n".join(reversed(selected))


def _compress_memory_summary(provider: dict, model_id: str, title: str, summary: str) -> str:
    source = str(summary or "").strip()
    if len(source) <= MEMORY_INDEX_SUMMARY_MAX_CHARS:
        return source
    messages = [
        {"role": "system", "content": MEMORY_SUMMARY_COMPRESS_SYSTEM},
        {"role": "user", "content": f"标题：{title}\n\n原摘要：\n{source}"},
    ]
    chunks = []
    try:
        for event_type, data in stream_chat(provider, model_id, messages, ""):
            if event_type == "content":
                chunks.append(str(data.get("text") or ""))
    except Exception:
        return ""
    compressed = re.sub(r"\s+", " ", "".join(chunks)).strip()
    compressed = compressed.strip(" #*`'\"“”‘’")
    if not compressed or len(compressed) > MEMORY_INDEX_SUMMARY_MAX_CHARS:
        return ""
    return compressed


def _memory_scan_worker() -> None:
    with FILE_LOCK:
        config = load_config()
        memory_settings = (config.get("settings") or {}).get("memory") or {}
        if not memory_settings.get("enabled"):
            return
        provider_id = str(memory_settings.get("provider_id") or "")
        model_id = str(memory_settings.get("model_id") or "")
        review_prompt = str(memory_settings.get("review_prompt") or MEMORY_REVIEW_SYSTEM).strip()
        try:
            provider = provider_by_id(config, provider_id)
            model_info = find_model(provider, model_id)
        except ApiError:
            return
        if not model_info:
            return
        conversations = load_conversations()
    memory = load_memory()
    sync_memory_entries(memory, conversations)
    oversized_summaries = [
        entry for entry in (memory.get("conversations") or [])
        if len(str(entry.get("summary") or "")) > MEMORY_INDEX_SUMMARY_MAX_CHARS
    ][:MEMORY_REVIEW_MAX_CONVERSATIONS]
    for entry in oversized_summaries:
        summary = str(entry.get("summary") or "")
        compressed = _compress_memory_summary(
            provider,
            model_id,
            str(entry.get("title") or "对话"),
            summary,
        )
        if not compressed:
            continue
        with FILE_LOCK:
            fresh = load_memory()
            fresh_entry = _memory_entry(fresh, str(entry.get("id") or ""))
            if fresh_entry and len(str(fresh_entry.get("summary") or "")) > MEMORY_INDEX_SUMMARY_MAX_CHARS:
                fresh_entry["summary"] = compressed
                fresh_entry["updated_at"] = utc_now()
                save_memory(fresh)
    memory = load_memory()
    sync_memory_entries(memory, conversations)
    now_ts = datetime.now(timezone.utc).timestamp()
    due = []
    for conv in conversations:
        if conv.get("archived") or conv.get("memory_excluded"):
            continue
        messages = conv.get("messages") or []
        if len(messages) < 2:
            continue
        last = _conversation_last_activity(conv)
        if now_ts - _parse_iso_ts(last) < MEMORY_SCAN_MINUTES * 60:
            continue
        review = (memory.get("reviews") or {}).get(str(conv.get("id") or "")) or {}
        reviewed_count = int(review.get("reviewed_message_count") or 0)
        if not reviewed_count:
            entry = _memory_entry(memory, str(conv.get("id") or ""))
            reviewed_count = int(entry.get("last_memory_message_count") or 0) if entry else 0
        if reviewed_count and len(messages) <= reviewed_count:
            continue
        due.append(conv)
    for conv in due[:MEMORY_REVIEW_MAX_CONVERSATIONS]:
        conv_id = str(conv.get("id") or "")
        turns = split_turns(conv.get("messages") or [])
        transcript = _memory_review_transcript(turns, MEMORY_REVIEW_CONTEXT_CHARS)
        entry = _memory_entry(memory, conv_id)
        prompt = "【现有用户画像】\n" + (str(memory.get("profile") or "") or "（暂无）") + "\n\n【当前对话已有摘要】\n" + (str(entry.get("summary") or "") if entry else "（暂无）") + "\n\n【最近对话】\n" + transcript
        messages = [{"role": "system", "content": review_prompt}, {"role": "user", "content": prompt}]
        chunks = []
        try:
            for event_type, data in stream_chat(provider, model_id, messages, ""):
                if event_type == "content":
                    chunks.append(str(data.get("text") or ""))
        except Exception as exc:
            with FILE_LOCK:
                fresh = load_memory()
                fresh_conversations = load_conversations()
                fresh_conv = next((x for x in fresh_conversations if x.get("id") == conv_id), conv)
                _mark_review_failure(fresh, fresh_conv, str(exc))
                save_memory(fresh)
            continue
        raw = "".join(chunks).strip()
        if raw.startswith(chr(96) * 3):
            raw = re.sub(r"^" + chr(96) * 3 + r"[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*" + chr(96) * 3 + r"$", "", raw)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            result = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            with FILE_LOCK:
                fresh = load_memory()
                fresh_conversations = load_conversations()
                fresh_conv = next((x for x in fresh_conversations if x.get("id") == conv_id), conv)
                _mark_review_failure(fresh, fresh_conv, f"invalid memory JSON: {exc}")
                save_memory(fresh)
            continue
        profile = _clean_profile(result.get("profile") or memory.get("profile"))
        if len(profile) > PROFILE_MAX_CHARS:
            with FILE_LOCK:
                fresh = load_memory()
                fresh_conversations = load_conversations()
                fresh_conv = next((x for x in fresh_conversations if x.get("id") == conv_id), conv)
                _mark_review_failure(
                    fresh,
                    fresh_conv,
                    f"profile_too_long: 画像 {len(profile)} 字，超过 {PROFILE_MAX_CHARS} 字，必须缩短",
                )
                save_memory(fresh)
            continue
        summary = str(result.get("conversation_summary") or "").strip()
        if len(summary) > MEMORY_INDEX_SUMMARY_MAX_CHARS:
            summary = _compress_memory_summary(
                provider,
                model_id,
                str(conv.get("title") or "对话"),
                summary,
            )
            if not summary:
                with FILE_LOCK:
                    fresh = load_memory()
                    fresh_conversations = load_conversations()
                    fresh_conv = next((x for x in fresh_conversations if x.get("id") == conv_id), conv)
                    _mark_review_failure(
                        fresh,
                        fresh_conv,
                        f"summary_too_long: 索引摘要超过 {MEMORY_INDEX_SUMMARY_MAX_CHARS} 字，压缩失败",
                    )
                    save_memory(fresh)
                continue
        keywords = result.get("keywords") if isinstance(result.get("keywords"), list) else []
        existing_notes = (entry.get("notes") or []) if entry else []
        worth = result.get("worth_remembering")
        if not isinstance(worth, bool):
            worth = bool(summary or existing_notes or (entry and entry.get("summary")))
        if existing_notes:
            worth = True
        if worth and not (summary or existing_notes or keywords):
            worth = False
        with FILE_LOCK:
            fresh = load_memory()
            fresh_conversations = load_conversations()
            fresh_conv = next((x for x in fresh_conversations if x.get("id") == conv_id), None)
            if fresh_conv is None:
                continue
            if worth:
                fresh["profile"] = profile
                _upsert_memory_entry(
                    fresh,
                    fresh_conv,
                    summary or (entry.get("summary") if entry else ""),
                    keywords,
                    mark_reviewed=True,
                    worth_remembering=True,
                )
            else:
                _remove_memory_entry(fresh, conv_id)
            _mark_review_success(fresh, fresh_conv, worth)
            save_memory(fresh)


def start_memory_scan() -> bool:
    if not MEMORY_SCAN_LOCK.acquire(blocking=False):
        return False
    def worker():
        try:
            _memory_scan_worker()
        finally:
            MEMORY_SCAN_LOCK.release()
    threading.Thread(target=worker, daemon=True).start()
    return True


def build_upstream_messages(conv: dict, exclude_turn_index: int | None = None) -> list[dict]:
    # 每轮对话绑定创建时选中的提示词原文，后续修改预设不会改变旧对话。
    if "system_prompt" in conv:
        system_text = str(conv.get("system_prompt") or "").strip()
    else:
        system_text = SYSTEM_PROMPT
    # 这一段工具规则始终固定，开关只决定是否允许执行工具，避免破坏前缀缓存。
    tool_rules = str(conv.get("tool_instructions") or DEFAULT_TOOL_INSTRUCTIONS).strip()
    environment_note = str(conv.get("environment") or "").strip() or build_environment_note()
    messages = [{"role": "system", "content": tool_rules}]
    memory_snapshot = str(conv.get("memory_snapshot") or "").strip()
    if memory_snapshot:
        messages.append({"role": "system", "content": memory_snapshot})
    messages.append({"role": "system", "content": environment_note})
    if system_text:
        messages.append({"role": "system", "content": system_text})
    digest = conv.get("digest") or {}
    digest_text = str(digest.get("text") or "").strip()
    if digest_text:
        # 摘要放在固定系统提示词之后，位置稳定，不破坏前缀缓存。
        messages.append(
            {"role": "system", "content": "【早前对话摘要，用于延续上下文】\n" + digest_text}
        )
    try:
        covered_turns = int(digest.get("turns") or 0)
    except (TypeError, ValueError):
        covered_turns = 0
    for index, (user, assistants) in enumerate(split_turns(conv.get("messages") or [])):
        if index < covered_turns:
            continue
        content = user.get("content")
        text = content if isinstance(content, str) else ""
        images = user.get("images")
        files = user.get("files")
        blocks: list[str] = []
        doc_images: list[tuple[str, int, str]] = []
        if isinstance(files, list):
            for file in files:
                if not isinstance(file, dict):
                    continue
                block = file_text_block(file)
                if block:
                    blocks.append(block)
                file_name = str(file.get("name") or "附件")
                for image in file.get("page_images") or []:
                    if not isinstance(image, dict):
                        continue
                    data_url = image.get("data_url")
                    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
                        continue
                    try:
                        page_number = int(image.get("page") or 0)
                    except (TypeError, ValueError):
                        page_number = 0
                    doc_images.append((file_name, page_number, data_url))
        if text:
            blocks.append(text)
        combined_text = "\n\n".join(blocks).strip()
        if (isinstance(images, list) and images) or doc_images:
            parts: list[dict] = []
            if combined_text:
                parts.append({"type": "text", "text": combined_text})
            if isinstance(images, list):
                for url in images:
                    if isinstance(url, str) and url.startswith("data:image/"):
                        parts.append({"type": "image_url", "image_url": {"url": url}})
            for file_name, page_number, data_url in doc_images:
                label = f"【文档页面图片：{file_name}"
                if page_number:
                    label += f" 第 {page_number} 页"
                label += "】"
                parts.append({"type": "text", "text": label})
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            if parts:
                messages.append({"role": "user", "content": parts})
        elif combined_text:
            messages.append({"role": "user", "content": combined_text})

        if index == exclude_turn_index:
            continue
        assistant, _variant = active_assistant(user, assistants)
        if assistant is None or assistant.get("error") or assistant.get("stopped"):
            continue
        assistant_text = assistant.get("content")
        if isinstance(assistant_text, str) and assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def compress_conversation(
    conv: dict,
    provider: dict,
    provider_id: str,
    model_id: str,
    model_info: dict,
    effort: str,
) -> tuple[dict, str, int]:
    """把对话前半段压缩成摘要，返回 (响应体, 摘要文本, 已覆盖轮数)。"""
    digest = dict(conv.get("digest") or {})
    turns = split_turns(conv.get("messages") or [])
    try:
        covered = int(digest.get("turns") or 0)
    except (TypeError, ValueError):
        covered = 0
    remaining = turns[covered:]
    if len(remaining) < MIN_COMPRESS_TURNS:
        raise ApiError("这个对话还很短，暂时不需要压缩；继续聊一段时间再压缩效果更好。", 400)

    half = max(1, len(remaining) // 2)
    old_turns = remaining[:half]
    previous = str(digest.get("text") or "").strip()
    before_tokens = conversation_context(conv)["tokens"]

    transcript = _turns_transcript(old_turns)
    prompt_parts: list[str] = []
    if previous:
        prompt_parts.append("【已有摘要】\n" + previous)
    prompt_parts.append("【需要并入摘要的对话】\n" + transcript)
    prompt_parts.append(
        "请输出一份合并后的新摘要，控制在原文的三分之一左右，分条列点，"
        "保留结论、数据、待办和用户偏好。"
    )
    compress_messages = [
        {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]

    chunks: list[str] = []
    try:
        for event_type, data in stream_chat(provider, model_id, compress_messages, effort):
            if event_type == "content":
                chunks.append(str(data.get("text") or ""))
    except UpstreamError as exc:
        raise ApiError(f"压缩失败：{exc.message}", 502) from exc
    summary_text = "".join(chunks).strip()
    if not summary_text:
        raise ApiError("模型没有返回摘要内容，请稍后重试", 502)
    if len(summary_text) > MAX_DIGEST_CHARS:
        summary_text = summary_text[:MAX_DIGEST_CHARS] + "…"

    return (
        {
            "before_tokens": before_tokens,
            "summarized_turns": covered + half,
        },
        summary_text,
        covered + half,
    )


def _turns_transcript(turns: list[tuple[dict, list[dict]]], limit_chars: int = 120_000) -> str:
    parts: list[str] = []
    total = 0
    for user, assistants in turns:
        text = user.get("content")
        text = text if isinstance(text, str) else ""
        if not text and user.get("images"):
            text = "[图片]"
        if not text and user.get("files"):
            text = "[附件：" + "、".join(
                str(item.get("name") or "文件")
                for item in (user.get("files") or [])
                if isinstance(item, dict)
            ) + "]"
        block = "用户：" + re.sub(r"\s+", " ", text).strip()
        assistant, _index = active_assistant(user, assistants)
        reply = ""
        if assistant and not assistant.get("error") and isinstance(assistant.get("content"), str):
            reply = re.sub(r"\s+", " ", assistant.get("content") or "").strip()
        block += "\n助手：" + (reply or "（无有效回答）")
        total += len(block)
        if total > limit_chars:
            parts.append(block[: max(0, limit_chars - (total - len(block)))])
            parts.append("……（更早内容已省略）")
            break
        parts.append(block)
    return "\n\n".join(parts)


def web_search_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索公开网络并读取搜索结果的正文。仅在需要实时信息、外部资料或用户明确要求搜索时使用。"
                "普通知识问答、写作、代码和数学计算不要使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "简洁、具体的搜索关键词"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def build_chat_payload(
    model: str,
    messages: list[dict],
    effort: str,
    tools: list[dict] | None = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if effort:
        payload["reasoning_effort"] = effort
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def _extract_firecrawl_results(data: dict, max_results: int) -> list[dict]:
    candidates = []
    if isinstance(data.get("data"), dict):
        for key in ("web", "results", "links"):
            if isinstance(data["data"].get(key), list):
                candidates = data["data"][key]
                break
    elif isinstance(data.get("data"), list):
        candidates = data["data"]
    elif isinstance(data.get("results"), list):
        candidates = data["results"]
    results = []
    for item in candidates[:max_results]:
        if not isinstance(item, dict):
            continue
        markdown = item.get("markdown") or item.get("content") or ""
        if isinstance(markdown, dict):
            markdown = markdown.get("markdown") or markdown.get("text") or ""
        description = item.get("description") or item.get("snippet") or item.get("summary") or ""
        page_url = str(item.get("url") or item.get("link") or "")[:2000]
        try:
            if urlparse(page_url).scheme.lower() not in ("http", "https"):
                continue
        except Exception:
            continue
        results.append(
            {
                "title": str(item.get("title") or item.get("name") or "未命名页面")[:300],
                "url": page_url,
                "description": str(description)[:1200],
                "markdown": str(markdown)[:WEB_RESULT_MARKDOWN_CHARS],
            }
        )
    return results


def firecrawl_search(api_key: str, query: str, max_results: int = WEB_SEARCH_DEFAULT_RESULTS) -> list[dict]:
    try:
        limit = int(max_results or WEB_SEARCH_DEFAULT_RESULTS)
    except (TypeError, ValueError):
        limit = WEB_SEARCH_DEFAULT_RESULTS
    limit = max(1, min(WEB_SEARCH_MAX_RESULTS, limit))
    body = json.dumps(
        {
            "query": query,
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        FIRECRAWL_SEARCH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"XiaoXiaoFatFish/{VERSION}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60, context=ssl.create_default_context()) as resp:
            raw = resp.read(8 * 1024 * 1024)
    except error.HTTPError as exc:
        detail = _extract_error_message(exc.read())
        raise UpstreamError(f"Firecrawl 搜索失败（{exc.code}）：{detail}", exc.code, detail) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpstreamError(f"无法连接 Firecrawl：{reason}") from exc
    except TimeoutError as exc:
        raise UpstreamError("Firecrawl 搜索等待超过 60 秒") from exc
    try:
        data = json.loads(_decode_body(raw))
    except json.JSONDecodeError as exc:
        raise UpstreamError("Firecrawl 返回的不是有效 JSON") from exc
    if not isinstance(data, dict) or data.get("success") is False:
        detail = _extract_error_message(raw)
        raise UpstreamError(f"Firecrawl 搜索失败：{detail}")
    results = _extract_firecrawl_results(data, limit)
    if not results:
        raise UpstreamError("Firecrawl 没有返回搜索结果")
    return results


def normalize_tool_calls(calls: list) -> list[dict]:
    normalized = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or f"call_{index}")
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False),
                },
                "_args": args if isinstance(args, dict) else {},
            }
        )
    return normalized


def execute_web_tool(call: dict, api_key: str, max_results: int) -> tuple[str, list[dict]]:
    if call["function"]["name"] != "web_search":
        return f"未知工具：{call['function']['name']}", []
    query = str(call["_args"].get("query") or "").strip()
    if not query:
        return "搜索失败：query 不能为空", []
    try:
        results = firecrawl_search(api_key, query, max_results)
    except UpstreamError as exc:
        return f"搜索失败：{exc.message}", []
    sources = [{"title": item["title"], "url": item["url"]} for item in results]
    return format_web_results(results), sources


def format_web_results(results: list[dict]) -> str:
    parts = []
    used = 0
    for index, item in enumerate(results, 1):
        body = item.get("markdown") or item.get("description") or ""
        remain = WEB_TOTAL_CONTEXT_CHARS - used
        if remain <= 0:
            break
        body = body[:remain]
        used += len(body)
        parts.append(
            f"[{index}] {item.get('title')}\nURL: {item.get('url')}\n{body}"
        )
    return "\n\n---\n\n".join(parts)


def _open_chat_stream(provider: dict, payload: dict):
    url = api_endpoint(str(provider.get("base_url") or ""), "/chat/completions")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {provider.get('api_key', '')}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": f"XiaoXiaoFatFish/{VERSION}",
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        return request.urlopen(req, timeout=provider_timeout(provider), context=ssl.create_default_context())
    except error.HTTPError as exc:
        detail = _extract_error_message(exc.read())
        raise UpstreamError(friendly_http_error(exc.code, detail), exc.code, detail) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise UpstreamError(f"无法连接模型服务：{reason}。请检查 Base URL 和网络。") from exc
    except TimeoutError as exc:
        raise UpstreamError(f"等待模型响应超过 {provider_timeout(provider)} 秒") from exc


def _optional_parameter_error(exc: UpstreamError) -> bool:
    text = (exc.detail or exc.message or "").lower()
    markers = (
        "stream_options", "reasoning_effort", "reasoning", "unsupported", "not supported",
        "unknown parameter", "unexpected keyword", "extra inputs", "unrecognized",
        "tools", "tool_choice", "function calling", "function_call",
    )
    return exc.status in (400, 422) and any(marker in text for marker in markers)


WEB_SEARCH_HINTS = (
    "搜索", "搜一下", "搜搜", "联网查", "查一下最新", "最新", "实时",
    "今天", "明天", "后天", "现在", "新闻", "天气", "气温", "价格",
    "汇率", "股价", "比分", "发布", "上映", "刚刚",
)


def contains_tool_markup(text: str) -> bool:
    lowered = (text or "").lower()
    return "dsml" in lowered or (
        "web_search" in lowered and ("invoke" in lowered or "tool_calls" in lowered or "parameter name" in lowered)
    )


def clean_search_query(text: str) -> str:
    query = re.sub(r"^(?:请|帮我|麻烦|能不能|可以)?\s*(?:搜搜|搜一下|搜索一下|搜索|联网查|查一下|查查)\s*", "", text.strip())
    query = re.sub(r"\s+", " ", query).strip()
    return (query or text.strip())[:240]


def start_tool_event(assistant: dict, emit, name: str, query: str) -> int:
    events = assistant.setdefault("tool_events", [])
    event = {"name": name, "query": query, "status": "searching"}
    events.append(event)
    index = len(events) - 1
    emit("tool_event", {"index": index, "event": dict(event)})
    return index


def finish_tool_event(assistant: dict, emit, index: int, result_count: int = 0, error: str = "") -> None:
    events = assistant.get("tool_events") or []
    if index < 0 or index >= len(events):
        return
    event = events[index]
    event["status"] = "error" if error else "done"
    event["result_count"] = result_count
    if error:
        event["error"] = error[:300]
    emit("tool_event", {"index": index, "event": dict(event)})


def should_force_web_search(user_text: str, model_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    if any(hint in text for hint in WEB_SEARCH_HINTS):
        return True
    short_model = (model_text or "").strip()
    return len(short_model) < 120 and any(word in short_model for word in ("查一下", "搜索一下", "帮你查", "我来搜"))


def run_tool_loop(
    provider: dict,
    model_id: str,
    messages: list[dict],
    effort: str,
    tools: list[dict] | None,
    web_enabled: bool,
    web_api_key: str,
    web_max_results: int,
    memory_enabled: bool,
    current_conversation_id: str,
    emit,
    assistant: dict,
) -> None:
    active = list(messages)
    active_tools = tools
    disabled_notice_sent = False
    for round_index in range(TOOL_MAX_ROUNDS + 1):
        pending: list[dict] = []
        round_content: list[str] = []
        round_reasoning: list[str] = []
        events = stream_chat(provider, model_id, active, effort, tools=active_tools)
        try:
            for event_type, data in events:
                if event_type == "notice":
                    emit("notice", {"message": data.get("message", "已切换兼容模式")})
                elif event_type == "status":
                    assistant["status"] = str(data.get("message") or "")
                    emit("status", {"message": assistant["status"]})
                elif event_type == "reasoning":
                    text = str(data.get("text") or "")
                    round_reasoning.append(text)
                    assistant["reasoning"] += text
                    assistant["status"] = ""
                    emit("reasoning", {"text": text})
                elif event_type == "content":
                    text = str(data.get("text") or "")
                    round_content.append(text)
                    assistant["content"] += text
                    assistant["status"] = ""
                    emit("content", {"text": text})
                elif event_type == "usage":
                    assistant["usage"] = data.get("usage")
                    emit("usage", {"usage": data.get("usage")})
                elif event_type == "tool_calls":
                    calls = data.get("tool_calls")
                    if isinstance(calls, list):
                        pending.extend(calls)
                elif event_type == "finish" and data.get("finish_reason"):
                    assistant["finish_reason"] = data.get("finish_reason")
        finally:
            events.close()

        if not pending:
            round_text = "".join(round_content)
            tool_markup = contains_tool_markup(round_text)
            if tool_markup:
                assistant["content"] = ""
                emit("replace_content", {"content": ""})
                if not web_enabled:
                    if disabled_notice_sent:
                        return
                    active.append({
                        "role": "system",
                        "content": "模型刚才试图调用 web_search，但用户当前关闭了联网搜索。请用简体中文直接告诉用户：需要点击输入框中的地球按钮开启联网搜索，然后才能查询实时或网络信息。不要输出 DSML、XML 或工具调用代码，也不要假装已经完成搜索。",
                    })
                    active_tools = [t for t in (active_tools or []) if t.get("function", {}).get("name") != "web_search"] or None
                    disabled_notice_sent = True
                    continue
            if round_index == 0 and web_enabled and not assistant.get("search_sources"):
                user_text = ""
                for message in reversed(active):
                    if message.get("role") == "user":
                        content = message.get("content")
                        if isinstance(content, str):
                            user_text = content
                        elif isinstance(content, list):
                            user_text = " ".join(
                                str(part.get("text") or "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            )
                        break
                if tool_markup or should_force_web_search(user_text, round_text):
                    query = clean_search_query(user_text)
                    tool_index = start_tool_event(assistant, emit, "web_search", query)
                    assistant["status"] = f"正在联网搜索：{query}"
                    emit("status", {"message": assistant["status"]})
                    try:
                        search_results = firecrawl_search(web_api_key, query, web_max_results)
                    except UpstreamError as exc:
                        finish_tool_event(assistant, emit, tool_index, 0, f"搜索失败：{exc.message}")
                        emit("notice", {"message": f"搜索失败：{exc.message}"})
                        return
                    finish_tool_event(assistant, emit, tool_index, len(search_results))
                    result_text = format_web_results(search_results)
                    assistant["search_sources"] = [
                        {"title": item["title"], "url": item["url"]}
                        for item in search_results
                    ]
                    active.append({
                        "role": "system",
                        "content": "【本轮联网搜索结果，请基于这些结果回答用户问题，并在使用处附来源链接】\n" + result_text,
                    })
                    active_tools = [t for t in (active_tools or []) if t.get("function", {}).get("name") != "web_search"] or None
                    assistant["status"] = "正在整理搜索结果"
                    emit("status", {"message": assistant["status"]})
                    continue
            return

        if round_index >= TOOL_MAX_ROUNDS:
            raise UpstreamError("工具调用次数超过限制，已停止本次回答")

        calls = normalize_tool_calls(pending)
        active.append({
            "role": "assistant",
            "content": "".join(round_content) or None,
            "tool_calls": [
                {"id": item["id"], "type": "function", "function": item["function"]}
                for item in calls
            ],
        })
        web_disabled_hit = False
        for item in calls:
            name = item["function"]["name"]
            args = item["_args"]
            if name == "web_search":
                display_query = str(args.get("query") or "").strip()
            elif name == "memory_search":
                display_query = str(args.get("query") or "").strip()
            elif name == "memory_read_conversation":
                display_query = str(args.get("conversation_id") or "").strip()
            elif name == "memory_save":
                display_query = str(args.get("content") or "").strip()[:80]
            else:
                display_query = ""
            tool_index = start_tool_event(assistant, emit, name, display_query)
            assistant["status"] = f"正在使用工具：{name}"
            emit("status", {"message": assistant["status"]})
            sources: list[dict] = []
            error_text = ""
            if name == "web_search":
                if not web_enabled:
                    text = json.dumps({
                        "error": "web_search_disabled",
                        "message": "用户当前关闭了联网搜索。不要重试。请用简体中文直接告诉用户：需要点击输入框中的地球按钮开启联网搜索，然后才能查询实时或网络信息。",
                    }, ensure_ascii=False)
                    error_text = "联网搜索未开启"
                    web_disabled_hit = True
                else:
                    text, sources = execute_web_tool(item, web_api_key, web_max_results)
                    if text.startswith("搜索失败"):
                        error_text = text
                        emit("notice", {"message": text})
            elif name in ("memory_search", "memory_read_conversation", "memory_save"):
                if not memory_enabled:
                    text = json.dumps({"error": "memory_disabled", "message": "记忆功能当前未开启"}, ensure_ascii=False)
                    error_text = "记忆功能未开启"
                else:
                    text, sources = execute_memory_tool(name, args, current_conversation_id)
                    try:
                        result_obj = json.loads(text)
                        if isinstance(result_obj, dict) and result_obj.get("error"):
                            error_text = str(result_obj.get("error"))
                    except (TypeError, json.JSONDecodeError):
                        pass
            else:
                text = json.dumps({"error": "unknown_tool", "message": f"未知工具：{name}"}, ensure_ascii=False)
                error_text = "未知工具"
            if sources:
                known = assistant.setdefault("search_sources", [])
                for source in sources:
                    if source not in known:
                        known.append(source)
            finish_tool_event(assistant, emit, tool_index, len(sources), error_text)
            active.append({"role": "tool", "tool_call_id": item["id"], "content": text})
        if web_disabled_hit:
            active_tools = [t for t in (active_tools or []) if t.get("function", {}).get("name") != "web_search"] or None
        assistant["status"] = "正在整理工具结果"
        emit("status", {"message": assistant["status"]})


def stream_chat(
    provider: dict,
    model: str,
    messages: list[dict],
    effort: str,
    tools: list[dict] | None = None,
):
    """生成 (event, data) 事件。网络抖动或中转站响应慢时按配置自动重试。"""
    full_payload = build_chat_payload(model, messages, effort, tools=tools)
    variants: list[tuple[dict, str | None]] = [(full_payload, None)]
    if tools:
        no_tools = dict(full_payload)
        no_tools.pop("tools", None)
        no_tools.pop("tool_choice", None)
        variants.append((no_tools, "Provider does not support tools; continuing without tool calls."))
    if "stream_options" in full_payload:
        no_usage = dict(full_payload)
        no_usage.pop("stream_options", None)
        variants.append((no_usage, "该供应商不支持流式用量统计，已自动使用兼容模式。"))
    if effort:
        minimal = dict(full_payload)
        minimal.pop("stream_options", None)
        minimal.pop("reasoning_effort", None)
        variants.append((minimal, "Provider does not accept the effort parameter; using the model default."))

    retries = provider_retries(provider)
    for index, (payload, fallback_notice) in enumerate(variants):
        attempt = 0
        while True:
            response = None
            started = False
            try:
                response = _open_chat_stream(provider, payload)
                if fallback_notice:
                    yield "notice", {"message": fallback_notice}

                content_type = (response.headers.get("Content-Type") or "").lower()
                if "text/event-stream" not in content_type:
                    raw = response.read(2 * 1024 * 1024)
                    text = raw.decode("utf-8", "replace")
                    stripped = text.lstrip()
                    if "text/html" in content_type or stripped.startswith("<"):
                        raise UpstreamError(
                            "该地址返回了网页而不是模型接口。请检查 Base URL："
                            "有些中转站的 API 域名和官网不同（例如把 www 换成 api）。",
                            400,
                        )
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise UpstreamError(
                            "模型接口返回了无法解析的内容，请检查 Base URL", 400
                        ) from exc
                    if isinstance(obj, dict) and obj.get("error"):
                        message = _extract_error_message(raw)
                        raise StreamBusinessError(f"模型服务返回错误：{message}")
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        raise UpstreamError("模型接口没有返回任何回答", 400)
                    choice = choices[0] or {}
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") for part in content if isinstance(part, dict)
                        )
                    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    if reasoning:
                        yield "reasoning", {"text": str(reasoning)}
                    if content:
                        yield "content", {"text": str(content)}
                    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
                    if isinstance(tool_calls, list) and tool_calls:
                        yield "tool_calls", {"tool_calls": tool_calls}
                    usage = normalize_usage(obj.get("usage"))
                    if usage:
                        yield "usage", {"usage": usage}
                    yield "finish", {"finish_reason": str(choice.get("finish_reason") or "")}
                    return

                finish_reason = ""
                last_usage = None
                stream_tool_calls: dict[int, dict] = {}
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        message = _extract_error_message(data_text.encode("utf-8"))
                        raise StreamBusinessError(f"模型服务返回错误：{message}")
                    if not isinstance(obj, dict):
                        continue
                    usage = normalize_usage(obj.get("usage"))
                    if usage:
                        last_usage = usage
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0] or {}
                    if choice.get("finish_reason"):
                        finish_reason = str(choice.get("finish_reason"))
                    delta = choice.get("delta") or {}
                    source = delta if isinstance(delta, dict) else {}
                    reasoning = source.get("reasoning_content") or source.get("reasoning") or ""
                    content = source.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") for part in content if isinstance(part, dict)
                        )
                    if reasoning:
                        started = True
                        yield "reasoning", {"text": str(reasoning)}
                    if content:
                        started = True
                        yield "content", {"text": str(content)}
                    raw_tool_calls = source.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        for position, raw_call in enumerate(raw_tool_calls):
                            if not isinstance(raw_call, dict):
                                continue
                            try:
                                call_index = int(raw_call.get("index", position))
                            except (TypeError, ValueError):
                                call_index = position
                            current = stream_tool_calls.setdefault(
                                call_index,
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                            )
                            if raw_call.get("id"):
                                current["id"] = str(raw_call["id"])
                            function = raw_call.get("function") or {}
                            if isinstance(function, dict):
                                if function.get("name") and not current["function"]["name"]:
                                    current["function"]["name"] = str(function["name"])
                                args = function.get("arguments")
                                if args:
                                    current["function"]["arguments"] += (
                                        args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                                    )
                            started = True

                if stream_tool_calls:
                    yield "tool_calls", {
                        "tool_calls": [stream_tool_calls[key] for key in sorted(stream_tool_calls)]
                    }
                if last_usage:
                    yield "usage", {"usage": last_usage}
                yield "finish", {"finish_reason": finish_reason}
                return
            except StreamBusinessError:
                raise
            except UpstreamError as exc:
                if not started and _optional_parameter_error(exc) and index < len(variants) - 1:
                    break
                if not started and attempt < retries and _retryable_error(exc):
                    attempt += 1
                    delay = _retry_delay(attempt)
                    yield "status", {
                        "message": f"{exc.message}，{delay:g} 秒后重试（{attempt}/{retries}）…"
                    }
                    time.sleep(delay)
                    continue
                raise
            except (error.URLError, TimeoutError, OSError) as exc:
                wrapped = UpstreamError(f"读取模型响应失败：{exc}")
                if not started and attempt < retries:
                    attempt += 1
                    delay = _retry_delay(attempt)
                    yield "status", {
                        "message": f"等待响应超时，{delay:g} 秒后重试（{attempt}/{retries}）…"
                    }
                    time.sleep(delay)
                    continue
                raise wrapped from exc
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass


INDEX_HTML = r'''
<!doctype html>
<html lang="zh-CN" data-font-size="normal">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#090b0d">
<title>小小肥鱼</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous" onerror="this.onerror=null;this.href='https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.11/katex.min.css';">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%230b0e10'/%3E%3Cpath d='M32 11l4.8 15.2L52 31l-15.2 4.8L32 51l-4.8-15.2L12 31l15.2-4.8z' fill='%2378a9ff'/%3E%3C/svg%3E">
<style>
:root {
  color-scheme: dark;
  --bg: #090b0d;
  --bg-soft: #0c0f12;
  --panel: #101419;
  --panel-2: #151a21;
  --panel-3: #1a2028;
  --line: rgba(222, 232, 246, .075);
  --line-strong: rgba(222, 232, 246, .13);
  --text: #eef2f8;
  --text-soft: #c4cbd6;
  --muted: #8b94a3;
  --faint: #5d6675;
  --accent: #78a9ff;
  --accent-strong: #5b93f0;
  --accent-rgb: 120, 169, 255;
  --blue: #8fc6ff;
  --danger: #ff8290;
  --danger-bg: rgba(255, 100, 118, .09);
  --warning: #f1c77b;
  --sidebar-w: 270px;
  --radius-lg: 20px;
  --radius-md: 14px;
  --shadow: 0 20px 60px rgba(0, 0, 0, .34);
  --font: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", "Noto Sans SC", ui-sans-serif, sans-serif;
  --mono: "Cascadia Code", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
button, input, textarea, select { font: inherit; color: inherit; }
button { cursor: pointer; }
button:disabled { cursor: not-allowed; opacity: .48; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
::selection { background: rgba(120, 169, 255, .24); color: #fff; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, .12); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, .2); }
.hidden { display: none !important; }
:root { --reading-font-scale: 1; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }

.ambient {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(700px 480px at 13% -8%, rgba(120,169,255,.085), transparent 68%),
    radial-gradient(760px 600px at 90% 100%, rgba(78,144,180,.075), transparent 70%),
    linear-gradient(180deg, #090b0d 0%, #080a0c 100%);
}
.ambient::after {
  content: ""; position: absolute; inset: 0; opacity: .16;
  background-image: radial-gradient(rgba(255,255,255,.14) .5px, transparent .5px);
  background-size: 22px 22px;
  mask-image: linear-gradient(to bottom, rgba(0,0,0,.35), transparent 58%);
}

.app-shell { position: relative; z-index: 1; display: flex; height: 100dvh; }

/* ---------- sidebar ---------- */
.sidebar {
  width: var(--sidebar-w); flex: 0 0 var(--sidebar-w);
  display: flex; flex-direction: column; min-height: 0;
  padding: 20px 14px 15px;
  background: rgba(11, 14, 17, .78);
  border-right: 1px solid var(--line);
  backdrop-filter: blur(24px) saturate(1.12);
}
.brand { display: flex; align-items: center; gap: 11px; padding: 3px 8px 20px; }
.brand-mark {
  width: 38px; height: 38px; flex: 0 0 38px; display: grid; place-items: center;
  border-radius: 12px; color: #0a1424;
  background: linear-gradient(145deg, #d7e6ff 0%, var(--accent) 52%, #4f7fd6 100%);
  box-shadow: 0 0 0 1px rgba(120,169,255,.25), 0 8px 24px rgba(82,128,220,.12);
}
.brand-mark svg { width: 22px; height: 22px; }
.brand-name { font-size: 15px; font-weight: 650; letter-spacing: .01em; line-height: 1.3; }
.brand-sub { margin-top: 1px; color: var(--faint); font-family: var(--mono); font-size: 8.5px; letter-spacing: .22em; }
.nav-list { display: flex; flex-direction: column; gap: 3px; }
.nav-item {
  display: flex; align-items: center; gap: 10px; width: 100%; height: 41px;
  padding: 0 11px; border: 0; border-radius: 10px; background: transparent;
  color: var(--muted); font-size: 13.5px; text-align: left; transition: .18s ease;
}
.nav-item svg { width: 17px; height: 17px; opacity: .9; }
.nav-item:hover { color: var(--text-soft); background: rgba(255,255,255,.035); }
.nav-item.active { color: var(--text); background: rgba(120,169,255,.085); }
.nav-item.active svg { color: var(--accent); }
.nav-divider { height: 1px; margin: 16px 9px 14px; background: var(--line); }
.new-chat {
  display: flex; align-items: center; justify-content: center; gap: 7px; height: 40px;
  border: 1px solid rgba(120,169,255,.24); border-radius: 11px;
  background: rgba(120,169,255,.075); color: var(--accent);
  font-size: 13px; font-weight: 600; letter-spacing: .01em; transition: .18s ease;
}
.new-chat:hover { background: rgba(120,169,255,.13); border-color: rgba(120,169,255,.38); transform: translateY(-1px); }
.new-chat svg { width: 15px; height: 15px; }
.history-search-btn { margin-left: auto; width: 26px; height: 26px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 7px; background: rgba(255,255,255,.02); color: var(--faint); }
.history-search-btn:hover { color: var(--accent); border-color: rgba(120,169,255,.3); background: rgba(120,169,255,.08); }
.history-search-btn svg { width: 13px; height: 13px; }
.history-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 10px 9px; color: var(--faint); font-size: 10px; font-weight: 650;
  letter-spacing: .14em; text-transform: uppercase;
}
.history-count { font-family: var(--mono); font-size: 9px; letter-spacing: 0; }
.history-list { flex: 1; min-height: 0; overflow-y: auto; margin: 0 -4px; padding: 0 4px 12px; }
.history-group-label { padding: 14px 9px 6px; color: var(--faint); font-size: 10.5px; letter-spacing: .04em; }
.history-group-label:first-child { padding-top: 3px; }
.history-item {
  position: relative; width: 100%; display: block; padding: 9px 58px 9px 11px;
  margin: 1px 0; border: 1px solid transparent; border-radius: 10px;
  background: transparent; text-align: left; transition: .15s ease;
}
.history-item:hover { background: rgba(255,255,255,.036); }
.history-item.active { background: rgba(255,255,255,.055); border-color: var(--line); }
.history-item.active::before {
  content: ""; position: absolute; left: 0; top: 13px; bottom: 13px; width: 2px;
  border-radius: 2px; background: var(--accent);
}
.history-title { display: block; overflow: hidden; color: var(--text-soft); font-size: 12.5px; line-height: 1.5; white-space: nowrap; text-overflow: ellipsis; }
.history-live { display: inline-block; width: 5px; height: 5px; margin-left: 5px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 7px rgba(120,169,255,.6); animation: history-live-pulse 1.2s infinite ease-in-out; vertical-align: middle; }
@keyframes history-live-pulse { 50% { opacity: .3; } }
.history-item.active .history-title { color: var(--text); }
.history-meta { display: block; margin-top: 2px; color: var(--faint); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.history-archive, .history-edit {
  position: absolute; top: 50%; transform: translateY(-50%);
  width: 25px; height: 25px; display: grid; place-items: center;
  border: 0; border-radius: 7px; background: transparent; color: var(--faint);
  opacity: 0; transition: .15s ease;
}
.history-archive { right: 5px; }
.history-edit { right: 29px; }
.history-item:hover .history-archive, .history-item.active .history-archive,
.history-item:hover .history-edit, .history-item.active .history-edit { opacity: 1; }
.history-archive:hover { color: var(--accent); background: rgba(120,169,255,.1); }
.history-edit:hover { color: var(--text); background: rgba(255,255,255,.06); }
.history-archive svg, .history-edit svg { width: 13px; height: 13px; }
.history-empty { padding: 10px 10px; color: var(--faint); font-size: 11.5px; line-height: 1.8; }
.sidebar-foot {
  display: flex; align-items: center; gap: 7px; min-height: 31px; padding: 10px 9px 0;
  border-top: 1px solid var(--line); color: var(--faint);
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .03em;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px rgba(120,169,255,.55); }
.status-dot.offline { background: var(--danger); box-shadow: 0 0 8px rgba(255,100,118,.5); }

/* ---------- main / topbar ---------- */
.main { flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.topbar {
  position: relative; z-index: 6; flex: 0 0 66px; display: flex; align-items: center;
  justify-content: space-between; gap: 20px; padding: 0 27px;
  border-bottom: 1px solid var(--line); background: rgba(9, 11, 13, .55);
  backdrop-filter: blur(18px);
}
.topbar-left { min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.eyebrow { color: var(--faint); font-family: var(--mono); font-size: 8.5px; letter-spacing: .18em; line-height: 1.45; text-transform: uppercase; }
.page-title { max-width: min(38vw, 460px); overflow: hidden; color: var(--text); font-size: 14px; font-weight: 600; line-height: 1.45; white-space: nowrap; text-overflow: ellipsis; }
.topbar-actions { display: flex; align-items: center; gap: 7px; min-width: 0; }
.select-shell {
  display: flex; align-items: center; gap: 7px; height: 36px; padding: 0 10px;
  border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.026);
  color: var(--faint); font-size: 9.5px; letter-spacing: .03em; transition: .15s ease;
}
.select-shell:hover, .select-shell:focus-within { border-color: var(--line-strong); background: rgba(255,255,255,.04); }
.select-shell > span { flex: 0 0 auto; }
.select-shell select {
  max-width: 172px; border: 0; outline: none; background: transparent;
  color: var(--text-soft); font-size: 11.5px; letter-spacing: 0; cursor: pointer;
}
.select-shell select option { background: #14191f; color: var(--text); }
#effortSelect { width: 54px; }
.icon-btn {
  width: 36px; height: 36px; display: grid; place-items: center;
  border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.02);
  color: var(--muted); transition: .15s ease;
}
.icon-btn:hover { color: var(--text); border-color: var(--line-strong); background: rgba(255,255,255,.045); }
.icon-btn svg { width: 17px; height: 17px; }
.mobile-only { display: none; }

.page { flex: 1; min-height: 0; }
.chat-page { position: relative; display: flex; flex-direction: column; }
.chat-page.has-turn-nav .chat-scroll { padding-right: 30px; }
.turn-nav { position: absolute; z-index: 8; top: 18px; right: 7px; width: 26px; max-height: min(62vh, 620px); display: flex; flex-direction: column; padding: 6px 4px; overflow: hidden; border: 1px solid transparent; border-radius: 14px; background: transparent; box-shadow: none; backdrop-filter: blur(0); transition: width .18s ease, padding .18s ease, background .18s ease, border-color .18s ease, box-shadow .18s ease; }
.turn-nav:hover, .turn-nav:focus-within { width: 226px; padding: 12px; border-color: rgba(255,255,255,.1); background: rgba(17,21,26,.9); box-shadow: 0 18px 50px rgba(0,0,0,.34); backdrop-filter: blur(16px); }
.turn-nav-head { display: none; justify-content: space-between; gap: 8px; padding: 0 3px 8px; color: var(--text-soft); font-size: 10px; font-weight: 600; border-bottom: 1px solid var(--line); }
.turn-nav:hover .turn-nav-head, .turn-nav:focus-within .turn-nav-head { display: flex; }
.turn-nav-count { color: var(--faint); font-family: var(--mono); font-size: 8.5px; font-weight: 400; }
.turn-nav-list { display: flex; flex-direction: column; align-items: center; gap: 5px; margin-top: 0; overflow-y: auto; scrollbar-width: none; }
.turn-nav-list::-webkit-scrollbar { display: none; }
.turn-nav:hover .turn-nav-list, .turn-nav:focus-within .turn-nav-list { align-items: stretch; gap: 3px; margin-top: 7px; scrollbar-width: thin; }
.turn-nav-item { width: 18px; height: 18px; display: grid; place-items: center; padding: 0; border: 0; border-left: 2px solid transparent; border-radius: 50%; background: transparent; color: var(--muted); text-align: left; font-size: 10.5px; line-height: 1.45; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.turn-nav-dot { width: 5px; height: 5px; border-radius: 50%; background: rgba(180,190,205,.42); transition: .15s ease; }
.turn-nav-text { display: none; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.turn-nav:hover .turn-nav-item, .turn-nav:focus-within .turn-nav-item { width: 100%; height: auto; min-height: 28px; display: block; padding: 7px 7px 7px 10px; border-radius: 7px; }
.turn-nav:hover .turn-nav-dot, .turn-nav:focus-within .turn-nav-dot { display: none; }
.turn-nav:hover .turn-nav-text, .turn-nav:focus-within .turn-nav-text { display: block; }
.turn-nav-item:hover { color: var(--text-soft); background: rgba(255,255,255,.045); }
.turn-nav-item.active { color: var(--accent); border-left-color: var(--accent); background: rgba(120,169,255,.08); }
.turn-nav-item.active .turn-nav-dot { width: 7px; height: 7px; background: var(--accent); box-shadow: 0 0 0 3px rgba(120,169,255,.12); }
@media (max-width: 760px) and (hover: none) { .turn-nav { display: none !important; } }
.chat-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 24px 30px 0; scroll-behavior: smooth; }
.message-list { width: min(100%, 880px); margin: 0 auto; padding: 0 0 42px; }

/* ---------- empty state ---------- */
.empty-state { width: min(100%, 780px); min-height: 100%; margin: 0 auto; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px 16px 70px; text-align: center; }
.hero-mark {
  position: relative; width: 66px; height: 66px; display: grid; place-items: center;
  margin-bottom: 22px; border: 1px solid rgba(120,169,255,.2); border-radius: 21px;
  color: var(--accent); background: radial-gradient(circle at 42% 35%, rgba(120,169,255,.17), rgba(120,169,255,.035));
  box-shadow: 0 0 50px rgba(120,169,255,.08), inset 0 0 20px rgba(120,169,255,.04);
}
.hero-mark::after { content: ""; position: absolute; inset: -12px; border: 1px solid rgba(120,169,255,.055); border-radius: 29px; }
.hero-mark svg { width: 32px; height: 32px; }
.hero-title { margin: 0; color: var(--text); font-size: clamp(23px, 3vw, 31px); font-weight: 570; letter-spacing: -.035em; line-height: 1.3; }
.hero-title em { color: var(--accent); font-style: normal; font-weight: 620; }
.hero-sub { max-width: 530px; margin: 12px 0 0; color: var(--muted); font-size: 13px; line-height: 1.8; }
.suggestions { width: min(100%, 680px); display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin-top: 32px; }
.suggestion {
  min-height: 93px; padding: 14px 14px 13px; border: 1px solid var(--line); border-radius: 13px;
  background: linear-gradient(155deg, rgba(255,255,255,.033), rgba(255,255,255,.012));
  text-align: left; transition: .18s ease;
}
.suggestion:hover { border-color: rgba(120,169,255,.21); background: linear-gradient(155deg, rgba(120,169,255,.055), rgba(255,255,255,.015)); transform: translateY(-2px); }
.suggestion svg { width: 15px; height: 15px; color: var(--accent); opacity: .8; }
.suggestion strong { display: block; margin: 8px 0 2px; color: var(--text-soft); font-size: 12px; font-weight: 580; }
.suggestion span { display: block; color: var(--faint); font-size: 10.5px; line-height: 1.5; }

/* ---------- messages ---------- */
.message { position: relative; margin: 0 0 30px; animation: message-in .28s ease both; }
@keyframes message-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.message.user { display: flex; flex-direction: column; align-items: flex-end; }
.user-bubble {
  max-width: min(80%, 650px); padding: 11px 15px; border: 1px solid rgba(255,255,255,.085);
  border-radius: 16px 16px 4px 16px; background: #171c22; color: #e8edf5;
  font-size: 13.5px; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere;
}
.user-images { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 7px; }
.user-image { display: block; max-width: 220px; max-height: 220px; min-width: 74px; min-height: 74px; object-fit: cover; border-radius: 10px; border: 1px solid rgba(255,255,255,.12); cursor: zoom-in; }
.user-text { white-space: pre-wrap; }
.user-files { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 7px; }
.user-file { display: inline-flex; align-items: center; gap: 6px; max-width: 100%; padding: 5px 8px; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.03); color: var(--text-soft); text-align: left; font-size: 11px; }
.user-file:hover { border-color: var(--line-strong); background: rgba(255,255,255,.06); }
.user-file svg { flex: 0 0 13px; width: 13px; height: 13px; color: var(--accent); }
.user-file span { max-width: 180px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.user-file small { color: var(--faint); font-family: var(--mono); font-size: 8.5px; white-space: nowrap; }
.msg-time { margin-top: 6px; color: var(--faint); font-family: var(--mono); font-size: 9px; letter-spacing: .02em; }
.message.assistant { display: grid; grid-template-columns: 29px minmax(0, 1fr); gap: 12px; }
.assistant-avatar {
  width: 29px; height: 29px; display: grid; place-items: center; margin-top: 1px;
  border: 1px solid rgba(120,169,255,.19); border-radius: 9px;
  background: rgba(120,169,255,.07); color: var(--accent);
}
.assistant-avatar svg { width: 16px; height: 16px; }
.assistant-body { min-width: 0; }
.assistant-head { display: flex; align-items: baseline; gap: 9px; min-height: 24px; margin-bottom: 5px; }
.assistant-name { color: var(--text-soft); font-size: 12px; font-weight: 600; letter-spacing: .01em; }
.assistant-model { color: var(--faint); font-family: var(--mono); font-size: 9px; }
.thinking-indicator { display: inline-flex; align-items: center; gap: 4px; padding: 7px 0; color: var(--muted); font-size: 12px; }
.thinking-indicator i { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); opacity: .35; animation: thinking 1.15s infinite ease-in-out; }
.thinking-indicator i:nth-child(2) { animation-delay: .14s; }
.thinking-indicator i:nth-child(3) { animation-delay: .28s; }
@keyframes thinking { 0%, 80%, 100% { transform: translateY(0); opacity: .25; } 40% { transform: translateY(-3px); opacity: .9; } }

.reasoning { margin: 5px 0 10px; border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.018); overflow: hidden; }
.reasoning summary {
  display: flex; align-items: center; gap: 7px; min-height: 33px; padding: 6px 11px;
  color: var(--muted); font-size: 10.5px; cursor: pointer; user-select: none; list-style: none;
}
.reasoning summary::-webkit-details-marker { display: none; }
.reasoning summary::after { content: ""; width: 5px; height: 5px; margin-left: auto; border-right: 1px solid currentColor; border-bottom: 1px solid currentColor; transform: rotate(45deg); transition: .18s ease; }
.reasoning[open] summary::after { transform: rotate(225deg); }
.reasoning summary:hover { color: var(--text-soft); }
.reasoning summary svg { width: 13px; height: 13px; color: var(--accent); opacity: .7; }
.reasoning-text {
  max-height: 300px; overflow-y: auto; padding: 10px 12px 12px;
  color: #8792a3; font-size: 11.5px; line-height: 1.8; white-space: pre-wrap; overflow-wrap: anywhere;
  border-top: 1px solid rgba(255,255,255,.04);
}
.markdown { color: var(--text-soft); font-size: 13.6px; line-height: 1.82; overflow-wrap: anywhere; }
.markdown > :first-child { margin-top: 0 !important; }
.markdown > :last-child { margin-bottom: 0 !important; }
.markdown p { margin: 0 0 12px; }
.markdown h1, .markdown h2, .markdown h3, .markdown h4 { margin: 24px 0 10px; color: var(--text); font-weight: 620; line-height: 1.4; letter-spacing: -.015em; }
.markdown h1 { font-size: 21px; } .markdown h2 { font-size: 18px; } .markdown h3 { font-size: 15.5px; } .markdown h4 { font-size: 14px; }
.markdown ul, .markdown ol { margin: 4px 0 13px; padding-left: 21px; }
.markdown li { margin: 3px 0; padding-left: 2px; }
.markdown li::marker { color: var(--accent); }
.markdown strong { color: #f2f5fa; font-weight: 620; }
.markdown em { color: #dde5f0; }
.markdown blockquote { margin: 11px 0; padding: 5px 13px; border-left: 2px solid rgba(120,169,255,.45); background: rgba(120,169,255,.035); color: var(--muted); }
.markdown hr { border: 0; border-top: 1px solid var(--line); margin: 18px 0; }
.table-wrap { margin: 12px 0 15px; overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.012); }
.markdown table { width: 100%; min-width: 420px; border-collapse: collapse; font-size: 12.5px; line-height: 1.65; }
.markdown th, .markdown td { padding: 8px 11px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
.markdown th:last-child, .markdown td:last-child { border-right: 0; }
.markdown tbody tr:last-child td { border-bottom: 0; }
.markdown th { background: rgba(120,169,255,.09); color: var(--text); font-weight: 600; }
.markdown tbody tr:nth-child(even) { background: rgba(255,255,255,.018); }
.markdown td { color: var(--text-soft); }
.markdown .align-center { text-align: center; }
.markdown .align-right { text-align: right; }
.markdown code { padding: 2px 5px; border-radius: 5px; background: rgba(255,255,255,.065); color: #d6e6ff; font-family: var(--mono); font-size: .89em; }
.math-inline { display: inline-block; vertical-align: middle; font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif; font-size: 1.03em; line-height: 1.6; }
.math-display { margin: 13px 0; padding: 10px 14px 12px; overflow-x: auto; text-align: center; background: rgba(120,169,255,.035); border: 1px solid rgba(120,169,255,.14); border-radius: 10px; font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif; font-size: 15px; line-height: 2.05; }
.math-frac { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 .18em; line-height: 1.25; }
.math-num { padding: 0 .2em .08em; border-bottom: 1px solid currentColor; }
.math-den { padding: .08em .2em 0; }
.math-box { display: inline-block; padding: .08em .35em; border: 1px solid rgba(120,169,255,.5); border-radius: 4px; background: rgba(120,169,255,.05); }
.math-sqrt { display: inline-flex; align-items: flex-start; vertical-align: middle; }
.math-radical-body { position: relative; display: inline-block; padding-left: 13px; }
.math-radical-hook { position: absolute; left: 0; top: 0; display: block; width: 14px; height: 100%; overflow: visible; }
.math-radicand { display: inline-block; padding: .12em .16em 0; border-top: 1px solid currentColor; margin-left: -1px; }
.math-root-index { position: relative; top: -.18em; z-index: 1; margin-right: -.28em; padding-top: .02em; font-size: .62em; line-height: 1; }
.math-underbrace, .math-overbrace { display: inline-flex; flex-direction: column; align-items: stretch; vertical-align: middle; margin: 0 .14em; }
.math-brace-content { display: block; text-align: center; }
.math-brace-label { display: block; color: var(--text-soft); font-family: var(--font); font-size: .68em; line-height: 1.25; text-align: center; white-space: nowrap; }
.math-brace-svg { display: block; width: 100%; height: 7px; overflow: visible; color: var(--accent); }
.math-overbrace .math-brace-svg { transform: scaleY(-1); }
.math-xarrow { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 .28em; min-width: 2.2em; line-height: 1.1; }
.math-xarrow-above, .math-xarrow-below { color: var(--text-soft); font-family: var(--font); font-size: .68em; line-height: 1.2; text-align: center; white-space: nowrap; }
.math-xarrow-line { position: relative; display: block; width: 100%; height: 8px; margin: 1px 0; }
.math-xarrow-line::before { content: ""; position: absolute; left: 0; right: 6px; top: 50%; border-top: 1.5px solid var(--accent); }
.math-xarrow-line::after { content: ""; position: absolute; right: 0; top: 50%; width: 6px; height: 6px; border-top: 1.5px solid var(--accent); border-right: 1.5px solid var(--accent); transform: translateY(-50%) rotate(45deg); }
.math-xarrow-left .math-xarrow-line::before { left: 6px; right: 0; }
.math-xarrow-left .math-xarrow-line::after { left: 0; right: auto; transform: translateY(-50%) rotate(-135deg); }
.math-op { display: inline-flex; align-items: center; vertical-align: middle; margin: 0 .1em 0 .04em; }
.math-op-symbol { color: var(--accent); font-size: 1.58em; line-height: .95; }
.math-op-limits { display: inline-flex; flex-direction: column; justify-content: space-between; min-height: 1.85em; margin-left: -.04em; font-size: .64em; line-height: 1; }
.math-op-upper { align-self: flex-start; padding-left: .05em; }
.math-op-lower { align-self: flex-start; padding-left: .05em; }
.math-op-stack { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 .14em; line-height: 1; }
.math-op-stack .math-op-upper, .math-op-stack .math-op-lower { font-size: .64em; }
.math-op-stack .math-op-symbol { font-size: 1.5em; }
.math-inline .math-op-symbol { font-size: 1.2em; }
.math-op-lim { display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 .12em; line-height: 1; }
.math-op-lim .math-op-symbol { color: var(--text-soft); font-size: 1em; font-style: normal; }
.math-op-lim .math-op-lower { align-self: center; padding: 0; margin-top: 1px; font-size: .66em; }
.math-rel { margin: 0 .28em; }
.math-bin { margin: 0 .12em; }
.math-inline sub, .math-display sub, .math-inline sup, .math-display sup { font-size: .68em; line-height: 0; }
.math-text, .math-fn { font-family: var(--font); font-style: normal; }
.math-overline { text-decoration: overline; }
.math-line { display: block; height: .65em; }
.math-vec { position: relative; display: inline-block; }
.math-vec::after { content: "→"; position: absolute; top: -.62em; left: 0; right: 0; text-align: center; font-size: .72em; line-height: 1; }
.markdown pre { margin: 10px 0 14px; }
.markdown pre { margin: 10px 0 14px; }
.code-wrap { border: 1px solid var(--line); border-radius: 11px; overflow: hidden; background: #0a0d10; }
.code-head { display: flex; align-items: center; justify-content: space-between; min-height: 31px; padding: 0 10px 0 12px; border-bottom: 1px solid rgba(255,255,255,.055); background: rgba(255,255,255,.02); }
.code-lang { color: var(--faint); font-family: var(--mono); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }
.copy-code { border: 0; border-radius: 5px; background: transparent; color: var(--faint); font-size: 9.5px; padding: 3px 6px; }
.copy-code:hover { color: var(--text-soft); background: rgba(255,255,255,.05); }
.code-wrap pre { margin: 0; padding: 12px 13px; overflow-x: auto; }
.code-wrap code { padding: 0; background: transparent; color: #cdd8e8; font-size: 11.7px; line-height: 1.72; white-space: pre; }
.markdown.streaming::after { content: "▍"; display: inline-block; margin-left: 2px; color: var(--accent); animation: caret 1s steps(2, end) infinite; }
@keyframes caret { 50% { opacity: 0; } }
.message-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; min-height: 20px; margin-top: 8px; color: var(--faint); font-family: var(--mono); font-size: 9px; letter-spacing: .01em; }
.message-meta .dot { opacity: .4; }
.cache-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 6px; border: 1px solid rgba(255,255,255,.07); border-radius: 5px; color: var(--faint); background: rgba(255,255,255,.025); }
.cache-badge.hit { color: #a9c8ff; border-color: rgba(120,169,255,.17); background: rgba(120,169,255,.06); }
.cache-badge.dot-badge::before { content: ""; width: 4px; height: 4px; border-radius: 50%; background: currentColor; opacity: .8; }
.msg-action { border: 0; background: transparent; color: var(--faint); font-family: var(--mono); font-size: 9px; padding: 2px 4px; border-radius: 4px; }
.msg-action:hover { color: var(--text-soft); background: rgba(255,255,255,.05); }
.error-box { margin: 8px 0; padding: 10px 12px; border: 1px solid rgba(255,107,125,.2); border-radius: 9px; background: var(--danger-bg); color: #ffb6bf; font-size: 12px; line-height: 1.7; white-space: pre-wrap; }
.notice-line { margin: 5px 0 9px; color: var(--warning); font-size: 10.5px; opacity: .84; }
.search-sources { display: flex; flex-wrap: wrap; gap: 6px; margin: 9px 0 3px; }
.search-sources a { max-width: 320px; overflow: hidden; padding: 3px 8px; border: 1px solid rgba(120,169,255,.16); border-radius: 999px; background: rgba(120,169,255,.05); color: var(--muted); font-size: 9.5px; text-decoration: none; white-space: nowrap; text-overflow: ellipsis; }
.search-sources a:hover { color: var(--accent); border-color: rgba(120,169,255,.38); text-decoration: none; }
.source-more { display: inline-flex; align-items: center; }
.source-more > summary { list-style: none; cursor: pointer; padding: 3px 8px; border: 1px solid rgba(120,169,255,.16); border-radius: 999px; color: var(--accent); background: rgba(120,169,255,.05); font-size: 9.5px; }
.source-more > summary::-webkit-details-marker { display: none; }
.source-more-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.chat-search-modal { width: min(100%, 760px); max-height: min(90dvh, 760px); display: flex; flex-direction: column; }
.chat-search-results { display: flex; flex-direction: column; gap: 7px; min-height: 120px; overflow-y: auto; padding-right: 2px; }
.chat-search-result { display: flex; flex-direction: column; gap: 5px; width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.018); text-align: left; }
.chat-search-result:hover { border-color: rgba(120,169,255,.3); background: rgba(120,169,255,.055); }
.chat-search-result-head { display: flex; justify-content: space-between; gap: 12px; color: var(--text-soft); font-size: 11px; }
.chat-search-result-head small { flex: 0 0 auto; color: var(--faint); font-size: 9px; }
.chat-search-result-snippet { color: var(--muted); font-size: 10.5px; line-height: 1.6; }
.tool-events { display: flex; flex-direction: column; gap: 5px; margin: 4px 0 9px; }
.tool-events.hidden { display: none; }
.tool-event { display: flex; align-items: center; gap: 6px; padding: 5px 9px; border: 1px solid rgba(120,169,255,.16); border-radius: 8px; background: rgba(120,169,255,.04); color: var(--muted); font-size: 10px; line-height: 1.4; }
.tool-event svg { flex: 0 0 12px; width: 12px; height: 12px; color: var(--accent); }
.tool-event-name { flex: 0 0 auto; color: var(--text-soft); font-weight: 600; }
.tool-event-query { min-width: 0; overflow: hidden; color: var(--muted); white-space: nowrap; text-overflow: ellipsis; }
.tool-event-status { margin-left: auto; flex: 0 0 auto; color: var(--faint); font-family: var(--mono); font-size: 8.5px; }
.tool-event.searching .tool-event-status { color: var(--accent); }
.tool-event.error { border-color: rgba(255,100,118,.25); background: rgba(255,100,118,.05); }
.tool-event.error .tool-event-status { color: var(--danger); }

/* ---------- composer ---------- */
.composer-wrap { position: relative; flex: 0 0 auto; padding: 2px 28px 10px; background: linear-gradient(to top, var(--bg) 56%, rgba(9,11,13,0)); }
.composer { width: min(100%, 880px); margin: 0 auto; padding: 10px 12px 8px 16px; border: 1px solid var(--line-strong); border-radius: 18px; background: rgba(18,22,27,.87); box-shadow: 0 14px 46px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.025); backdrop-filter: blur(18px); transition: .18s ease; }
.composer:focus-within { border-color: rgba(120,169,255,.34); box-shadow: 0 14px 46px rgba(0,0,0,.28), 0 0 0 3px rgba(120,169,255,.035); }
.composer textarea { display: block; width: 100%; min-height: 27px; max-height: 180px; padding: 0 4px 0 0; border: 0; outline: none; resize: none; background: transparent; color: var(--text); font-size: 13.5px; line-height: 1.7; }
.composer textarea::placeholder { color: #68736a; }
.composer-bottom { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; margin-top: 7px; }
.composer-tools { flex: 1; min-width: 0; display: flex; align-items: center; gap: 8px; }
.attach-btn { flex: 0 0 30px; width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.02); color: var(--faint); transition: .15s ease; }
.attach-btn:hover:not(:disabled) { color: var(--text-soft); border-color: var(--line-strong); background: rgba(255,255,255,.05); }
.attach-btn svg { width: 15px; height: 15px; }
.attach-btn.active { color: #0b1524; border-color: rgba(120,169,255,.6); background: var(--accent); }
.attach-btn.active svg { color: #0b1524; }
.settings-card { padding: 18px; border: 1px solid var(--line); border-radius: 15px; background: linear-gradient(150deg, rgba(255,255,255,.03), rgba(255,255,255,.012)); }
.settings-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; padding: 4px 0 14px; }
.settings-row strong { display: block; color: var(--text); font-size: 13px; font-weight: 600; }
.settings-row > div > span { display: block; margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.6; }
.switch { position: relative; flex: 0 0 42px; width: 42px; height: 24px; margin-top: 1px; }
.switch input { position: absolute; opacity: 0; width: 0; height: 0; }
.switch span { position: absolute; inset: 0; border: 1px solid var(--line-strong); border-radius: 999px; background: rgba(255,255,255,.05); transition: .18s ease; }
.switch span::after { content: ""; position: absolute; top: 3px; left: 3px; width: 16px; height: 16px; border-radius: 50%; background: #8d98a8; transition: .18s ease; }
.switch input:checked + span { background: rgba(120,169,255,.22); border-color: rgba(120,169,255,.5); }
.switch input:checked + span::after { transform: translateX(18px); background: var(--accent); }
.settings-hint { margin: 6px 0 14px; color: var(--faint); font-size: 10.5px; line-height: 1.7; }
.settings-card .field { margin-top: 14px; }
.memory-model-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.memory-model-row select { width: 100%; }
.memory-review-prompt-textarea { min-height: 190px; max-height: 42vh; }
.memory-prompt-details { margin-top: 14px; border: 1px solid var(--line); border-radius: 11px; background: rgba(255,255,255,.018); overflow: hidden; }
.memory-prompt-details > summary { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 11px 13px; color: var(--text-soft); font-size: 10.5px; font-weight: 600; cursor: pointer; list-style: none; }
.memory-prompt-details > summary::-webkit-details-marker { display: none; }
.memory-prompt-toggle { color: var(--faint); font-family: var(--mono); font-size: 8.5px; font-weight: 400; }
.memory-prompt-details[open] > summary { border-bottom: 1px solid var(--line); background: rgba(120,169,255,.035); }
.memory-prompt-details[open] .memory-prompt-toggle { font-size: 0; }
.memory-prompt-details[open] .memory-prompt-toggle::after { content: "收起"; font-size: 8.5px; }
.memory-prompt-body { padding: 12px 13px 13px; }
.memory-prompt-body .field { margin-top: 0; }
.memory-prompt-detail-actions { display: flex; justify-content: flex-end; }
.memory-actions { display: flex; gap: 8px; margin-top: 16px; }
.memory-summary-card { margin-top: 18px; padding: 16px; border: 1px solid var(--line); border-radius: 15px; background: rgba(255,255,255,.018); }
.memory-section-title { margin: 4px 0 8px; color: var(--text); font-size: 12px; font-weight: 600; }
.memory-section-title span { color: var(--faint); font-family: var(--mono); font-size: 9px; font-weight: 400; }
.memory-profile-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 4px 0 8px; }
.memory-profile-head .memory-section-title { margin: 0; }
.memory-profile { width: 100%; min-height: 150px; max-height: 360px; margin: 0 0 8px; overflow: auto; padding: 11px 12px; border: 1px solid var(--line); border-radius: 9px; outline: none; resize: vertical; box-sizing: border-box; background: #0d1115; color: var(--text-soft); font: 11.5px/1.75 var(--font); white-space: pre-wrap; }
.memory-profile:focus { border-color: rgba(120,169,255,.45); box-shadow: 0 0 0 3px rgba(120,169,255,.08); }
#memoryProfileCount.over-limit { color: var(--danger); }
#memoryIndexSummaryCount.over-limit { color: var(--danger); }
.memory-index-list { display: flex; flex-direction: column; gap: 7px; }
.memory-index-item { padding: 9px 11px; border: 1px solid var(--line); border-radius: 9px; background: rgba(255,255,255,.018); }
.memory-index-title { color: var(--text-soft); font-size: 11.5px; font-weight: 600; }
.memory-index-summary { margin-top: 3px; color: var(--muted); font-size: 10.5px; line-height: 1.65; }
.memory-index-meta { margin-top: 4px; color: var(--faint); font-family: var(--mono); font-size: 8.5px; }
.memory-index-actions { display: flex; justify-content: flex-end; margin-top: 8px; }
.memory-index-actions .secondary-btn { min-height: 28px; padding: 0 9px; font-size: 10px; }
.memory-index-textarea { min-height: 220px; }
.memory-index-sync-title { margin: 2px 0 14px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.02); color: var(--muted); font-size: 10.5px; }
.memory-index-sync-title strong { color: var(--text-soft); font-weight: 600; }
.attach-btn.active { color: #0b1524; border-color: rgba(120,169,255,.6); background: var(--accent); }
.attach-btn.active svg { color: #0b1524; }
.attachment-strip { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 2px 10px; }
.attachment-item { position: relative; width: 62px; height: 62px; border: 1px solid var(--line-strong); border-radius: 10px; overflow: hidden; background: #0a0d10; }
.attachment-item img { width: 100%; height: 100%; object-fit: cover; display: block; cursor: zoom-in; }
.attachment-remove { position: absolute; top: 3px; right: 3px; width: 18px; height: 18px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,.18); border-radius: 50%; background: rgba(8,11,13,.78); color: #fff; }
.attachment-remove:hover { background: rgba(255,100,118,.85); }
.attachment-remove svg { width: 9px; height: 9px; }
.attachment-file { position: relative; display: flex; align-items: center; gap: 7px; height: 36px; max-width: 270px; padding: 0 8px; border: 1px solid var(--line-strong); border-radius: 9px; background: #0a0d10; color: var(--accent); }
.attachment-file > svg { flex: 0 0 14px; width: 14px; height: 14px; }
.attachment-file-name { min-width: 0; max-width: 160px; overflow: hidden; color: var(--text-soft); font-size: 11px; white-space: nowrap; text-overflow: ellipsis; }
.attachment-file-meta { flex: 0 0 auto; color: var(--faint); font-family: var(--mono); font-size: 8.5px; }
.attachment-file-remove { flex: 0 0 18px; width: 18px; height: 18px; display: grid; place-items: center; border: 0; border-radius: 50%; background: transparent; color: var(--faint); }
.attachment-file-remove:hover { color: var(--danger); background: rgba(255,100,118,.1); }
.attachment-file-remove svg { width: 9px; height: 9px; }
.composer.dragging { border-color: rgba(120,169,255,.5); box-shadow: 0 0 0 3px rgba(120,169,255,.06); }
.composer-hint { min-width: 0; color: var(--faint); font-family: var(--mono); font-size: 9px; line-height: 1.5; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.composer-hint kbd { padding: 0 3px; border: 1px solid var(--line); border-radius: 3px; font-family: var(--mono); font-size: 8.5px; }
.send-btn { flex: 0 0 34px; width: 34px; height: 34px; display: grid; place-items: center; border: 0; border-radius: 10px; background: var(--accent); color: #0b1524; transition: .16s ease; }
.send-btn:hover:not(:disabled) { background: #9cc4ff; transform: translateY(-1px); box-shadow: 0 6px 17px rgba(120,169,255,.14); }
.send-btn svg { width: 16px; height: 16px; }
.send-btn.loading svg { animation: send-pulse 1.1s infinite ease-in-out; }
@keyframes send-pulse { 50% { opacity: .35; transform: scale(.88); } }
.stop-btn { flex: 0 0 34px; width: 34px; height: 34px; display: grid; place-items: center; border: 1px solid rgba(255,100,118,.28); border-radius: 10px; background: rgba(255,100,118,.08); color: var(--danger); transition: .16s ease; }
.stop-btn:hover { background: rgba(255,100,118,.14); }
.stop-btn span { width: 10px; height: 10px; border-radius: 2px; background: currentColor; }
.turn-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; min-height: 22px; margin-top: 7px; }
.variant-bar { display: inline-flex; align-items: center; gap: 4px; height: 24px; padding: 0 4px; border: 1px solid var(--line); border-radius: 7px; background: rgba(255,255,255,.02); color: var(--faint); font-family: var(--mono); font-size: 9px; }
.variant-btn { width: 20px; height: 20px; display: grid; place-items: center; border: 0; border-radius: 5px; background: transparent; color: var(--muted); font-size: 14px; line-height: 1; }
.variant-btn:hover:not(:disabled) { color: var(--text); background: rgba(255,255,255,.06); }
.variant-btn:disabled { opacity: .35; }
.turn-action-btn { border: 0; border-radius: 4px; background: transparent; color: var(--faint); font-family: var(--mono); font-size: 9px; padding: 2px 4px; }
.turn-action-btn:hover:not(:disabled) { color: var(--text-soft); background: rgba(255,255,255,.05); }
.context-bar {
  flex: 0 0 auto; display: flex; align-items: center; gap: 5px;
  margin-left: auto; margin-right: 2px; padding: 2px 7px;
  border: 1px solid rgba(222,232,246,.06); border-radius: 999px;
  background: rgba(255,255,255,.022);
  color: var(--faint); font-family: var(--mono); font-size: 9px; line-height: 1.45;
  white-space: nowrap; transition: border-color .15s ease, background .15s ease;
}
.context-bar.warn { border-color: rgba(255,186,92,.3); background: rgba(255,186,92,.07); color: #ffca85; }
.context-bar.over { border-color: rgba(255,100,118,.34); background: rgba(255,100,118,.07); color: #ffb0ba; }
.context-bar.hidden { display: none; }
.context-bar.warn { border-color: rgba(255, 186, 92, .3); background: rgba(255, 186, 92, .06); color: #ffca85; }
.context-bar.over { border-color: rgba(255, 100, 118, .34); background: rgba(255, 100, 118, .07); color: #ffb0ba; }
.context-text { flex: 1; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.context-text b { color: var(--text-soft); font-weight: 600; }
.context-track { flex: 0 0 26px; height: 2px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,.12); }
.context-fill { display: block; height: 100%; width: 0; border-radius: 999px; background: var(--accent); transition: width .3s ease; }
.context-bar.warn .context-fill { background: #ffba5c; }
.context-bar.over .context-fill { background: var(--danger); }
.context-btn {
  flex: 0 0 auto; height: 17px; padding: 0 6px; border: 1px solid rgba(120,169,255,.28);
  border-radius: 999px; background: rgba(120,169,255,.1); color: var(--accent);
  font-family: var(--font); font-size: 9px; line-height: 1; white-space: nowrap;
}
.context-btn:hover:not(:disabled) { background: rgba(120,169,255,.18); }
.context-btn:disabled { opacity: .45; cursor: default; }
.context-jump {
  flex: 0 0 auto; height: 28px; padding: 0 11px; border: 1px solid var(--line-strong);
  border-radius: 999px; background: rgba(16,20,25,.94); color: var(--text-soft);
  font-size: 10.5px; box-shadow: 0 8px 24px rgba(0,0,0,.32); backdrop-filter: blur(12px);
}
.context-jump:hover { color: var(--text); border-color: rgba(120,169,255,.36); }
.jump-wrap {
  position: absolute; left: 50%; bottom: calc(100% + 10px); z-index: 5;
  transform: translateX(-50%);
}
.jump-wrap.hidden { display: none; }

/* ---------- settings ---------- */
.settings-page { overflow-y: auto; }
.settings-scroll { width: min(100%, 1000px); margin: 0 auto; padding: 46px 34px 60px; }
.settings-tabs { display: inline-flex; gap: 4px; margin-bottom: 26px; padding: 4px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.02); }
.settings-tab { display: inline-flex; align-items: center; gap: 6px; height: 34px; padding: 0 14px; border: 0; border-radius: 8px; background: transparent; color: var(--muted); font-size: 12px; transition: .15s ease; }
.settings-tab:hover { color: var(--text-soft); background: rgba(255,255,255,.04); }
.settings-tab.active { color: var(--text); background: rgba(120,169,255,.13); }
.tab-count { min-width: 18px; height: 18px; padding: 0 5px; display: inline-grid; place-items: center; border-radius: 999px; background: rgba(255,255,255,.08); color: var(--text-soft); font-family: var(--mono); font-size: 9px; line-height: 1; }
.prompt-picker-shell select { max-width: 142px; }
.prompt-lock { max-width: 176px; flex: 0 1 auto; min-width: 0; cursor: default; }
.prompt-lock span.prompt-lock-label { flex: 0 0 auto; color: var(--faint); }
.prompt-lock .prompt-lock-name { min-width: 0; overflow: hidden; color: var(--text-soft); font-size: 11.5px; white-space: nowrap; text-overflow: ellipsis; }
.prompt-lock .prompt-lock-icon { flex: 0 0 11px; width: 11px; height: 11px; color: var(--faint); opacity: .75; }
.prompt-list { display: flex; flex-direction: column; gap: 10px; }
.prompt-card { padding: 15px 16px; border: 1px solid var(--line); border-radius: 13px; background: linear-gradient(150deg, rgba(255,255,255,.028), rgba(255,255,255,.012)); }
.prompt-card.active { border-color: rgba(120,169,255,.34); background: rgba(120,169,255,.055); }
.prompt-card-head { display: flex; align-items: center; gap: 8px; }
.prompt-card-title { color: var(--text); font-size: 13px; font-weight: 620; }
.prompt-tag { padding: 2px 6px; border-radius: 5px; background: rgba(255,255,255,.06); color: var(--faint); font-size: 8.5px; }
.prompt-tag.custom { color: var(--accent); background: rgba(120,169,255,.1); }
.prompt-active-mark { margin-left: auto; color: var(--accent); font-size: 9.5px; }
.prompt-preview { max-height: 4.8em; margin-top: 8px; overflow: hidden; color: var(--muted); font-size: 11.5px; line-height: 1.65; white-space: pre-wrap; }
.prompt-card-actions { display: flex; gap: 6px; margin-top: 11px; }
.prompt-card-actions .secondary-btn { min-height: 29px; padding: 0 10px; font-size: 10.5px; }
.prompt-empty { padding: 36px 20px; border: 1px dashed var(--line-strong); border-radius: 13px; color: var(--faint); text-align: center; font-size: 12px; }
.archive-list { display: flex; flex-direction: column; gap: 9px; }
.archive-item { display: flex; align-items: center; gap: 16px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 13px; background: linear-gradient(150deg, rgba(255,255,255,.028), rgba(255,255,255,.012)); }
.archive-item:hover { border-color: var(--line-strong); }
.archive-item-main { flex: 1; min-width: 0; }
.archive-title { overflow: hidden; color: var(--text); font-size: 13px; font-weight: 600; white-space: nowrap; text-overflow: ellipsis; }
.archive-preview { margin-top: 3px; overflow: hidden; color: var(--muted); font-size: 11px; white-space: nowrap; text-overflow: ellipsis; }
.archive-meta { margin-top: 5px; color: var(--faint); font-family: var(--mono); font-size: 9px; }
.archive-actions { flex: 0 0 auto; display: flex; gap: 6px; }
.archive-actions .secondary-btn { min-height: 30px; padding: 0 10px; font-size: 11px; }
.archive-empty { padding: 40px 20px; border: 1px dashed var(--line-strong); border-radius: 13px; color: var(--faint); font-size: 12px; text-align: center; line-height: 1.9; }
.settings-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 27px; }
.settings-heading h1 { margin: 5px 0 6px; color: var(--text); font-size: 29px; font-weight: 590; letter-spacing: -.035em; line-height: 1.25; }
.settings-heading p { max-width: 610px; margin: 0; color: var(--muted); font-size: 12.5px; line-height: 1.8; }
.primary-btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 7px;
  min-height: 38px; padding: 0 15px; border: 1px solid rgba(120,169,255,.28);
  border-radius: 10px; background: var(--accent); color: #0b1524;
  font-size: 12px; font-weight: 650; white-space: nowrap; transition: .17s ease;
}
.primary-btn:hover:not(:disabled) { background: #9cc4ff; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(120,169,255,.14); }
.primary-btn svg { width: 14px; height: 14px; }
.secondary-btn {
  min-height: 36px; padding: 0 13px; border: 1px solid var(--line-strong); border-radius: 9px;
  background: rgba(255,255,255,.025); color: var(--text-soft); font-size: 11.5px; transition: .15s ease;
}
.secondary-btn:hover:not(:disabled) { background: rgba(255,255,255,.055); color: var(--text); }
.danger-btn { color: var(--danger); border-color: rgba(255,100,118,.18); background: rgba(255,100,118,.055); }
.danger-btn:hover:not(:disabled) { color: #ffc1c8; background: rgba(255,100,118,.1); }
.cache-tip {
  display: flex; align-items: flex-start; gap: 13px; margin-bottom: 26px; padding: 14px 16px;
  border: 1px solid rgba(120,169,255,.13); border-radius: 13px;
  background: linear-gradient(110deg, rgba(120,169,255,.055), rgba(120,169,255,.014));
}
.cache-tip-icon { flex: 0 0 29px; width: 29px; height: 29px; display: grid; place-items: center; border-radius: 8px; background: rgba(120,169,255,.1); color: var(--accent); }
.cache-tip-icon svg { width: 15px; height: 15px; }
.cache-tip-body { flex: 1; min-width: 0; }
.cache-tip-title { color: var(--text-soft); font-size: 11.5px; font-weight: 620; }
.cache-tip-text { margin-top: 2px; color: var(--muted); font-size: 10.5px; line-height: 1.65; }
.cache-tip-tag { flex: 0 0 auto; padding: 3px 7px; border: 1px solid rgba(120,169,255,.18); border-radius: 5px; color: var(--accent); font-family: var(--mono); font-size: 8px; letter-spacing: .08em; }
.provider-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 13px; }
.provider-card {
  position: relative; display: flex; flex-direction: column; min-height: 213px; padding: 17px;
  border: 1px solid var(--line); border-radius: 15px;
  background: linear-gradient(150deg, rgba(255,255,255,.032), rgba(255,255,255,.012));
  transition: .18s ease; overflow: hidden;
}
.provider-card::after { content: ""; position: absolute; width: 150px; height: 150px; right: -75px; top: -90px; border-radius: 50%; background: rgba(120,169,255,.045); filter: blur(20px); pointer-events: none; }
.provider-card:hover { border-color: var(--line-strong); background: linear-gradient(150deg, rgba(255,255,255,.045), rgba(255,255,255,.015)); }
.provider-top { display: flex; align-items: flex-start; gap: 11px; }
.provider-icon { flex: 0 0 37px; width: 37px; height: 37px; display: grid; place-items: center; border: 1px solid rgba(120,169,255,.16); border-radius: 10px; background: rgba(120,169,255,.065); color: var(--accent); font-family: var(--mono); font-size: 14px; font-weight: 700; text-transform: uppercase; }
.provider-info { flex: 1; min-width: 0; padding-top: 1px; }
.provider-name { display: flex; align-items: center; gap: 7px; color: var(--text); font-size: 13.5px; font-weight: 610; }
.provider-live { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 7px rgba(120,169,255,.55); }
.provider-url { margin-top: 3px; overflow: hidden; color: var(--faint); font-family: var(--mono); font-size: 9px; white-space: nowrap; text-overflow: ellipsis; }
.provider-actions { display: flex; gap: 3px; margin-left: auto; }
.mini-btn { width: 27px; height: 27px; display: grid; place-items: center; border: 1px solid transparent; border-radius: 7px; background: transparent; color: var(--faint); transition: .15s ease; }
.mini-btn:hover { color: var(--text); border-color: var(--line); background: rgba(255,255,255,.04); }
.mini-btn.danger:hover { color: var(--danger); border-color: rgba(255,100,118,.16); background: rgba(255,100,118,.06); }
.mini-btn svg { width: 13px; height: 13px; }
.mini-btn.spinning svg { animation: spin .8s linear infinite; }
.provider-stats { display: flex; align-items: center; gap: 10px; margin-top: 16px; color: var(--faint); font-size: 10px; }
.provider-stats strong { color: var(--text-soft); font-family: var(--mono); font-size: 10.5px; font-weight: 500; }
.model-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; min-height: 24px; }
.model-chip { max-width: 100%; overflow: hidden; padding: 3px 7px; border: 1px solid rgba(255,255,255,.075); border-radius: 5px; background: rgba(255,255,255,.028); color: var(--muted); font-family: var(--mono); font-size: 9px; white-space: nowrap; text-overflow: ellipsis; }
.model-chip.more { color: var(--accent); border-color: rgba(120,169,255,.14); background: rgba(120,169,255,.035); }
.provider-foot { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: auto; padding-top: 15px; border-top: 1px solid rgba(255,255,255,.045); color: var(--faint); font-size: 9.5px; }
.key-hint { display: inline-flex; align-items: center; gap: 5px; font-family: var(--mono); font-size: 9px; }
.key-hint svg { width: 11px; height: 11px; opacity: .7; }
.empty-providers { grid-column: 1 / -1; min-height: 270px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px; border: 1px dashed var(--line-strong); border-radius: 15px; text-align: center; }
.empty-providers-icon { width: 47px; height: 47px; display: grid; place-items: center; margin-bottom: 13px; border: 1px solid var(--line); border-radius: 14px; color: var(--faint); }
.empty-providers-icon svg { width: 22px; height: 22px; }
.empty-providers strong { color: var(--text-soft); font-size: 13px; font-weight: 600; }
.empty-providers p { max-width: 340px; margin: 5px 0 15px; color: var(--faint); font-size: 10.5px; line-height: 1.7; }

/* ---------- modal / toast ---------- */
.modal-backdrop { position: fixed; z-index: 50; inset: 0; display: grid; place-items: center; padding: 20px; background: rgba(3,5,6,.72); backdrop-filter: blur(9px); animation: fade-in .18s ease; }
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
.modal {
  width: min(100%, 455px); max-height: min(90dvh, 700px); overflow-y: auto;
  padding: 22px; border: 1px solid var(--line-strong); border-radius: 18px;
  background: #12171c; box-shadow: 0 30px 90px rgba(0,0,0,.6); animation: modal-in .22s cubic-bezier(.2,.8,.2,1) both;
}
@keyframes modal-in { from { opacity: 0; transform: translateY(8px) scale(.985); } to { opacity: 1; transform: none; } }
.modal-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; margin-bottom: 19px; }
.modal-title { margin: 0; color: var(--text); font-size: 17px; font-weight: 620; letter-spacing: -.02em; line-height: 1.35; }
.modal-subtitle { margin: 4px 0 0; color: var(--muted); font-size: 11px; line-height: 1.65; }
.modal-close { flex: 0 0 29px; width: 29px; height: 29px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,.02); color: var(--faint); }
.modal-close:hover { color: var(--text); background: rgba(255,255,255,.05); }
.modal-close svg { width: 14px; height: 14px; }
.image-viewer { position: relative; width: min(94vw, 1280px); height: min(90dvh, 900px); display: flex; align-items: center; justify-content: center; overflow: hidden; touch-action: none; }
.image-viewer img { display: block; max-width: 100%; max-height: 100%; border: 1px solid var(--line-strong); border-radius: 12px; box-shadow: 0 30px 90px rgba(0,0,0,.65); transform-origin: center center; cursor: zoom-in; user-select: none; -webkit-user-drag: none; }
.image-viewer.zoomed img { cursor: grab; }
.image-viewer.dragging img { cursor: grabbing; }
.image-viewer-hint { position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%); padding: 5px 10px; border: 1px solid rgba(255,255,255,.12); border-radius: 999px; background: rgba(8,11,13,.74); color: rgba(232,237,245,.7); font-size: 10px; white-space: nowrap; pointer-events: none; backdrop-filter: blur(8px); }
.image-viewer-close { position: absolute; top: 10px; right: 10px; width: 32px; height: 32px; display: grid; place-items: center; border: 1px solid rgba(255,255,255,.2); border-radius: 50%; background: rgba(8,11,13,.82); color: #fff; }
.image-viewer-close:hover { background: rgba(255,255,255,.14); }
.image-viewer-close svg { width: 14px; height: 14px; }
.file-modal { width: min(100%, 760px); max-height: 92dvh; display: flex; flex-direction: column; }
.file-preview-text { flex: 1; min-height: 180px; max-height: 62dvh; overflow: auto; margin: 0; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px; background: #0a0d10; color: var(--text-soft); font-family: var(--mono); font-size: 11px; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
.file-page-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 8px; max-height: 220px; overflow: auto; margin-bottom: 12px; }
.file-page-thumb { padding: 4px; border: 1px solid var(--line); border-radius: 8px; background: #0a0d10; color: var(--faint); font-size: 9px; text-align: center; }
.file-page-thumb:hover { border-color: rgba(120,169,255,.35); color: var(--text-soft); }
.file-page-thumb img { display: block; width: 100%; height: 88px; margin-bottom: 3px; border-radius: 5px; object-fit: cover; }
.field { display: block; margin-bottom: 14px; }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.field-label { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; color: var(--text-soft); font-size: 11.5px; font-weight: 560; }
.field-label span { color: var(--faint); font-size: 9.5px; font-weight: 400; }
.input-wrap { display: flex; align-items: center; height: 38px; border: 1px solid var(--line-strong); border-radius: 9px; background: #0d1115; transition: .15s ease; }
.input-wrap:focus-within { border-color: rgba(120,169,255,.36); box-shadow: 0 0 0 3px rgba(120,169,255,.03); }
.input-wrap input { width: 100%; min-width: 0; height: 100%; padding: 0 11px; border: 0; outline: none; background: transparent; color: var(--text); font-size: 12px; }
.input-wrap input::placeholder { color: #505b53; }
.modal-textarea { width: 100%; min-height: 150px; max-height: 50dvh; padding: 10px 11px; border: 1px solid var(--line-strong); border-radius: 9px; background: #0d1115; color: var(--text); font-size: 12.5px; line-height: 1.7; resize: vertical; outline: none; }
.modal-textarea:focus { border-color: rgba(120,169,255,.36); box-shadow: 0 0 0 3px rgba(120,169,255,.03); }
.input-action { flex: 0 0 31px; width: 31px; height: 28px; margin-right: 4px; display: grid; place-items: center; border: 0; border-radius: 6px; background: transparent; color: var(--faint); }
.input-action:hover { color: var(--text-soft); background: rgba(255,255,255,.05); }
.input-action svg { width: 14px; height: 14px; }
.field-hint { margin-top: 5px; color: var(--faint); font-size: 9.5px; line-height: 1.55; }
.form-error { margin: 2px 0 13px; padding: 9px 11px; border: 1px solid rgba(255,100,118,.18); border-radius: 8px; background: var(--danger-bg); color: #ffadb7; font-size: 10.5px; line-height: 1.6; white-space: pre-wrap; }
.modal-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 7px; }
.modal-actions .primary-btn { min-width: 128px; }
.spinner { width: 12px; height: 12px; border: 1.5px solid rgba(11,21,36,.28); border-top-color: #0b1524; border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.confirm-copy { margin: 0 0 19px; color: var(--muted); font-size: 12px; line-height: 1.75; }
.confirm-copy strong { color: var(--text-soft); }
.toast-root { position: fixed; z-index: 80; right: 20px; bottom: 20px; display: flex; flex-direction: column; gap: 7px; width: min(340px, calc(100vw - 40px)); pointer-events: none; }
.toast {
  display: flex; align-items: flex-start; gap: 9px; padding: 10px 12px;
  border: 1px solid var(--line-strong); border-radius: 10px; background: rgba(20,26,31,.96);
  color: var(--text-soft); box-shadow: 0 16px 44px rgba(0,0,0,.4);
  font-size: 11.5px; line-height: 1.55; animation: toast-in .22s ease both; backdrop-filter: blur(16px);
}
.toast.out { animation: toast-out .2s ease both; }
.toast svg { flex: 0 0 14px; width: 14px; height: 14px; margin-top: 1px; color: var(--accent); }
.toast.error { border-color: rgba(255,100,118,.2); }
.toast.error svg { color: var(--danger); }
@keyframes toast-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes toast-out { to { opacity: 0; transform: translateY(4px); } }

/* ---------- responsive ---------- */
@media (max-width: 980px) {
  .context-track { display: none; }
  .context-bar { font-size: 9px; }
  .sidebar { position: fixed; z-index: 30; left: 0; top: 0; bottom: 0; transform: translateX(-102%); transition: transform .24s cubic-bezier(.2,.8,.2,1); box-shadow: 20px 0 60px rgba(0,0,0,.3); }
  .sidebar.open { transform: none; }
  .mobile-only { display: grid; }
  .topbar { padding: 0 18px; }
  .topbar-left { flex-direction: row; align-items: center; gap: 10px; }
  .topbar-left .eyebrow { display: none; }
  .mobile-backdrop { position: fixed; z-index: 25; inset: 0; display: none; background: rgba(0,0,0,.5); backdrop-filter: blur(3px); }
  .mobile-backdrop.show { display: block; }
  .chat-scroll { padding-left: 20px; padding-right: 20px; }
  .composer-wrap { padding-left: 18px; padding-right: 18px; }
  .settings-scroll { padding: 35px 22px 50px; }
}
@media (max-width: 760px) {
  .topbar { flex: 0 0 auto; min-height: 64px; padding-top: 10px; padding-bottom: 10px; flex-wrap: wrap; gap: 8px; }
  .topbar-left { flex: 1 1 100%; }
  .topbar-actions { width: 100%; overflow-x: auto; padding-bottom: 1px; }
  .select-shell { flex: 1 1 auto; min-width: 0; height: 34px; padding: 0 8px; }
  .select-shell select { max-width: none; width: 100%; }
  #effortSelect { width: 100%; }
  .chat-scroll { padding-top: 23px; }
  .message-list { padding-bottom: 20px; }
  .suggestions { grid-template-columns: 1fr; gap: 7px; margin-top: 24px; }
  .suggestion { min-height: 0; }
  .empty-state { padding-bottom: 35px; }
  .hero-title { font-size: 25px; }
  .composer-wrap { padding: 8px 12px 11px; }
  .composer { border-radius: 15px; padding: 10px 9px 8px 13px; }
  .composer-hint { max-width: 68%; }
  .field-row { grid-template-columns: 1fr; }
  .settings-tabs { display: flex; width: 100%; }
  .settings-tab { flex: 1; justify-content: center; }
  .archive-item { align-items: flex-start; flex-direction: column; gap: 10px; }
  .archive-actions { width: 100%; }
  .archive-actions .secondary-btn { flex: 1; }
  .settings-heading { align-items: flex-start; flex-direction: column; gap: 16px; }
  .settings-heading h1 { font-size: 25px; }
  .provider-grid { grid-template-columns: 1fr; }
  .user-bubble { max-width: 88%; }
  .message.assistant { grid-template-columns: 25px minmax(0, 1fr); gap: 9px; }
  .assistant-avatar { width: 25px; height: 25px; border-radius: 8px; }
  .assistant-avatar svg { width: 14px; height: 14px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
}
/* KaTeX: professional formula layout, with the built-in lightweight renderer kept as fallback. */
.math-display > .katex-display { margin: 0; }
.math-display .katex-display { max-width: 100%; }
.math-display .katex-display > .katex { max-width: 100%; }
.math-inline > .katex { font-size: 1.06em; }
.math-display > .katex-display > .katex { font-size: 1.08em; }
.katex-error { color: var(--danger) !important; }
.katex .mord.text, .katex .mord.textord { font-family: var(--font); }
.font-stepper { display: inline-flex; align-items: center; gap: 7px; }
.font-step-btn { width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid var(--line-strong); border-radius: 8px; background: rgba(255,255,255,.03); color: var(--text-soft); font-size: 16px; line-height: 1; }
.font-step-btn:hover:not(:disabled) { color: var(--accent); border-color: rgba(120,169,255,.35); background: rgba(120,169,255,.08); }
.font-step-btn:disabled { opacity: .35; cursor: default; }
.font-stepper strong { min-width: 52px; color: var(--text-soft); font-family: var(--mono); font-size: 11px; text-align: center; }
.markdown { font-size: calc(13.6px * var(--reading-font-scale)); }
.user-bubble { font-size: calc(13.5px * var(--reading-font-scale)); }
.composer textarea { font-size: calc(13.5px * var(--reading-font-scale)); }
.history-title { font-size: calc(12.5px * var(--reading-font-scale)); }
.history-meta { font-size: calc(10px * var(--reading-font-scale)); }
</style>
</head>
<body>
<svg class="sr-only" aria-hidden="true" focusable="false">
  <symbol id="i-spark" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2.2l2.15 6.65L20.8 11l-6.65 2.15L12 19.8l-2.15-6.65L3.2 11l6.65-2.15L12 2.2z"/></symbol>
  <symbol id="i-chat" viewBox="0 0 24 24"><path d="M20 11.5a7.7 7.7 0 0 1-8 7.4 9.4 9.4 0 0 1-3.3-.6L4 20l1.7-4.2A7.1 7.1 0 0 1 4 11.5 7.7 7.7 0 0 1 12 4a7.7 7.7 0 0 1 8 7.5Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></symbol>
  <symbol id="i-settings" viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M8 14v6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="16" cy="7" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="8" cy="17" r="2" fill="none" stroke="currentColor" stroke-width="1.6"/></symbol>
  <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="i-trash" viewBox="0 0 24 24"><path d="M4 7h16M9 7V4.8c0-.4.4-.8.8-.8h4.4c.4 0 .8.4.8.8V7M6.5 7l.8 12.2c0 .4.4.8.8.8h7.8c.4 0 .8-.4.8-.8L17.5 7M10 11v5M14 11v5" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-edit" viewBox="0 0 24 24"><path d="m14.5 5.5 4 4M5 19l2.3-.4L19 7a2.1 2.1 0 0 0-3-3L4.4 15.7 4 19h1Z" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-key" viewBox="0 0 24 24"><circle cx="8" cy="15.5" r="3.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m10.5 13 8-8M16 7l2 2M14 9l1.5 1.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></symbol>
  <symbol id="i-close" viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></symbol>
  <symbol id="i-search" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="m15.3 15.3 4.2 4.2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="i-menu" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round"/></symbol>
  <symbol id="i-send" viewBox="0 0 24 24"><path d="M12 19V5M6.5 10.5 12 5l5.5 5.5" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3 5 6v5.4c0 4.3 2.9 7.9 7 9.6 4.1-1.7 7-5.3 7-9.6V6l-7-3Z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="m9 12 2 2 4-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-code" viewBox="0 0 24 24"><path d="m8.5 7-5 5 5 5M15.5 7l5 5-5 5M13.5 4l-3 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-bulb" viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M8.3 14.8A6 6 0 1 1 15.7 14.8c-.7.5-1.2 1.4-1.2 2.2h-5c0-.8-.5-1.7-1.2-2.2Z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-pen" viewBox="0 0 24 24"><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16v4ZM13.5 6.5l4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-alert" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M12 7.5v5.8M12 16.6v.1" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
  <symbol id="i-check" viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-eye" viewBox="0 0 24 24"><path d="M3 12s3.2-5.5 9-5.5S21 12 21 12s-3.2 5.5-9 5.5S3 12 3 12Z" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="12" cy="12" r="2.4" fill="none" stroke="currentColor" stroke-width="1.5"/></symbol>
  <symbol id="i-eye-off" viewBox="0 0 24 24"><path d="M3 3l18 18M10.2 6.8A9.8 9.8 0 0 1 12 6.5c5.8 0 9 5.5 9 5.5a16 16 0 0 1-2.2 2.9M6.4 7.8A16.8 16.8 0 0 0 3 12s3.2 5.5 9 5.5c1 0 2-.2 2.8-.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></symbol>
  <symbol id="i-copy" viewBox="0 0 24 24"><rect x="8" y="8" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" fill="none" stroke="currentColor" stroke-width="1.5"/></symbol>
  <symbol id="i-bolt" viewBox="0 0 24 24"><path d="M13.2 2 5 13h5.6L9.8 22 19 10.5h-5.9L13.2 2Z" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linejoin="round"/></symbol>
  <symbol id="i-image" viewBox="0 0 24 24"><rect x="3" y="4.5" width="18" height="15" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="8.5" cy="9.5" r="1.6" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="m4.5 17 5.2-5.2 3.4 3.4 2.4-2.4 4 4.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-globe" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4 12h16M12 3.8c2.1 2.3 3.1 5.1 3.1 8.2s-1 5.9-3.1 8.2c-2.1-2.3-3.1-5.1-3.1-8.2S9.9 6.1 12 3.8Z" fill="none" stroke="currentColor" stroke-width="1.4"/></symbol>
  <symbol id="i-globe" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M4 12h16M12 3.8c2.1 2.3 3.1 5.1 3.1 8.2s-1 5.9-3.1 8.2c-2.1-2.3-3.1-5.1-3.1-8.2S9.9 6.1 12 3.8Z" fill="none" stroke="currentColor" stroke-width="1.4"/></symbol>
  <symbol id="i-file" viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6V3Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M14 3v4h4M9 12h6M9 16h6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.4 6.7M20 4.5V11h-6.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
  <symbol id="i-archive" viewBox="0 0 24 24"><path d="M4 7.5h16V19a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19V7.5ZM3 4.5h18v3H3v-3ZM9.5 11.5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
</svg>
<div class="ambient"></div>
<div class="app-shell">
  <aside class="sidebar" id="sidebar">
    <div class="brand">
      <div class="brand-mark"><svg><use href="#i-spark"></use></svg></div>
      <div>
        <div class="brand-name">小小肥鱼</div>
        <div class="brand-sub">LOCAL AGENT</div>
      </div>
    </div>
    <nav class="nav-list" aria-label="主导航">
      <button class="nav-item active" id="navChat" type="button">
        <svg><use href="#i-chat"></use></svg><span>对话</span>
      </button>
      <button class="nav-item" id="navSettings" type="button">
        <svg><use href="#i-settings"></use></svg><span>设置</span>
      </button>
    </nav>
    <div class="nav-divider"></div>
    <button class="new-chat" id="newChatBtn" type="button">
      <svg><use href="#i-plus"></use></svg><span>新建对话</span>
    </button>
    <div class="history-head"><span>历史对话</span><span class="history-count" id="historyCount">0</span><button class="history-search-btn" id="chatSearchBtn" type="button" aria-label="搜索历史对话" title="搜索标题和原文"><svg><use href="#i-search"></use></svg></button></div>
    <div class="history-list" id="historyList"></div>
    <div class="sidebar-foot">
      <span class="status-dot" id="statusDot"></span>
      <span id="localStatus">本地运行 · 数据仅存本机</span>
    </div>
  </aside>
  <div class="mobile-backdrop" id="mobileBackdrop"></div>

  <main class="main">
    <header class="topbar">
      <div class="topbar-left">
        <button class="icon-btn mobile-only" id="menuBtn" type="button" aria-label="打开菜单"><svg><use href="#i-menu"></use></svg></button>
        <div>
          <div class="eyebrow" id="pageEyebrow">CONVERSATION</div>
          <div class="page-title" id="pageTitle">新对话</div>
        </div>
      </div>
      <div class="topbar-actions" id="chatControls">
        <label class="select-shell prompt-picker-shell" id="promptPickerShell"><span>提示词</span><select id="promptPicker" aria-label="选择系统提示词"></select></label>
        <div class="select-shell prompt-lock hidden" id="promptLock" role="status"><svg class="prompt-lock-icon"><use href="#i-shield"></use></svg><span class="prompt-lock-label">提示词</span><span class="prompt-lock-name" id="promptLockName"></span></div>
        <label class="select-shell"><span>供应商</span><select id="providerSelect" aria-label="选择供应商"></select></label>
        <label class="select-shell"><span>模型</span><select id="modelSelect" aria-label="选择模型"></select></label>
        <label class="select-shell"><span>思考强度</span><select id="effortSelect" aria-label="选择思考强度"></select></label>
      </div>
    </header>

    <section class="page chat-page" id="chatPage">
      <div class="chat-scroll" id="chatScroll">
        <div class="empty-state" id="emptyState">
          <div class="hero-mark"><svg><use href="#i-spark"></use></svg></div>
          <h1 class="hero-title">你好，我是<em>小小肥鱼</em></h1>
          <p class="hero-sub">一个在本地运行、只连接你自己的模型供应商的轻量 Agent。历史对话保存在项目文件夹里，刷新或重启都不会丢。</p>
          <div class="suggestions">
            <button class="suggestion" type="button" data-prompt="请解释下面这段代码的核心逻辑，并指出潜在问题：\n\n```\n在这里粘贴代码\n```">
              <svg><use href="#i-code"></use></svg><strong>看懂一段代码</strong><span>解释逻辑、边界与潜在问题</span>
            </button>
            <button class="suggestion" type="button" data-prompt="帮我把这个想法拆解成可执行的步骤：\n\n">
              <svg><use href="#i-bulb"></use></svg><strong>梳理一个思路</strong><span>把模糊的问题变成行动清单</span>
            </button>
            <button class="suggestion" type="button" data-prompt="帮我起草一份简洁、专业的邮件，主题是：\n\n">
              <svg><use href="#i-pen"></use></svg><strong>起草一点东西</strong><span>润色文案、邮件或内容提纲</span>
            </button>
          </div>
        </div>
        <div class="message-list" id="messageList"></div>
      </div>
      <nav class="turn-nav hidden" id="turnNav" aria-label="会话问题导航">
        <div class="turn-nav-head"><span>问题导航</span><span class="turn-nav-count" id="turnNavCount"></span></div>
        <div class="turn-nav-list" id="turnNavList"></div>
      </nav>
      <div class="composer-wrap">
        <div class="jump-wrap hidden" id="jumpWrap">
          <button class="context-jump" id="jumpBtn" type="button">↓ 回到最新</button>
        </div>
        <form class="composer" id="composerForm">
          <div class="attachment-strip hidden" id="attachmentStrip"></div>
          <textarea id="composerInput" rows="1" placeholder="输入消息，和模型聊聊…" autocomplete="off"></textarea>
          <div class="composer-bottom">
            <div class="composer-tools">
              <button class="attach-btn" id="fileBtn" type="button" aria-label="添加文件" title="添加 PDF / Word / 文本文件"><svg><use href="#i-file"></use></svg></button>
              <button class="attach-btn" id="attachBtn" type="button" aria-label="添加图片" title="添加图片（需模型支持视觉）"><svg><use href="#i-image"></use></svg></button>
              <button class="attach-btn" id="webSearchBtn" type="button" aria-label="联网搜索" title="联网搜索（需先配置 Firecrawl）"><svg><use href="#i-globe"></use></svg></button>
              <div class="composer-hint" id="composerHint"><kbd>Enter</kbd> 发送 <span class="sep">·</span> <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行</div>
            </div>
            <div class="context-bar hidden" id="contextBar">
              <span class="context-text" id="contextText"></span>
              <span class="context-track"><span class="context-fill" id="contextFill"></span></span>
              <button class="context-btn hidden" id="compressBtn" type="button">压缩</button>
            </div>
            <button class="stop-btn hidden" id="stopBtn" type="button" aria-label="停止生成"><span></span></button>
            <button class="send-btn" id="sendBtn" type="submit" aria-label="发送"><svg><use href="#i-send"></use></svg></button>
          </div>
          <input class="sr-only" id="imageInput" type="file" accept="image/*" multiple>
          <input class="sr-only" id="fileInput" type="file" accept=".pdf,.doc,.docx,.txt,.md,.markdown,.csv,.json" multiple>
        </form>
      </div>
    </section>

    <section class="page settings-page hidden" id="settingsPage">
      <div class="settings-scroll">
        <div class="settings-tabs" id="settingsTabs">
          <button class="settings-tab" type="button" data-settings-tab="general">通用</button>
          <button class="settings-tab active" type="button" data-settings-tab="models">模型配置</button>
          <button class="settings-tab" type="button" data-settings-tab="prompts">系统提示词</button>
          <button class="settings-tab" type="button" data-settings-tab="memory">记忆</button>
          <button class="settings-tab" type="button" data-settings-tab="web">联网搜索</button>
          <button class="settings-tab" type="button" data-settings-tab="archive">归档 <span class="tab-count" id="archiveCount">0</span></button>
        </div>
        <div id="generalSettingsPanel" class="hidden">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">GENERAL</div>
              <h1>通用设置</h1>
              <p>这里保存界面相关的个人偏好。当前字体设置只影响这个浏览器，不会修改对话内容。</p>
            </div>
          </div>
          <section class="settings-card">
            <div class="settings-row">
              <div><strong>对话字体大小</strong><span>调整聊天正文、用户消息和输入框的字号。</span></div>
              <div class="font-stepper" aria-label="调整对话字体大小">
                <button class="font-step-btn" id="generalFontMinus" type="button" aria-label="减小字体">−</button>
                <strong id="generalFontValue">100%</strong>
                <button class="font-step-btn" id="generalFontPlus" type="button" aria-label="增大字体">+</button>
              </div>
            </div>
            <div class="settings-hint">选择后会立即生效，并保存在当前浏览器本地。</div>
            <div class="memory-actions"><button class="secondary-btn" id="resetGeneralSettingsBtn" type="button">恢复默认外观</button></div>
          </section>
        </div>
        <div id="modelSettingsPanel">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">MODEL PROVIDERS</div>
              <h1>模型供应商</h1>
              <p>保存只写入设置，模型列表在供应商卡片上单独刷新。支持多个供应商，API Key 只保存在本地 config.json。</p>
            </div>
            <button class="primary-btn" id="addProviderBtn" type="button">
              <svg><use href="#i-plus"></use></svg><span>添加供应商</span>
            </button>
          </div>
          <div class="cache-tip">
            <div class="cache-tip-icon"><svg><use href="#i-bolt"></use></svg></div>
            <div class="cache-tip-body">
              <div class="cache-tip-title">上下文缓存已优化</div>
              <div class="cache-tip-text">系统提示词保持固定，历史消息不插入时间戳或随机内容，并按原始顺序持续追加。相同前缀更容易命中供应商的自动上下文缓存。</div>
            </div>
            <span class="cache-tip-tag">AUTO</span>
          </div>
          <div class="provider-grid" id="providerGrid"></div>
        </div>
        <div id="promptSettingsPanel" class="hidden">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">SYSTEM PROMPTS</div>
              <h1>系统提示词</h1>
              <p>新对话开始时选择一个提示词，之后会锁定在该对话中。修改预设不会改变已有对话，从而保持前缀缓存稳定。</p>
            </div>
            <button class="primary-btn" id="addPromptBtn" type="button">
              <svg><use href="#i-plus"></use></svg><span>新建提示词</span>
            </button>
          </div>
          <div class="prompt-list" id="promptList"></div>
        </div>
        <div id="memorySettingsPanel" class="hidden">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">LONG-TERM MEMORY</div>
              <h1>记忆</h1>
              <p>用户画像和最近 40 条对话索引会在新对话开始时冻结注入。历史细节可以在对话中由模型按需读取。</p>
            </div>
            <button class="secondary-btn" id="refreshMemoryBtn" type="button">刷新</button>
          </div>
          <section class="settings-card">
            <div class="settings-row">
              <div><strong>启用长期记忆</strong><span>关闭后不再注入记忆，也不向模型提供记忆工具；已有记忆文件会保留。</span></div>
              <label class="switch"><input type="checkbox" id="memoryEnabledToggle"><span></span></label>
            </div>
            <div class="memory-model-row">
              <label class="field"><span>整理记忆的供应商</span><select class="input" id="memoryProviderSelect"></select></label>
              <label class="field"><span>整理记忆的模型</span><select class="input" id="memoryModelSelect"></select></label>
            </div>
            <details class="memory-prompt-details">
              <summary><span>记忆整理提示词（画像与索引）</span><span class="memory-prompt-toggle">展开</span></summary>
              <div class="memory-prompt-body">
                <div class="settings-hint">这段提示词同时决定用户画像的分类整理，以及对话索引的摘要、关键词和价值判断。</div>
                <label class="field"><textarea class="modal-textarea memory-review-prompt-textarea" id="memoryReviewPrompt" maxlength="30000" spellcheck="false"></textarea></label>
                <div class="settings-hint">保存后从下一次记忆检查开始生效，不修改已有快照。</div>
                <div class="memory-prompt-detail-actions"><button class="secondary-btn" id="resetMemoryPromptBtn" type="button">恢复默认提示词</button></div>
              </div>
            </details>
            <div class="memory-actions">
              <button class="primary-btn" id="saveMemorySettingsBtn" type="button">保存记忆设置</button>
              <button class="secondary-btn" id="scanMemoryBtn" type="button">立即检查更新</button>
            </div>
          </section>
          <section class="memory-summary-card">
            <div class="memory-profile-head">
              <div class="memory-section-title">用户画像 <span id="memoryProfileCount"></span></div>
              <button class="secondary-btn" id="saveMemoryProfileBtn" type="button">保存画像</button>
            </div>
            <textarea class="memory-profile" id="memoryProfile" rows="7" maxlength="2500" spellcheck="false" placeholder="手动填写或修改长期用户画像……"></textarea>
            <div class="settings-hint">这里可以直接编辑并保存。下一次新建对话时会使用保存后的画像快照。</div>
            <div class="memory-section-title">对话索引 <span id="memoryIndexCount"></span></div>
            <div class="settings-hint" id="memoryRetryInfo"></div>
            <div class="memory-index-list" id="memoryIndexList"></div>
          </section>
        </div>
        <div id="webSettingsPanel" class="hidden">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">WEB SEARCH</div>
              <h1>联网搜索</h1>
              <p>使用 Firecrawl 搜索网页并读取正文。只有输入框的联网按钮开启且模型判断需要时才会搜索。</p>
            </div>
          </div>
          <section class="settings-card">
            <div class="settings-row">
              <div><strong>启用联网搜索</strong><span>关闭后输入框按钮不可用，也不会向模型发送搜索工具。</span></div>
              <label class="switch"><input type="checkbox" id="webEnabledToggle"><span></span></label>
            </div>
            <label class="field"><span>Firecrawl API Key</span><input class="input" id="firecrawlKey" type="password" autocomplete="off" placeholder="fc-..."></label>
            <div class="settings-hint" id="firecrawlKeyHint"></div>
            <label class="field"><span>每次搜索结果数 <small>1–50，默认 10</small></span><input class="input" id="webMaxResults" type="number" min="1" max="50" value="10"></label>
            <div style="margin-top:16px"><button class="primary-btn" id="saveWebSettingsBtn" type="button">保存搜索设置</button></div>
          </section>
        </div>
        <div id="archiveSettingsPanel" class="hidden">
          <div class="settings-heading">
            <div>
              <div class="eyebrow">ARCHIVE</div>
              <h1>归档对话</h1>
              <p>归档的对话不会出现在左侧历史列表。删除只能在这里进行，删除后无法恢复。</p>
            </div>
            <button class="secondary-btn" id="refreshArchiveBtn" type="button">刷新</button>
          </div>
          <div class="archive-list" id="archiveList"></div>
        </div>
      </div>
    </section>
  </main>
</div>
<div id="modalRoot"></div>
<div class="toast-root" id="toastRoot"></div>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous" onerror="window.__katexLoadFailed=true"></script>
<script>
if (!window.katex) document.write('<script src="https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.11/katex.min.js"><\/script>');
</script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/mhchem.min.js" crossorigin="anonymous" onerror="window.__mhchemLoadFailed=true"></script>
<script>
if (window.__mhchemLoadFailed) document.write('<script src="https://cdn.bootcdn.net/ajax/libs/KaTeX/0.16.11/contrib/mhchem.min.js"><\/script>');
</script>
<script>
(() => {
  "use strict";

  const state = {
    providers: [],
    conversations: [],
    archivedConversations: [],
    settingsTab: "models",
    fontSize: 100,
    currentId: null,
    current: null,
    messages: [],
    route: "chat",
    streaming: false,
    uidSeq: 0,
    modalOpen: false,
    attachments: [],
    fileAttachments: [],
    streamJobs: new Set(),
    autoScroll: true,
    contextInfo: null,
    lastScrollTop: 0,
    scrollPauseUntil: 0,
    prompts: [],
    activePromptId: "",
    promptSelection: "",
    webSettings: { enabled: false, configured: false, key_hint: "", max_results: 10 },
    webSearchEnabled: false,
    memorySettings: { enabled: false, provider_id: "", model_id: "", review_prompt: "", default_review_prompt: "" },
    memoryData: { profile: "", profile_limit: 2500, conversations: [], updated_at: null },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const EFFORT_LABELS = {
    minimal: "Minimal",
    low: "Low",
    medium: "Medium",
    high: "High",
    xhigh: "XHigh",
    max: "Max",
    ultra: "Ultra",
    none: "None",
  };
  const STORE_KEYS = {
    provider: "xxh.provider",
    model: "xxh.model.",
    effort: "xxh.effort.",
    prompt: "xxh.prompt",
    fontSize: "xxh.fontSize",
  };
  const MAX_IMAGE_EDGE = 1600;

  const el = {
    sidebar: $("#sidebar"),
    mobileBackdrop: $("#mobileBackdrop"),
    menuBtn: $("#menuBtn"),
    navChat: $("#navChat"),
    navSettings: $("#navSettings"),
    newChatBtn: $("#newChatBtn"),
    historyList: $("#historyList"),
    historyCount: $("#historyCount"),
    chatSearchBtn: $("#chatSearchBtn"),
    statusDot: $("#statusDot"),
    localStatus: $("#localStatus"),
    pageEyebrow: $("#pageEyebrow"),
    pageTitle: $("#pageTitle"),
    chatControls: $("#chatControls"),
    providerSelect: $("#providerSelect"),
    modelSelect: $("#modelSelect"),
    effortSelect: $("#effortSelect"),
    chatPage: $("#chatPage"),
    turnNav: $("#turnNav"),
    turnNavList: $("#turnNavList"),
    turnNavCount: $("#turnNavCount"),
    settingsPage: $("#settingsPage"),
    settingsTabs: $("#settingsTabs"),
    generalSettingsPanel: $("#generalSettingsPanel"),
    generalFontMinus: $("#generalFontMinus"),
    generalFontPlus: $("#generalFontPlus"),
    generalFontValue: $("#generalFontValue"),
    resetGeneralSettingsBtn: $("#resetGeneralSettingsBtn"),
    modelSettingsPanel: $("#modelSettingsPanel"),
    promptSettingsPanel: $("#promptSettingsPanel"),
    promptList: $("#promptList"),
    addPromptBtn: $("#addPromptBtn"),
    memorySettingsPanel: $("#memorySettingsPanel"),
    memoryEnabledToggle: $("#memoryEnabledToggle"),
    memoryProviderSelect: $("#memoryProviderSelect"),
    memoryModelSelect: $("#memoryModelSelect"),
    memoryReviewPrompt: $("#memoryReviewPrompt"),
    resetMemoryPromptBtn: $("#resetMemoryPromptBtn"),
    memoryProfile: $("#memoryProfile"),
    memoryProfileCount: $("#memoryProfileCount"),
    saveMemoryProfileBtn: $("#saveMemoryProfileBtn"),
    memoryIndexList: $("#memoryIndexList"),
    memoryIndexCount: $("#memoryIndexCount"),
    memoryRetryInfo: $("#memoryRetryInfo"),
    saveMemorySettingsBtn: $("#saveMemorySettingsBtn"),
    scanMemoryBtn: $("#scanMemoryBtn"),
    refreshMemoryBtn: $("#refreshMemoryBtn"),
    archiveSettingsPanel: $("#archiveSettingsPanel"),
    archiveList: $("#archiveList"),
    archiveCount: $("#archiveCount"),
    refreshArchiveBtn: $("#refreshArchiveBtn"),
    chatScroll: $("#chatScroll"),
    contextBar: $("#contextBar"),
    contextText: $("#contextText"),
    contextFill: $("#contextFill"),
    compressBtn: $("#compressBtn"),
    jumpWrap: $("#jumpWrap"),
    jumpBtn: $("#jumpBtn"),
    emptyState: $("#emptyState"),
    messageList: $("#messageList"),
    composerForm: $("#composerForm"),
    composerInput: $("#composerInput"),
    composerHint: $("#composerHint"),
    sendBtn: $("#sendBtn"),
    stopBtn: $("#stopBtn"),
    attachBtn: $("#attachBtn"),
    imageInput: $("#imageInput"),
    fileBtn: $("#fileBtn"),
    fileInput: $("#fileInput"),
    webSearchBtn: $("#webSearchBtn"),
    webSettingsPanel: $("#webSettingsPanel"),
    webEnabledToggle: $("#webEnabledToggle"),
    firecrawlKey: $("#firecrawlKey"),
    firecrawlKeyHint: $("#firecrawlKeyHint"),
    webMaxResults: $("#webMaxResults"),
    saveWebSettingsBtn: $("#saveWebSettingsBtn"),
    attachmentStrip: $("#attachmentStrip"),
    addProviderBtn: $("#addProviderBtn"),
    providerGrid: $("#providerGrid"),
    promptPickerShell: $("#promptPickerShell"),
    promptPicker: $("#promptPicker"),
    promptLock: $("#promptLock"),
    promptLockName: $("#promptLockName"),
    modalRoot: $("#modalRoot"),
    toastRoot: $("#toastRoot"),
  };

  const store = {
    get(key, fallback = "") {
      try {
        const value = localStorage.getItem(key);
        return value === null ? fallback : value;
      } catch (_) {
        return fallback;
      }
    },
    set(key, value) {
      try { localStorage.setItem(key, String(value)); } catch (_) {}
    },
    remove(key) {
      try { localStorage.removeItem(key); } catch (_) {}
    },
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function icon(id) {
    return `<svg aria-hidden="true"><use href="#${id}"></use></svg>`;
  }

  function modelById(provider, modelId) {
    if (!provider) return null;
    return (provider.models || []).find((model) => model.id === modelId) || null;
  }

  function selectedProvider() {
    return state.providers.find((provider) => provider.id === el.providerSelect.value) || null;
  }

  function selectedModel() {
    return modelById(selectedProvider(), el.modelSelect.value);
  }

  function effortLabel(level) {
    if (!level) return "None";
    return EFFORT_LABELS[level] || level;
  }

  function formatTime(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  }

  function formatShortDate(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const now = new Date();
    if (date.toDateString() === now.toDateString()) return formatTime(iso);
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) return "昨天";
    return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
  }

  function dateGroup(iso) {
    if (!iso) return "更早";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "更早";
    const now = new Date();
    if (date.toDateString() === now.toDateString()) return "今天";
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) return "昨天";
    const weekAgo = new Date(now);
    weekAgo.setDate(now.getDate() - 7);
    return date >= weekAgo ? "最近 7 天" : "更早";
  }

  function formatNumber(value) {
    const number = Number(value) || 0;
    if (number >= 1_000_000) return (number / 1_000_000).toFixed(number >= 10_000_000 ? 0 : 1) + "m";
    if (number >= 1000) return (number / 1000).toFixed(number >= 10_000 ? 0 : 1) + "k";
    return String(number);
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function api(path, options = {}) {
    const init = { cache: "no-store", ...options };
    init.headers = { Accept: "application/json", ...(options.headers || {}) };
    if (init.body && typeof init.body !== "string") {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(init.body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (error) {
      throw new Error("无法连接本地服务，请确认 app.py 仍在运行");
    }
    const text = await response.text();
    let data = {};
    if (text) {
      try { data = JSON.parse(text); } catch (_) { data = { error: text }; }
    }
    if (!response.ok) {
      throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
    }
    return data;
  }

  function toast(message, type = "success", duration = 3400) {
    const node = document.createElement("div");
    node.className = `toast ${type === "error" ? "error" : ""}`;
    node.innerHTML = `${icon(type === "error" ? "i-alert" : "i-check")}<span></span>`;
    node.querySelector("span").textContent = message;
    el.toastRoot.appendChild(node);
    window.setTimeout(() => {
      node.classList.add("out");
      window.setTimeout(() => node.remove(), 220);
    }, duration);
  }

  function showStatus(ok, text) {
    el.statusDot.classList.toggle("offline", !ok);
    el.localStatus.textContent = text || (ok ? "本地运行 · 数据仅存本机" : "服务未连接");
  }

  function closeMobileSidebar() {
    el.sidebar.classList.remove("open");
    el.mobileBackdrop.classList.remove("show");
  }

  function openMobileSidebar() {
    el.sidebar.classList.add("open");
    el.mobileBackdrop.classList.add("show");
  }

  function setPageMeta(eyebrow, title) {
    el.pageEyebrow.textContent = eyebrow;
    el.pageTitle.textContent = title;
  }

  function currentTitle() {
    if (state.current && state.current.title) return state.current.title;
    return "新对话";
  }

  function nextUid() {
    state.uidSeq += 1;
    return `m${state.uidSeq}`;
  }

  function openPromptModal(prompt = null) {
    const isEdit = Boolean(prompt?.id);
    showModal(`
      <div class="modal prompt-modal" role="dialog" aria-modal="true" aria-labelledby="promptModalTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="promptModalTitle">${isEdit ? "编辑系统提示词" : "新建系统提示词"}</h2>
            <p class="modal-subtitle">这段内容会作为每条新对话的第一条 system 消息发送给模型。</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <form id="promptForm" novalidate>
          <label class="field"><span>名称</span><input class="input" id="promptName" maxlength="80" value="${escapeHtml(prompt?.name || "")}" placeholder="例如：教学辅助" required></label>
          <label class="field"><span>系统提示词</span><textarea class="modal-textarea prompt-textarea" id="promptContent" maxlength="30000" placeholder="可以完全留空">${escapeHtml(prompt?.content || "")}</textarea></label>
          <div class="form-error hidden" id="promptFormError"></div>
          <div class="modal-actions">
            <button class="secondary-btn" type="button" data-modal-close>取消</button>
            <button class="primary-btn" id="promptSubmit" type="submit"><span>保存</span></button>
          </div>
        </form>
      </div>`);
    const form = $("#promptForm", el.modalRoot);
    const errorBox = $("#promptFormError", el.modalRoot);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const submit = $("#promptSubmit", form);
      const name = $("#promptName", form).value.trim();
      const content = $("#promptContent", form).value.trim();
      if (!name) {
        errorBox.textContent = "提示词名称不能为空。";
        errorBox.classList.remove("hidden");
        return;
      }
      submit.disabled = true;
      try {
        const data = await api("/api/prompts", {
          method: "POST",
          body: { id: prompt?.id || "", name, content, set_active: !isEdit },
        });
        state.prompts = data.prompts || state.prompts;
        state.activePromptId = data.active_prompt_id || state.activePromptId;
        if (!isEdit) state.promptSelection = data.prompt?.id || state.promptSelection;
        closeModal();
        renderPromptList();
        renderPromptPicker();
        toast(isEdit ? "提示词已保存" : "提示词已创建");
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        submit.disabled = false;
      }
    });
  }

  async function deletePrompt(prompt) {
    const confirmed = await confirmDialog(
      "删除系统提示词",
      `确定删除 <strong>${escapeHtml(prompt.name)}</strong> 吗？已有对话不受影响。`,
      "删除",
    );
    if (!confirmed) return;
    try {
      const data = await api(`/api/prompts/${encodeURIComponent(prompt.id)}`, { method: "DELETE" });
      state.prompts = data.prompts || [];
      state.activePromptId = data.active_prompt_id || state.prompts[0]?.id || "";
      if (!state.prompts.some((item) => item.id === state.promptSelection)) {
        state.promptSelection = state.activePromptId;
      }
      renderPromptList();
      renderPromptPicker();
      toast("提示词已删除");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function setDefaultPrompt(prompt) {
    try {
      const data = await api("/api/prompts", {
        method: "POST",
        body: { id: prompt.id, name: prompt.name, content: prompt.content, set_active: true },
      });
      state.prompts = data.prompts || state.prompts;
      state.activePromptId = data.active_prompt_id || prompt.id;
      state.promptSelection = prompt.id;
      store.set(STORE_KEYS.prompt, prompt.id);
      renderPromptList();
      renderPromptPicker();
      toast("已设为新对话默认提示词");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function loadPrompts(options = {}) {
    try {
      const data = await api("/api/prompts");
      state.prompts = data.prompts || [];
      state.activePromptId = data.active_prompt_id || state.prompts[0]?.id || "";
      if (!state.prompts.some((p) => p.id === state.promptSelection)) {
        const saved = store.get(STORE_KEYS.prompt, "");
        state.promptSelection = state.prompts.some((p) => p.id === saved)
          ? saved
          : state.activePromptId;
      }
      renderPromptList();
      renderPromptPicker();
      return state.prompts;
    } catch (error) {
      if (!options.silent) toast(error.message, "error");
      return [];
    }
  }

  function renderPromptList() {
    if (!state.prompts.length) {
      el.promptList.innerHTML = '<div class="prompt-empty">还没有系统提示词。</div>';
      return;
    }
    el.promptList.innerHTML = state.prompts.map((prompt) => {
      const active = prompt.id === state.activePromptId;
      const builtin = prompt.builtin ? '<span class="prompt-tag">内置</span>' : '<span class="prompt-tag custom">自定义</span>';
      return `<article class="prompt-card${active ? " active" : ""}" data-prompt-id="${escapeHtml(prompt.id)}">
        <div class="prompt-card-head">
          <div class="prompt-card-title">${escapeHtml(prompt.name)}</div>
          ${builtin}
          ${active ? '<span class="prompt-active-mark">新对话默认</span>' : ""}
        </div>
        <div class="prompt-preview">${prompt.content ? escapeHtml(prompt.content) : "（空提示词）"}</div>
        <div class="prompt-card-actions">
          <button class="secondary-btn" type="button" data-prompt-action="edit">编辑</button>
          <button class="secondary-btn" type="button" data-prompt-action="default"${active ? " disabled" : ""}>设为默认</button>
          <button class="secondary-btn danger-btn" type="button" data-prompt-action="delete">删除</button>
        </div>
      </article>`;
    }).join("");
  }

  function renderPromptPicker() {
    const editable = !state.currentId && state.messages.length === 0;
    el.promptPickerShell?.classList.toggle("hidden", !editable);
    if (!el.promptPicker) return;
    el.promptPicker.innerHTML = state.prompts.map((prompt) => (
      `<option value="${escapeHtml(prompt.id)}"${prompt.id === state.promptSelection ? " selected" : ""}>${escapeHtml(prompt.name)}</option>`
    )).join("");
    el.promptPicker.disabled = !editable || !state.prompts.length;
    if (el.promptLock) {
      el.promptLock.classList.toggle("hidden", editable);
      if (!editable) {
        const selected = state.prompts.find((prompt) => prompt.id === state.promptSelection);
        const name = state.current?.prompt_name || selected?.name || "已锁定";
        el.promptLockName.textContent = name;
        el.promptLock.title = state.current?.system_prompt || selected?.content || name;
      }
      // A brand-new local conversation shows the picker and has not locked anything yet.
    }
  }

  async function loadMemory(options = {}) {
    try {
      const data = await api("/api/memory");
      state.memorySettings = {
        enabled: Boolean(data.enabled),
        provider_id: data.provider_id || "",
        model_id: data.model_id || "",
        review_prompt: data.review_prompt || data.default_review_prompt || "",
        default_review_prompt: data.default_review_prompt || "",
      };
      state.memoryData = {
        profile: data.profile || "",
        profile_limit: Number(data.profile_limit) || 2500,
        conversations: data.conversations || [],
        retry_count: Number(data.retry_count) || 0,
        updated_at: data.updated_at || null,
      };
      renderMemorySettings();
      return data;
    } catch (error) {
      if (!options.silent) toast(error.message, "error");
      return null;
    }
  }

  function populateMemoryModels(preferredProvider = "", preferredModel = "") {
    if (!el.memoryProviderSelect || !el.memoryModelSelect) return;
    const currentProvider = preferredProvider || el.memoryProviderSelect.value || "";
    el.memoryProviderSelect.innerHTML = '<option value="">选择供应商</option>' + state.providers.map((provider) => (
      '<option value="' + escapeHtml(provider.id) + '">' + escapeHtml(provider.name) + '</option>'
    )).join("");
    if (state.providers.some((provider) => provider.id === currentProvider)) {
      el.memoryProviderSelect.value = currentProvider;
    }
    const provider = state.providers.find((item) => item.id === el.memoryProviderSelect.value);
    const models = provider ? (provider.models || []) : [];
    const currentModel = preferredModel || el.memoryModelSelect.value || "";
    el.memoryModelSelect.innerHTML = '<option value="">选择模型</option>' + models.map((model) => (
      '<option value="' + escapeHtml(model.id) + '">' + escapeHtml(model.name && model.name !== model.id ? model.name + " · " + model.id : model.id) + '</option>'
    )).join("");
    if (models.some((model) => model.id === currentModel)) {
      el.memoryModelSelect.value = currentModel;
    }
  }

  function updateMemoryProfileCount() {
    if (!el.memoryProfile || !el.memoryProfileCount) return;
    const length = el.memoryProfile.value.length;
    const limit = Number(state.memoryData.profile_limit) || 2500;
    const overLimit = length > limit;
    el.memoryProfileCount.textContent = length + " / " + limit + " 字" + (overLimit ? "，请缩短" : "");
    el.memoryProfileCount.classList?.toggle("over-limit", overLimit);
    if (el.saveMemoryProfileBtn) el.saveMemoryProfileBtn.disabled = overLimit;
  }

  function renderMemorySettings() {
    if (!el.memoryEnabledToggle) return;
    el.memoryEnabledToggle.checked = Boolean(state.memorySettings.enabled);
    if (el.memoryReviewPrompt) {
      el.memoryReviewPrompt.value = state.memorySettings.review_prompt || state.memorySettings.default_review_prompt || "";
    }
    populateMemoryModels(state.memorySettings.provider_id, state.memorySettings.model_id);
    const profile = state.memoryData.profile || "";
    const profileLimit = Number(state.memoryData.profile_limit) || 2500;
    el.memoryProfile.value = profile;
    el.memoryProfile.maxLength = profileLimit;
    updateMemoryProfileCount();
    const entries = [...(state.memoryData.conversations || [])].sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
    el.memoryIndexCount.textContent = entries.length + " 条";
    el.memoryIndexList.innerHTML = entries.length
      ? entries.map((item) => {
          const archived = item.archived ? " · 已归档" : "";
          return '<div class="memory-index-item"><div class="memory-index-title">' + escapeHtml(item.title || "对话") + '</div><div class="memory-index-summary">' + escapeHtml(item.summary || "暂无摘要") + '</div><div class="memory-index-meta">ID ' + escapeHtml(item.id || "") + ' · ' + escapeHtml(String(item.turns || 0)) + ' 轮' + archived + '</div><div class="memory-index-actions"><button class="secondary-btn" type="button" data-index-action="edit" data-index-id="' + escapeHtml(item.id || "") + '">编辑</button><button class="secondary-btn danger-btn" type="button" data-index-action="delete" data-index-id="' + escapeHtml(item.id || "") + '">删除索引</button></div></div>';
        }).join("")
      : '<div class="memory-index-empty">暂无对话索引。新对话结束后会逐步建立。</div>';
    el.memoryRetryInfo.textContent = state.memoryData.retry_count
      ? '待重试：' + state.memoryData.retry_count + ' 条。失败任务会保留并在下次检查时继续重试。'
      : '';
  }

  async function saveMemoryProfile() {
    el.saveMemoryProfileBtn.disabled = true;
    try {
      const data = await api("/api/memory/profile", {
        method: "POST",
        body: { profile: el.memoryProfile.value },
      });
      state.memoryData.profile = data.profile || "";
      el.memoryProfile.value = data.profile || "";
      updateMemoryProfileCount();
      toast("用户画像已保存");
    } catch (error) {
      toast(error.message, "error");
      updateMemoryProfileCount();
    }
  }

  function editMemoryIndex(convId) {
    const item = (state.memoryData.conversations || []).find((entry) => entry.id === convId);
    if (!item) {
      toast("记忆索引不存在", "error");
      return;
    }
    showModal(`
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="memoryIndexEditTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="memoryIndexEditTitle">编辑记忆索引</h2>
            <p class="modal-subtitle">内部 ID 保持不变：${escapeHtml(item.id || "")}</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <div class="memory-index-sync-title">索引标题跟随对话：<strong>${escapeHtml(item.title || "对话")}</strong></div>
        <label class="field"><span>索引摘要 <small id="memoryIndexSummaryCount"></small></span><textarea class="modal-textarea memory-index-textarea" id="memoryIndexSummary" maxlength="300" spellcheck="false">${escapeHtml(item.summary || "")}</textarea></label>
        <label class="field"><span>关键词 <small>用逗号分隔</small></span><input class="input" id="memoryIndexKeywords" maxlength="400" value="${escapeHtml((item.keywords || []).join("，"))}"></label>
        <div class="form-error hidden" id="memoryIndexError"></div>
        <div class="modal-actions">
          <button class="secondary-btn" type="button" data-modal-close>取消</button>
          <button class="primary-btn" id="saveMemoryIndexBtn" type="button">保存索引</button>
        </div>
      </div>`);
    const summaryInput = $("#memoryIndexSummary", el.modalRoot);
    const keywordsInput = $("#memoryIndexKeywords", el.modalRoot);
    const summaryCount = $("#memoryIndexSummaryCount", el.modalRoot);
    const errorBox = $("#memoryIndexError", el.modalRoot);
    const saveIndexButton = $("#saveMemoryIndexBtn", el.modalRoot);
    const updateSummaryCount = () => {
      const overLimit = summaryInput.value.length > 300;
      summaryCount.textContent = summaryInput.value.length + " / 300 字" + (overLimit ? "，请压缩" : "");
      summaryCount.classList?.toggle("over-limit", overLimit);
      saveIndexButton.disabled = overLimit;
    };
    summaryInput.addEventListener("input", updateSummaryCount);
    updateSummaryCount();
    saveIndexButton.addEventListener("click", async () => {
      const summary = summaryInput.value.trim();
      if (!summary) {
        errorBox.textContent = "索引摘要不能为空。";
        errorBox.classList.remove("hidden");
        return;
      }
      const button = saveIndexButton;
      button.disabled = true;
      errorBox.classList.add("hidden");
      try {
        await api(`/api/memory/index/${encodeURIComponent(convId)}`, {
          method: "POST",
          body: { summary, keywords: keywordsInput.value },
        });
        closeModal();
        await loadMemory({ silent: true });
        toast("记忆索引已保存");
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        button.disabled = false;
      }
    });
  }

  async function deleteMemoryIndex(convId) {
    const item = (state.memoryData.conversations || []).find((entry) => entry.id === convId);
    const confirmed = await confirmDialog(
      "删除记忆索引",
      `确定永久排除 <strong>${escapeHtml(item?.title || "这条对话")}</strong> 吗？它会从当前索引移除，已有快照保持不变，并且以后不会进入新对话的记忆快照。`,
      "删除索引",
    );
    if (!confirmed) return;
    try {
      await api(`/api/memory/index/${encodeURIComponent(convId)}`, { method: "DELETE" });
      await loadMemory({ silent: true });
      toast("记忆索引已删除");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function saveMemorySettings() {
    el.saveMemorySettingsBtn.disabled = true;
    try {
      const data = await api("/api/memory", {
        method: "POST",
        body: {
          enabled: Boolean(el.memoryEnabledToggle.checked),
          provider_id: el.memoryProviderSelect.value,
          model_id: el.memoryModelSelect.value,
          review_prompt: el.memoryReviewPrompt.value,
        },
      });
      state.memorySettings = { ...state.memorySettings, ...(data.memory || {}) };
      await loadMemory({ silent: true });
      toast("记忆设置已保存");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.saveMemorySettingsBtn.disabled = false;
    }
  }

  async function scanMemory() {
    el.scanMemoryBtn.disabled = true;
    try {
      const data = await api("/api/memory/scan", { method: "POST", body: {} });
      toast(data.started ? "已开始检查可更新的对话" : "记忆检查已在运行");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.scanMemoryBtn.disabled = false;
    }
  }

  async function loadWebSettings(options = {}) {
    try {
      const data = await api("/api/settings");
      state.webSettings = { enabled: false, configured: false, key_hint: "", max_results: 10, ...(data.web || {}) };
      const savedWeb = store.get("xxh.webSearch", "");
      if (!state.webSettings.enabled || !state.webSettings.configured) {
        state.webSearchEnabled = false;
        store.set("xxh.webSearch", "0");
      } else {
        state.webSearchEnabled = savedWeb === "1";
      }
      renderWebSettings();
      return state.webSettings;
    } catch (error) {
      if (!options.silent) toast(error.message, "error");
      return state.webSettings;
    }
  }

  function renderWebSettings() {
    if (!el.webEnabledToggle) return;
    el.webEnabledToggle.checked = Boolean(state.webSettings.enabled);
    el.firecrawlKey.value = "";
    el.firecrawlKey.placeholder = state.webSettings.configured
      ? "已保存 " + (state.webSettings.key_hint || "Key") + "，留空则不修改"
      : "fc-...";
    el.firecrawlKeyHint.textContent = state.webSettings.configured
      ? "已保存本地 Key，只写入 config.json，不会发送给模型。"
      : "请填写 Firecrawl API Key 后再启用联网搜索。";
    el.webMaxResults.value = String(state.webSettings.max_results || 10);
    updateWebSearchButton();
  }

  function updateWebSearchButton() {
    if (!el.webSearchBtn) return;
    const usable = Boolean(state.webSettings.enabled && state.webSettings.configured);
    if (!usable) state.webSearchEnabled = false;
    el.webSearchBtn.classList.toggle("active", state.webSearchEnabled);
    el.webSearchBtn.setAttribute("aria-pressed", state.webSearchEnabled ? "true" : "false");
    el.webSearchBtn.title = !state.webSettings.enabled
      ? "联网搜索已在设置中关闭"
      : !state.webSettings.configured
        ? "请先在设置中填写 Firecrawl API Key"
        : state.webSearchEnabled
          ? "联网搜索已开启，再次点击关闭"
          : "点击开启联网搜索（模型自主决定是否搜索）";
  }

  function toggleWebSearch() {
    if (!state.webSettings.enabled || !state.webSettings.configured) {
      toast("请先在设置中启用联网搜索并填写 Firecrawl API Key", "error");
      setRoute("settings");
      setSettingsTab("web");
      return;
    }
    state.webSearchEnabled = !state.webSearchEnabled;
    store.set("xxh.webSearch", state.webSearchEnabled ? "1" : "0");
    updateWebSearchButton();
    toast(state.webSearchEnabled ? "联网搜索已开启，模型会自行判断是否搜索" : "联网搜索已关闭");
  }

  async function saveWebSettings() {
    const enabled = Boolean(el.webEnabledToggle.checked);
    const apiKey = el.firecrawlKey.value.trim();
    const maxResults = Math.max(1, Math.min(50, Number(el.webMaxResults.value) || 10));
    el.saveWebSettingsBtn.disabled = true;
    try {
      const data = await api("/api/settings/web", {
        method: "POST",
        body: { enabled, api_key: apiKey, max_results: maxResults },
      });
      state.webSettings = { enabled: false, configured: false, key_hint: "", max_results: 10, ...(data.web || {}) };
      const savedWeb = store.get("xxh.webSearch", "");
      if (!state.webSettings.enabled || !state.webSettings.configured) {
        state.webSearchEnabled = false;
        store.set("xxh.webSearch", "0");
      } else {
        state.webSearchEnabled = savedWeb === "1";
      }
      renderWebSettings();
      toast("搜索设置已保存");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.saveWebSettingsBtn.disabled = false;
    }
  }

  async function loadProviders(options = {}) {
    try {
      const data = await api("/api/providers");
      state.providers = data.providers || [];
      renderProviderGrid();
      populateProviderSelect();
      showStatus(true);
      return state.providers;
    } catch (error) {
      showStatus(false, "服务未连接");
      if (!options.silent) toast(error.message, "error");
      state.providers = [];
      renderProviderGrid();
      populateProviderSelect();
      return [];
    }
  }

  function renderProviderGrid() {
    if (!state.providers.length) {
      el.providerGrid.innerHTML = `
        <div class="empty-providers">
          <div class="empty-providers-icon">${icon("i-settings")}</div>
          <strong>还没有模型供应商</strong>
          <p>添加一个兼容 OpenAI 接口的服务，填写名称、Base URL 和 API Key 后会立即拉取可用模型。</p>
          <button class="primary-btn" type="button" data-empty-add>${icon("i-plus")}<span>添加第一个供应商</span></button>
        </div>`;
      return;
    }
    el.providerGrid.innerHTML = state.providers.map((provider) => {
      const models = provider.models || [];
      const shown = models.slice(0, 7);
      const hidden = Math.max(0, models.length - shown.length);
      const chips = shown.map((model) => {
        const label = model.name && model.name !== model.id ? `${model.name} · ${model.id}` : model.id;
        return `<span class="model-chip" title="${escapeHtml(label)}">${escapeHtml(model.id)}</span>`;
      }).join("");
      const more = hidden ? `<span class="model-chip more">+${hidden}</span>` : "";
      const initial = Array.from(provider.name || "?")[0] || "?";
      const updated = provider.updated_at ? new Date(provider.updated_at).toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" }) : "—";
      const timeoutValue = Number.isFinite(Number(provider.timeout)) ? Number(provider.timeout) : 20;
      const retryValue = Number.isFinite(Number(provider.retries)) ? Number(provider.retries) : 3;
      return `
        <article class="provider-card" data-provider-id="${escapeHtml(provider.id)}">
          <div class="provider-top">
            <div class="provider-icon">${escapeHtml(initial)}</div>
            <div class="provider-info">
              <div class="provider-name"><span>${escapeHtml(provider.name)}</span><span class="provider-live" title="可用"></span></div>
              <div class="provider-url" title="${escapeHtml(provider.base_url)}">${escapeHtml(provider.base_url)}</div>
            </div>
            <div class="provider-actions">
              <button class="mini-btn" type="button" data-provider-action="refresh" tabindex="0" aria-label="刷新模型" title="刷新模型">${icon("i-refresh")}</button>
              <button class="mini-btn" type="button" data-provider-action="edit" tabindex="0" aria-label="编辑供应商">${icon("i-edit")}</button>
              <button class="mini-btn danger" type="button" data-provider-action="delete" aria-label="删除供应商">${icon("i-trash")}</button>
            </div>
          </div>
          <div class="provider-stats"><span><strong>${models.length}</strong> 个模型</span><span>·</span><span>更新于 ${escapeHtml(updated)}</span></div>
          <div class="model-chips">${models.length ? chips + more : '<span class="model-chip">暂无模型</span>'}</div>
          <div class="provider-foot">
            <span class="key-hint">${icon("i-key")}${escapeHtml(provider.key_hint || "已保存")}</span>
            <span>${timeoutValue}s · 重试 ${retryValue} 次</span>
          </div>
        </article>`;
    }).join("");
  }

  function populateProviderSelect(preferredId = "") {
    const previous = preferredId || el.providerSelect.value || store.get(STORE_KEYS.provider, "");
    if (!state.providers.length) {
      el.providerSelect.innerHTML = '<option value="">未配置</option>';
      el.modelSelect.innerHTML = '<option value="">—</option>';
      el.effortSelect.innerHTML = '<option value="">—</option>';
      updateSendAvailability();
      return;
    }
    el.providerSelect.innerHTML = state.providers.map((provider) => (
      `<option value="${escapeHtml(provider.id)}">${escapeHtml(provider.name)}</option>`
    )).join("");
    const chosen = state.providers.some((provider) => provider.id === previous)
      ? previous
      : state.providers[0].id;
    el.providerSelect.value = chosen;
    store.set(STORE_KEYS.provider, chosen);
    populateModelSelect();
  }

  function populateModelSelect(preferredModel = "") {
    const provider = selectedProvider();
    if (!provider) {
      el.modelSelect.innerHTML = '<option value="">—</option>';
      el.effortSelect.innerHTML = '<option value="">—</option>';
      updateSendAvailability();
      return;
    }
    const saved = preferredModel || el.modelSelect.value || store.get(STORE_KEYS.model + provider.id, "");
    const models = provider.models || [];
    if (!models.length) {
      el.modelSelect.innerHTML = '<option value="">暂无模型</option>';
      el.effortSelect.innerHTML = '<option value="">—</option>';
      updateSendAvailability();
      return;
    }
    el.modelSelect.innerHTML = models.map((model) => {
      const label = model.name && model.name !== model.id ? `${model.name} · ${model.id}` : model.id;
      return `<option value="${escapeHtml(model.id)}">${escapeHtml(label)}</option>`;
    }).join("");
    const chosen = models.some((model) => model.id === saved) ? saved : models[0].id;
    el.modelSelect.value = chosen;
    store.set(STORE_KEYS.model + provider.id, chosen);
    populateEffortSelect();
  }

  function populateEffortSelect(preferredEffort = "") {
    const provider = selectedProvider();
    const model = selectedModel();
    if (!provider || !model) {
      el.effortSelect.innerHTML = '<option value="">—</option>';
      updateSendAvailability();
      return;
    }
    // 部分 OpenAI 兼容服务不在 /models 里声明思考强度，仍然允许选择通用档位。
    const declared = Array.isArray(model.efforts) ? model.efforts : [];
    const levels = declared.length ? declared : ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"];
    el.effortSelect.innerHTML = levels.map((level) => (
      `<option value="${escapeHtml(level)}">${escapeHtml(effortLabel(level))}</option>`
    )).join("");
    const saved = preferredEffort || store.get(STORE_KEYS.effort + provider.id + "." + model.id, "");
    let chosen = "";
    if (levels.includes(saved)) chosen = saved;
    else if (declared.length && levels.includes(model.default_effort)) chosen = model.default_effort;
    else if (levels.includes("medium")) chosen = "medium";
    else chosen = levels[0] || "";
    el.effortSelect.value = chosen;
    store.set(STORE_KEYS.effort + provider.id + "." + model.id, chosen);
    updateSendAvailability();
  }

  function syncControlsFromConversation(conv) {
    if (!conv) return;
    if (state.providers.some((provider) => provider.id === conv.provider_id)) {
      el.providerSelect.value = conv.provider_id;
      store.set(STORE_KEYS.provider, conv.provider_id);
    }
    populateModelSelect(conv.model || "");
    if (conv.model && el.modelSelect.value !== conv.model) {
      // 模型可能已被供应商移除，保持页面可用，但不要在沉默中切换旧对话的模型。
      toast("原对话使用的模型已不在列表中，已切换到可用模型", "error");
    }
    if (conv.reasoning_effort) populateEffortSelect(conv.reasoning_effort);
    updateSendAvailability();
  }

  function jobForMessages(messages) {
    for (const job of state.streamJobs) {
      if (job.messages === messages) return job;
    }
    return null;
  }

  function jobForConversation(convId) {
    for (const job of state.streamJobs) {
      if (job.conversationId && job.conversationId === convId) return job;
    }
    return null;
  }

  function jobForAssistant(assistant) {
    for (const job of state.streamJobs) {
      if (job.assistant === assistant) return job;
    }
    return null;
  }

  function currentJob() {
    return jobForMessages(state.messages);
  }

  function refreshStreamingFlag() {
    state.streaming = Boolean(currentJob());
  }

  function updateSendAvailability() {
    const provider = selectedProvider();
    const model = selectedModel();
    const job = currentJob();
    const disabled = Boolean(job) || !provider || !model;
    el.sendBtn.disabled = disabled;
    el.sendBtn.classList.toggle("hidden", Boolean(job));
    el.stopBtn.classList.toggle("hidden", !job);
    if (job) {
      el.composerInput.placeholder = "正在生成，按 Esc 停止…";
      el.composerHint.innerHTML = "<kbd>Esc</kbd> 停止生成";
      return;
    }
    if (!state.providers.length) {
      el.composerInput.placeholder = "请先在“模型配置”中添加供应商…";
      el.composerHint.textContent = "尚未配置模型";
    } else if (!provider || !model) {
      el.composerInput.placeholder = "请选择可用的供应商和模型…";
      el.composerHint.textContent = "模型不可用";
    } else if (!state.streaming) {
      el.composerInput.placeholder = `和 ${provider.name} / ${model.id} 对话…`;
      el.composerHint.innerHTML = '<kbd>Enter</kbd> 发送 <span class="sep">·</span> <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行';
    }
  }

  function showModal(html) {
    el.modalRoot.innerHTML = `<div class="modal-backdrop" data-modal-backdrop>${html}</div>`;
    state.modalOpen = true;
  }

  function closeModal() {
    el.modalRoot.innerHTML = "";
    state.modalOpen = false;
  }

  async function refreshProviderModels(provider, button) {
    if (button) {
      button.disabled = true;
      button.classList.add("spinning");
    }
    try {
      const data = await api(`/api/providers/${encodeURIComponent(provider.id)}/refresh`, {
        method: "POST",
        body: {},
      });
      const count = (data.provider?.models || []).length;
      toast(`模型已刷新：${count} 个`);
      await loadProviders();
      if (el.providerSelect.value === provider.id) populateModelSelect();
    } catch (error) {
      toast(error.message, "error");
      if (button) {
        button.disabled = false;
        button.classList.remove("spinning");
      }
    }
  }

  function openProviderModal(provider = null) {
    const editing = Boolean(provider);
    const keyHint = editing && provider.key_hint ? `当前 ${provider.key_hint}，留空保持不变` : "必填";
    const basePlaceholder = "https://api.openai.com/v1";
    const timeoutValue = Number.isFinite(Number(provider?.timeout)) ? Number(provider?.timeout) : 20;
    const retryValue = Number.isFinite(Number(provider?.retries)) ? Number(provider?.retries) : 3;
    showModal(`
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="providerModalTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="providerModalTitle">${editing ? "编辑供应商" : "添加供应商"}</h2>
            <p class="modal-subtitle">保存只写入设置；模型列表在供应商卡片上用刷新按钮单独获取。</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <form id="providerForm" novalidate>
          <label class="field">
            <span class="field-label">供应商名称</span>
            <span class="input-wrap"><input id="providerName" name="name" type="text" maxlength="60" placeholder="例如 DeepSeek" value="${escapeHtml(provider?.name || "")}" autocomplete="off" required></span>
          </label>
          <label class="field">
            <span class="field-label">Base URL <span>到 /v1 或服务根地址</span></span>
            <span class="input-wrap"><input id="providerBaseUrl" name="base_url" type="url" placeholder="${basePlaceholder}" value="${escapeHtml(provider?.base_url || "")}" autocomplete="off" spellcheck="false" required></span>
            <span class="field-hint">例如 https://api.deepseek.com 或 https://api.openai.com/v1；中转站请填 API 域名，不要填官网首页</span>
          </label>
          <label class="field">
            <span class="field-label">API Key <span>${escapeHtml(keyHint)}</span></span>
            <span class="input-wrap">
              <input id="providerApiKey" name="api_key" type="password" placeholder="${editing ? "留空则保留原 Key" : "sk-..."}" autocomplete="new-password" spellcheck="false" ${editing ? "" : "required"}>
              <button class="input-action" type="button" id="toggleKey" aria-label="显示 API Key">${icon("i-eye")}</button>
            </span>
            <span class="field-hint">Key 只写入本机项目目录下的 config.json，不会上传到其他位置。</span>
          </label>
          <div class="field-row">
            <label class="field">
              <span class="field-label">等待时间 <span>秒</span></span>
              <span class="input-wrap"><input id="providerTimeout" name="timeout" type="number" min="3" max="600" step="1" value="${timeoutValue}" required></span>
              <span class="field-hint">连接或无响应超过该时间就重试</span>
            </label>
            <label class="field">
              <span class="field-label">重试次数 <span>不含首次</span></span>
              <span class="input-wrap"><input id="providerRetries" name="retries" type="number" min="0" max="10" step="1" value="${retryValue}" required></span>
              <span class="field-hint">默认 3 次，失败后退避重试</span>
            </label>
          </div>
          <div class="form-error hidden" id="providerFormError"></div>
          <div class="modal-actions">
            <button class="secondary-btn" type="button" data-modal-close>取消</button>
            <button class="primary-btn" type="submit" id="providerSubmit"><span>保存</span></button>
          </div>
        </form>
      </div>`);
    const form = $("#providerForm");
    const keyInput = $("#providerApiKey");
    const toggle = $("#toggleKey");
    toggle.addEventListener("click", () => {
      const showing = keyInput.type === "text";
      keyInput.type = showing ? "password" : "text";
      toggle.innerHTML = icon(showing ? "i-eye" : "i-eye-off");
      toggle.setAttribute("aria-label", showing ? "显示 API Key" : "隐藏 API Key");
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await saveProvider(form, provider);
    });
    window.setTimeout(() => $("#providerName").focus(), 30);
  }

  async function saveProvider(form, provider) {
    const errorBox = $("#providerFormError");
    const submit = $("#providerSubmit");
    const name = $("#providerName").value.trim();
    const baseUrl = $("#providerBaseUrl").value.trim();
    const apiKey = $("#providerApiKey").value.trim();
    const timeoutValue = Number.parseInt($("#providerTimeout").value, 10);
    const retryValue = Number.parseInt($("#providerRetries").value, 10);
    errorBox.classList.add("hidden");
    if (!name) { showFormError("请填写供应商名称"); return; }
    if (!baseUrl) { showFormError("请填写 Base URL"); return; }
    if (!provider && !apiKey) { showFormError("请填写 API Key"); return; }
    if (!Number.isFinite(timeoutValue) || timeoutValue < 3 || timeoutValue > 600) {
      showFormError("等待时间需在 3-600 秒之间");
      return;
    }
    if (!Number.isFinite(retryValue) || retryValue < 0 || retryValue > 10) {
      showFormError("重试次数需在 0-10 之间");
      return;
    }
    submit.disabled = true;
    submit.innerHTML = '<span class="spinner"></span><span>保存中…</span>';
    try {
      const data = await api("/api/providers", {
        method: "POST",
        body: {
          id: provider?.id || "",
          name,
          base_url: baseUrl,
          api_key: apiKey,
          timeout: timeoutValue,
          retries: retryValue,
        },
      });
      closeModal();
      const modelCount = (data.provider?.models || []).length;
      toast(modelCount ? "供应商已保存" : "供应商已保存，请点卡片上的刷新按钮获取模型");
      await loadProviders();
      if (data.provider?.id) {
        el.providerSelect.value = data.provider.id;
        store.set(STORE_KEYS.provider, data.provider.id);
        populateModelSelect();
      }
      setRoute(modelCount ? "chat" : "settings");
    } catch (error) {
      showFormError(error.message);
      submit.disabled = false;
      submit.innerHTML = "<span>保存</span>";
    }
  }

  function showFormError(message) {
    const errorBox = $("#providerFormError");
    if (!errorBox) { toast(message, "error"); return; }
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function confirmDialog(title, message, confirmLabel = "确认删除") {
    return new Promise((resolve) => {
      showModal(`
        <div class="modal" role="dialog" aria-modal="true">
          <div class="modal-head">
            <div><h2 class="modal-title">${escapeHtml(title)}</h2></div>
            <button class="modal-close" type="button" data-confirm-cancel aria-label="关闭">${icon("i-close")}</button>
          </div>
          <p class="confirm-copy">${message}</p>
          <div class="modal-actions">
            <button class="secondary-btn" type="button" data-confirm-cancel>取消</button>
            <button class="primary-btn danger-btn" type="button" data-confirm-ok>${escapeHtml(confirmLabel)}</button>
          </div>
        </div>`);
      const finish = (value) => {
        closeModal();
        resolve(value);
      };
      $$("[data-confirm-cancel]", el.modalRoot).forEach((button) => button.addEventListener("click", () => finish(false)));
      $("[data-confirm-ok]", el.modalRoot).addEventListener("click", () => finish(true));
      $("[data-modal-backdrop]", el.modalRoot).addEventListener("click", (event) => {
        if (event.target === event.currentTarget) finish(false);
      });
    });
  }

  async function deleteProvider(provider) {
    const confirmed = await confirmDialog(
      "删除供应商",
      `确定删除 <strong>${escapeHtml(provider.name)}</strong> 吗？已有对话不会被删除，但将无法再用这个供应商继续发送消息。`,
      "删除供应商",
    );
    if (!confirmed) return;
    try {
      await api(`/api/providers/${encodeURIComponent(provider.id)}`, { method: "DELETE" });
      toast("供应商已删除");
      await loadProviders();
      if (!state.providers.length) setRoute("settings");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function loadConversations(options = {}) {
    try {
      const data = await api("/api/conversations");
      state.conversations = data.conversations || [];
      renderHistory();
      return state.conversations;
    } catch (error) {
      if (!options.silent) toast(error.message, "error");
      return [];
    }
  }

  async function loadArchivedConversations(options = {}) {
    try {
      const data = await api("/api/conversations?archived=1");
      state.archivedConversations = data.conversations || [];
      renderArchiveList();
      return state.archivedConversations;
    } catch (error) {
      if (!options.silent) toast(error.message, "error");
      return [];
    }
  }

  function renderArchiveList() {
    const count = state.archivedConversations.length;
    el.archiveCount.textContent = String(count);
    if (!count) {
      el.archiveList.innerHTML = '<div class="archive-empty">归档区是空的。<br>在左侧历史对话上点归档按钮，对话会移动到这里。</div>';
      return;
    }
    el.archiveList.innerHTML = state.archivedConversations.map((conv) => {
      const model = conv.model_name || conv.model || "";
      const metaParts = [
        formatShortDate(conv.updated_at || conv.created_at),
        `${conv.message_count || 0} 条`,
      ];
      if (model) metaParts.push(model);
      return `
        <div class="archive-item" data-archive-id="${escapeHtml(conv.id)}">
          <div class="archive-item-main">
            <div class="archive-title">${escapeHtml(conv.title || "新对话")}</div>
            <div class="archive-preview">${escapeHtml(conv.preview || "（无预览）")}</div>
            <div class="archive-meta">${escapeHtml(metaParts.join(" · "))}</div>
          </div>
          <div class="archive-actions">
            <button class="secondary-btn" type="button" data-restore-conv="${escapeHtml(conv.id)}">恢复</button>
            <button class="secondary-btn danger-btn" type="button" data-delete-archived="${escapeHtml(conv.id)}">删除</button>
          </div>
        </div>`;
    }).join("");
  }

  function renderHistory() {
    el.historyCount.textContent = String(state.conversations.length);
    if (!state.conversations.length) {
      el.historyList.innerHTML = '<div class="history-empty">还没有历史对话。<br>发出第一条消息后，会自动保存在这里。</div>';
      return;
    }
    const groups = new Map();
    for (const conv of state.conversations) {
      const label = dateGroup(conv.updated_at || conv.created_at);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(conv);
    }
    const html = [];
    for (const [label, conversations] of groups) {
      html.push(`<div class="history-group-label">${escapeHtml(label)}</div>`);
      for (const conv of conversations) {
        const active = conv.id === state.currentId ? " active" : "";
        const meta = `${formatShortDate(conv.updated_at || conv.created_at)} · ${conv.message_count || 0} 条`;
        const generating = Array.from(state.streamJobs).some(
          (job) => job.conversationId && job.conversationId === conv.id
        );
        html.push(`
          <div class="history-item${active}" data-conv-id="${escapeHtml(conv.id)}" role="button" tabindex="0" title="${escapeHtml(conv.preview || conv.title)}">
            <span class="history-title">${escapeHtml(conv.title || "新对话")}${generating ? '<i class="history-live"></i>' : ""}</span>
            <span class="history-meta">${escapeHtml(meta)}</span>
            <span class="history-edit" data-rename-conv="${escapeHtml(conv.id)}" role="button" tabindex="-1" title="重命名对话">${icon("i-edit")}</span>
            <span class="history-archive" data-archive-conv="${escapeHtml(conv.id)}" role="button" tabindex="-1" title="归档对话">${icon("i-archive")}</span>
          </div>`);
      }
    }
    el.historyList.innerHTML = html.join("");
  }

  async function archiveConversation(convId) {
    if (jobForConversation(convId)) {
      toast("这个对话还在生成，请先停止或等它完成", "error");
      return;
    }
    try {
      await api(`/api/conversations/${encodeURIComponent(convId)}`, {
        method: "PATCH",
        body: { archived: true },
      });
      if (state.currentId === convId) {
        state.currentId = null;
        state.current = null;
        state.messages = [];
        refreshStreamingFlag();
        clearAttachments();
        renderMessages();
        setPageMeta("CONVERSATION", "新对话");
      }
      toast("已归档，可在设置 → 归档中删除");
      await Promise.all([loadConversations(), loadArchivedConversations({ silent: true })]);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function restoreConversation(convId) {
    try {
      await api(`/api/conversations/${encodeURIComponent(convId)}`, {
        method: "PATCH",
        body: { archived: false },
      });
      toast("对话已恢复");
      await Promise.all([loadConversations(), loadArchivedConversations({ silent: true })]);
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function deleteArchivedConversation(convId) {
    const conv = state.archivedConversations.find((item) => item.id === convId);
    const confirmed = await confirmDialog(
      "永久删除对话",
      `确定永久删除 <strong>${escapeHtml(conv?.title || "新对话")}</strong> 吗？删除后无法恢复。`,
      "永久删除",
    );
    if (!confirmed) return;
    try {
      await api(`/api/conversations/${encodeURIComponent(convId)}`, { method: "DELETE" });
      toast("对话已永久删除");
      await loadArchivedConversations();
      if (state.currentId === convId) {
        state.currentId = null;
        state.current = null;
        state.messages = [];
        refreshStreamingFlag();
        renderMessages();
        setPageMeta("CONVERSATION", "新对话");
      }
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function normalizeFontSize(value) {
    const legacy = { small: 90, normal: 100, large: 110, xlarge: 120 };
    const raw = legacy[value] ?? Number(value);
    const safe = Number.isFinite(raw) ? raw : 100;
    return Math.max(80, Math.min(140, Math.round(safe / 10) * 10));
  }

  function applyFontSize(value) {
    const next = normalizeFontSize(value);
    state.fontSize = next;
    document.documentElement.style.setProperty("--reading-font-scale", String(next / 100));
    store.set(STORE_KEYS.fontSize, String(next));
    if (el.generalFontValue) el.generalFontValue.textContent = next + "%";
    if (el.generalFontMinus) el.generalFontMinus.disabled = next <= 80;
    if (el.generalFontPlus) el.generalFontPlus.disabled = next >= 140;
  }

  function loadGeneralSettings() {
    applyFontSize(store.get(STORE_KEYS.fontSize, "100"));
  }

  function setSettingsTab(tab) {
    const next = ["general", "models", "prompts", "memory", "web", "archive"].includes(tab) ? tab : "models";
    state.settingsTab = next;
    $$("[data-settings-tab]", el.settingsTabs).forEach((button) => {
      button.classList.toggle("active", button.getAttribute("data-settings-tab") === next);
    });
    el.generalSettingsPanel.classList.toggle("hidden", next !== "general");
    el.modelSettingsPanel.classList.toggle("hidden", next !== "models");
    el.promptSettingsPanel.classList.toggle("hidden", next !== "prompts");
    el.memorySettingsPanel.classList.toggle("hidden", next !== "memory");
    el.webSettingsPanel.classList.toggle("hidden", next !== "web");
    el.archiveSettingsPanel.classList.toggle("hidden", next !== "archive");
    if (next === "prompts") loadPrompts();
    if (next === "memory") loadMemory({ silent: true });
    if (next === "web") loadWebSettings();
    if (next === "archive") loadArchivedConversations({ silent: true });
  }

  function renameConversation(convId) {
    const conv = state.conversations.find((item) => item.id === convId);
    if (!conv) return;
    showModal(`
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="renameTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="renameTitle">重命名对话</h2>
            <p class="modal-subtitle">只修改显示名称，不影响上下文和消息内容。</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <form id="renameForm" novalidate>
          <label class="field">
            <span class="field-label">对话名称 <span>最多 60 字</span></span>
            <span class="input-wrap"><input id="renameInput" type="text" maxlength="60" value="${escapeHtml(conv.title || "新对话")}" autocomplete="off"></span>
          </label>
          <div class="form-error hidden" id="renameError"></div>
          <div class="modal-actions">
            <button class="secondary-btn" type="button" data-modal-close>取消</button>
            <button class="primary-btn" type="submit" id="renameSubmit"><span>保存名称</span></button>
          </div>
        </form>
      </div>`);
    const form = $("#renameForm");
    const input = $("#renameInput");
    const submit = $("#renameSubmit");
    const errorBox = $("#renameError");
    window.setTimeout(() => { input.focus(); input.select(); }, 30);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const title = input.value.trim();
      errorBox.classList.add("hidden");
      if (!title) {
        errorBox.textContent = "对话名称不能为空";
        errorBox.classList.remove("hidden");
        return;
      }
      submit.disabled = true;
      submit.innerHTML = '<span class="spinner"></span><span>保存中…</span>';
      try {
        const data = await api(`/api/conversations/${encodeURIComponent(convId)}`, {
          method: "PATCH",
          body: { title },
        });
        const updated = data.conversation || { ...conv, title };
        const index = state.conversations.findIndex((item) => item.id === convId);
        if (index >= 0) state.conversations[index] = { ...state.conversations[index], ...updated };
        if (state.currentId === convId) {
          state.current = { ...(state.current || {}), ...updated };
          setPageMeta("CONVERSATION", updated.title || title);
        }
        closeModal();
        renderHistory();
        toast("对话已重命名");
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        submit.disabled = false;
        submit.innerHTML = "<span>保存名称</span>";
      }
    });
  }

  function openChatSearch() {
    showModal(`
      <div class="modal chat-search-modal" role="dialog" aria-modal="true" aria-labelledby="chatSearchTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="chatSearchTitle">搜索历史对话</h2>
            <p class="modal-subtitle">同时搜索标题、用户消息、回答和附件文字，包含已归档对话。</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <label class="field"><span>关键词</span><input class="input" id="chatSearchInput" maxlength="200" autocomplete="off" placeholder="输入标题或原文关键词…"></label>
        <div class="chat-search-results" id="chatSearchResults"><div class="memory-index-empty">输入关键词开始搜索。</div></div>
      </div>`);
    const input = $("#chatSearchInput", el.modalRoot);
    const results = $("#chatSearchResults", el.modalRoot);
    let timer = null;
    let sequence = 0;
    const runSearch = async () => {
      const query = input.value.trim();
      const current = ++sequence;
      if (!query) {
        results.innerHTML = '<div class="memory-index-empty">输入关键词开始搜索。</div>';
        return;
      }
      results.innerHTML = '<div class="memory-index-empty">正在搜索…</div>';
      try {
        const data = await api(`/api/conversations/search?q=${encodeURIComponent(query)}`);
        if (current !== sequence) return;
        const items = data.results || [];
        results.innerHTML = items.length
          ? items.map((item) => `<button class="chat-search-result" type="button" data-search-open="${escapeHtml(item.id)}" data-search-archived="${item.archived ? "1" : "0"}"><span class="chat-search-result-head"><strong>${escapeHtml(item.title || "对话")}</strong><small>${escapeHtml(item.match_type || "")}${item.archived ? " · 已归档" : ""}</small></span><span class="chat-search-result-snippet">${escapeHtml(item.snippet || "")}</span></button>`).join("")
          : '<div class="memory-index-empty">没有找到匹配的对话。</div>';
      } catch (error) {
        if (current === sequence) results.innerHTML = `<div class="form-error">${escapeHtml(error.message)}</div>`;
      }
    };
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = window.setTimeout(runSearch, 180);
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        clearTimeout(timer);
        runSearch();
      }
    });
    results.addEventListener("click", (event) => {
      const button = event.target.closest("[data-search-open]");
      if (!button) return;
      const convId = button.getAttribute("data-search-open") || "";
      if (!convId) return;
      const archived = button.getAttribute("data-search-archived") === "1";
      closeModal();
      openConversation(convId);
      if (archived) toast("该对话已归档，可以查看但不能继续提问");
    });
    window.setTimeout(() => input.focus(), 30);
  }

  async function openConversation(convId) {
    state.autoScroll = true;
    state.lastScrollTop = 0;
    const job = jobForConversation(convId);
    if (job) {
      state.currentId = convId;
      if (job.conversation) state.current = { ...job.conversation };
      state.messages = job.messages;
      applyConversationContext(job.conversation || state.current);
      refreshStreamingFlag();
      setRoute("chat", false);
      if (state.current) syncControlsFromConversation(state.current);
      setPageMeta("CONVERSATION", job.title || state.current?.title || "新对话");
      renderPromptPicker();
      renderMessages();
      renderHistory();
      closeMobileSidebar();
      requestAnimationFrame(() => scrollToBottom(false));
      return;
    }
    try {
      const data = await api(`/api/conversations/${encodeURIComponent(convId)}`);
      const conv = data.conversation;
      clearAttachments();
      state.current = conv;
      state.currentId = conv.id;
      state.messages = (conv.messages || []).map((message) => ({ ...message, uid: nextUid() }));
      applyConversationContext(conv);
      refreshStreamingFlag();
      setRoute("chat");
      syncControlsFromConversation(conv);
      setPageMeta("CONVERSATION", conv.title || "新对话");
      renderPromptPicker();
      renderMessages();
      renderHistory();
      closeMobileSidebar();
      requestAnimationFrame(() => scrollToBottom(false));
    } catch (error) {
      toast(error.message, "error");
      await loadConversations({ silent: true });
    }
  }

  function newChat() {
    state.currentId = null;
    state.current = null;
    state.messages = [];
    state.contextInfo = null;
    state.autoScroll = true;
    state.lastScrollTop = 0;
    refreshStreamingFlag();
    clearAttachments();
    setRoute("chat");
    setPageMeta("CONVERSATION", "新对话");
    renderPromptPicker();
    renderMessages();
    renderHistory();
    closeMobileSidebar();
    if (state.memorySettings.enabled) {
      api("/api/memory/scan", { method: "POST", body: {} }).catch(() => {});
    }
    el.composerInput.focus();
  }

  function setRoute(route, push = true) {
    const next = route === "settings" ? "settings" : "chat";
    state.route = next;
    el.chatPage.classList.toggle("hidden", next !== "chat");
    el.settingsPage.classList.toggle("hidden", next !== "settings");
    el.chatControls.classList.toggle("hidden", next !== "chat");
    el.navChat.classList.toggle("active", next === "chat");
    el.navSettings.classList.toggle("active", next === "settings");
    if (next === "settings") {
      setPageMeta("SETTINGS", "设置");
      renderProviderGrid();
      setSettingsTab(state.settingsTab);
    } else {
      setPageMeta("CONVERSATION", currentTitle());
      updateSendAvailability();
    }
    if (push) {
      const desired = next === "settings" ? "#settings" : "#chat";
      if (window.location.hash !== desired) {
        try { history.replaceState(null, "", desired); } catch (_) { window.location.hash = desired; }
      }
    }
    closeMobileSidebar();
  }

  function onProviderChanged() {
    const providerId = el.providerSelect.value;
    if (providerId) store.set(STORE_KEYS.provider, providerId);
    populateModelSelect();
    updateSendAvailability();
  }

  function onModelChanged() {
    const provider = selectedProvider();
    const model = selectedModel();
    if (provider && model) store.set(STORE_KEYS.model + provider.id, model.id);
    populateEffortSelect();
    updateSendAvailability();
  }

  function onEffortChanged() {
    const provider = selectedProvider();
    const model = selectedModel();
    if (provider && model && el.effortSelect.value) {
      store.set(STORE_KEYS.effort + provider.id + "." + model.id, el.effortSelect.value);
    }
  }

  const MATH_SYMBOLS = {
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
    zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
    lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ",
    tau: "τ", upsilon: "υ", phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
    Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
    Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
    partial: "∂", nabla: "∇", infty: "∞", cdot: "·", times: "×", div: "÷",
    pm: "±", mp: "∓", le: "≤", leq: "≤", ge: "≥", geq: "≥", ne: "≠", neq: "≠",
    approx: "≈", equiv: "≡", propto: "∝", sim: "∼", simeq: "≃",
    to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔",
    Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", implies: "⇒",
    Longrightarrow: "⟹", Longleftarrow: "⟸", Longleftrightarrow: "⟺",
    longrightarrow: "⟶", longleftarrow: "⟵", longleftrightarrow: "⟷", iff: "⇔",
    sum: "∑", prod: "∏", int: "∫", oint: "∮",
    ldots: "…", cdots: "⋯", dots: "…", therefore: "∴", because: "∵",
    in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆", cup: "∪", cap: "∩",
    forall: "∀", exists: "∃", angle: "∠", perp: "⊥", parallel: "∥",
    degree: "°", circ: "°", prime: "′",
  };
  const MATH_FUNCTIONS = new Set([
    "ln", "log", "lg", "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "exp",
    "lim", "max", "min", "sup", "inf", "det", "dim", "gcd",
  ]);
  const MATH_SPACES = new Set([",", ";", ":", "!", " ", "quad", "qquad"]);
  const MATH_BIG_OPS = { int: "∫", iint: "∬", iiint: "∭", oint: "∮", sum: "∑", prod: "∏", coprod: "∐" };
  const MATH_RELATIONS = new Set(["le", "leq", "ge", "geq", "ne", "neq", "approx", "equiv", "propto", "sim", "simeq", "to", "rightarrow", "leftarrow", "leftrightarrow", "Rightarrow", "Leftarrow", "Leftrightarrow", "implies", "Longrightarrow", "Longleftarrow", "Longleftrightarrow", "longrightarrow", "longleftarrow", "longleftrightarrow", "iff", "in", "notin", "subset", "subseteq"]);
  const MATH_BINARY = new Set(["pm", "mp", "cdot", "times", "div", "cup", "cap"]);

  function readMathGroup(text, start) {
    let depth = 0;
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (ch === "\\") { i += 1; continue; }
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) return { content: text.slice(start + 1, i), next: i + 1 };
      }
    }
    return { content: text.slice(start + 1), next: text.length };
  }

  function readMathAtom(text, start) {
    let i = start;
    while (i < text.length && /\s/.test(text[i])) i += 1;
    if (i >= text.length) return { content: "", next: i };
    if (text[i] === "{") return readMathGroup(text, i);
    if (text[i] === "\\") {
      let j = i + 1;
      if (/[A-Za-z]/.test(text[j] || "")) {
        while (j < text.length && /[A-Za-z]/.test(text[j])) j += 1;
      } else {
        j += 1;
      }
      while (text[j] === "{") {
        const group = readMathGroup(text, j);
        j = group.next;
      }
      return { content: text.slice(i, j), next: j };
    }
    return { content: text[i], next: i + 1 };
  }

  function readMathScripts(text, start) {
    let i = start;
    let upper = "";
    let lower = "";
    while (i < text.length && (text[i] === "_" || text[i] === "^")) {
      const isUpper = text[i] === "^";
      const arg = readMathAtom(text, i + 1);
      if (isUpper) upper = arg.content;
      else lower = arg.content;
      i = arg.next;
    }
    return { upper, lower, next: i };
  }

  function renderBigOperator(symbol, upper, lower, display, name) {
    const up = upper ? renderMathExpr(upper, false) : "";
    const lo = lower ? renderMathExpr(lower, false) : "";
    if (display && (name === "sum" || name === "prod" || name === "coprod")) {
      return `<span class="math-op math-op-stack">${up ? `<span class="math-op-upper">${up}</span>` : ""}<span class="math-op-symbol">${symbol}</span>${lo ? `<span class="math-op-lower">${lo}</span>` : ""}</span>`;
    }
    const limits = (up || lo)
      ? `<span class="math-op-limits">${up ? `<span class="math-op-upper">${up}</span>` : ""}${lo ? `<span class="math-op-lower">${lo}</span>` : ""}</span>`
      : "";
    return `<span class="math-op"><span class="math-op-symbol">${symbol}</span>${limits}</span>`;
  }

  function renderLimit(lower) {
    const lo = lower ? renderMathExpr(lower, false) : "";
    return `<span class="math-op math-op-lim"><span class="math-op-symbol">lim</span>${lo ? `<span class="math-op-lower">${lo}</span>` : ""}</span>`;
  }

  function readMathBracket(text, start) {
    if (text[start] !== "[") return { content: "", next: start, found: false };
    let depth = 0;
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (ch === "\\") { i += 1; continue; }
      if (ch === "[") depth += 1;
      else if (ch === "]") {
        depth -= 1;
        if (depth === 0) {
          return { content: text.slice(start + 1, i), next: i + 1, found: true };
        }
      }
    }
    return { content: text.slice(start + 1), next: text.length, found: true };
  }

  function renderXArrow(direction, above, below) {
    const aboveHtml = above ? `<span class="math-xarrow-above">${renderMathExpr(above, false)}</span>` : "";
    const belowHtml = below ? `<span class="math-xarrow-below">${renderMathExpr(below, false)}</span>` : "";
    return `<span class="math-xarrow ${direction === "left" ? "math-xarrow-left" : "math-xarrow-right"}">${aboveHtml}<span class="math-xarrow-line"></span>${belowHtml}</span>`;
  }

  function renderMathExpr(text, display) {
    let out = "";
    let i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (ch === "\\") {
        if (text[i + 1] === "\\") {
          i += 2;
          if (text[i] === ";") i += 1;
          out += display ? '<span class="math-line"></span>' : " ";
          continue;
        }
        let j = i + 1;
        let name = "";
        if (/[A-Za-z]/.test(text[j] || "")) {
          while (j < text.length && /[A-Za-z]/.test(text[j])) {
            name += text[j];
            j += 1;
          }
        } else {
          name = text[j] || "";
          j += 1;
        }
        if (MATH_BIG_OPS[name]) {
          const scripts = readMathScripts(text, j);
          out += renderBigOperator(MATH_BIG_OPS[name], scripts.upper, scripts.lower, display, name);
          out += "&thinsp;";
          i = scripts.next;
          continue;
        }
        if (name === "lim") {
          const scripts = readMathScripts(text, j);
          out += renderLimit(scripts.lower);
          out += "&thinsp;";
          i = scripts.next;
          continue;
        }
        if (name === "xrightarrow" || name === "xleftarrow") {
          const firstBracket = readMathBracket(text, j);
          let k = firstBracket.found ? firstBracket.next : j;
          let above = "";
          if (text[k] === "{") {
            const aboveArg = readMathAtom(text, k);
            above = aboveArg.content;
            k = aboveArg.next;
          }
          let below = firstBracket.content;
          if (!below) {
            const trailingBracket = readMathBracket(text, k);
            if (trailingBracket.found) {
              below = trailingBracket.content;
              k = trailingBracket.next;
            }
          }
          out += renderXArrow(name === "xleftarrow" ? "left" : "right", above, below);
          i = k;
          continue;
        }
        if (name === "underbrace" || name === "overbrace") {
          const arg = readMathAtom(text, j);
          const scripts = readMathScripts(text, arg.next);
          const label = name === "underbrace" ? scripts.lower : scripts.upper;
          const content = renderMathExpr(arg.content, false);
          const labelHtml = label ? `<span class="math-brace-label">${renderMathExpr(label, false)}</span>` : "";
          const brace = `<svg class="math-brace-svg" viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true"><path d="M1 1 C25 1 38 2 46 8 C48 10 52 10 54 8 C62 2 75 1 99 1" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
          if (name === "underbrace") {
            out += `<span class="math-underbrace"><span class="math-brace-content">${content}</span>${brace}${labelHtml}</span>`;
          } else {
            out += `<span class="math-overbrace">${labelHtml}${brace}<span class="math-brace-content">${content}</span></span>`;
          }
          i = scripts.next;
          continue;
        }
        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          const num = readMathAtom(text, j);
          const den = readMathAtom(text, num.next);
          out += `<span class="math-frac"><span class="math-num">${renderMathExpr(num.content, false)}</span><span class="math-den">${renderMathExpr(den.content, false)}</span></span>`;
          i = den.next;
          continue;
        }
        if (name === "sqrt") {
          let k = j;
          let index = "";
          if (text[k] === "[") {
            const close = text.indexOf("]", k);
            if (close !== -1) {
              index = renderMathExpr(text.slice(k + 1, close), false);
              k = close + 1;
            }
          }
          const rad = readMathAtom(text, k);
          const hook = `<svg class="math-radical-hook" viewBox="0 0 14 20" preserveAspectRatio="none" aria-hidden="true"><path d="M0 11 L5 19 L11 0 L14 0" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>`;
          out += `<span class="math-sqrt">${index ? `<span class="math-root-index">${index}</span>` : ""}<span class="math-radical-body">${hook}<span class="math-radicand">${renderMathExpr(rad.content, false)}</span></span></span>`;
          i = rad.next;
          continue;
        }
        if (name === "boxed") {
          const arg = readMathAtom(text, j);
          out += `<span class="math-box">${renderMathExpr(arg.content, false)}</span>`;
          i = arg.next;
          continue;
        }
        if (name === "text" || name === "mathrm" || name === "operatorname" || name === "textrm") {
          const arg = readMathAtom(text, j);
          out += `<span class="math-text">${escapeHtml(arg.content)}</span>`;
          i = arg.next;
          continue;
        }
        if (name === "mathbf" || name === "boldsymbol" || name === "textbf") {
          const arg = readMathAtom(text, j);
          out += `<strong>${renderMathExpr(arg.content, false)}</strong>`;
          i = arg.next;
          continue;
        }
        if (name === "overline") {
          const arg = readMathAtom(text, j);
          out += `<span class="math-overline">${renderMathExpr(arg.content, false)}</span>`;
          i = arg.next;
          continue;
        }
        if (name === "vec") {
          const arg = readMathAtom(text, j);
          out += `<span class="math-vec">${renderMathExpr(arg.content, false)}</span>`;
          i = arg.next;
          continue;
        }
        if (name === "left" || name === "right" || name === "big" || name === "Big" || name === "bigl" || name === "bigr" || name === "Bigl" || name === "Bigr") {
          i = j;
          continue;
        }
        if (name === "_") {
          const arg = readMathAtom(text, j);
          out += `<sub>${renderMathExpr(arg.content, false)}</sub>`;
          i = arg.next;
          continue;
        }
        if (name === "{" || name === "}" || name === "%" || name === "$" || name === "&" || name === "#") {
          out += escapeHtml(name);
          i = j;
          continue;
        }
        if (name === " ") {
          out += " ";
          i = j;
          continue;
        }
        if (MATH_SPACES.has(name)) {
          out += name === "quad" || name === "qquad" ? "&emsp;" : "&thinsp;";
          i = j;
          continue;
        }
        if (MATH_FUNCTIONS.has(name)) {
          out += `<span class="math-fn">${escapeHtml(name)}</span>`;
          i = j;
          continue;
        }
        if (MATH_SYMBOLS[name]) {
          const symbol = MATH_SYMBOLS[name];
          if (MATH_RELATIONS.has(name)) out += `<span class="math-rel">${symbol}</span>`;
          else if (MATH_BINARY.has(name)) out += `<span class="math-bin">${symbol}</span>`;
          else out += symbol;
          i = j;
          continue;
        }
        out += escapeHtml(text.slice(i, j));
        i = j;
        continue;
      }
      if (ch === "{" || ch === "}") {
        i += 1;
        continue;
      }
      if (ch === "^" || ch === "_") {
        const arg = readMathAtom(text, i + 1);
        out += ch === "^" ? `<sup>${renderMathExpr(arg.content, false)}</sup>` : `<sub>${renderMathExpr(arg.content, false)}</sub>`;
        i = arg.next;
        continue;
      }
      if (ch === "&") {
        out += "&thinsp;";
        i += 1;
        continue;
      }
      if (/\s/.test(ch)) {
        let k = i;
        while (k < text.length && /\s/.test(text[k])) k += 1;
        const prev = text[i - 1] || "";
        const next = text[k] || "";
        const isWord = (c) => /[A-Za-z0-9]/.test(c) || c === "\\" || c === "(" || c === "[";
        if (isWord(prev) && isWord(next)) out += "&thinsp;";
        i = k;
        continue;
      }
      if (ch === "=" || ch === "<" || ch === ">") {
        out += `<span class="math-rel">${escapeHtml(ch)}</span>`;
        i += 1;
        continue;
      }
      if (ch === "-") {
        out += "−";
        i += 1;
        continue;
      }
      out += escapeHtml(ch);
      i += 1;
    }
    return out;
  }

  function repairLatexBraces(text) {
    let value = String(text);
    let depth = 0;
    let out = "";
    for (let i = 0; i < value.length; i += 1) {
      const ch = value[i];
      if (ch === "\\") {
        out += ch;
        if (i + 1 < value.length) { out += value[i + 1]; i += 1; }
        continue;
      }
      if (ch === "{") { depth += 1; out += ch; continue; }
      if (ch === "}") {
        if (depth === 0) continue; // Drop a stray closing brace that would break the whole formula.
        depth -= 1; out += ch; continue;
      }
      out += ch;
    }
    if (depth > 0) out += "}".repeat(depth);
    return out;
  }

  function stripUnsupportedLatex(text) {
    let value = String(text);
    // mhchem is not part of KaTeX core. Render common chemistry as upright text
    // instead of showing the entire expression as a red KaTeX error.
    const subDigits = { "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉" };
    const supDigits = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
    const plainChemistry = (inner) => String(inner)
      .replace(/\s+/g, "")
      .replace(/[{}]/g, "")
      .replace(/\\/g, "")
      .replace(/([A-Za-z)])([0-9]+)/g, (_, token, digits) => token + Array.from(digits).map((ch) => subDigits[ch] || ch).join(""))
    let previous;
    do {
      previous = value;
      value = value.replace(/\\ce\s*\{([^{}]*)\}/g, (_, inner) => `\\mathrm{${plainChemistry(inner)}}`);
    } while (value !== previous);
    return value;
  }

  function renderMath(expr, display) {
    let text = repairLatexBraces(String(expr).trim());
    if (!window.katex || window.__mhchemLoadFailed) text = stripUnsupportedLatex(text);
    if (window.katex && typeof window.katex.renderToString === "function") {
      try {
        const body = window.katex.renderToString(text, {
          displayMode: Boolean(display),
          throwOnError: false,
          strict: "ignore",
          trust: false,
          output: "htmlAndMathml",
          errorColor: "#ff8290",
        });
        if (body.includes("katex-error")) throw new Error("KaTeX could not parse the expression");
        return display
          ? `<div class="math-display">${body}</div>`
          : `<span class="math-inline">${body}</span>`;
      } catch (error) {
        console.warn("KaTeX 渲染失败，改用内置降级渲染器", error);
      }
    }
    let fallback = stripUnsupportedLatex(text);
    fallback = fallback.replace(/\\(?:displaystyle|textstyle|limits|nolimits)/g, "");
    fallback = fallback.replace(/\\begin\{[^}]*\}/g, "").replace(/\\end\{[^}]*\}/g, "");
    const body = renderMathExpr(fallback, display);
    return display ? `<div class="math-display">${body}</div>` : `<span class="math-inline">${body}</span>`;
  }

  function renderInline(text) {
    let value = String(text);
    const inlineCodes = [];
    value = value.replace(/`([^`\n]+)`/g, (_match, code) => {
      const token = `\u0001IC${inlineCodes.length}\u0001`;
      inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
      return token;
    });
    const inlineMath = [];
    value = value.replace(/\$\$([^$\n]+?)\$\$/g, (_match, expr) => {
      const token = `\u0002IM${inlineMath.length}\u0002`;
      inlineMath.push(renderMath(expr, true));
      return token;
    });
    value = value.replace(/(?<!\\)\$([^$\n]+?)\$/g, (_match, expr) => {
      if (/^\s|\s$/.test(expr)) return _match;
      const token = `\u0002IM${inlineMath.length}\u0002`;
      inlineMath.push(renderMath(expr, false));
      return token;
    });
    value = value.replace(/\\\(([\s\S]*?)\\\)/g, (_match, expr) => {
      const token = `\u0002IM${inlineMath.length}\u0002`;
      inlineMath.push(renderMath(expr, false));
      return token;
    });
    value = escapeHtml(value);
    value = value.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
    value = value.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    value = value.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    value = value.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    value = value.replace(/\u0001IC(\d+)\u0001/g, (_match, index) => inlineCodes[Number(index)] || "");
    value = value.replace(/\u0002IM(\d+)\u0002/g, (_match, index) => inlineMath[Number(index)] || "");
    return value;
  }

  function splitTableRow(line) {
    let text = String(line).trim();
    if (text.startsWith("|")) text = text.slice(1);
    if (text.endsWith("|")) text = text.slice(0, -1);
    const cells = [];
    let current = "";
    let inCode = false;
    let mathMode = "";
    for (let i = 0; i < text.length; i += 1) {
      const ch = text[i];
      const next = text[i + 1] || "";
      if (ch === "\\") {
        if (next === "(") { mathMode = "paren"; current += "\\("; i += 1; continue; }
        if (next === ")" && mathMode === "paren") { mathMode = ""; current += "\\)"; i += 1; continue; }
        if (next === "[") { mathMode = "bracket"; current += "\\["; i += 1; continue; }
        if (next === "]" && mathMode === "bracket") { mathMode = ""; current += "\\]"; i += 1; continue; }
        if (next === "|") {
          current += (inCode || mathMode) ? "\\|" : "|";
          i += 1;
          continue;
        }
        if (next === "\\") { current += "\\\\"; i += 1; continue; }
        current += ch;
        continue;
      }
      if (ch === "`" && !mathMode) {
        inCode = !inCode;
        current += ch;
        continue;
      }
      if (ch === "$" && !inCode) {
        if (!mathMode && next === "$") {
          mathMode = "dollar2";
          current += "$$";
          i += 1;
          continue;
        }
        if (!mathMode) {
          mathMode = "dollar";
          current += ch;
          continue;
        }
        if (mathMode === "dollar") {
          mathMode = "";
          current += ch;
          continue;
        }
        if (mathMode === "dollar2" && next === "$") {
          mathMode = "";
          current += "$$";
          i += 1;
          continue;
        }
        current += ch;
        continue;
      }
      if (ch === "|" && !inCode && !mathMode) {
        cells.push(current.trim());
        current = "";
        continue;
      }
      current += ch;
    }
    cells.push(current.trim());
    return cells;
  }

  function isTableSeparator(line) {
    const cells = splitTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  function renderTable(lines) {
    const headers = splitTableRow(lines[0]);
    const aligns = splitTableRow(lines[1]).map((cell) => {
      const left = cell.startsWith(":");
      const right = cell.endsWith(":");
      if (left && right) return "center";
      if (right) return "right";
      if (left) return "left";
      return "";
    });
    const bodyRows = lines.slice(2).filter((line) => line.trim()).map(splitTableRow);
    const headHtml = headers.map((cell, index) => {
      const align = aligns[index] ? ` class="align-${aligns[index]}"` : "";
      return `<th${align}>${renderInline(cell)}</th>`;
    }).join("");
    const bodyHtml = bodyRows.map((cells) => {
      const rowCells = headers.map((_header, index) => {
        const cell = cells[index] || "";
        const align = aligns[index] ? ` class="align-${aligns[index]}"` : "";
        return `<td${align}>${renderInline(cell)}</td>`;
      }).join("");
      return `<tr>${rowCells}</tr>`;
    }).join("");
    return `<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
  }

  function renderMarkdown(raw) {
    if (!raw) return "";
    let text = String(raw).replace(/\r\n?/g, "\n");
    const codeBlocks = [];
    text = text.replace(/```([^\n`]*)\n([\s\S]*?)(?:```|$)/g, (_match, info, code) => {
      const lang = String(info || "").trim().split(/\s+/)[0] || "text";
      const index = codeBlocks.length;
      codeBlocks.push({ lang, code: String(code).replace(/\n$/, "") });
      return `\n\n\u0000CB${index}\u0000\n\n`;
    });

    const mathBlocks = [];
    const storeDisplayMath = (_match, expr) => {
      const token = `\u0002DM${mathBlocks.length}\u0002`;
      mathBlocks.push(renderMath(expr, true));
      return `\n\n${token}\n\n`;
    };
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, storeDisplayMath);
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, storeDisplayMath);

    const blocks = text.split(/\n{2,}/);
    const html = [];
    for (let block of blocks) {
      block = block.replace(/^\n+|\n+$/g, "");
      if (!block.trim()) continue;
      const codeMatch = block.trim().match(/^\u0000CB(\d+)\u0000$/);
      if (codeMatch) {
        html.push(`\u0000CB${codeMatch[1]}\u0000`);
        continue;
      }
      const mathMatch = block.trim().match(/^\u0002DM(\d+)\u0002$/);
      if (mathMatch) {
        html.push(mathBlocks[Number(mathMatch[1])] || "");
        continue;
      }
      const lines = block.split("\n");
      if (lines.length >= 2 && lines[0].includes("|") && isTableSeparator(lines[1])) {
        html.push(renderTable(lines));
        continue;
      }
      const heading = block.match(/^(#{1,4})\s+(.+)$/);
      if (heading && lines.length === 1) {
        const level = heading[1].length;
        html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }
      if (/^([-*_])(?:\s*\1){2,}$/.test(block.trim())) {
        html.push("<hr>");
        continue;
      }
      if (lines.every((line) => /^\s*>/.test(line))) {
        const body = lines.map((line) => line.replace(/^\s*>\s?/, "")).join("<br>");
        html.push(`<blockquote>${renderInline(body)}</blockquote>`);
        continue;
      }
      if (lines.every((line) => /^\s*[-*+]\s+/.test(line))) {
        const items = lines.map((line) => `<li>${renderInline(line.replace(/^\s*[-*+]\s+/, ""))}</li>`).join("");
        html.push(`<ul>${items}</ul>`);
        continue;
      }
      if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) {
        const items = lines.map((line) => `<li>${renderInline(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("");
        html.push(`<ol>${items}</ol>`);
        continue;
      }
      html.push(`<p>${lines.map(renderInline).join("<br>")}</p>`);
    }

    let result = html.join("");
    result = result.replace(/\u0002DM(\d+)\u0002/g, (_match, index) => mathBlocks[Number(index)] || "");
    result = result.replace(/\u0000CB(\d+)\u0000/g, (_match, index) => {
      const block = codeBlocks[Number(index)];
      if (!block) return "";
      return `<div class="code-wrap"><div class="code-head"><span class="code-lang">${escapeHtml(block.lang)}</span><button class="copy-code" type="button" data-copy-code>复制</button></div><pre><code>${escapeHtml(block.code)}</code></pre></div>`;
    });
    return result;
  }

  function cacheBadge(usage) {
    if (!usage) return "";
    const prompt = Number(usage.prompt_tokens) || 0;
    const cached = Number(usage.cached_tokens) || 0;
    if (!prompt) return "";
    const ratio = Math.max(0, Math.min(100, Math.round((cached / prompt) * 100)));
    const title = `本次请求 ${formatNumber(prompt)} 输入 tokens，其中 ${formatNumber(cached)} tokens 命中缓存`;
    return `<span class="cache-badge dot-badge ${ratio > 0 ? "hit" : ""}" title="${escapeHtml(title)}">缓存 ${ratio}%</span>`;
  }

  function usageMeta(usage) {
    if (!usage) return "";
    const parts = [];
    if (Number(usage.total_tokens)) parts.push(`${formatNumber(usage.total_tokens)} tokens`);
    if (Number(usage.reasoning_tokens)) parts.push(`${formatNumber(usage.reasoning_tokens)} 思考`);
    return parts.map((part) => `<span>${escapeHtml(part)}</span>`).join('<span class="dot">•</span>');
  }

  function renderToolEvents(events) {
    if (!Array.isArray(events) || !events.length) return "";
    const labels = { web_search: "联网搜索", memory_search: "搜索记忆", memory_read_conversation: "读取历史对话", memory_save: "写入记忆" };
    return events.map((event) => {
      const name = event.name || "web_search";
      const label = labels[name] || name;
      const status = event.status === "searching"
        ? "调用中…"
        : event.status === "error"
          ? "失败"
          : name === "web_search"
            ? `完成 · ${event.result_count || 0} 条`
            : "完成";
      const cls = event.status === "error" ? " error" : event.status === "searching" ? " searching" : "";
      const iconId = name === "web_search" ? "i-globe" : "i-chat";
      return `<div class="tool-event${cls}">${icon(iconId)}<span class="tool-event-name">${escapeHtml(label)}</span><span class="tool-event-query">${escapeHtml(event.query || "自动判断")}</span><span class="tool-event-status">${escapeHtml(status)}</span></div>`;
    }).join("");
  }

  function renderSearchSources(sources) {
    const items = Array.isArray(sources) ? sources : [];
    if (!items.length) return "";
    const visible = items.slice(0, 3);
    const rest = items.slice(3);
    const link = (source) => `<a href="${escapeHtml(source.url || "#")}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(source.title || source.url || "")}">${escapeHtml(source.title || source.url || "来源")}</a>`;
    return `<div class="search-sources">${visible.map(link).join("")}${rest.length ? `<details class="source-more"><summary>其余 ${rest.length} 个来源</summary><div class="source-more-list">${rest.map(link).join("")}</div></details>` : ""}</div>`;
  }

  function messageHtml(message, options = {}) {
    const uid = message.uid || (message.uid = nextUid());
    if (message.role === "user") {
      const images = Array.isArray(message.images) ? message.images : [];
      const files = Array.isArray(message.files) ? message.files : [];
      const imageHtml = images.length
        ? `<div class="user-images">${images.map((url, index) => `<img class="user-image" src="${escapeHtml(url)}" alt="图片 ${index + 1}" data-view-image="${index}" data-view-uid="${escapeHtml(uid)}" loading="lazy">`).join("")}</div>`
        : "";
      const fileHtml = files.length
        ? `<div class="user-files">${files.map((file, index) => `<button class="user-file" type="button" data-view-file="${index}" data-view-uid="${escapeHtml(uid)}" title="查看提取的文字">${icon("i-file")}<span>${escapeHtml(file.name || "附件")}</span><small>${formatNumber(file.chars || (file.text || "").length)} 字${Array.isArray(file.page_images) && file.page_images.length ? ` · ${file.page_images.length} 图` : ""}</small></button>`).join("")}</div>`
        : "";
      const textHtml = message.content ? `<div class="user-text">${escapeHtml(message.content)}</div>` : "";
      const editButton = options.editable
        ? `<button class="msg-action" type="button" data-edit-turn="${options.turnIndex}">编辑</button>`
        : "";
      return `
        <div class="message user" data-uid="${escapeHtml(uid)}">
          <div class="user-bubble">${fileHtml}${imageHtml}${textHtml}</div>
          <div class="msg-time">${escapeHtml(formatTime(message.created_at))}${editButton ? ` · ${editButton}` : ""}</div>
        </div>`;
    }
    const reasoning = message.reasoning || "";
    const content = message.content || "";
    const modelLabel = message.model || state.current?.model || "";
    const meta = [];
    if (modelLabel) meta.push(`<span>${escapeHtml(modelLabel)}</span>`);
    if (message.created_at) meta.push(`<span>${escapeHtml(formatTime(message.created_at))}</span>`);
    const usagePieces = usageMeta(message.usage);
    if (usagePieces) meta.push(usagePieces);
    const badge = cacheBadge(message.usage);
    if (badge) meta.push(badge);
    if (content) meta.push(`<button class="msg-action" type="button" data-copy-message="${escapeHtml(uid)}">复制</button>`);
    return `
      <div class="message assistant" data-uid="${escapeHtml(uid)}">
        <div class="assistant-avatar">${icon("i-spark")}</div>
        <div class="assistant-body">
          <div class="assistant-head"><span class="assistant-name">小小肥鱼</span><span class="assistant-model">${escapeHtml(message.model_name || modelLabel || "")}</span></div>
          <details class="reasoning ${reasoning ? "" : "hidden"}" ${reasoning ? "" : "data-empty"}>
            <summary><span>思考过程</span><span class="reasoning-count">${reasoning ? ` · ${reasoning.length} 字` : ""}</span></summary>
            <div class="reasoning-text">${escapeHtml(reasoning)}</div>
          </details>
          <div class="tool-events ${Array.isArray(message.tool_events) && message.tool_events.length ? "" : "hidden"}">${renderToolEvents(message.tool_events)}</div>
          <div class="thinking-indicator ${message.streaming && !content ? "" : "hidden"}"><span class="thinking-text">${escapeHtml(message.status || "正在思考")}</span><i></i><i></i><i></i></div>
          <div class="markdown ${message.streaming ? "streaming" : ""} ${content ? "" : "hidden"}">${content ? renderMarkdown(content) : ""}</div>
          ${renderSearchSources(message.search_sources)}
          ${message.notice ? `<div class="notice-line">${escapeHtml(message.notice)}</div>` : ""}
          ${message.stopped ? `<div class="notice-line">已停止生成</div>` : ""}
          ${message.error ? `<div class="error-box">${escapeHtml(message.error)}</div>` : ""}
          <div class="message-meta">${meta.join('<span class="dot">•</span>')}</div>
        </div>
      </div>`;
  }

  function buildTurns(messages) {
    const turns = [];
    for (const message of messages) {
      if (message.role === "user") {
        turns.push({ user: message, assistants: [] });
      } else if (message.role === "assistant" && turns.length) {
        turns[turns.length - 1].assistants.push(message);
      }
    }
    return turns;
  }

  function clampVariant(turn) {
    if (!turn.assistants.length) return 0;
    const value = Number(turn.user.active_variant);
    const index = Number.isFinite(value) ? value : 0;
    return Math.max(0, Math.min(index, turn.assistants.length - 1));
  }

  const CONTEXT_WARN_RATIO = 0.85;
  const IMAGE_TOKEN_ESTIMATE = 900;
  const CONTEXT_LIMIT_FALLBACK = 1_000_000;

  function estimateTextTokens(text) {
    if (!text) return 0;
    let cjk = 0;
    let other = 0;
    for (const ch of String(text)) {
      const code = ch.codePointAt(0) || 0;
      const isCjk = (code >= 0x2e80 && code <= 0x9fff) || (code >= 0xf900 && code <= 0xfaff)
        || (code >= 0x3040 && code <= 0x30ff) || (code >= 0xac00 && code <= 0xd7af);
      if (isCjk) cjk += 1; else other += 1;
    }
    return Math.round(cjk * 0.75 + other / 3.2) + 1;
  }

  function localContextTokens() {
    let total = 0;
    for (const message of state.messages) {
      total += 4;
      if (message.role === "user") {
        total += estimateTextTokens(message.content || "");
        total += (message.images || []).length * IMAGE_TOKEN_ESTIMATE;
        for (const file of message.files || []) {
          total += estimateTextTokens(file.text || "");
          total += (file.page_images || []).length * IMAGE_TOKEN_ESTIMATE;
        }
      } else {
        total += estimateTextTokens(message.content || "");
      }
    }
    return total;
  }

  function contextLimit() {
    const model = selectedModel();
    const fromModel = Number(model?.context_window);
    if (Number.isFinite(fromModel) && fromModel > 0) return fromModel;
    const fromServer = Number(state.contextInfo?.limit);
    if (Number.isFinite(fromServer) && fromServer > 0) return fromServer;
    return CONTEXT_LIMIT_FALLBACK;
  }

  function compressableTurns() {
    const turns = buildTurns(state.messages).length;
    const covered = Number(state.contextInfo?.digest_turns) || 0;
    return turns - covered;
  }

  function overflowFlag() {
    return state.messages.some((message) => message.context_overflow);
  }

  function renderContextBar() {
    const bar = el.contextBar;
    if (!bar) return;
    if (!state.messages.length) {
      bar.classList.add("hidden");
      return;
    }
    const limit = contextLimit();
    const serverTokens = Number(state.contextInfo?.tokens);
    const tokens = Number.isFinite(serverTokens) && serverTokens > 0 ? serverTokens : localContextTokens();
    const ratio = limit > 0 ? tokens / limit : 0;
    const over = overflowFlag() || ratio >= 1;
    const warn = !over && ratio >= CONTEXT_WARN_RATIO;
    bar.classList.remove("hidden");
    bar.classList.toggle("warn", warn);
    bar.classList.toggle("over", over);
    const percent = Math.min(100, Math.round(ratio * 100));
    el.contextFill.style.width = Math.max(2, Math.min(100, ratio * 100)).toFixed(1) + "%";
    const digestTurns = Number(state.contextInfo?.digest_turns) || 0;
    const digestTag = digestTurns ? " · 已压缩" : "";
    el.contextText.innerHTML = `<b>${formatNumber(tokens)}</b> / ${formatNumber(limit)}${escapeHtml(digestTag)}`;
    el.contextBar.title = over
      ? "上下文已到上限，请压缩上下文或换个新对话"
      : warn
        ? "上下文接近上限，建议压缩"
        : `上下文占用 ${percent}%${digestTag ? "，较早内容已压缩" : ""}`;
    const canCompress = compressableTurns() >= 3 && !currentJob() && Boolean(state.currentId);
    el.compressBtn.classList.toggle("hidden", !canCompress);
    el.compressBtn.disabled = !canCompress;
  }

  async function compressContext() {
    if (!state.currentId) {
      toast("先发送一条消息，再压缩上下文", "error");
      return;
    }
    if (currentJob()) {
      toast("当前回答还在生成，请稍后压缩", "error");
      return;
    }
    if (compressableTurns() < 3) {
      toast("对话还太短，暂时不需要压缩", "error");
      return;
    }
    const provider = selectedProvider() || state.providers.find((item) => item.id === state.current?.provider_id);
    const model = selectedModel() || modelById(provider, state.current?.model);
    if (!provider || !model) {
      toast("请选择可用的供应商和模型", "error");
      return;
    }
    el.compressBtn.disabled = true;
    el.contextText.textContent = "正在压缩上下文…";
    try {
      const data = await api(`/api/conversations/${encodeURIComponent(state.currentId)}/compress`, {
        method: "POST",
        body: {
          provider_id: provider.id,
          model: model.id,
          reasoning_effort: el.effortSelect.value || "",
        },
      });
      const conv = data.conversation;
      if (conv) {
        state.current = { ...(state.current || {}), ...conv };
        state.contextInfo = conv.context || null;
      }
      toast(`上下文已压缩：${formatNumber(data.before_tokens)} → ${formatNumber(data.after_tokens)} tokens`);
      renderContextBar();
      loadConversations({ silent: true });
    } catch (error) {
      toast(error.message, "error");
    } finally {
      el.compressBtn.disabled = false;
      renderContextBar();
    }
  }

  function applyConversationContext(conv) {
    if (conv && conv.context) state.contextInfo = conv.context;
  }

  function renderMessages() {
    el.emptyState.classList.toggle("hidden", state.messages.length > 0);
    const turns = buildTurns(state.messages);
    const lastIndex = turns.length - 1;
    el.messageList.innerHTML = turns.map((turn, index) => {
      const activeIndex = clampVariant(turn);
      if (turn.assistants.length) turn.user.active_variant = activeIndex;
      const assistant = turn.assistants[activeIndex];
      const userHtml = messageHtml(turn.user, {
        editable: index === lastIndex && !currentJob(),
        turnIndex: index,
      });
      const assistantHtml = assistant ? messageHtml(assistant) : "";
      const variantBar = turn.assistants.length > 1
        ? `<div class="variant-bar">
            <button class="variant-btn" type="button" data-variant-prev="${index}" ${currentJob() ? "disabled" : ""} aria-label="上一条回复">‹</button>
            <span>${activeIndex + 1} / ${turn.assistants.length}</span>
            <button class="variant-btn" type="button" data-variant-next="${index}" ${currentJob() ? "disabled" : ""} aria-label="下一条回复">›</button>
          </div>`
        : "";
      const regenerate = index === lastIndex && assistant && !currentJob()
        ? `<button class="turn-action-btn" type="button" data-regenerate-turn="${index}">重新生成</button>`
        : "";
      const actions = (variantBar || regenerate)
        ? `<div class="turn-actions">${variantBar}${regenerate}</div>`
        : "";
      return `<div class="turn" data-turn-index="${index}">${userHtml}${assistantHtml}${actions}</div>`;
    }).join("");
    updateSendAvailability();
    renderContextBar();
    renderTurnNav();
    if (state.autoScroll) {
      requestAnimationFrame(() => followBottom());
    }
  }

  function renderTurnNav() {
    if (!el.turnNav || !el.turnNavList) return;
    const turns = buildTurns(state.messages);
    const visible = turns.length > 1;
    el.turnNav.classList.toggle("hidden", !visible);
    el.chatPage.classList.toggle("has-turn-nav", visible);
    el.turnNavCount.textContent = turns.length ? String(turns.length) : "";
    if (!visible) return;
    el.turnNavList.innerHTML = turns.map((turn, index) => {
      const text = String(turn.user.content || "").trim()
        || (turn.user.images?.length ? "[图片]" : turn.user.files?.length ? "[附件]" : "问题");
      return `<button class="turn-nav-item" type="button" data-nav-turn="${index}" title="${escapeHtml(text)}"><span class="turn-nav-dot"></span><span class="turn-nav-text">${escapeHtml(text)}</span></button>`;
    }).join("");
    syncTurnNav();
  }

  function scrollTurnIntoView(index) {
    const turn = el.messageList.querySelector(`[data-turn-index="${index}"]`);
    if (!turn) return;
    pauseAutoScroll(1200);
    const node = el.chatScroll;
    const top = turn.getBoundingClientRect().top - node.getBoundingClientRect().top + node.scrollTop - 12;
    node.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }

  function syncTurnNav() {
    if (!el.turnNavList || el.turnNav.classList.contains("hidden")) return;
    const node = el.chatScroll;
    const top = node.getBoundingClientRect().top + 28;
    const items = Array.from(el.turnNavList.querySelectorAll("[data-nav-turn]"));
    let active = 0;
    for (const item of items) {
      const index = Number(item.getAttribute("data-nav-turn"));
      const turn = el.messageList.querySelector(`[data-turn-index="${index}"]`);
      if (turn && turn.getBoundingClientRect().top <= top) active = index;
    }
    items.forEach((item) => item.classList.toggle("active", Number(item.getAttribute("data-nav-turn")) === active));
  }

  const STICK_THRESHOLD = 120;

  function isNearBottom() {
    const node = el.chatScroll;
    return node.scrollHeight - node.scrollTop - node.clientHeight < STICK_THRESHOLD;
  }

  function syncAutoScroll() {
    // 只有用户主动向上滚动时才停止跟随；回答继续变长不会改变 scrollTop，
    // 因此自动跟随不会被内容增长误判成"用户在看上面"。
    const node = el.chatScroll;
    const top = node.scrollTop;
    const delta = top - state.lastScrollTop;
    state.lastScrollTop = top;
    if (delta < -2) {
      pauseAutoScroll();
    } else if (delta > 0 && isNearBottom() && performance.now() >= state.scrollPauseUntil) {
      state.autoScroll = true;
    }
    el.jumpWrap.classList.toggle("hidden", state.autoScroll && isNearBottom());
    syncTurnNav();
  }

  function pauseAutoScroll(durationMs = 1200) {
    state.autoScroll = false;
    state.scrollPauseUntil = performance.now() + durationMs;
    el.jumpWrap.classList.remove("hidden");
  }

  function followBottom() {
    if (!state.autoScroll || performance.now() < state.scrollPauseUntil) return;
    const node = el.chatScroll;
    node.scrollTop = node.scrollHeight;
  }

  function followStreamingRow(row) {
    if (!state.autoScroll || performance.now() < state.scrollPauseUntil) return;
    const node = el.chatScroll;
    const rowTop = row.getBoundingClientRect().top - node.getBoundingClientRect().top;
    if (rowTop <= 8) {
      pauseAutoScroll(0);
      return;
    }
    node.scrollTop = node.scrollHeight;
  }

  function scrollToBottom(smooth = true) {
    const node = el.chatScroll;
    state.autoScroll = true;
    state.scrollPauseUntil = 0;
    el.jumpWrap.classList.add("hidden");
    node.scrollTo({ top: node.scrollHeight, behavior: smooth ? "smooth" : "instant" });
  }

  function stickToBottomIfFollowing() {
    followBottom();
  }

  function updateStreamingAssistant(message) {
    const row = el.messageList.querySelector(`[data-uid="${message.uid}"]`);
    if (!row) {
      renderMessages();
      return;
    }
    const reasoningBox = row.querySelector(".reasoning");
    const reasoningText = row.querySelector(".reasoning-text");
    const reasoningCount = row.querySelector(".reasoning-count");
    if (message.reasoning) {
      reasoningBox.classList.remove("hidden");
      reasoningText.textContent = message.reasoning;
      reasoningCount.textContent = ` · ${message.reasoning.length} 字`;
      if (!message._reasoningOpened && !message.content) {
        reasoningBox.open = true;
        message._reasoningOpened = true;
        message._reasoningAutoOpened = true;
      }
    }
    const toolBox = row.querySelector(".tool-events");
    if (toolBox) {
      toolBox.innerHTML = renderToolEvents(message.tool_events);
      toolBox.classList.toggle("hidden", !Array.isArray(message.tool_events) || !message.tool_events.length);
    }
    const thinking = row.querySelector(".thinking-indicator");
    const thinkingText = row.querySelector(".thinking-text");
    if (thinkingText) thinkingText.textContent = message.status || "正在思考";
    const markdown = row.querySelector(".markdown");
    if (message.content) {
      if (message._reasoningAutoOpened) {
        reasoningBox.open = false;
        message._reasoningAutoOpened = false;
      }
      thinking.classList.add("hidden");
      markdown.classList.remove("hidden");
      markdown.innerHTML = renderMarkdown(message.content);
      markdown.classList.add("streaming");
    } else {
      thinking.classList.toggle("hidden", Boolean(message.reasoning || message.error));
      markdown.classList.add("hidden");
    }
    if (message.notice) {
      let notice = row.querySelector(".notice-line");
      if (!notice) {
        notice = document.createElement("div");
        notice.className = "notice-line";
        row.querySelector(".markdown").before(notice);
      }
      notice.textContent = message.notice;
    }
    followStreamingRow(row);
  }

  function queueStreamRender(message) {
    const job = jobForAssistant(message);
    if (!job || job.messages !== state.messages) return;
    if (message._frame) return;
    message._frame = requestAnimationFrame(() => {
      message._frame = null;
      updateStreamingAssistant(message);
    });
  }

  async function copyText(text, label = "已复制") {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      toast(label);
    } catch (_) {
      const helper = document.createElement("textarea");
      helper.value = text;
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      try {
        document.execCommand("copy");
        toast(label);
      } catch (error) {
        toast("复制失败，请手动选择文本", "error");
      } finally {
        helper.remove();
      }
    }
  }
  async function readSSE(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    const handleChunk = (chunk) => {
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let data;
        try { data = JSON.parse(payload); } catch (_) { continue; }
        if (data && typeof data.type === "string") onEvent(data);
      }
    };
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf("\n\n")) !== -1) {
          handleChunk(buffer.slice(0, index));
          buffer = buffer.slice(index + 2);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) handleChunk(buffer);
    } finally {
      try { reader.releaseLock(); } catch (_) {}
    }
  }

  function autoResizeTextarea() {
    const node = el.composerInput;
    node.style.height = "auto";
    node.style.height = Math.min(node.scrollHeight, 180) + "px";
  }

  function loadImageElement(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("图片读取失败，格式可能不受支持"));
      img.src = url;
    });
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("图片读取失败，请重试或换一张图片"));
      reader.readAsDataURL(file);
    });
  }

  async function compressImage(file) {
    if (!file.type || !file.type.startsWith("image/")) throw new Error("不是图片文件");
    if (file.size > 20 * 1024 * 1024) throw new Error("图片超过 20 MB");
    const img = await loadImageElement(await readFileAsDataUrl(file));
    const width = img.naturalWidth || img.width || 1;
    const height = img.naturalHeight || img.height || 1;
    const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("浏览器不支持图片处理");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    let quality = 0.86;
    let dataUrl = canvas.toDataURL("image/jpeg", quality);
    while (dataUrl.length > 2_500_000 && quality > 0.5) {
      quality -= 0.12;
      dataUrl = canvas.toDataURL("image/jpeg", quality);
    }
    let outWidth = canvas.width;
    let outHeight = canvas.height;
    while (dataUrl.length > 2_500_000 && outWidth > 640 && outHeight > 640) {
      outWidth = Math.round(outWidth * 0.8);
      outHeight = Math.round(outHeight * 0.8);
      canvas.width = outWidth;
      canvas.height = outHeight;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, outWidth, outHeight);
      ctx.drawImage(img, 0, 0, outWidth, outHeight);
      dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    }
    return dataUrl;
  }

  async function addImageFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file.type && file.type.startsWith("image/"));
    if (!files.length) return;
    el.attachBtn.disabled = true;
    try {
      for (const file of files) {
        try {
          const dataUrl = await compressImage(file);
          const projectedTokens = localContextTokens()
            + pendingAttachmentTokens()
            + estimateTextTokens(el.composerInput.value || "")
            + IMAGE_TOKEN_ESTIMATE;
          if (projectedTokens > contextLimit()) {
            toast(`${file.name} 加入后会超出当前模型上下文，已跳过`, "error");
            continue;
          }
          state.attachments.push(dataUrl);
          renderAttachments();
        } catch (error) {
          toast(error.message || "图片处理失败", "error");
        }
      }
    } finally {
      el.attachBtn.disabled = false;
    }
  }

  async function extractFileAttachment(file) {
    if (file.size > 25 * 1024 * 1024) throw new Error(`${file.name} 超过 25 MB`);
    let response;
    try {
      response = await fetch("/api/files/extract", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-File-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
    } catch (_) {
      throw new Error("无法连接本地服务，请确认 app.py 仍在运行");
    }
    let data = {};
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(data.error || `读取失败（HTTP ${response.status}）`);
    if (!data.file || !data.file.text) throw new Error("文件没有可读取的文字");
    return data.file;
  }

  function pendingAttachmentTokens() {
    let total = 0;
    total += (state.attachments || []).length * IMAGE_TOKEN_ESTIMATE;
    for (const file of state.fileAttachments || []) {
      total += estimateTextTokens(file.text || "");
      total += (file.page_images || []).length * IMAGE_TOKEN_ESTIMATE;
    }
    return total;
  }

  function fileAttachmentTokens(file) {
    return estimateTextTokens(file?.text || "") + (file?.page_images || []).length * IMAGE_TOKEN_ESTIMATE;
  }

  async function addFileAttachments(fileList) {
    const candidates = Array.from(fileList || []);
    const files = candidates.filter((file) => /\.(pdf|docx?|txt|md|markdown|csv|json)$/i.test(file.name));
    if (!files.length) {
      if (candidates.length) toast("只支持 PDF、DOC/DOCX 和文本文件", "error");
      return;
    }
    el.fileBtn.disabled = true;
    try {
      for (const file of files) {
        try {
          toast(`正在读取 ${file.name}…`, "success", 1800);
          const extracted = await extractFileAttachment(file);
          const sourceImages = Array.isArray(extracted.page_images) ? extracted.page_images : [];
          extracted.page_images = sourceImages;
          const projectedTokens = localContextTokens()
            + pendingAttachmentTokens()
            + estimateTextTokens(el.composerInput.value || "")
            + fileAttachmentTokens(extracted);
          if (projectedTokens > contextLimit()) {
            toast(`${extracted.name} 加入后会超出当前模型上下文，已跳过`, "error");
            continue;
          }
          state.fileAttachments.push(extracted);
          renderAttachments();
          if (extracted.truncated) toast(`${extracted.name} 内容过长，已截断`, "error");
        } catch (error) {
          toast(error.message || "文件读取失败", "error");
        }
      }
    } finally {
      el.fileBtn.disabled = false;
    }
  }

  function renderAttachments() {
    const images = state.attachments.map((url, index) => `
      <div class="attachment-item">
        <img src="${escapeHtml(url)}" alt="待发送图片 ${index + 1}" data-preview-src="${escapeHtml(url)}" title="滚轮放大">
        <button class="attachment-remove" type="button" data-remove-image="${index}" aria-label="移除图片">${icon("i-close")}</button>
      </div>`).join("");
    const files = state.fileAttachments.map((file, index) => `
      <div class="attachment-file" title="${escapeHtml(file.truncated ? "内容过长，发送时会带上已截断的部分" : file.name)}">
        <svg aria-hidden="true"><use href="#i-file"></use></svg>
        <span class="attachment-file-name">${escapeHtml(file.name)}</span>
        <small class="attachment-file-meta">${formatNumber(file.chars || (file.text || "").length)} 字${Array.isArray(file.page_images) && file.page_images.length ? ` · ${file.page_images.length} 图` : ""}</small>
        <button class="attachment-file-remove" type="button" data-remove-file="${index}" aria-label="移除文件">${icon("i-close")}</button>
      </div>`).join("");
    const items = images + files;
    el.attachmentStrip.innerHTML = items;
    el.attachmentStrip.classList.toggle("hidden", !items);
  }

  function clearAttachments() {
    state.attachments = [];
    state.fileAttachments = [];
    renderAttachments();
  }

  function viewImage(url) {
    showModal(`
      <div class="image-viewer" role="dialog" aria-modal="true">
        <button class="image-viewer-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        <img src="${escapeHtml(url)}" alt="图片预览" draggable="false">
        <div class="image-viewer-hint">滚轮缩放 · 拖拽平移 · 双击重置</div>
      </div>`);
    const viewer = el.modalRoot.querySelector(".image-viewer");
    const img = viewer?.querySelector("img");
    if (!viewer || !img) return;
    let scale = 1;
    let tx = 0;
    let ty = 0;
    let dragging = false;
    let startX = 0;
    let startY = 0;
    let originX = 0;
    let originY = 0;
    const apply = () => {
      img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
      viewer.classList.toggle("zoomed", scale > 1.01);
    };
    const clampScale = (value) => Math.max(0.35, Math.min(6, value));
    const reset = () => {
      scale = 1;
      tx = 0;
      ty = 0;
      apply();
    };
    viewer.addEventListener("wheel", (event) => {
      event.preventDefault();
      const next = clampScale(scale * Math.exp(-event.deltaY * 0.0015));
      if (Math.abs(next - scale) < 0.0005) return;
      const rect = img.getBoundingClientRect();
      const cx = event.clientX - (rect.left + rect.width / 2);
      const cy = event.clientY - (rect.top + rect.height / 2);
      tx = cx - (cx - tx) * (next / scale);
      ty = cy - (cy - ty) * (next / scale);
      scale = next;
      if (scale <= 1.01) { tx = 0; ty = 0; }
      apply();
    }, { passive: false });
    img.addEventListener("dblclick", (event) => {
      event.preventDefault();
      reset();
    });
    img.addEventListener("pointerdown", (event) => {
      if (scale <= 1.01) return;
      event.preventDefault();
      dragging = true;
      startX = event.clientX;
      startY = event.clientY;
      originX = tx;
      originY = ty;
      viewer.classList.add("dragging");
      img.setPointerCapture?.(event.pointerId);
    });
    img.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      tx = originX + event.clientX - startX;
      ty = originY + event.clientY - startY;
      apply();
    });
    const endDrag = () => {
      dragging = false;
      viewer.classList.remove("dragging");
    };
    img.addEventListener("pointerup", endDrag);
    img.addEventListener("pointercancel", endDrag);
    img.addEventListener("pointerleave", () => { if (dragging) endDrag(); });
    apply();
  }

  function viewFile(file) {
    const chars = formatNumber(file.chars || (file.text || "").length);
    const pageImages = Array.isArray(file.page_images) ? file.page_images : [];
    const extra = [
      file.pages ? `${file.pages} 页` : "",
      pageImages.length ? `${pageImages.length} 张页面图片` : "",
      file.truncated ? "文字已截断" : "",
      file.images_truncated ? "页面图片已截断" : "",
      file.pages_truncated ? "仅处理前若干页" : "",
    ].filter(Boolean).join(" · ");
    const pagesHtml = pageImages.length
      ? `<div class="file-page-grid">${pageImages.map((image, index) => `<button class="file-page-thumb" type="button" data-page-index="${index}"><img src="${escapeHtml(image.data_url)}" alt="第 ${image.page || index + 1} 页"><span>第 ${image.page || index + 1} 页</span></button>`).join("")}</div>`
      : "";
    showModal(`
      <div class="modal file-modal" role="dialog" aria-modal="true">
        <div class="modal-head">
          <div>
            <h2 class="modal-title">${escapeHtml(file.name || "附件")}</h2>
            <p class="modal-subtitle">${chars} 字${extra ? ` · ${escapeHtml(extra)}` : ""}</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        ${pagesHtml}
        <pre class="file-preview-text">${escapeHtml(file.text || "")}</pre>
      </div>`);
    $$("[data-page-index]", el.modalRoot).forEach((button) => {
      button.addEventListener("click", () => {
        const image = pageImages[Number(button.getAttribute("data-page-index"))];
        if (image?.data_url) viewImage(image.data_url);
      });
    });
  }

  function setupComposer() {
    el.composerInput.addEventListener("input", autoResizeTextarea);
    el.composerInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!currentJob()) el.composerForm.requestSubmit();
      }
    });
    el.attachBtn.addEventListener("click", () => el.imageInput.click());
    el.webSearchBtn.addEventListener("click", toggleWebSearch);
    el.imageInput.addEventListener("change", async () => {
      await addImageFiles(el.imageInput.files);
      el.imageInput.value = "";
    });
    el.fileBtn.addEventListener("click", () => el.fileInput.click());
    el.fileInput.addEventListener("change", async () => {
      await addFileAttachments(el.fileInput.files);
      el.fileInput.value = "";
    });
    el.composerInput.addEventListener("paste", (event) => {
      const clipboard = event.clipboardData;
      if (!clipboard) return;
      const files = Array.from(clipboard.files || []).filter(
        (file) => file.type && file.type.startsWith("image/"),
      );
      if (!files.length) {
        for (const item of Array.from(clipboard.items || [])) {
          if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
            const file = item.getAsFile();
            if (file) files.push(file);
          }
        }
      }
      if (files.length) {
        event.preventDefault();
        addImageFiles(files);
      }
    });
    el.composerForm.addEventListener("dragover", (event) => {
      event.preventDefault();
      el.composerForm.classList.add("dragging");
    });
    el.composerForm.addEventListener("dragleave", (event) => {
      if (!el.composerForm.contains(event.relatedTarget)) el.composerForm.classList.remove("dragging");
    });
    el.composerForm.addEventListener("drop", (event) => {
      event.preventDefault();
      el.composerForm.classList.remove("dragging");
      const all = Array.from(event.dataTransfer?.files || []);
      if (!all.length) return;
      const images = all.filter((file) => file.type && file.type.startsWith("image/"));
      const documents = all.filter((file) => !file.type || !file.type.startsWith("image/"));
      if (images.length) addImageFiles(images);
      if (documents.length) addFileAttachments(documents);
    });
    renderAttachments();
    autoResizeTextarea();
  }

  function cancelStreamFrame(message) {
    if (message._frame) {
      cancelAnimationFrame(message._frame);
      message._frame = null;
    }
  }

  function finishAssistant(assistant, errorMessage, stopped = false) {
    const job = jobForAssistant(assistant);
    const visible = Boolean(job && job.messages === state.messages);
    if (errorMessage) assistant.error = errorMessage;
    if (stopped) assistant.stopped = true;
    cancelStreamFrame(assistant);
    assistant.streaming = false;
    assistant.status = "";
    if (job) state.streamJobs.delete(job);
    refreshStreamingFlag();
    el.sendBtn.classList.remove("loading");
    updateSendAvailability();
    if (visible) {
      renderMessages();
      stickToBottomIfFollowing();
    } else if (job) {
      const name = job.title || "对话";
      toast(stopped ? `「${name}」已停止生成` : `「${name}」已生成完毕`);
    }
    loadConversations({ silent: true });
    if (visible && window.matchMedia("(min-width: 981px)").matches) el.composerInput.focus();
  }

  function stopStreaming() {
    const job = currentJob();
    if (!job) return;
    job.stopping = true;
    job.controller.abort();
  }

  function resyncConversation(convId = state.currentId) {
    if (!convId || currentJob() || convId !== state.currentId) return;
    openConversation(convId);
  }

  async function streamAssistant(payload, assistant) {
    const controller = new AbortController();
    const turnAction = payload.edit_turn_index !== undefined || payload.regenerate_turn_index !== undefined;
    const job = {
      messages: state.messages,
      assistant,
      controller,
      conversationId: state.currentId || null,
      conversation: state.current ? { ...state.current } : null,
      title: state.current?.title || "",
      stopping: false,
    };
    state.streamJobs.add(job);
    refreshStreamingFlag();
    el.sendBtn.classList.add("loading");
    updateSendAvailability();
    renderMessages();
    scrollToBottom(false);
    let response;
    try {
      response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    } catch (error) {
      const stopped = error?.name === "AbortError" || job.stopping;
      finishAssistant(
        assistant,
        stopped ? "" : "无法连接本地服务，请确认 app.py 仍在运行",
        stopped,
      );
      return;
    }

    if (!response.ok) {
      let message = `请求失败（HTTP ${response.status}）`;
      try {
        const data = await response.json();
        if (data && data.error) message = data.error;
      } catch (_) {}
      finishAssistant(assistant, message);
      if (turnAction) resyncConversation(job.conversationId);
      return;
    }
    if (!response.body) {
      finishAssistant(assistant, "当前浏览器不支持流式读取");
      return;
    }

    let stopped = false;
    try {
      await readSSE(response, (event) => {
        if (event.type === "meta") {
          if (event.conversation) {
            job.conversation = event.conversation;
            job.conversationId = event.conversation.id;
            job.title = event.conversation.title || job.title;
            if (job.messages === state.messages) {
              state.current = { ...(state.current || {}), ...event.conversation };
              if (event.conversation.id) state.currentId = event.conversation.id;
              applyConversationContext(event.conversation);
              setPageMeta("CONVERSATION", event.conversation.title || currentTitle());
            }
            loadConversations({ silent: true });
            renderContextBar();
          }
          if (event.notice) {
            assistant.notice = event.notice;
            queueStreamRender(assistant);
          }
        } else if (event.type === "notice") {
          assistant.notice = event.message || "";
          queueStreamRender(assistant);
        } else if (event.type === "status") {
          assistant.status = event.message || "";
          queueStreamRender(assistant);
        } else if (event.type === "reasoning") {
          assistant.reasoning += event.text || "";
          assistant.status = "";
          queueStreamRender(assistant);
        } else if (event.type === "content") {
          assistant.content += event.text || "";
          assistant.status = "";
          queueStreamRender(assistant);
        } else if (event.type === "replace_content") {
          assistant.content = event.content || "";
          queueStreamRender(assistant);
        } else if (event.type === "tool_event") {
          if (event.event) {
            assistant.tool_events = Array.isArray(assistant.tool_events) ? assistant.tool_events : [];
            assistant.tool_events[Number(event.index) || 0] = event.event;
          }
          queueStreamRender(assistant);
        } else if (event.type === "usage") {
          if (event.usage) assistant.usage = event.usage;
        } else if (event.type === "error") {
          assistant.error = event.message || "回答失败";
          if (event.context_overflow) assistant.context_overflow = true;
          queueStreamRender(assistant);
        } else if (event.type === "done") {
          if (event.conversation) {
            job.conversation = event.conversation;
            job.conversationId = event.conversation.id;
            job.title = event.conversation.title || job.title;
            if (job.messages === state.messages) {
              state.current = { ...(state.current || {}), ...event.conversation };
              if (event.conversation.id) state.currentId = event.conversation.id;
              applyConversationContext(event.conversation);
              setPageMeta("CONVERSATION", event.conversation.title || currentTitle());
              renderContextBar();
            }
          }
          if (event.usage) assistant.usage = event.usage;
        }
      });
      if (!assistant.content && !assistant.reasoning && !assistant.error && !job.stopping) {
        assistant.error = "模型没有返回内容";
      }
    } catch (error) {
      stopped = error?.name === "AbortError" || job.stopping;
      if (!stopped) assistant.error = error.message || "读取回答时连接中断";
    } finally {
      const wasStopped = stopped || job.stopping;
      finishAssistant(assistant, "", wasStopped);
      if (turnAction && !wasStopped && assistant.error) resyncConversation(job.conversationId);
    }
  }

  function makeAssistantMessage(model, effort, variant) {
    return {
      role: "assistant",
      content: "",
      reasoning: "",
      created_at: new Date().toISOString(),
      model: model.id,
      model_name: model.name || model.id,
      reasoning_effort: effort || null,
      variant,
      streaming: true,
      uid: nextUid(),
    };
  }

  async function sendMessage() {
    if (currentJob()) return;
    const text = el.composerInput.value.trim();
    const images = state.attachments.slice();
    const files = state.fileAttachments.slice();
    if (!text && !images.length && !files.length) return;
    const provider = selectedProvider();
    const model = selectedModel();
    if (!provider) {
      toast("请先在模型配置中添加供应商", "error");
      setRoute("settings");
      return;
    }
    if (!model) {
      toast("请选择可用模型", "error");
      return;
    }

    const effort = el.effortSelect.value || "";
    const stamp = new Date().toISOString();
    const userMessage = {
      role: "user",
      content: text,
      images,
      files,
      created_at: stamp,
      uid: nextUid(),
      active_variant: 0,
    };
    const assistant = makeAssistantMessage(model, effort, 0);
    state.messages.push(userMessage, assistant);
    renderPromptPicker();
    el.composerInput.value = "";
    state.attachments = [];
    state.fileAttachments = [];
    renderAttachments();
    autoResizeTextarea();
    await streamAssistant({
      conversation_id: state.currentId || "",
      provider_id: provider.id,
      model: model.id,
      prompt_id: state.currentId ? "" : state.promptSelection,
      web_search: Boolean(state.webSearchEnabled),
      reasoning_effort: effort,
      message: text,
      images,
      files: files.map((file) => ({
        name: file.name,
        text: file.text,
        chars: file.chars,
        pages: file.pages,
        truncated: file.truncated,
        page_images: file.page_images || [],
        images_truncated: file.images_truncated,
        pages_truncated: file.pages_truncated,
      })),
    }, assistant);
  }

  async function regenerateLastTurn() {
    if (currentJob()) {
      toast("当前回答还在生成，请稍后重新生成", "error");
      return;
    }
    const turns = buildTurns(state.messages);
    if (!turns.length || !state.currentId) return;
    const turnIndex = turns.length - 1;
    const turn = turns[turnIndex];
    const provider = selectedProvider();
    const model = selectedModel();
    if (!provider || !model) {
      toast("请选择可用的供应商和模型", "error");
      return;
    }
    const effort = el.effortSelect.value || "";
    const variant = turn.assistants.length;
    turn.user.active_variant = variant;
    const assistant = makeAssistantMessage(model, effort, variant);
    state.messages.push(assistant);
    await streamAssistant({
      conversation_id: state.currentId,
      provider_id: provider.id,
      model: model.id,
      reasoning_effort: effort,
      regenerate_turn_index: turnIndex,
    }, assistant);
  }

  async function switchVariant(turnIndex, direction) {
    if (currentJob()) {
      toast("当前回答还在生成，请稍后切换", "error");
      return;
    }
    const turns = buildTurns(state.messages);
    const turn = turns[turnIndex];
    if (!turn || turn.assistants.length < 2) return;
    const current = clampVariant(turn);
    const next = Math.max(0, Math.min(current + direction, turn.assistants.length - 1));
    if (next === current) return;
    turn.user.active_variant = next;
    renderMessages();
    if (!state.currentId) return;
    try {
      await api(`/api/conversations/${encodeURIComponent(state.currentId)}`, {
        method: "PATCH",
        body: { turn_index: turnIndex, active_variant: next },
      });
      loadConversations({ silent: true });
    } catch (error) {
      turn.user.active_variant = current;
      renderMessages();
      toast(error.message, "error");
    }
  }

  function editLastTurn(turnIndex) {
    if (currentJob()) {
      toast("当前回答还在生成，请稍后编辑", "error");
      return;
    }
    const turns = buildTurns(state.messages);
    const turn = turns[turnIndex];
    if (!turn || turnIndex !== turns.length - 1) return;
    const user = turn.user;
    const images = Array.isArray(user.images) ? user.images.slice() : [];
    const files = Array.isArray(user.files) ? user.files.slice() : [];
    showModal(`
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="editQuestionTitle">
        <div class="modal-head">
          <div>
            <h2 class="modal-title" id="editQuestionTitle">编辑上一条问题</h2>
            <p class="modal-subtitle">保存后会新增一个回答版本，旧回答仍然保留，可用下方箭头切换查看。</p>
          </div>
          <button class="modal-close" type="button" data-modal-close aria-label="关闭">${icon("i-close")}</button>
        </div>
        <label class="field">
          <span class="field-label">问题内容 ${(images.length || files.length) ? "<span>图片/文件会保留</span>" : ""}</span>
          <textarea class="modal-textarea" id="editQuestionInput">${escapeHtml(user.content || "")}</textarea>
        </label>
        <div class="form-error hidden" id="editQuestionError"></div>
        <div class="modal-actions">
          <button class="secondary-btn" type="button" data-modal-close>取消</button>
          <button class="primary-btn" type="button" id="editQuestionSubmit">保存并重新回答</button>
        </div>
      </div>`);
    const input = $("#editQuestionInput");
    const errorBox = $("#editQuestionError");
    const submit = $("#editQuestionSubmit");
    window.setTimeout(() => { input.focus(); input.select(); }, 30);
    submit.addEventListener("click", async () => {
      const text = input.value.trim();
      errorBox.classList.add("hidden");
      if (!text && !images.length && !files.length) {
        errorBox.textContent = "问题不能为空";
        errorBox.classList.remove("hidden");
        return;
      }
      const provider = selectedProvider();
      const model = selectedModel();
      if (!provider || !model) {
        toast("请选择可用的供应商和模型", "error");
        return;
      }
      if (!state.currentId) {
        toast("对话尚未保存，无法编辑", "error");
        return;
      }
      closeModal();
      const effort = el.effortSelect.value || "";
      user.content = text;
      if (images.length) user.images = images;
      else delete user.images;
      if (files.length) user.files = files;
      else delete user.files;
      const variant = turn.assistants.length;
      user.active_variant = variant;
      const assistant = makeAssistantMessage(model, effort, variant);
      state.messages.push(assistant);
      await streamAssistant({
        conversation_id: state.currentId,
        provider_id: provider.id,
        model: model.id,
        reasoning_effort: effort,
        edit_turn_index: turnIndex,
        message: text,
        images,
        files,
      }, assistant);
    });
  }

  function dismissModal() {
    const cancel = el.modalRoot.querySelector("[data-confirm-cancel]");
    if (cancel) {
      cancel.click();
      return;
    }
    closeModal();
  }

  function bindEvents() {
    el.chatScroll.addEventListener("scroll", () => {
      syncAutoScroll();
    }, { passive: true });
    el.chatScroll.addEventListener("wheel", (event) => {
      if (event.deltaY < 0) pauseAutoScroll(1200);
    }, { passive: true });
    el.chatScroll.addEventListener("touchmove", () => pauseAutoScroll(1200), { passive: true });
    el.jumpBtn.addEventListener("click", () => scrollToBottom(false));
    el.compressBtn.addEventListener("click", compressContext);
    el.navChat.addEventListener("click", () => setRoute("chat"));
    el.navSettings.addEventListener("click", () => setRoute("settings"));
    el.newChatBtn.addEventListener("click", newChat);
    el.menuBtn.addEventListener("click", openMobileSidebar);
    el.mobileBackdrop.addEventListener("click", closeMobileSidebar);

    el.generalFontMinus.addEventListener("click", () => applyFontSize(state.fontSize - 10));
    el.generalFontPlus.addEventListener("click", () => applyFontSize(state.fontSize + 10));
    el.resetGeneralSettingsBtn.addEventListener("click", () => {
      applyFontSize(100);
      toast("已恢复默认字体大小");
    });
    el.addProviderBtn.addEventListener("click", () => openProviderModal());
    el.saveWebSettingsBtn.addEventListener("click", saveWebSettings);
    el.saveMemorySettingsBtn.addEventListener("click", saveMemorySettings);
    el.resetMemoryPromptBtn.addEventListener("click", () => {
      el.memoryReviewPrompt.value = state.memorySettings.default_review_prompt || "";
      toast("已恢复默认提示词，点击“保存记忆设置”后生效");
    });
    el.saveMemoryProfileBtn.addEventListener("click", saveMemoryProfile);
    el.memoryProfile.addEventListener("input", updateMemoryProfileCount);
    el.scanMemoryBtn.addEventListener("click", scanMemory);
    el.refreshMemoryBtn.addEventListener("click", () => loadMemory());
    el.memoryProviderSelect.addEventListener("change", () => populateMemoryModels(el.memoryProviderSelect.value, ""));
    el.memoryIndexList.addEventListener("click", (event) => {
      const action = event.target.closest("[data-index-action]");
      if (!action) return;
      const convId = action.getAttribute("data-index-id") || "";
      if (!convId) return;
      if (action.getAttribute("data-index-action") === "edit") editMemoryIndex(convId);
      else deleteMemoryIndex(convId);
    });
    el.addPromptBtn.addEventListener("click", () => openPromptModal());
    el.promptList.addEventListener("click", (event) => {
      const action = event.target.closest("[data-prompt-action]");
      if (!action) return;
      const card = action.closest("[data-prompt-id]");
      const prompt = state.prompts.find((item) => item.id === card?.getAttribute("data-prompt-id"));
      if (!prompt) return;
      const actionName = action.getAttribute("data-prompt-action");
      if (actionName === "edit") openPromptModal(prompt);
      else if (actionName === "delete") deletePrompt(prompt);
      else if (actionName === "default") setDefaultPrompt(prompt);
    });
    el.promptPicker.addEventListener("change", () => {
      state.promptSelection = el.promptPicker.value;
      store.set(STORE_KEYS.prompt, state.promptSelection);
      renderPromptPicker();
    });

    el.providerGrid.addEventListener("click", (event) => {
      if (event.target.closest("[data-empty-add]")) {
        openProviderModal();
        return;
      }
      const action = event.target.closest("[data-provider-action]");
      if (!action) return;
      const card = action.closest("[data-provider-id]");
      const provider = state.providers.find((item) => item.id === card?.getAttribute("data-provider-id"));
      if (!provider) return;
      const actionName = action.getAttribute("data-provider-action");
      if (actionName === "edit") openProviderModal(provider);
      else if (actionName === "refresh") refreshProviderModels(provider, action);
      else deleteProvider(provider);
    });

    el.chatSearchBtn.addEventListener("click", openChatSearch);
    el.turnNavList.addEventListener("click", (event) => {
      const item = event.target.closest("[data-nav-turn]");
      if (item) scrollTurnIntoView(Number(item.getAttribute("data-nav-turn")));
    });
    el.historyList.addEventListener("click", (event) => {
      const rename = event.target.closest("[data-rename-conv]");
      if (rename) {
        event.stopPropagation();
        renameConversation(rename.getAttribute("data-rename-conv"));
        return;
      }
      const archive = event.target.closest("[data-archive-conv]");
      if (archive) {
        event.stopPropagation();
        archiveConversation(archive.getAttribute("data-archive-conv"));
        return;
      }
      const item = event.target.closest(".history-item");
      if (item) openConversation(item.getAttribute("data-conv-id"));
    });
    el.historyList.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const item = event.target.closest(".history-item");
      if (!item) return;
      event.preventDefault();
      openConversation(item.getAttribute("data-conv-id"));
    });

    el.settingsTabs.addEventListener("click", (event) => {
      const tab = event.target.closest("[data-settings-tab]");
      if (tab) setSettingsTab(tab.getAttribute("data-settings-tab"));
    });

    el.refreshArchiveBtn.addEventListener("click", () => loadArchivedConversations());

    el.archiveList.addEventListener("click", (event) => {
      const restore = event.target.closest("[data-restore-conv]");
      if (restore) {
        restoreConversation(restore.getAttribute("data-restore-conv"));
        return;
      }
      const remove = event.target.closest("[data-delete-archived]");
      if (remove) deleteArchivedConversation(remove.getAttribute("data-delete-archived"));
    });

    el.providerSelect.addEventListener("change", onProviderChanged);
    el.modelSelect.addEventListener("change", onModelChanged);
    el.effortSelect.addEventListener("change", onEffortChanged);

    el.emptyState.addEventListener("click", (event) => {
      const button = event.target.closest("[data-prompt]");
      if (!button) return;
      el.composerInput.value = (button.getAttribute("data-prompt") || "").replace(/\\n/g, "\n");
      autoResizeTextarea();
      el.composerInput.focus();
      const end = el.composerInput.value.length;
      el.composerInput.setSelectionRange(end, end);
    });

    el.messageList.addEventListener("click", (event) => {
      const edit = event.target.closest("[data-edit-turn]");
      if (edit) {
        editLastTurn(Number(edit.getAttribute("data-edit-turn")));
        return;
      }
      const regen = event.target.closest("[data-regenerate-turn]");
      if (regen) {
        regenerateLastTurn();
        return;
      }
      const prev = event.target.closest("[data-variant-prev]");
      if (prev) {
        switchVariant(Number(prev.getAttribute("data-variant-prev")), -1);
        return;
      }
      const next = event.target.closest("[data-variant-next]");
      if (next) {
        switchVariant(Number(next.getAttribute("data-variant-next")), 1);
        return;
      }
      const fileButton = event.target.closest("[data-view-file]");
      if (fileButton) {
        const uid = fileButton.getAttribute("data-view-uid");
        const message = state.messages.find((item) => item.uid === uid);
        const file = message?.files?.[Number(fileButton.getAttribute("data-view-file"))];
        if (file) viewFile(file);
        return;
      }
      const view = event.target.closest("[data-view-image], [data-view-file-image]");
      if (view) {
        const uid = view.getAttribute("data-view-uid");
        const message = state.messages.find((item) => item.uid === uid);
        const imageIndex = view.hasAttribute("data-view-image") ? Number(view.getAttribute("data-view-image")) : null;
        const fileIndex = view.hasAttribute("data-view-file-image") ? Number(view.getAttribute("data-view-file-image")) : null;
        const file = fileIndex !== null ? message?.files?.[fileIndex] : null;
        const url = imageIndex !== null
          ? message?.images?.[imageIndex]
          : file?.page_images?.[Number(view.getAttribute("data-page-image"))]?.data_url;
        if (url) viewImage(url);
        return;
      }
      const copyCode = event.target.closest("[data-copy-code]");
      if (copyCode) {
        const code = copyCode.closest(".code-wrap")?.querySelector("code")?.textContent || "";
        copyText(code, "代码已复制");
        return;
      }
      const copyMessage = event.target.closest("[data-copy-message]");
      if (copyMessage) {
        const uid = copyMessage.getAttribute("data-copy-message");
        const message = state.messages.find((item) => item.uid === uid);
        if (message?.content) copyText(message.content, "回答已复制");
      }
    });

    el.composerForm.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage();
    });

    el.stopBtn.addEventListener("click", stopStreaming);

    el.attachmentStrip.addEventListener("click", (event) => {
      const preview = event.target.closest("[data-preview-src]");
      if (preview) {
        viewImage(preview.getAttribute("data-preview-src") || "");
        return;
      }
      const removeImage = event.target.closest("[data-remove-image]");
      if (removeImage) {
        state.attachments.splice(Number(removeImage.getAttribute("data-remove-image")), 1);
        renderAttachments();
        return;
      }
      const removeFile = event.target.closest("[data-remove-file]");
      if (removeFile) {
        state.fileAttachments.splice(Number(removeFile.getAttribute("data-remove-file")), 1);
        renderAttachments();
      }
    });

    el.modalRoot.addEventListener("click", (event) => {
      if (event.target.closest("[data-modal-close]")) {
        dismissModal();
        return;
      }
      const backdrop = event.target.closest("[data-modal-backdrop]");
      if (backdrop && event.target === backdrop) dismissModal();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (currentJob()) {
        stopStreaming();
        return;
      }
      if (state.modalOpen) dismissModal();
    });

    window.addEventListener("hashchange", () => {
      setRoute(window.location.hash === "#settings" ? "settings" : "chat", false);
    });
  }

  async function init() {
    loadGeneralSettings();
    state.webSearchEnabled = store.get("xxh.webSearch", "") === "1";
    setRoute(window.location.hash === "#settings" ? "settings" : "chat", false);
    setupComposer();
    bindEvents();
    renderMessages();
    await Promise.all([loadProviders(), loadPrompts(), loadWebSettings({ silent: true })]);
    await loadMemory({ silent: true });
    await Promise.all([
      loadConversations(),
      loadArchivedConversations({ silent: true }),
    ]);
    setPageMeta(
      state.route === "settings" ? "SETTINGS" : "CONVERSATION",
      state.route === "settings" ? "设置" : "新对话",
    );
    updateSendAvailability();
    if (window.matchMedia("(min-width: 981px)").matches) el.composerInput.focus();
  }

  init();
})();
</script>
</body>
</html>
'''


class HarnessHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"XiaoXiaoFatFish/{VERSION}"

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, fmt: str, *args) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"[{stamp}] {fmt % args}\n")

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.close_connection = True

    def _send_json(self, data, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self._common_headers()
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def _send_html(self) -> None:
        raw = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.bootcdn.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.bootcdn.net; "
            "img-src 'self' data: blob:; connect-src 'self'; "
            "font-src 'self' data: https://cdn.jsdelivr.net https://cdn.bootcdn.net; "
            "base-uri 'none'; form-action 'self'",
        )
        self._common_headers()
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json_body(self) -> dict:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise ApiError("请求必须使用 application/json", 415)
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ApiError("Content-Length 无效") from exc
        if length <= 0:
            raise ApiError("请求体为空")
        if length > MAX_BODY_BYTES:
            raise ApiError("请求体过大", 413)
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError("请求体不是有效 JSON") from exc
        if not isinstance(data, dict):
            raise ApiError("请求体必须是 JSON 对象")
        return data

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Allow", "GET, POST, PATCH, DELETE, OPTIONS")
        self._common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_html()
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self._common_headers()
                self.end_headers()
                return
            if path == "/api/health":
                config = load_config()
                self._send_json(
                    {
                        "ok": True,
                        "version": VERSION,
                        "providers": len(config.get("providers", [])),
                    }
                )
                return
            if path == "/api/providers":
                config = load_config()
                providers = [public_provider(item) for item in config.get("providers", [])]
                self._send_json({"providers": providers})
                return
            if path == "/api/prompts":
                config = load_config()
                self._send_json(
                    {
                        "prompts": config.get("prompts") or [],
                        "active_prompt_id": config.get("active_prompt_id"),
                    }
                )
                return
            if path == "/api/memory":
                config = load_config()
                conversations = load_conversations()
                memory = sync_memory_entries(load_memory(), conversations)
                settings = (config.get("settings") or {}).get("memory") or {}
                reviews = memory.get("reviews") or {}
                snapshots = [
                    memory_snapshot_summary(conv)
                    for conv in conversations
                    if conv.get("memory_snapshot") or conv.get("memory_snapshot_manual")
                ]
                retry_count = sum(
                    1 for conv in conversations
                    if not conv.get("archived")
                    and int((reviews.get(str(conv.get("id") or "")) or {}).get("failures") or 0) > 0
                )
                self._send_json({
                    "enabled": bool(settings.get("enabled")),
                    "provider_id": settings.get("provider_id") or "",
                    "model_id": settings.get("model_id") or "",
                    "review_prompt": settings.get("review_prompt") or MEMORY_REVIEW_SYSTEM,
                    "default_review_prompt": MEMORY_REVIEW_SYSTEM,
                    "profile": memory.get("profile") or "",
                    "profile_limit": PROFILE_MAX_CHARS,
                    "conversations": [item for item in (memory.get("conversations") or []) if _entry_is_meaningful(item)],
                    "snapshots": snapshots,
                    "retry_count": retry_count,
                    "updated_at": memory.get("updated_at"),
                })
                return
            snapshot_match = re.fullmatch(r"/api/memory/snapshots/([^/]+)", path)
            if snapshot_match:
                conv_id = unquote(snapshot_match.group(1))
                with FILE_LOCK:
                    conversations = load_conversations()
                conv = next((item for item in conversations if item.get("id") == conv_id), None)
                if conv is None:
                    raise ApiError("对话不存在", 404)
                result = memory_snapshot_summary(conv)
                result["snapshot"] = str(conv.get("memory_snapshot") or "")
                result["limit"] = MEMORY_SNAPSHOT_EDIT_MAX_CHARS
                self._send_json({"snapshot": result})
                return
            if path == "/api/settings":
                config = load_config()
                web = (config.get("settings") or {}).get("web") or {}
                key = str(web.get("api_key") or "")
                self._send_json({
                    "web": {
                        "enabled": bool(web.get("enabled")),
                        "configured": bool(key),
                        "key_hint": ("••••" + key[-4:]) if key else "",
                        "max_results": int(web.get("max_results") or WEB_SEARCH_DEFAULT_RESULTS),
                    }
                })
                return
            if path == "/api/conversations/search":
                query = parse_qs(urlparse(self.path).query)
                needle = str(query.get("q", [""])[0] or "").strip()
                try:
                    limit = int(query.get("limit", ["80"])[0])
                except (TypeError, ValueError):
                    limit = 80
                with FILE_LOCK:
                    results = search_conversation_index(needle, limit)
                self._send_json({"results": results})
                return
            if path == "/api/conversations":
                query = parse_qs(urlparse(self.path).query)
                want_archived = str(query.get("archived", ["0"])[0]).lower() in ("1", "true", "yes")
                with FILE_LOCK:
                    items = load_conversation_index()
                summaries = [
                    {key: value for key, value in item.items() if key not in ("search_text", "cache_file", "fingerprint")}
                    for item in items
                    if bool(item.get("archived")) == want_archived
                ]
                summaries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
                self._send_json({"conversations": summaries})
                return
            match = re.fullmatch(r"/api/conversations/([^/]+)", path)
            if match:
                conv_id = unquote(match.group(1))
                with FILE_LOCK:
                    conv = load_conversation_by_id(conv_id)
                if not conv:
                    raise ApiError("对话不存在", 404)
                self._send_json({"conversation": public_conversation(conv)})
                return
            raise ApiError("页面不存在", 404)
        except ApiError as exc:
            self._send_error_json(exc.message, exc.status)
        except Exception as exc:  # noqa: BLE001 - 本地工具需要把异常返回给界面
            self.log_message("GET error: %s", exc)
            self._send_error_json(f"服务器内部错误：{exc}", 500)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        try:
            match = re.fullmatch(r"/api/providers/([^/]+)", path)
            if match:
                provider_id = unquote(match.group(1))
                with FILE_LOCK:
                    config = load_config()
                    before = len(config.get("providers", []))
                    config["providers"] = [
                        item for item in config.get("providers", []) if item.get("id") != provider_id
                    ]
                    if len(config["providers"]) == before:
                        raise ApiError("供应商不存在", 404)
                    save_config(config)
                self._send_json({"ok": True})
                return
            prompt_delete_match = re.fullmatch(r"/api/prompts/([^/]+)", path)
            if prompt_delete_match:
                self._handle_delete_prompt(unquote(prompt_delete_match.group(1)))
                return
            index_delete_match = re.fullmatch(r"/api/memory/index/([^/]+)", path)
            if index_delete_match:
                self._handle_delete_memory_index(unquote(index_delete_match.group(1)))
                return
            snapshot_delete_match = re.fullmatch(r"/api/memory/snapshots/([^/]+)", path)
            if snapshot_delete_match:
                self._handle_delete_memory_snapshot(unquote(snapshot_delete_match.group(1)))
                return
            match = re.fullmatch(r"/api/conversations/([^/]+)", path)
            if match:
                conv_id = unquote(match.group(1))
                with FILE_LOCK:
                    conversations = load_conversations()
                    conv = next((item for item in conversations if item.get("id") == conv_id), None)
                    if conv is None:
                        raise ApiError("对话不存在", 404)
                    if not conv.get("archived"):
                        raise ApiError("请先归档再删除对话", 400)
                    conversations = [item for item in conversations if item.get("id") != conv_id]
                    save_conversations(conversations)
                self._send_json({"ok": True})
                return
            raise ApiError("接口不存在", 404)
        except ApiError as exc:
            self._send_error_json(exc.message, exc.status)
        except Exception as exc:  # noqa: BLE001
            self.log_message("DELETE error: %s", exc)
            self._send_error_json(f"服务器内部错误：{exc}", 500)

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json_body()
            match = re.fullmatch(r"/api/conversations/([^/]+)", path)
            if not match:
                raise ApiError("接口不存在", 404)
            conv_id = unquote(match.group(1))
            has_title = "title" in payload
            has_archived = "archived" in payload
            has_variant = "turn_index" in payload and "active_variant" in payload
            if not has_title and not has_archived and not has_variant:
                raise ApiError("没有要修改的字段")
            title = ""
            if has_title:
                title = re.sub(r"\s+", " ", str(payload.get("title") or "")).strip()
                if not title:
                    raise ApiError("对话名称不能为空")
                if len(title) > 60:
                    raise ApiError("对话名称不能超过 60 个字符")
            with FILE_LOCK:
                conversations = load_conversations()
                conv = next((item for item in conversations if item.get("id") == conv_id), None)
                if conv is None:
                    raise ApiError("对话不存在", 404)
                if has_title:
                    conv["title"] = title
                if has_archived:
                    archived = payload.get("archived")
                    if isinstance(archived, str):
                        archived = archived.lower() in ("1", "true", "yes")
                    conv["archived"] = bool(archived)
                if has_variant:
                    try:
                        turn_index = int(payload.get("turn_index"))
                        variant_index = int(payload.get("active_variant"))
                    except (TypeError, ValueError) as exc:
                        raise ApiError("回复版本参数无效") from exc
                    turns = split_turns(conv.get("messages") or [])
                    if turn_index < 0 or turn_index >= len(turns):
                        raise ApiError("轮次不存在", 404)
                    user, assistants = turns[turn_index]
                    if variant_index < 0 or variant_index >= len(assistants):
                        raise ApiError("回复版本不存在", 400)
                    user["active_variant"] = variant_index
                save_conversations(conversations)
                result = conversation_summary(conv)
            self._send_json({"conversation": result})
        except ApiError as exc:
            self._send_error_json(exc.message, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self.log_message("PATCH error: %s", exc)
            self._send_error_json(f"服务器内部错误：{exc}", 500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/files/extract":
                self._handle_extract_file()
                return
            payload = self._read_json_body()
            refresh_match = re.fullmatch(r"/api/providers/([^/]+)/refresh", path)
            if refresh_match:
                self._handle_refresh_provider(unquote(refresh_match.group(1)))
                return
            if path == "/api/providers":
                self._handle_save_provider(payload)
                return
            if path == "/api/prompts":
                self._handle_save_prompt(payload)
                return
            prompt_delete = re.fullmatch(r"/api/prompts/([^/]+)", path)
            if prompt_delete:
                self._handle_delete_prompt(unquote(prompt_delete.group(1)))
                return
            if path == "/api/memory/profile":
                self._handle_save_memory_profile(payload)
                return
            index_save_match = re.fullmatch(r"/api/memory/index/([^/]+)", path)
            if index_save_match:
                self._handle_save_memory_index(unquote(index_save_match.group(1)), payload)
                return
            snapshot_save_match = re.fullmatch(r"/api/memory/snapshots/([^/]+)", path)
            if snapshot_save_match:
                self._handle_save_memory_snapshot(unquote(snapshot_save_match.group(1)), payload)
                return
            if path == "/api/memory":
                self._handle_save_memory_settings(payload)
                return
            if path == "/api/memory/scan":
                started = start_memory_scan()
                self._send_json({"started": started})
                return
            if path == "/api/settings/web":
                self._handle_save_web_settings(payload)
                return
            if path == "/api/chat":
                self._handle_chat(payload)
                return
            compress_match = re.fullmatch(r"/api/conversations/([^/]+)/compress", path)
            if compress_match:
                self._handle_compress(unquote(compress_match.group(1)), payload)
                return
            raise ApiError("接口不存在", 404)
        except ApiError as exc:
            self._send_error_json(exc.message, exc.status)
        except UpstreamError as exc:
            self._send_error_json(exc.message, 502)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self.log_message("POST error: %s", exc)
            try:
                self._send_error_json(f"服务器内部错误：{exc}", 500)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_extract_file(self) -> None:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/octet-stream":
            raise ApiError("文件上传必须使用 application/octet-stream", 415)
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ApiError("文件大小无效") from exc
        if length <= 0:
            raise ApiError("文件内容为空")
        if length > MAX_FILE_BYTES:
            raise ApiError("单个文件不能超过 25 MB", 413)
        raw = self.rfile.read(length)
        name = unquote(self.headers.get("X-File-Name") or "file")
        result = extract_file_text(name, raw)
        self._send_json({"file": result})

    def _handle_save_memory_profile(self, payload: dict) -> None:
        profile = str(payload.get("profile") or "").strip()
        if len(profile) > PROFILE_MAX_CHARS:
            raise ApiError(f"用户画像不能超过 {PROFILE_MAX_CHARS} 字")
        with FILE_LOCK:
            memory = load_memory()
            memory["profile"] = profile
            save_memory(memory)
        self._send_json({"ok": True, "profile": profile, "profile_chars": len(profile)})

    def _handle_save_memory_snapshot(self, conv_id: str, payload: dict) -> None:
        snapshot = str(payload.get("snapshot") or "").strip()
        if len(snapshot) > MEMORY_SNAPSHOT_EDIT_MAX_CHARS:
            raise ApiError(f"记忆快照不能超过 {MEMORY_SNAPSHOT_EDIT_MAX_CHARS} 字")
        with FILE_LOCK:
            conversations = load_conversations()
            conv = next((item for item in conversations if item.get("id") == conv_id), None)
            if conv is None:
                raise ApiError("对话不存在", 404)
            conv["memory_snapshot"] = snapshot
            conv["memory_snapshot_manual"] = True
            conv["memory_snapshot_updated_at"] = utc_now()
            save_conversations(conversations)
            result = memory_snapshot_summary(conv)
            result["snapshot"] = snapshot
        self._send_json({"ok": True, "snapshot": result})

    def _handle_save_memory_index(self, conv_id: str, payload: dict) -> None:
        summary = str(payload.get("summary") or "").strip()
        raw_keywords = payload.get("keywords")
        if isinstance(raw_keywords, list):
            keywords = [str(item).strip()[:40] for item in raw_keywords if str(item).strip()][:10]
        else:
            keywords = [item.strip()[:40] for item in re.split(r"[,，\n]+", str(raw_keywords or "")) if item.strip()][:10]
        if not summary:
            raise ApiError("索引摘要不能为空")
        if len(summary) > MEMORY_INDEX_SUMMARY_MAX_CHARS:
            raise ApiError(f"索引摘要不能超过 {MEMORY_INDEX_SUMMARY_MAX_CHARS} 字，请压缩后保存")
        with FILE_LOCK:
            memory = load_memory()
            entry = _memory_entry(memory, conv_id)
            if entry is None:
                raise ApiError("记忆索引不存在", 404)
            entry["summary"] = summary
            entry["keywords"] = keywords
            entry["manual_edit"] = True
            entry["updated_at"] = utc_now()
            save_memory(memory)
        self._send_json({"ok": True, "index": entry})

    def _handle_delete_memory_index(self, conv_id: str) -> None:
        with FILE_LOCK:
            memory = load_memory()
            if _memory_entry(memory, conv_id) is None:
                raise ApiError("记忆索引不存在", 404)
            _remove_memory_entry(memory, conv_id)
            conversations = load_conversations()
            conv = next((item for item in conversations if item.get("id") == conv_id), None)
            if conv is not None:
                conv["memory_excluded"] = True
                _mark_review_success(memory, conv, False)
            save_conversations(conversations)
            save_memory(memory)
        self._send_json({"ok": True})

    def _handle_delete_memory_snapshot(self, conv_id: str) -> None:
        with FILE_LOCK:
            conversations = load_conversations()
            conv = next((item for item in conversations if item.get("id") == conv_id), None)
            if conv is None:
                raise ApiError("对话不存在", 404)
            conv["memory_snapshot"] = ""
            conv["memory_snapshot_manual"] = True
            conv["memory_snapshot_updated_at"] = utc_now()
            save_conversations(conversations)
            result = memory_snapshot_summary(conv)
            result["snapshot"] = ""
        self._send_json({"ok": True, "snapshot": result})

    def _handle_save_memory_settings(self, payload: dict) -> None:
        raw_enabled = payload.get("enabled")
        enabled = raw_enabled is True or str(raw_enabled).lower() in ("1", "true", "yes")
        provider_id = str(payload.get("provider_id") or "").strip()
        model_id = str(payload.get("model_id") or "").strip()
        review_prompt = str(payload.get("review_prompt") or "").strip()
        if len(review_prompt) > MAX_PROMPT_CHARS:
            raise ApiError(f"记忆整理提示词不能超过 {MAX_PROMPT_CHARS} 个字符")
        effective_review_prompt = review_prompt or MEMORY_REVIEW_SYSTEM
        with FILE_LOCK:
            config = load_config()
            settings = config.setdefault("settings", {}).setdefault("memory", {})
            settings["enabled"] = enabled
            settings["provider_id"] = provider_id
            settings["model_id"] = model_id
            settings["review_prompt"] = review_prompt
            save_config(config)
        self._send_json({"memory": {"enabled": enabled, "provider_id": provider_id, "model_id": model_id, "review_prompt": effective_review_prompt, "default_review_prompt": MEMORY_REVIEW_SYSTEM}})

    def _handle_save_web_settings(self, payload: dict) -> None:
        api_key = str(payload.get("api_key") or "").strip()
        raw_enabled = payload.get("enabled")
        enabled = raw_enabled is True or str(raw_enabled).lower() in ("1", "true", "yes")
        try:
            max_results = int(payload.get("max_results") or WEB_SEARCH_DEFAULT_RESULTS)
        except (TypeError, ValueError):
            max_results = WEB_SEARCH_DEFAULT_RESULTS
        max_results = max(1, min(WEB_SEARCH_MAX_RESULTS, max_results))
        with FILE_LOCK:
            config = load_config()
            web = ((config.get("settings") or {}).get("web") or {})
            existing = str(web.get("api_key") or "")
            if payload.get("clear_key"):
                api_key = ""
            elif not api_key:
                api_key = existing
            if enabled and not api_key:
                raise ApiError("开启联网搜索前请先填写 Firecrawl API Key", 400)
            web["enabled"] = enabled
            web["api_key"] = api_key
            web["max_results"] = max_results
            config.setdefault("settings", {})["web"] = web
            save_config(config)
        self._send_json({
            "web": {
                "enabled": enabled,
                "configured": bool(api_key),
                "key_hint": ("••••" + api_key[-4:]) if api_key else "",
                "max_results": max_results,
            }
        })

    def _handle_save_prompt(self, payload: dict) -> None:
        prompt_id = str(payload.get("id") or "").strip()
        name = str(payload.get("name") or "").strip()
        content = str(payload.get("content") or "").strip()
        if not name:
            raise ApiError("请填写提示词名称")
        if len(name) > 80:
            raise ApiError("提示词名称不能超过 80 个字符")
        if len(content) > MAX_PROMPT_CHARS:
            raise ApiError(f"提示词不能超过 {MAX_PROMPT_CHARS} 个字符")
        with FILE_LOCK:
            config = load_config()
            prompts = config["prompts"]
            existing = prompt_by_id(config, prompt_id) if prompt_id else None
            if prompt_id and existing is None:
                raise ApiError("提示词不存在，可能已被删除", 404)
            if existing is None:
                existing = {
                    "id": new_id("p_"),
                    "name": name,
                    "content": content,
                    "builtin": False,
                }
                prompts.append(existing)
            else:
                existing["name"] = name
                existing["content"] = content
            if payload.get("set_active"):
                config["active_prompt_id"] = existing["id"]
            save_config(config)
            result = {
                "prompt": existing,
                "prompts": config["prompts"],
                "active_prompt_id": config["active_prompt_id"],
            }
        self._send_json(result)

    def _handle_delete_prompt(self, prompt_id: str) -> None:
        with FILE_LOCK:
            config = load_config()
            prompts = config["prompts"]
            if prompt_by_id(config, prompt_id) is None:
                raise ApiError("提示词不存在", 404)
            if len(prompts) <= 1:
                raise ApiError("至少保留一个系统提示词", 400)
            config["prompts"] = [item for item in prompts if item.get("id") != prompt_id]
            if config.get("active_prompt_id") == prompt_id:
                config["active_prompt_id"] = config["prompts"][0]["id"]
            save_config(config)
            result = {
                "prompts": config["prompts"],
                "active_prompt_id": config["active_prompt_id"],
            }
        self._send_json(result)

    def _handle_save_provider(self, payload: dict) -> None:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ApiError("请填写供应商名称")
        if len(name) > 60:
            raise ApiError("供应商名称不能超过 60 个字符")
        base_url = normalize_base_url(str(payload.get("base_url") or ""))
        provider_id = str(payload.get("id") or "").strip()
        api_key = str(payload.get("api_key") or "").strip()

        with FILE_LOCK:
            config = load_config()
            existing = None
            if provider_id:
                existing = next(
                    (item for item in config.get("providers", []) if item.get("id") == provider_id),
                    None,
                )
                if existing is None:
                    raise ApiError("要编辑的供应商不存在", 404)
            if not api_key and existing:
                api_key = str(existing.get("api_key") or "")
            if not api_key:
                raise ApiError("请填写 API Key")

        timeout = _coerce_int(
            payload.get("timeout"),
            provider_timeout(existing) if existing else DEFAULT_TIMEOUT,
        )
        retries = _coerce_int(
            payload.get("retries"),
            provider_retries(existing) if existing else DEFAULT_RETRIES,
        )
        if not (MIN_TIMEOUT <= timeout <= MAX_TIMEOUT):
            raise ApiError(f"等待时间需在 {MIN_TIMEOUT}-{MAX_TIMEOUT} 秒之间")
        if not (0 <= retries <= MAX_RETRIES):
            raise ApiError(f"重试次数需在 0-{MAX_RETRIES} 之间")

        now = utc_now()
        with FILE_LOCK:
            config = load_config()
            providers = config.setdefault("providers", [])
            existing = None
            if provider_id:
                existing = next((item for item in providers if item.get("id") == provider_id), None)
            if existing is None:
                provider_id = new_id("p_")
                existing = {
                    "id": provider_id,
                    "created_at": now,
                }
                providers.append(existing)
                models = []
            else:
                models = list(existing.get("models") or [])
                if existing.get("base_url") != base_url or existing.get("api_key") != api_key:
                    # 地址或 Key 变了，旧模型列表不再可信，等用户手动刷新。
                    models = []
            existing.update(
                {
                    "name": name,
                    "base_url": base_url,
                    "api_key": api_key,
                    "models": models,
                    "timeout": timeout,
                    "retries": retries,
                    "updated_at": now,
                }
            )
            save_config(config)
            result = public_provider(existing)
        self._send_json({"provider": result})

    def _handle_refresh_provider(self, provider_id: str) -> None:
        with FILE_LOCK:
            config = load_config()
            provider = provider_by_id(config, provider_id)
            base_url = str(provider.get("base_url") or "")
            api_key = str(provider.get("api_key") or "")
            timeout = provider_timeout(provider)
            retries = provider_retries(provider)

        try:
            models = fetch_provider_models(base_url, api_key, timeout, retries)
        except UpstreamError as exc:
            raise ApiError(exc.message, 502) from exc

        with FILE_LOCK:
            config = load_config()
            provider = provider_by_id(config, provider_id)
            provider["models"] = models
            provider["updated_at"] = utc_now()
            save_config(config)
            result = public_provider(provider)
        self._send_json({"provider": result})

    def _handle_compress(self, conversation_id: str, payload: dict) -> None:
        provider_id = str(payload.get("provider_id") or "").strip()
        model_id = str(payload.get("model") or "").strip()
        requested_effort = str(payload.get("reasoning_effort") or "").strip()

        with FILE_LOCK:
            conversations = load_conversations()
            conv = next(
                (item for item in conversations if item.get("id") == conversation_id),
                None,
            )
            if conv is None:
                raise ApiError("对话不存在，可能已被删除", 404)
            config = load_config()
            provider_id = provider_id or str(conv.get("provider_id") or "")
            model_id = model_id or str(conv.get("model") or "")
            if not provider_id or not model_id:
                raise ApiError("该对话还没有可用的模型，请先发送一条消息", 400)
            provider = provider_by_id(config, provider_id)
            model_info = find_model(provider, model_id)
            if not model_info:
                raise ApiError("所选模型不在该供应商的模型列表中，请刷新模型配置", 400)
            levels = [str(item) for item in (model_info.get("efforts") or [])]
            effort = ""
            if levels:
                effort = requested_effort if requested_effort in levels else (
                    str(model_info.get("default_effort") or "") or levels[0]
                )
            elif requested_effort in STANDARD_EFFORTS:
                effort = requested_effort
            snapshot = dict(conv)
            snapshot["messages"] = list(conv.get("messages") or [])
            digest = dict(conv.get("digest") or {})

        result_payload, summary_text, summarized_turns = compress_conversation(
            snapshot, provider, provider_id, model_id, model_info, effort
        )
        with FILE_LOCK:
            conversations = load_conversations()
            conv = next(
                (item for item in conversations if item.get("id") == conversation_id),
                None,
            )
            if conv is None:
                raise ApiError("对话不存在，可能已被删除", 404)
            conv["digest"] = {
                "text": summary_text,
                "turns": summarized_turns,
                "created_at": utc_now(),
            }
            conv["context_limit"] = int(
                model_info.get("context_window") or conv.get("context_limit") or CONTEXT_LIMIT_DEFAULT
            )
            conv["updated_at"] = utc_now()
            save_conversations(conversations)
            result_payload["after_tokens"] = conversation_context(conv)["tokens"]
            result_payload["conversation"] = public_conversation(conv)

        self._send_json(result_payload)

    def _handle_chat(self, payload: dict) -> None:
        message = str(payload.get("message") or "").strip()
        images = normalize_images(payload.get("images"))
        files = normalize_files(payload.get("files"))
        provider_id = str(payload.get("provider_id") or "").strip()
        model_id = str(payload.get("model") or "").strip()
        conversation_id = str(payload.get("conversation_id") or "").strip()
        requested_prompt_id = str(payload.get("prompt_id") or "").strip()
        requested_effort = str(payload.get("reasoning_effort") or "").strip()
        raw_web_search = payload.get("web_search")
        web_search_enabled = raw_web_search is True or str(raw_web_search).lower() in ("1", "true", "yes")
        raw_regenerate = payload.get("regenerate_turn_index")
        raw_edit = payload.get("edit_turn_index")
        is_regenerate = raw_regenerate is not None and str(raw_regenerate).strip() != ""
        is_edit = raw_edit is not None and str(raw_edit).strip() != ""
        if is_regenerate and is_edit:
            raise ApiError("不能同时重新生成和编辑问题")
        if not is_regenerate and not message and not images and not files:
            raise ApiError("消息不能为空")
        if len(message) > 120_000:
            raise ApiError("单条消息过长")
        fallback_title = "图片对话" if images else ("文件对话" if files else "新对话")
        if not provider_id:
            raise ApiError("请选择供应商")
        if not model_id:
            raise ApiError("请选择模型")
        if (is_regenerate or is_edit) and not conversation_id:
            raise ApiError("缺少对话 ID")

        def parse_turn_index(value, label: str) -> int:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ApiError(f"{label}无效") from exc

        with FILE_LOCK:
            config = load_config()
            app_settings = config.get("settings") or default_settings()
            web_settings = app_settings.get("web") or {}
            web_api_key = str(web_settings.get("api_key") or "").strip()
            web_enabled = bool(web_search_enabled and web_settings.get("enabled"))
            web_capable = bool(web_settings.get("enabled") and web_api_key)
            memory_settings = app_settings.get("memory") or {}
            memory_enabled = bool(memory_settings.get("enabled"))
            if web_search_enabled and not web_settings.get("enabled"):
                raise ApiError("联网搜索尚未在设置中开启", 400)
            if web_search_enabled and not web_api_key:
                raise ApiError("请先在设置中填写 Firecrawl API Key", 400)
            provider = provider_by_id(config, provider_id)
            model_info = find_model(provider, model_id)
            if not model_info:
                raise ApiError("所选模型不在该供应商的模型列表中，请刷新模型配置", 400)
            levels = [str(item) for item in (model_info.get("efforts") or [])]
            default_effort = str(model_info.get("default_effort") or "")
            if levels:
                effort = requested_effort or default_effort or levels[0]
                if effort not in levels:
                    raise ApiError(f"Unsupported effort: {effort}")
            else:
                effort = requested_effort if requested_effort in STANDARD_EFFORTS else ""

            conversations = load_conversations()
            conv = None
            new_conversation_created = False
            if conversation_id:
                conv = next(
                    (item for item in conversations if item.get("id") == conversation_id),
                    None,
                )
                if conv is None:
                    raise ApiError("对话不存在，可能已被删除", 404)
                if conv.get("archived"):
                    raise ApiError("该对话已归档，请先恢复后再继续", 400)

            target_index: int | None = None
            target_user: dict | None = None
            target_assistants: list[dict] = []
            if is_regenerate or is_edit:
                turns = split_turns(conv.get("messages") or [])
                target_index = parse_turn_index(
                    raw_regenerate if is_regenerate else raw_edit,
                    "turn_index",
                )
                if target_index < 0 or target_index >= len(turns):
                    raise ApiError("要修改的轮次不存在", 404)
                if target_index != len(turns) - 1:
                    raise ApiError("目前只支持修改或重新生成最后一轮问题", 400)
                target_user, target_assistants = turns[target_index]
            elif conv is None:
                new_conversation_created = True
                prompt_id = requested_prompt_id or str(config.get("active_prompt_id") or "")
                prompt = prompt_by_id(config, prompt_id) or (config.get("prompts") or [DEFAULT_PROMPTS[0]])[0]
                now = utc_now()
                conv = {
                    "id": new_id("c_"),
                    "title": make_title(message or fallback_title),
                    "created_at": now,
                    "updated_at": now,
                    "messages": [],
                    "archived": False,
                    "title_auto_generated": False,
                    "prompt_id": prompt.get("id"),
                    "prompt_name": prompt.get("name"),
                    "system_prompt": prompt.get("content"),
                }
                conversations.append(conv)
            elif "system_prompt" not in conv:
                prompt_id = requested_prompt_id or str(config.get("active_prompt_id") or "")
                prompt = prompt_by_id(config, prompt_id) or (config.get("prompts") or [DEFAULT_PROMPTS[0]])[0]
                conv["prompt_id"] = conv.get("prompt_id") or prompt.get("id")
                conv["prompt_name"] = conv.get("prompt_name") or prompt.get("name")
                conv["system_prompt"] = prompt.get("content")

            environment_text = str(conv.get("environment") or "")
            if not environment_text or "运行环境" in environment_text:
                conv["environment"] = build_environment_note()
            if memory_enabled and not conv.get("memory_snapshot") and not conv.get("memory_snapshot_manual"):
                conv["memory_snapshot"] = build_memory_snapshot(load_memory(), conversations)

            now = utc_now()
            conv["provider_id"] = provider_id
            conv["provider_name"] = provider.get("name") or provider_id
            conv["model"] = model_id
            conv["model_name"] = model_info.get("name") or model_id
            conv["reasoning_effort"] = effort
            conv["updated_at"] = now

            if is_edit:
                assert target_user is not None
                target_user["content"] = message
                if images:
                    target_user["images"] = images
                else:
                    target_user.pop("images", None)
                if files:
                    target_user["files"] = files
                else:
                    target_user.pop("files", None)
                assistant_variant = len(target_assistants)
                target_user["active_variant"] = assistant_variant
                if len(turns) == 1:
                    conv["title"] = make_title(message or fallback_title)
                upstream_messages = build_upstream_messages(conv, exclude_turn_index=target_index)
            elif is_regenerate:
                assert target_user is not None
                assistant_variant = len(target_assistants)
                target_user["active_variant"] = assistant_variant
                upstream_messages = build_upstream_messages(conv, exclude_turn_index=target_index)
            else:
                assistant_variant = 0
                user_message = {
                    "role": "user",
                    "content": message,
                    "created_at": now,
                    "active_variant": 0,
                }
                if images:
                    user_message["images"] = images
                if files:
                    user_message["files"] = files
                conv["messages"].append(user_message)
                if not conv.get("title"):
                    conv["title"] = make_title(message or fallback_title)
                upstream_messages = build_upstream_messages(conv)

            save_conversations(conversations)
            summary = conversation_summary(conv, with_context=True)

        if new_conversation_created and memory_enabled:
            start_memory_scan()

        # 从这里开始改用 SSE。所有校验均已完成。
        self._start_sse()
        self._emit_sse("meta", {"conversation": summary})

        assistant = {
            "role": "assistant",
            "content": "",
            "reasoning": "",
            "created_at": utc_now(),
            "model": model_id,
            "model_name": model_info.get("name") or model_id,
            "reasoning_effort": effort or None,
            "variant": assistant_variant,
        }
        error_message: str | None = None
        client_gone = False
        tool_schema = []
        if web_capable:
            tool_schema.append(web_search_tool_schema())
        if memory_enabled:
            tool_schema.extend([memory_search_tool_schema(), memory_read_tool_schema(), memory_save_tool_schema()])
        tool_schema = tool_schema or None
        active_messages = list(upstream_messages)
        tool_round = 0
        try:
            run_tool_loop(
                provider=provider,
                model_id=model_id,
                messages=active_messages,
                effort=effort,
                tools=tool_schema,
                web_enabled=web_enabled,
                web_api_key=web_api_key,
                web_max_results=int(web_settings.get("max_results") or WEB_SEARCH_DEFAULT_RESULTS),
                memory_enabled=memory_enabled,
                current_conversation_id=str(summary.get("id") or ""),
                emit=self._emit_sse,
                assistant=assistant,
            )
        except ClientDisconnected:
            client_gone = True
        except UpstreamError as exc:
            error_message = exc.message
            self.log_message("upstream error: %s", exc.message)
            if is_context_error(exc.message, exc.detail):
                assistant["context_overflow"] = True
                error_message = "上下文超出模型可接收的长度，本次请求未完成。请压缩上下文，或新建一个对话继续。"
            try:
                self._emit_sse("error", {"message": error_message, "context_overflow": bool(assistant.get("context_overflow"))})
            except ClientDisconnected:
                client_gone = True
        except Exception as exc:
            error_message = f"对话中断：{exc}"
            self.log_message("chat error: %s", exc)
            try:
                self._emit_sse("error", {"message": error_message})
            except ClientDisconnected:
                client_gone = True

        if client_gone and not assistant.get("finish_reason"):
            assistant["stopped"] = True
        if error_message:
            assistant["error"] = error_message
        if (
            not assistant["content"]
            and not assistant["reasoning"]
            and not error_message
            and not assistant.get("stopped")
        ):
            assistant["error"] = "模型没有返回内容。"
        if not assistant.get("usage") and assistant["content"]:
            assistant.pop("usage", None)
        assistant.pop("status", None)

        auto_title_needed = False
        auto_title_base = ""
        with FILE_LOCK:
            conversations = load_conversations()
            conv = next(
                (item for item in conversations if item.get("id") == summary.get("id")),
                None,
            )
            if conv is not None:
                conv["messages"].append(assistant)
                conv["updated_at"] = utc_now()
                conv["provider_id"] = provider_id
                conv["provider_name"] = provider.get("name") or provider_id
                conv["model"] = model_id
                conv["model_name"] = model_info.get("name") or model_id
                conv["reasoning_effort"] = effort
                if not conv.get("title"):
                    conv["title"] = make_title(message or fallback_title)
                save_conversations(conversations)
                if memory_enabled:
                    memory = load_memory()
                    if _touch_memory_entry(memory, conv) is not None:
                        save_memory(memory)
                auto_title_base = str(conv.get("title") or "")
                auto_title_needed = bool(
                    new_conversation_created
                    and not assistant.get("error")
                    and not assistant.get("stopped")
                    and str(assistant.get("content") or "").strip()
                    and not conv.get("title_auto_generated")
                    and len(split_turns(conv.get("messages") or [])) == 1
                )
                final_summary = conversation_summary(conv, with_context=True)
            else:
                final_summary = summary

        if auto_title_needed:
            generated_title = generate_first_turn_title(config, conv)
            if generated_title and generated_title != auto_title_base:
                with FILE_LOCK:
                    fresh_conversations = load_conversations()
                    fresh_conv = next(
                        (item for item in fresh_conversations if item.get("id") == summary.get("id")),
                        None,
                    )
                    if (
                        fresh_conv is not None
                        and not fresh_conv.get("title_auto_generated")
                        and str(fresh_conv.get("title") or "") == auto_title_base
                        and len(split_turns(fresh_conv.get("messages") or [])) == 1
                    ):
                        fresh_conv["title"] = generated_title
                        fresh_conv["title_auto_generated"] = True
                        fresh_conv["title_generated_at"] = utc_now()
                        fresh_conv["updated_at"] = utc_now()
                        save_conversations(fresh_conversations)
                        final_summary = conversation_summary(fresh_conv, with_context=True)

        if not client_gone:
            try:
                self._emit_sse("done", {"conversation": final_summary, "usage": assistant.get("usage")})
            except ClientDisconnected:
                pass

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self._common_headers()
        self.end_headers()
        self.wfile.flush()

    def _emit_sse(self, event_type: str, data: dict) -> None:
        payload = {"type": event_type, **data}
        raw = (
            "data: "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        ).encode("utf-8")
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise ClientDisconnected() from exc


class HarnessServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _print_banner(port: int) -> None:
    print()
    print("  ┌────────────────────────────────────────────┐")
    print("  │                  小小肥鱼                  │")
    print("  │        local multi-turn AI agent           │")
    print("  └────────────────────────────────────────────┘")
    print(f"  本地地址  http://{HOST}:{port}")
    print(f"  数据目录  {APP_DIR}")
    print("  按 Ctrl+C 停止服务")
    print()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="小小肥鱼 本地多轮问答 Agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口，默认 8765")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    args = parser.parse_args()

    port = args.port
    server = None
    for candidate in range(port, port + 30):
        try:
            server = HarnessServer((HOST, candidate), HarnessHandler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print(f"无法在 {args.port}-{args.port + 29} 找到可用端口。", file=sys.stderr)
        return 1

    _print_banner(port)
    if not args.no_browser:
        url = f"http://{HOST}:{port}"
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
