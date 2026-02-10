import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

import requests
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
_GLOBAL_RATE_LIMITER_LOCK = threading.Lock()
_GLOBAL_RATE_LIMITER: Optional[threading.BoundedSemaphore] = None
_GLOBAL_RATE_LIMIT: Optional[int] = None


def _get_global_rate_limiter(rate_limit: int) -> threading.BoundedSemaphore:
    """Create/reuse process-global limiter shared across SearchTool instances."""
    global _GLOBAL_RATE_LIMIT, _GLOBAL_RATE_LIMITER
    safe_rate_limit = max(1, int(rate_limit))
    with _GLOBAL_RATE_LIMITER_LOCK:
        if _GLOBAL_RATE_LIMITER is None or _GLOBAL_RATE_LIMIT != safe_rate_limit:
            _GLOBAL_RATE_LIMITER = threading.BoundedSemaphore(safe_rate_limit)
            _GLOBAL_RATE_LIMIT = safe_rate_limit
            logger.info(f"Initialized global search rate limiter: rate_limit={safe_rate_limit}")
        return _GLOBAL_RATE_LIMITER


def _run_search_call_with_limit(
    limiter: threading.BoundedSemaphore, url: str, payload: list[dict], timeout: int
) -> tuple[Optional[list], Optional[str]]:
    """Synchronous helper executed in a worker thread."""
    with limiter:
        return call_search_api(url=url, payload=payload, timeout=timeout)


# ── HTTP call with retry logic ──────────────────────────────────────────

def call_search_api(
    url: str,
    payload: list[dict],
    timeout: int,
) -> tuple[Optional[list], Optional[str]]:
    """
    Sync HTTP POST with retry + linear backoff.
    Returns (response_json, error_message).
    """
    request_id = str(uuid.uuid4())[:8]
    headers = {"Content-Type": "application/json"}
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

            if resp.status_code in (500, 502, 503, 504):
                last_error = f"[{request_id}] Server {resp.status_code} on attempt {attempt+1}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue

            resp.raise_for_status()
            return resp.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"[{request_id}] ConnectionError: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"[{request_id}] Timeout: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"[{request_id}] RequestException: {e}"
            break
        except Exception as e:
            last_error = f"[{request_id}] Unexpected: {e}"
            break

    logger.error(f"Search API failed after {MAX_RETRIES} retries: {last_error}")
    return None, last_error



# ── SearchTool ──────────────────────────────────────────────────────────

class SearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.url = config.get("retrieval_service_url", "http://localhost:5002/search")
        self.timeout = config.get("timeout", 30)
        self.k = config.get("k", 10)

        # Keep configs for compatibility; execution uses asyncio.to_thread + global semaphore.
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self._rate_limiter = _get_global_rate_limiter(self.rate_limit)

        # Image path setup
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        local_image_root = config.get("local_image_root", "./search_engine/corpus/img")
        if local_image_root.startswith("./"):
            self.local_image_root = os.path.join(self.project_root, local_image_root[2:])
        else:
            self.local_image_root = local_image_root

        logger.info(
            f"Initialized SearchTool: url={self.url}, workers={self.num_workers}, "
            f"rate_limit={self.rate_limit}, image_root={self.local_image_root}"
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResponse:
        agent_data = kwargs.get('agent_data')
        query = parameters.get('query')

        # Extract sample_id (required for search server)
        sample_id = None
        if agent_data and hasattr(agent_data, 'sample_id'):
            sample_id = agent_data.sample_id

        if sample_id is None:
            error_msg = (
                "CRITICAL: sample_id is None! "
                "Check: 1) ray_trainer.py uid assignment, 2) tool_agent_loop.py kwargs extraction"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        numeric_id = sample_id.split("_")[-1] if sample_id else None
        payload = [{"query": query, "request_idx": 0, "id": numeric_id}]

        # HTTP search in a thread with process-global concurrency limit.
        try:
            response_json, error_msg = await asyncio.to_thread(
                _run_search_call_with_limit,
                self._rate_limiter,
                self.url,
                payload,
                self.timeout,
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return ToolResponse(text=f"Error: Search execution failed. {e}")
        if error_msg or response_json is None:
            return ToolResponse(text=f"Error: {error_msg or 'No response from search server.'}")

        # ② Process results + load images locally (needs agent_data)
        text_content = "No search results found."
        images_found = []
        image_paths_found = []
        selected_rank: Optional[int] = None
        selected_server_idx: Optional[int] = None
        selected_image_name: Optional[str] = None

        if isinstance(response_json, list) and len(response_json) > 0:
            search_result = response_json[0]

            if isinstance(search_result, dict) and 'results' in search_result:
                candidates = self._normalize_candidates(search_result['results'])
                existing_image_paths = set()
                if agent_data:
                    existing_image_paths = set(agent_data.extra_fields.get('image_paths', []))

                for rank0, item in enumerate(candidates):
                    image_path = item.get('image_file')
                    if not image_path:
                        continue

                    final_path = self._resolve_local_image_path(image_path)

                    if final_path in existing_image_paths:
                        continue

                    if os.path.exists(final_path):
                        try:
                            img_obj = Image.open(final_path).convert("RGB")
                            images_found.append(img_obj)
                            image_paths_found.append(final_path)
                            selected_rank = rank0 + 1
                            selected_server_idx = item.get("idx")
                            selected_image_name = os.path.basename(final_path)
                            break  # 첫 번째 유효 이미지만
                        except Exception as e:
                            logger.warning(f"Failed to load image {final_path}: {e}")

                text_content = self._format_results(
                    search_result,
                    selected_rank=selected_rank,
                    selected_server_idx=selected_server_idx,
                    selected_image_name=selected_image_name,
                )
            else:
                text_content = self._format_results(search_result)

        # Update agent_data with loaded image paths
        if agent_data and image_paths_found:
            if 'image_paths' not in agent_data.extra_fields:
                agent_data.extra_fields['image_paths'] = []
            agent_data.extra_fields['image_paths'].extend(image_paths_found)

        return ToolResponse(text=text_content, image=images_found)

    def _resolve_local_image_path(self, image_path: str) -> str:
        clean_path = image_path.lstrip("./")
        if "corpus/img/" in clean_path:
            relative_part = clean_path.split("corpus/img/", 1)[1]
            return os.path.join(self.local_image_root, relative_part)
        return os.path.join(self.local_image_root, os.path.basename(clean_path))

    def _normalize_candidates(self, raw_results: list[Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for rank0, item in enumerate(raw_results):
            if isinstance(item, dict):
                image_file = item.get("image_file", "")
                if not image_file:
                    continue
                idx = item.get("idx", rank0)
                try:
                    idx = int(idx)
                except Exception:
                    idx = rank0
                score = item.get("score")
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    score = None
                candidates.append({"idx": idx, "image_file": image_file, "score": score})
            elif isinstance(item, str):
                candidates.append({"idx": rank0, "image_file": item, "score": None})
        return candidates

    def _format_results(
        self,
        result_data: Any,
        selected_rank: Optional[int] = None,
        selected_server_idx: Optional[int] = None,
        selected_image_name: Optional[str] = None,
    ) -> str:
        """Format search results for model consumption."""
        if isinstance(result_data, dict) and 'results' in result_data:
            candidates = self._normalize_candidates(result_data['results'])
            if not candidates:
                return "No search results found."

            snippets = []
            if selected_rank is not None:
                selected_msg = f"Selected image: rank={selected_rank}"
                if selected_server_idx is not None:
                    selected_msg += f", idx={selected_server_idx}"
                if selected_image_name:
                    selected_msg += f", file={selected_image_name}"
                snippets.append(selected_msg)
            else:
                snippets.append("Selected image: none (all candidates were previously used or missing locally).")

            snippets.append("Top candidates:")
            for rank0, item in enumerate(candidates[:self.k]):
                image_name = os.path.basename(item.get("image_file", ""))
                score = item.get("score")
                line = f"[rank={rank0+1}] idx={item.get('idx')} file={image_name}"
                if score is not None:
                    line += f" score={score:.4f}"
                snippets.append(line)
            return "\n".join(snippets)
        return str(result_data)
