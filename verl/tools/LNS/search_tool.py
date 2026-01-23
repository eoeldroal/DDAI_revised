import logging
import asyncio
import aiohttp
import os
import json #디버깅
from PIL import Image
from typing import Any
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

class SearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.url = config.get("retrieval_service_url", "http://localhost:5002/search")
        self.timeout = config.get("timeout", 30)
        self.k = config.get("k", 5)
        # 프로젝트 루트 경로 계산 (search_tool.py 기준으로 상위 4단계)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # 이미지 파일의 로컬 루트 경로 (config에서 상대 경로로 설정 가능)
        local_image_root = config.get("local_image_root", "./search_engnie/corpus/img")
        # 상대 경로면 프로젝트 루트 기준으로 절대 경로로 변환
        if local_image_root.startswith("./"):
            self.local_image_root = os.path.join(self.project_root, local_image_root[2:])
        else:
            self.local_image_root = local_image_root
        logger.info(f"Initialized SearchTool with URL: {self.url}, local_image_root: {self.local_image_root}")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResponse:
        agent_data = kwargs.get('agent_data')
        query = parameters.get('query')

        # Extract sample_id from agent_data (original dataset id like "train_14")
        # This is REQUIRED - the search server needs the dataset id to find relevant documents
        sample_id = None
        if agent_data and hasattr(agent_data, 'sample_id'):
            sample_id = agent_data.sample_id

        if sample_id is None:
            error_msg = (
                "CRITICAL: sample_id is None! "
                "The data pipeline failed to pass the original dataset id (e.g., 'train_14'). "
                "Check: 1) ray_trainer.py uid assignment, 2) tool_agent_loop.py kwargs extraction"
            )
            logger.error(f"🚨 {error_msg}")
            raise ValueError(error_msg)

        # Build payload with id field for search server compatibility
        # sample_id에서 숫자만 추출 (예: "train_14" -> "14")
        numeric_id = sample_id.split("_")[-1] if sample_id else None
        payload = [{"query": query, "request_idx": 0, "id": numeric_id}]

        print(f"\n🚀 [SearchTool Request] ID: {sample_id} -> {numeric_id} | Query: {query}", flush=True) # [DEBUG] 요청 내용 확인

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Search API Error: {resp.status} - {error_text}")
                        return ToolResponse(text=f"Error: {resp.status}")

                    results = await resp.json()
                    # ⭐️ [DEBUG] 서치엔진 응답 원본 확인 (가장 중요한 부분)
                    # 서버가 실제로 뭘 줬는지 눈으로 확인해야 합니다.
                    print(f"📥 [SearchTool Raw Response]:\n{json.dumps(results, indent=2, ensure_ascii=False)}", flush=True)

                    text_content = "No search results found."
                    images_found = []
                    image_paths_found = []

                    if isinstance(results, list) and len(results) > 0:
                        search_result = results[0]
                        text_content = self._format_results(search_result)

                        if isinstance(search_result, dict) and 'results' in search_result:
                            # 기존에 로드된 이미지 경로들 수집 (중복 체크용)
                            existing_image_paths = set()
                            if agent_data:
                                # agent_data.extra_fields에 저장된 이미지 경로들
                                existing_paths = agent_data.extra_fields.get('image_paths', [])
                                existing_image_paths = set(existing_paths)

                            for item in search_result['results']:
                                image_path = item.get('image_file')
                                if image_path:
                                    # 서버 반환 예시: "./search_engine/corpus/img/38_15.jpg"
                                    # 목표: "{local_image_root}/38_15.jpg"

                                    # (1) 앞의 ./ 제거
                                    clean_path = image_path.lstrip("./")

                                    # (2) search_engine/corpus/img/ 부분 이후의 파일명만 추출
                                    if "corpus/img/" in clean_path:
                                        relative_part = clean_path.split("corpus/img/", 1)[1]
                                        final_path = os.path.join(self.local_image_root, relative_part)
                                    else:
                                        final_path = os.path.join(self.local_image_root, os.path.basename(clean_path))

                                    # (3) 중복 체크: 이미 로드된 이미지는 건너뜀
                                    if final_path in existing_image_paths:
                                        logger.debug(f"Skipping duplicate image: {final_path}")
                                        continue

                                    # (4) 파일 존재 확인 및 로딩 - 첫 번째 유효 이미지만 선택
                                    if os.path.exists(final_path):
                                        try:
                                            img_obj = Image.open(final_path).convert("RGB")
                                            images_found.append(img_obj)
                                            image_paths_found.append(final_path)
                                            print(f"✅ [SearchTool] Selected first non-duplicate image: {final_path}", flush=True)
                                            break  # 첫 번째 유효 이미지를 찾으면 즉시 종료
                                        except Exception as e:
                                            logger.warning(f"Failed to load image {final_path}: {e}")


                    # agent_data.extra_fields에 이미지 경로 정보 저장
                    if agent_data and image_paths_found:
                        if 'image_paths' not in agent_data.extra_fields:
                            agent_data.extra_fields['image_paths'] = []
                        agent_data.extra_fields['image_paths'].extend(image_paths_found)

                    return ToolResponse(
                        text=text_content,
                        image=images_found  # 빈 리스트라도 그대로 반환
                    )

        except asyncio.TimeoutError:
            logger.warning(f"Search API Timeout for query: {query}")
            return ToolResponse(text="Error: Search request timed out.") # 1개 값 반환
            
        except Exception as e:
            logger.error(f"Search Tool Exception: {e}", exc_info=True)
            return ToolResponse(text=f"Error: An unexpected error occurred. {str(e)}") # 1개 값 반환

    def _format_results(self, result_data: Any) -> str:
        # (기존 포맷팅 로직 유지)
        if isinstance(result_data, dict) and 'results' in result_data:
            result_list = result_data['results']
            snippets = []
            for idx, item in enumerate(result_list[:self.k]):
                title = item.get("title", "No Title")
                content = item.get("text", item.get("snippet", str(item)))
                snippets.append(f"[{idx+1}] Title: {title}\nContent: {content}")
            return "\n\n".join(snippets)
        return str(result_data)