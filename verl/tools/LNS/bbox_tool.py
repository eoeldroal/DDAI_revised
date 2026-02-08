import logging
import os
import uuid
import math
import traceback
from typing import Any
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

class ImageCropper(BaseTool):
    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.crops_dir = config.get("crops_dir", "./agent_crops")
        os.makedirs(self.crops_dir, exist_ok=True)
        self.debug_mode = config.get("debug", True)
        logger.info(f"Initialized ImageCropper. Crops will be saved to: {self.crops_dir}")

    def _process_image_resolution(self, image: Image.Image) -> Image.Image:
        """[generation.py:697-721] 해상도 정규화 (200k~400k 픽셀)"""
        max_pixels = 512 * 28 * 28
        min_pixels = 256 * 28 * 28
        
        w, h = image.size
        pixel_count = w * h

        if pixel_count > max_pixels:
            resize_factor = math.sqrt(max_pixels / pixel_count)
            new_w, new_h = int(w * resize_factor), int(h * resize_factor)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif pixel_count < min_pixels:
            resize_factor = math.sqrt(min_pixels / pixel_count)
            new_w, new_h = int(w * resize_factor), int(h * resize_factor)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResponse:
        agent_data = kwargs.get('agent_data')
        bbox = parameters.get('bbox')
        
        # 1. 모델이 본 이미지(메모리)와 원본 이미지(디스크) 둘 다 확보 시도
        seen_image = None
        raw_image = None
        
        # 1-1. 모델이 본 이미지 (좌표 스케일링 기준점)
        if agent_data and hasattr(agent_data, 'image_data') and agent_data.image_data:
             seen_image = agent_data.image_data[-1]

        # 1-2. 원본 이미지 (실제 크롭 대상)
        if agent_data and hasattr(agent_data, 'extra_fields'):
            image_paths = agent_data.extra_fields.get('image_paths', [])
            if image_paths:
                raw_path = image_paths[-1]
                if os.path.exists(raw_path):
                    try:
                        raw_image = Image.open(raw_path).convert("RGB")
                    except Exception as e:
                        logger.error(f"Failed to load raw image from {raw_path}: {e}")

        # [Fallback] 하나라도 없으면 있는 걸로 대체 (최소한의 방어)
        if seen_image is None and raw_image is not None:
            seen_image = raw_image 
        if raw_image is None and seen_image is not None:
            raw_image = seen_image 

        if raw_image is None:
            return ToolResponse(text="Error: No image found to crop. Please ensure 'search' was successful.", image=[])

        try:
            # 2. 좌표 변환 (Coordinate Mapping)
            # generation.py 로직: Display 좌표 -> Raw 좌표 변환
            seen_w, seen_h = seen_image.size
            raw_w, raw_h = raw_image.size
            
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}")
            
            x1, y1, x2, y2 = map(int, bbox)

            # 비례식 적용 (0으로 나누기 방지)
            scale_x = raw_w / seen_w if seen_w > 0 else 1.0
            scale_y = raw_h / seen_h if seen_h > 0 else 1.0
            
            raw_x1 = int(x1 * scale_x)
            raw_y1 = int(y1 * scale_y)
            raw_x2 = int(x2 * scale_x)
            raw_y2 = int(y2 * scale_y)

            # 3. Safety Clamping (경계값 보정)
            safe_x1 = max(0, min(raw_x1, raw_w))
            safe_y1 = max(0, min(raw_y1, raw_h))
            safe_x2 = max(0, min(raw_x2, raw_w))
            safe_y2 = max(0, min(raw_y2, raw_h))

            if safe_x1 >= safe_x2 or safe_y1 >= safe_y2:
                raise ValueError(f"Invalid crop dimensions after clamping: {[safe_x1, safe_y1, safe_x2, safe_y2]}")

            # 4. 크롭 및 해상도 정규화
            cropped_img = raw_image.crop((safe_x1, safe_y1, safe_x2, safe_y2))
            processed_img = self._process_image_resolution(cropped_img)
            
            # (선택) 디버그 모드일 때만 파일로 저장하고 로그 남김
            if self.debug_mode:
                # 원본 이미지 이름 추출 (예: "19_3.jpg" -> "19_3")
                original_name = "unknown"
                if image_paths:
                    original_name = os.path.splitext(os.path.basename(image_paths[-1]))[0]
                save_filename = f"{original_name}_crop_{uuid.uuid4().hex[:8]}.jpg"
                save_path = os.path.join(self.crops_dir, save_filename)
                processed_img.save(save_path)
                logger.debug(f"Saved crop to {save_path} (from {bbox} -> {[safe_x1, safe_y1, safe_x2, safe_y2]})")

            return ToolResponse(
                text=f"Cropped region {[safe_x1, safe_y1, safe_x2, safe_y2]}",
                image=[processed_img]
            )

        except Exception as e:
            logger.error(f"BBox Tool Execution Failed: {str(e)}", exc_info=True)
            return ToolResponse(text=f"Error processing bbox: {str(e)}", image=[])