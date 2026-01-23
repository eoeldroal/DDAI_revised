import logging
import os
import uuid
from typing import Any
from PIL import Image

# BaseTool 상속
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

logger = logging.getLogger(__name__)

class ImageCropper(BaseTool):
    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        # generation_phase1.py의 로직 유지: 크롭된 이미지 저장 경로 설정
        self.crops_dir = config.get("crops_dir", "./agent_crops")
        os.makedirs(self.crops_dir, exist_ok=True)
        logger.info(f"Initialized ImageCropper. Crops will be saved to: {self.crops_dir}")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResponse:
        """
        Executes the crop operation based on the bbox provided by the model.
        Args:
            parameters: {'bbox': [x1, y1, x2, y2]}
            kwargs: Contains 'agent_data' which holds the conversation history and images.
        """
        agent_data = kwargs.get('agent_data')
        bbox = parameters.get('bbox') # Expecting List[int/float]

        # 1. 안전 장치: 크롭할 원본 이미지가 있는지 확인
        if not agent_data or not agent_data.image_data:
            logger.warning("BBox Tool called but no images found in history.")
            return ToolResponse(text="Error: No image found to crop. Please search for an image first.")

        # 가장 최근 이미지를 가져옴 (모델이 방금 본 이미지)
        # SearchTool이 PIL.Image 객체를 반환했으므로 리스트에 PIL 객체가 들어있음
        last_image = agent_data.image_data[-1]

        # 이미지 경로 정보 가져오기 (SearchTool에서 저장한 경로)
        image_paths = agent_data.extra_fields.get('image_paths', [])
        last_image_path = image_paths[-1] if image_paths else None
        last_image_name = os.path.basename(last_image_path) if last_image_path else "unknown"
        
        # 2. BBox 유효성 검사 및 좌표 보정
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            return ToolResponse(text=f"Error: Invalid bbox format {bbox}. Expected [x1, y1, x2, y2].")

        try:
            width, height = last_image.size
            x1, y1, x2, y2 = bbox
            
            # generation_phase1.py의 안전 로직 이식:
            # 좌표가 이미지 범위를 벗어나지 않도록 Clamp 처리
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 크롭 영역이 유효하지 않은 경우 (너비나 높이가 0 이하)
            if x1 >= x2 or y1 >= y2:
                return ToolResponse(text=f"Error: Invalid crop dimensions. [{x1}, {y1}, {x2}, {y2}] results in zero or negative area.")

            # 3. 이미지 크롭 수행
            cropped_img = last_image.crop((x1, y1, x2, y2))
            
            # 4. (옵션) 디버깅 및 로깅을 위해 파일 저장 (generation_phase1.py 동작 유지)
            # 원본 이미지 이름을 활용하여 파일명 생성
            base_name = os.path.splitext(last_image_name)[0]
            save_filename = f"{base_name}_crop_{uuid.uuid4().hex[:8]}.jpg"
            save_path = os.path.join(self.crops_dir, save_filename)
            cropped_img.save(save_path)
            logger.debug(f"Saved cropped image to {save_path} (from original: {last_image_name})")

            # 5. 결과 반환
            # SearchTool과 마찬가지로 단일 ToolResponse 객체를 반환합니다.
            # 텍스트 메시지와 함께 크롭된 이미지 객체를 전달하면 ToolAgentLoop가 처리합니다.
            return ToolResponse(
                text=f"Image cropped to {bbox}. Saved to {save_path}", 
                image=cropped_img
            )

        except Exception as e:
            logger.error(f"BBox Tool Exception: {e}", exc_info=True)
            return ToolResponse(text=f"Error processing bbox: {str(e)}")