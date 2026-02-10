import logging
import os
import uuid
import math
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
        """Normalize crop resolution into a stable pixel range for VLM input."""
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

    def _fatal_response(self, code: str, message: str, hint: str) -> ToolResponse:
        return ToolResponse(
            text=(
                f"ERROR:{code}\n"
                f"Reason: {message}\n"
                f"Hint: {hint}"
            ),
            image=[],
        )

    def _load_last_image(self, agent_data: Any) -> tuple[Image.Image | None, str | None, str | None]:
        """Get latest image shown to the model, fallback to last local image path."""
        if agent_data and hasattr(agent_data, "image_data") and agent_data.image_data:
            latest = agent_data.image_data[-1]
            if isinstance(latest, Image.Image):
                return latest, "in_memory", None

        if not agent_data or not hasattr(agent_data, "extra_fields"):
            return None, None, "agent_data_missing"

        image_paths = agent_data.extra_fields.get("image_paths", [])
        if not image_paths:
            return None, None, "image_path_missing"

        last_path = image_paths[-1]
        if not os.path.exists(last_path):
            return None, last_path, "image_path_not_found"

        try:
            return Image.open(last_path).convert("RGB"), last_path, "fallback_path_used"
        except Exception as e:
            logger.error(f"Failed to load fallback image from {last_path}: {e}")
            return None, last_path, "image_open_failed"

    def _normalize_and_clamp_bbox(
        self, x1: int, y1: int, x2: int, y2: int, image_w: int, image_h: int
    ) -> tuple[tuple[int, int, int, int] | None, list[str]]:
        warnings: list[str] = []

        pre_clamp = (x1, y1, x2, y2)
        safe_x1, safe_y1, safe_x2, safe_y2 = self._clamp_box(x1, y1, x2, y2, image_w, image_h)
        post_clamp = (safe_x1, safe_y1, safe_x2, safe_y2)
        if post_clamp != pre_clamp:
            warnings.append(
                f"WARN:BBOX_CLAMPED - input {list(pre_clamp)} was clipped to image bounds -> {list(post_clamp)}."
            )

        if safe_x1 >= safe_x2 or safe_y1 >= safe_y2:
            return None, warnings
        return (safe_x1, safe_y1, safe_x2, safe_y2), warnings

    @staticmethod
    def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
        safe_x1 = max(0, min(x1, w))
        safe_y1 = max(0, min(y1, h))
        safe_x2 = max(0, min(x2, w))
        safe_y2 = max(0, min(y2, h))
        return safe_x1, safe_y1, safe_x2, safe_y2

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> ToolResponse:
        del instance_id
        agent_data = kwargs.get("agent_data")
        bbox = parameters.get("bbox")

        try:
            if bbox is None:
                return self._fatal_response(
                    code="BBOX_MISSING",
                    message="bbox is None.",
                    hint="Use <bbox>[left, top, right, bottom]</bbox> with 4 integers.",
                )
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return self._fatal_response(
                    code="BBOX_FORMAT",
                    message=f"Invalid bbox format: {bbox}",
                    hint="bbox must be a JSON array with exactly 4 numbers, e.g. [120, 80, 420, 320].",
                )

            try:
                x1, y1, x2, y2 = (int(v) for v in bbox)
            except Exception:
                return self._fatal_response(
                    code="BBOX_TYPE",
                    message=f"bbox contains non-integer values: {bbox}",
                    hint="Use integer coordinates only.",
                )

            if x1 > x2:
                return self._fatal_response(
                    code="BBOX_X_ORDER_INVALID",
                    message=(
                        f"Invalid x-order: left ({x1}) is greater than right ({x2}). "
                        "This request is rejected; no auto-correction is applied."
                    ),
                    hint=(
                        "Action required: provide bbox as [left, top, right, bottom] with left < right. "
                        "Example: [120, 80, 420, 320]."
                    ),
                )
            if y1 > y2:
                return self._fatal_response(
                    code="BBOX_Y_ORDER_INVALID",
                    message=(
                        f"Invalid y-order: top ({y1}) is greater than bottom ({y2}). "
                        "This request is rejected; no auto-correction is applied."
                    ),
                    hint=(
                        "Action required: provide bbox as [left, top, right, bottom] with top < bottom. "
                        "Example: [120, 80, 420, 320]."
                    ),
                )
            if x1 == x2:
                return self._fatal_response(
                    code="BBOX_ZERO_WIDTH",
                    message=(
                        f"Degenerate bbox width: left ({x1}) equals right ({x2}). "
                        "This request is rejected; no pixel expansion is applied."
                    ),
                    hint=(
                        "Action required: use a non-zero width box (right must be at least left + 1). "
                        "Example: [120, 80, 421, 320]."
                    ),
                )
            if y1 == y2:
                return self._fatal_response(
                    code="BBOX_ZERO_HEIGHT",
                    message=(
                        f"Degenerate bbox height: top ({y1}) equals bottom ({y2}). "
                        "This request is rejected; no pixel expansion is applied."
                    ),
                    hint=(
                        "Action required: use a non-zero height box (bottom must be at least top + 1). "
                        "Example: [120, 80, 420, 321]."
                    ),
                )

            image, source_ref, image_status = self._load_last_image(agent_data)

            if image is None:
                if image_status == "image_path_not_found":
                    return self._fatal_response(
                        code="IMAGE_PATH_NOT_FOUND",
                        message=f"Last image path does not exist: {source_ref}",
                        hint=(
                            "Run <search> again to refresh image_paths before calling <bbox>. "
                            "Or continue reasoning with the current image context without calling <bbox>."
                        ),
                    )
                if image_status == "image_open_failed":
                    return self._fatal_response(
                        code="IMAGE_OPEN_FAILED",
                        message=f"Failed to open last image path: {source_ref}",
                        hint=(
                            "Run <search> again or select another area. "
                            "Or continue reasoning with the current image context without calling <bbox>."
                        ),
                    )
                return self._fatal_response(
                    code="NO_IMAGE",
                    message="No image available for cropping.",
                    hint="Call <search> first, then call <bbox>.",
                )

            img_w, img_h = image.size
            normalized_bbox, warnings = self._normalize_and_clamp_bbox(x1, y1, x2, y2, img_w, img_h)
            if normalized_bbox is None:
                return self._fatal_response(
                    code="BBOX_OUT_OF_VIEW",
                    message="bbox has no valid overlap with current image after correction/clamping.",
                    hint="Use coordinates inside the visible image and keep right>left, bottom>top.",
                )
            safe_x1, safe_y1, safe_x2, safe_y2 = normalized_bbox

            cropped_img = image.crop((safe_x1, safe_y1, safe_x2, safe_y2))
            processed_img = self._process_image_resolution(cropped_img)

            if image_status == "fallback_path_used":
                warnings.append("WARN:FALLBACK_IMAGE_SOURCE - used last local image path instead of in-memory image.")

            if self.debug_mode:
                original_name = "unknown"
                if source_ref and source_ref != "in_memory":
                    original_name = os.path.splitext(os.path.basename(source_ref))[0]
                save_filename = f"{original_name}_crop_{uuid.uuid4().hex[:8]}.jpg"
                save_path = os.path.join(self.crops_dir, save_filename)
                try:
                    processed_img.save(save_path)
                    logger.debug(f"Saved crop to {save_path} (from {bbox} -> {[safe_x1, safe_y1, safe_x2, safe_y2]})")
                except Exception as e:
                    warnings.append(f"WARN:DEBUG_SAVE_FAILED - {e}")

            response_text = f"Cropped region {[safe_x1, safe_y1, safe_x2, safe_y2]}"
            if warnings:
                response_text = (
                    "WARNING: Crop completed with auto-correction.\n"
                    + "\n".join(warnings)
                    + f"\nFinal bbox: {[safe_x1, safe_y1, safe_x2, safe_y2]}"
                )
            return ToolResponse(
                text=response_text,
                image=[processed_img]
            )

        except Exception as e:
            logger.error(f"BBox Tool Execution Failed: {str(e)}", exc_info=True)
            return self._fatal_response(
                code="INTERNAL",
                message=f"Unexpected exception: {str(e)}",
                hint="Retry with a valid bbox JSON array: [left, top, right, bottom].",
            )
