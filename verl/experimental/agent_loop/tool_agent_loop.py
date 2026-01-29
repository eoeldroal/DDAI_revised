# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrajectoryLogger:
    """Thread-safe singleton logger for saving rollout trajectories to JSONL file.

    Groups agent responses by sample_id (e.g., train_13) so that all agents'
    responses for the same sample are written together.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = "logs", n_agent: int = 2):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: str = "logs", n_agent: int = 2):
        if self._initialized:
            return
        self._initialized = True
        self._write_lock = threading.Lock()
        self._buffer: dict[str, list[dict]] = {}  # sample_id -> list of agent entries
        self.n_agent = n_agent  # Expected number of agents per sample
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rollout_trajectory.jsonl"
        logger.info(f"TrajectoryLogger initialized. Logging to: {self.log_file}, n_agent={n_agent}")

    def set_n_agent(self, n_agent: int):
        """Set the expected number of agents per sample."""
        self.n_agent = n_agent

    def log(self, entry: dict):
        """Buffer an agent entry and write when all agents for the sample have completed."""
        sample_id = entry.get("sample_id", "unknown")
        entry["timestamp"] = datetime.now().isoformat()

        with self._write_lock:
            # Initialize buffer for this sample if needed
            if sample_id not in self._buffer:
                self._buffer[sample_id] = []

            # Add entry to buffer
            self._buffer[sample_id].append(entry)

            # Check if all agents have completed for this sample
            if len(self._buffer[sample_id]) >= self.n_agent:
                self._write_grouped_entry(sample_id)

    def _write_grouped_entry(self, sample_id: str):
        """Write all agent entries for a sample as a grouped entry."""
        entries = self._buffer.pop(sample_id, [])
        if not entries:
            return

        # Sort by agent_id for consistent ordering
        entries.sort(key=lambda x: x.get("agent_id", 0))

        # Build grouped entry
        grouped_entry = {
            "sample_id": sample_id,
            "reference_documents": entries[0].get("ndcg_details", {}).get("reference_documents", []),
            "agents": []
        }

        for entry in entries:
            agent_entry = {
                "agent_id": entry.get("agent_id"),
                "response": entry.get("response"),
                "num_turns": entry.get("num_turns"),
                "terminated": entry.get("terminated"),
                "termination_reason": entry.get("termination_reason"),
                "retrieved_documents": entry.get("ndcg_details", {}).get("retrieved_documents", []),
                "timestamp": entry.get("timestamp")
            }
            grouped_entry["agents"].append(agent_entry)

        # Write to file with pretty print for readability
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(grouped_entry, ensure_ascii=False, indent=2) + "\n\n")

    def flush(self):
        """Flush any remaining buffered entries (for cleanup)."""
        with self._write_lock:
            for sample_id in list(self._buffer.keys()):
                self._write_grouped_entry(sample_id)


# Global trajectory logger instance
trajectory_logger = TrajectoryLogger()


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        sample_id: Optional[str] = None,
        reference_page: Optional[list] = None,
        agent_index: Optional[int] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.sample_id = sample_id  # Original dataset id (e.g., "train_14") for search server
        self.reference_page = reference_page or []  # Ground truth reference pages
        self.agent_index = agent_index  # Agent index for GRPO (1, 2, ...)

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_turns = 0
        self.termination_reason: Optional[str] = None

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        # Get tool_settings from main config for overriding tool config values
        tool_settings = getattr(config.actor_rollout_ref.rollout.multi_turn, 'tool_settings', None)
        extra_config = dict(tool_settings) if tool_settings else {}
        tool_list = initialize_tools_from_config(tool_config_path, extra_config) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        )
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # Set n_agent for trajectory logger (for grouping responses by sample_id)
        n_agent = getattr(config.actor_rollout_ref.rollout, 'n', 2)
        trajectory_logger.set_n_agent(n_agent)

        # Initialize interactions from config file
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    def _set_termination_reason(self, agent_data: AgentData, reason: str) -> None:
        if agent_data.termination_reason is None:
            agent_data.termination_reason = reason

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Extract sample_id from uid (original dataset id like "train_14")
        sample_id = kwargs.get("uid", None)

        # Extract reference_page and agent_index for trajectory logging
        reference_page = kwargs.get("reference_page", [])
        agent_index = kwargs.get("rollout_n", 0) + 1  # GRPO agent index: 0-indexed -> 1-indexed (1, 2, ...)

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            sample_id=sample_id,
            reference_page=reference_page,
            agent_index=agent_index,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                self._set_termination_reason(agent_data, "invalid_state")
                state = AgentState.TERMINATED

        # Finalize output
        if agent_data.termination_reason is None:
            agent_data.termination_reason = "unknown"
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.assistant_turns,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores})

        # Log trajectory at episode termination
        try:
            # Decode response to get full trajectory text
            response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )

            # Get retrieved image paths from extra_fields
            retrieved_paths = agent_data.extra_fields.get('image_paths', [])
            # Extract just the filenames (e.g., "2_7.jpg" from full path)
            retrieved_docs = [os.path.basename(p) for p in retrieved_paths]

            # Build trajectory log entry
            trajectory_entry = {
                "sample_id": agent_data.sample_id,
                "agent_id": agent_data.agent_index,
                "response": response_text,
                "num_turns": agent_data.assistant_turns,
                "terminated": True,
                "termination_reason": agent_data.termination_reason,
                "ndcg_details": {
                    "retrieved_documents": retrieved_docs,
                    "reference_documents": agent_data.reference_page,
                }
            }
            trajectory_logger.log(trajectory_entry)
        except Exception as e:
            logger.warning(f"Failed to log trajectory: {e}")

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        if agent_data.tool_calls:
            agent_data.tool_turns += 1

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Check response_length termination (this must be checked before tool processing)
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            self._set_termination_reason(agent_data, "response_length")
            return AgentState.TERMINATED

        # If tool calls exist, process them first (termination check after tool execution)
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        # No tool calls - check other termination conditions
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            self._set_termination_reason(agent_data, "max_assistant_turns")
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            self._set_termination_reason(agent_data, "max_user_turns")
            return AgentState.TERMINATED

        # Determine next state
        if self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            self._set_termination_reason(agent_data, "no_tool_call")
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:

            # 1. 종료 신호 처리 (generation_phase1.py의 'dones.append(1)' 로직 이식)
            if tool_call.name == "search_complete":
                # search_complete가 나오면 즉시 종료 상태로 전이
                logger.info(f"Terminating sequence due to <search_complete>")
                self._set_termination_reason(agent_data, "search_complete")
                return AgentState.TERMINATED

            # 2. 실제 도구(search, bbox) 실행
            # 이 시점에서 yaml 설정에 'search', 'bbox'라는 이름의 툴이 등록되어 있어야 합니다.
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        for tool_response in responses:
            # Filter valid images first (before adding markers)
            valid_images = []
            if tool_response.image:
                if isinstance(tool_response.image, list):
                    valid_images = [img for img in tool_response.image if img is not None]
                elif tool_response.image is not None:
                    valid_images = [tool_response.image]

            # Create message from tool response
            if valid_images or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if valid_images:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)
            new_images_this_turn.extend(valid_images)

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn or None,  # Pass None instead of empty list
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            self._set_termination_reason(agent_data, "response_length")
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        # Check termination conditions after tool execution (before next generation)
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            self._set_termination_reason(agent_data, "max_assistant_turns")
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            self._set_termination_reason(agent_data, "max_user_turns")
            return AgentState.TERMINATED

        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            self._set_termination_reason(agent_data, "interaction_terminate")
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(text=f"Error when executing tool: {e}")
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
