import re
import json
import string
import random
import torch
import os
import numpy as np
import traceback
import asyncio
import logging
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import collections
import uuid
from utils_metric import compute_metrics, parse_result
logger = logging.getLogger(__name__)


# 常量定义 - 防止魔法数字
class ScoreConstants:
    """奖励分数相关常量"""

    MIN_SCORE = 0.05  # 降低最低保底分数
    INVALID_CALL_PENALTY = 0.1  # 无效调用次数*0.1分
    FORMAT_ERROR_PENALTY = 0.0  # 格式错误直接0分
    TARGET_MISS_PENALTY = 0.2  # 目标未找到的惩罚
    TURN_PENALTY_RATE = 0.1  # 每多一轮的惩罚率
    MAX_TURNS_WITHOUT_PENALTY = 100  # 不惩罚的最大轮数


@dataclass
class ScoreResult:
    """分数计算结果"""

    final_score: float
    breakdown: Dict[str, float]
    warnings: List[str]
    traj_id: str
    # debug_info: Dict


class RewardCalculator(object):
    """奖励计算器 - 主要业务逻辑"""

    def __init__(self):

        self.counter = collections.defaultdict(int)
        self.penalty_per_turn_multiplier = 0.8
        self.max_turns_without_penalty = ScoreConstants.MAX_TURNS_WITHOUT_PENALTY

    #     pass
    # self.constants = ScoreConstants()

    async def compute(self, result: Dict, only_last: bool = False) -> ScoreResult:
        details = {}
        """计算奖励分数的主入口"""
        try:

            messages = result["messages"]
            extra_info = result["extra_info"]
            stats = result["stats"]

            # 记录TRAJID
            input_text = result["messages"][0]["content"]
            cur_uuid = uuid.uuid5(uuid.NAMESPACE_URL, input_text)
            if cur_uuid not in self.counter:
                self.counter[cur_uuid] = str(len(self.counter)).zfill(6)
            cur_traj_id = self.counter[cur_uuid]
            # 解析解决方案
            if only_last:
                msg_to_parse = [messages[-1]["content"]]
            else:
                msg_to_parse = [msg["content"] for msg in messages if msg["role"] == "assistant"]
            f1_overall_list = []
            all_metrics_list = []
            for mi, msg in enumerate(msg_to_parse):
                status, output_fmt_actor = parse_result(msg)

                if status.startswith("ERROR"):
                    details["parse"] = status
                    return self._create_failure_result(
                        status, messages, extra_info, cur_traj_id
                    )

                labels = json.loads(extra_info["hallucination_labels"])

                if len(labels) == 0 and len(output_fmt_actor) == 0:
                    f1_overall, all_metrics = 1, {}
                else:
                    result_fmt = {
                        "n_turns": stats["turns"],
                        "response": extra_info["response"],
                        "labels": labels,
                        "output": output_fmt_actor,
                        "task_type": extra_info["task_type"],
                    }
                    record = collections.defaultdict(int)
                    f1_overall, all_metrics = compute_metrics([result_fmt], record)
                # Always append f1_overall to ensure f1_first_turn is available
                f1_overall_list.append(f1_overall)
                all_metrics_list.append(all_metrics)
            details["all_metrics"] = all_metrics_list
            details["f1_overall_list"] = f1_overall_list
            # FOR SOPO
            # final_score = sum(f1_overall_list) / len(f1_overall_list)
            # FOR CRITIQUE GRPO
            final_score = f1_overall_list[-1]
            # only_last是rollout中才开启
            if not only_last:
                result["stats"]['f1_first_turn'] = f1_overall_list[0]
            return ScoreResult(
                final_score=final_score,
                breakdown={
                    "details": details,
                },
                warnings=[],
                traj_id=cur_traj_id,
            )

        except Exception as e:
            logger.error(f"计算分数时发生错误: {e}")
            traceback.print_exc()
            return self._create_failure_result(
                f"计算错误: {str(e)}", messages, extra_info, cur_traj_id
            )

    # def _validate_output_format(self, content: str) -> bool:
    #     """验证输出格式"""
    #     try:
    #         think_token_count = content.count("</think>")
    #         assert think_token_count == 1, 'THINK TOKEN "</think>" COUNT NOT EQUAL TO 1'
    #         output_str = content.rsplit("</think>", 1)[-1]
    #         output_str = output_str.replace("<|im_end|>", "").strip()

    #         ## 抽取"{'hallucination list': []}"类似的内容，只抽取最后一组
    #         output_str = re.findall(r"""{"hallucination list": \[.*?\]}""", output_str)
    #         assert len(output_str) == 1, f"MULTIPLE HALLUCINATION LIST FOUND"
    #         output_str = output_str[0]

    #         # 使用Pydantic验证
    #         output = json.loads(output_str, strict=False)
    #         assert isinstance(output, dict), f"OUTPUT IS NOT A DICT"
    #         assert "hallucination list" in output, f"HALLUCINATION LIST NOT FOUND"
    #         assert isinstance(
    #             output["hallucination list"], list
    #         ), f"HALLUCINATION LIST IS NOT A LIST"
    #         return "SUCCESS", output["hallucination list"]

    #     except Exception as e:
    #         traceback.print_exc()
    #         return f"ERROR: {str(e)}", None

    def _create_failure_result(
        self, reason: str, messages: List[Dict], extra_info: Dict, cur_traj_id: str
    ) -> ScoreResult:
        """创建失败结果"""
        return ScoreResult(
            final_score=0.0,
            breakdown={"error": reason},
            warnings=[reason],
            traj_id=cur_traj_id,
            # debug_info=self._create_debug_info(messages, extra_info, 0.0),
        )


class DebugLogger:
    """调试日志记录器"""

    def __init__(self, debug_dir: str = "./havor_debug"):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    def log_score_result(self, result: Dict, score_result: ScoreResult):
        """记录分数结果"""
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            hour_dir = datetime.now().strftime("%Y_%m_%d_%H")
            minute_dir = datetime.now().strftime("%Y_%m_%d_%H_%M")
            minute_path = os.path.join(self.debug_dir, hour_dir, minute_dir)
            os.makedirs(minute_path, exist_ok=True)

            # Create a copy of result without the non-serializable completions field
            result_copy = dict(result)
            result_copy.pop("completions", None)  # Remove ChatCompletion objects

            debug_data = {
                "result": result_copy,
                "final_score": score_result.final_score,
                "breakdown": score_result.breakdown,
                "warnings": score_result.warnings,
                # "debug_info": score_result.debug_info,
            }
            traj_id = score_result.traj_id
            with open(
                os.path.join(minute_path, f"output_traj_{traj_id}_{timestamp}.json"),
                "w",
            ) as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"记录调试信息失败: {e}")


# 全局实例
reward_calculator = RewardCalculator()
debug_logger = DebugLogger()


async def compute_score(result: Dict, to_disk: bool = True, only_last: bool = False) -> float:
    """计算分数的公共接口"""
    score_result = await reward_calculator.compute(result, only_last)

    # 记录调试信息
    if to_disk:
        debug_logger.log_score_result(result, score_result)

    return score_result.final_score


# 向后兼容的同步接口
def compute_score_sync(result: Dict) -> float:
    """同步版本的分数计算接口"""
    return asyncio.run(compute_score(result))


if __name__ == "__main__":
    result = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": '{"hallucination list": ["His net worth was estimated to be $2.1 billion.", "Andrew Getty, an heir to the Getty oil fortune, died of natural causes"]}<|im_end|>',
            },
        ],
        "extra_info": {
            "response": "Andrew Getty, an heir to the Getty oil fortune, died of natural causes, the LA Police Department says. He was found dead at home with some medication present, but no foul play is suspected. Getty had filed for a restraining order against an ex-girlfriend, citing health concerns. His net worth was estimated to be $2.1 billion.",
            "hallucination_labels": '[{"start": 280, "end": 327, "text": "His net worth was estimated to be $2.1 billion.", "meta": "EVIDENT CONFLICT:\\nOriginal:  Gordon Getty whose net worth is stated as $2.1 billion.\\nAIGC: Andrew Getty\'s net worth was estimated to be $2.1", "label_type": "Evident Conflict", "implicit_true": false, "due_to_null": false}]',
            "task_type": "Summary",
        },
        "stats": {"turns": 1},
    }
    # result = {
    #     "messages": [
    #         {"role": "user", "content": "What is the capital of France?"},
    #         {"role": "assistant", "content": '{"hallucination list": []}<|im_end|>'},
    #     ],
    #     "extra_info": {
    #         "response": "Andrew Getty, an heir to the Getty oil fortune, died of natural causes, the LA Police Department says. He was found dead at home with some medication present, but no foul play is suspected. Getty had filed for a restraining order against an ex-girlfriend, citing health concerns. His net worth was estimated to be $2.1 billion.",
    #         "hallucination_labels": "[]",
    #         "task_type": "Summary",
    #     },
    #     "stats": {"turns": 1},
    # }
    print(compute_score_sync(result))
