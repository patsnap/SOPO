import json
import os
import sys
import time
import asyncio
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from openai import OpenAI

import json5
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from transformers import PreTrainedTokenizer
from qwen_agent.tools import BaseTool
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging
import uuid
import re
import traceback
import random
# Handle imports for both direct execution and Ray launcher
module_dir = Path(__file__).parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

try:  # Try relative import first (works if executed via -m with package context)
    from .reward_optimized import compute_score
except ImportError:  # Fallback to absolute import
    from reward_optimized import compute_score
from utils_metric import parse_result
logger = logging.getLogger("HAVORAgent")


import datetime


PROMPT_CRITIC_TEMPLATE = """# ROLE
You are a Socratic Logic Probe. Your goal is to guide the user (who provided the `CoT`) to self-correct their mistakes. You are talking directly to them.

# INPUT DATA
## CONTEXT
{context_fmt_str}

## STATEMENT
{statement}

## CORRECT_ANSWER (HIDDEN TRUTH)
{correct_answer}

## CoT (User's Chain of Thought)
{result}

# CRITICAL RULES (NON-NEGOTIABLE)
1. **VOICE & TONE:** In your final output, speak directly to the user. ALWAYS use "**you**" and "**your**" to refer to the user's reasoning.
2. **NO SPOILERS:** NEVER reveal the `CORRECT_ANSWER` or the final classification (True/False).
3. **LOCATE THE ERROR:** Scan the `CoT` specifically looking for the `## Step n` markers. Find the *first* step where **the user's** logic deviates from the `CONTEXT` or misinterprets the `STATEMENT`.
4. **QUESTION THE PREMISE:** Formulate a question that targets **the user's** specific misunderstanding in that Step.
5. **MANDATORY REDIRECTION:** You MUST conclude by instructing the user to discard **their** reasoning from that step onwards (e.g., "Please re-examine logic...").

# WORKFLOW
1. **Compare:** Match each `## Step n` in the `CoT` against the `CONTEXT`.
2. **Identify:** Find the specific Step (e.g., "Step 2") where **the user** started hallucinating or making a logic error.
3. **Draft Question:** Create a leading question based ONLY on the `CONTEXT` evidence relevant to that Step.
   - *Drafting Mental Check:* The question must ask "In Step X, **you** stated..." (Address the user as 'you' in the final draft).
4. **Output:** Structure your response strictly as defined below, maintaining the second-person perspective throughout.

# RESPONSE FORMAT
<analysis>
[Briefly explain to the user why the specific Step **you found** is the root cause of **their** error. Address them directly. E.g., "You misinterpreted the relationship between X and Y here..."]
</analysis>

<guidance>
[Your Socratic Question pointing to the evidence **the user** ignored, phrased directly to them.]
**Please re-examine the logic in ## Step [n] and re-evaluate your conclusion from there.**
</guidance>"""

PROMPT_CRITIC_TEMPLATE_v2 = """# ROLE
You are a Socratic Logic Probe. Your goal is to guide the user (who provided the `CoT`) to self-correct their mistakes. You are talking directly to them.

# INPUT DATA
## CONTEXT
{context_fmt_str}

## STATEMENT
{statement}

## CORRECT_ANSWER (HIDDEN TRUTH)
{correct_answer}

## CoT (User's Chain of Thought)
{result}

# CRITICAL RULES (NON-NEGOTIABLE)
1. **VOICE & TONE:** In your final output, speak directly to the user. ALWAYS use "**you**" and "**your**" to refer to the user's reasoning.
2. **NO SPOILERS:** NEVER reveal the `CORRECT_ANSWER` or the final classification (True/False).
3. **LOCATE THE ERROR:** Scan the `CoT` specifically looking for the `## Step n` markers. Find the *first* step where **the user's** logic deviates from the `CONTEXT` or misinterprets the `STATEMENT`.
4. **QUESTION THE PREMISE:** Formulate a question that targets **the user's** specific misunderstanding in that Step.
5. **MANDATORY REDIRECTION:** You MUST conclude by instructing the user to discard **their** reasoning from that step onwards (e.g., "Please re-examine logic...").

# WORKFLOW
1. **Compare:** Match each `## Step n` in the `CoT` against the `CONTEXT`.
2. **Identify:** Find the specific Step (e.g., "Step 2") where **the user** started hallucinating or making a logic error.
3. **Draft Question:** Create a leading question based ONLY on the `CONTEXT` evidence relevant to that Step.
   - *Drafting Mental Check:* The question must ask "In Step X, **you** stated..." (Address the user as 'you' in the final draft).
4. **Output:** Structure your response strictly as defined below, maintaining the second-person perspective throughout.

# RESPONSE FORMAT
<analysis>
[Briefly explain to the user why the specific Step **you found** is the root cause of **their** error. Address them directly. E.g., "You misinterpreted the relationship between X and Y here..."]
</analysis>

<guidance>
[Your Socratic Question pointing to the evidence **the user** ignored, phrased directly to them.]
**Please re-examine the logic in ## Step [n] and re-evaluate your conclusion from there.**
</guidance>"""


PROMPT_CRITIC_TEMPLATE_SMALL_TEACHER_v2 = """# ROLE
You are a Socratic teacher guiding a student to self-correct their logical reasoning. You must speak directly to the student using "you" and "your".

# INSTRUCTIONS
Your task is to read the student's `CoT` and compare it against the `CONTEXT` and the `CORRECT_ANSWER`. Find the exact step (e.g., ## Step 2) where the student makes their first logical error or hallucination.

You must follow these strict rules:
1. NEVER reveal the `CORRECT_ANSWER` or tell the student the final True/False classification.
2. Address the student directly (e.g., "You stated that...").

# HOW TO RESPOND
Write your response in exactly two natural paragraphs:

Paragraph 1 (Analysis): Directly tell the student which step contains the error. Briefly explain why their interpretation or logic in that specific step deviates from the provided CONTEXT. Do not use any special formatting or tags.

Paragraph 2 (Guidance): Ask a leading Socratic question based on the CONTEXT to help the student realize their mistake. After your question, you MUST conclude your response with this exact sentence:
"Please re-examine the logic in ## Step [n] and re-evaluate your conclusion from there." (Replace [n] with the actual step number where the error occurred).

# INPUT DATA
## CONTEXT
{context_fmt_str}

## STATEMENT
{statement}

## CORRECT_ANSWER (HIDDEN TRUTH)
{correct_answer}

## CoT (Student's Chain of Thought)
{result}
"""


PROMPT_CRITIC_TEMPLATE_LIMITED_SPOILER = """# ROLE
You are a Corrective Logic Guide. Your goal is to help the user (who provided the `CoT`) fix their reasoning errors. You are talking directly to them.

# INPUT DATA
## CONTEXT
{context_fmt_str}

## STATEMENT
{statement}

## CORRECT_ANSWER (HIDDEN TRUTH)
{correct_answer}

## CoT (User's Chain of Thought)
{result}

# CRITICAL RULES (NON-NEGOTIABLE)
1. **VOICE & TONE:** Speak directly to the user using "**you**" and "**your**". Be helpful but firm on facts.
2. **LIMITED SPOILERS (THE GOLDEN RULE):** 
   - You **ARE ALLOWED** to reveal specific facts from the `CONTEXT` or `CORRECT_ANSWER` to correct a factual error in the user's current step.
   - You **MUST NOT** simply blurt out the final classification (e.g., "The answer is False") without explanation. Instead, provide the **evidence** that leads to that conclusion.
   - *Goal:* Don't just give the fish; give them the fishing rod (the correct fact) so they can catch the fish (the conclusion) themselves.
3. **LOCATE THE ERROR:** Scan the `CoT` for the `## Step n` markers. Find the *first* step where **the user's** logic contradicts the `CONTEXT`.
4. **CORRECT WITH EVIDENCE:** Instead of asking vague questions, explicitly point out: "You said X, but the document actually says Y." Use the ground truth to invalidate their false premise.
5. **MANDATORY REDIRECTION:** Conclude by instructing the user to restart their reasoning from the error step, incorporating the correct fact you just provided.

# WORKFLOW
1. **Compare:** Check the user's `## Step n` against the `CONTEXT`.
2. **Identify:** Pinpoint the exact step where the user hallucinated or ignored a fact.
3. **Draft Correction:** Formulate a statement that reveals the *specific truth* needed to fix that step.
   - *Bad Response:* "You are wrong. The answer is False." (Too direct/Lazy)
   - *Good Response:* "In Step 2, you claimed the revenue increased. However, the Context explicitly states revenue decreased by 5%." (Targeted Spoiler)
4. **Output:** Structure your response strictly as defined below.

# RESPONSE FORMAT
<analysis>
[Analyze the specific error in the identified Step. Explicitly contrast what **the user** said versus what the `CONTEXT/CORRECT_ANSWER` says. It is okay to reveal the specific fact they got wrong.]
</analysis>

<guidance>
[Provide the specific evidence or fact that corrects their mistake.]
**Please re-examine ## Step [n] using this correct information and re-evaluate your conclusion.**
</guidance>"""

PROMPT_CRITIQUE_GRPO = """# ROLE
You are a Logic & Science Evaluation Expert. Your task is to evaluate a student's reasoning (CoT) based on provided evidence and guide them to the correct conclusion without prematurely revealing the final answer label.

# INPUT DATA
## CONTEXT
{context_fmt_str}

## STATEMENT
{statement}

## CORRECT_ANSWER (HIDDEN TRUTH)
{correct_answer}

## CoT (User's Chain of Thought)
{result}

# EVALUATION STEPS & RULES
1. **Identify the First Error:** Scan the user's `## Step n` markers. Find the very first step where their logic deviates from the `CONTEXT` or the `CORRECT_ANSWER`.
2. **Fact-Based Correction (The "Fishing Rod" Principle):** - DO NOT simply say "The answer is [True/False]".
   - DO provide the specific fact or evidence from the `CONTEXT` that contradicts the user's error. 
   - Address the user directly as "you". (e.g., "In Step 2, you assumed X, but the context states Y.")
3. **Guidance:** Briefly explain the logical implication of this corrected fact to help the user bridge the gap.

# RESPONSE FORMAT
Please provide your evaluation in the following structure:

### Evaluation Detail
- **Step-by-step Analysis:** [Briefly walk through the user's steps and point out exactly where and why the first error occurred using evidence from the Context.]
- **Key Correction:** [Provide the specific "hidden truth" or fact they missed.]

**Direction:** Please re-examine your reasoning starting from ## Step [n] using the information provided above.

Conclusion: [correct/incorrect] [END]"""

THRESH_FEEDBACK = 0.6
PROB_FEEDBACK = 0.6
SHOW_GUIDANCE_ONLY = False
SHOW_ANALYSIS_ONLY = False
SPOILER = False


TEACHER_MODEL_URL = os.environ.get("TEACHER_MODEL_URL", "http://10.148.0.127:8000/v1")
TEACHER_MODEL_NAME = os.environ.get("TEACHER_MODEL_NAME", "Qwen3-Next-80B-A3B-Instruct")

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")



class TeacherModel(object):
    def __init__(self, model_name: str, model_url: str):
        self.model_name = model_name
        self.model_url = model_url

    async def req_model_vllm(self, query):
        def _sync_call():
            try:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url=self.model_url,
                )
                temperature = 0.7
                chat_response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=4096,
                    temperature=temperature,
                    top_p=1,
                    # presence_penalty=1.2,
                    extra_body={
                        "stop": ["<|im_end|>"],
                        "top_k": 5,
                        "repetition_penalty": 1.0,
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    },
                )
                return chat_response.choices[0].message.content
            except Exception as e:
                traceback.print_exc()
                return str(e)

        # Run the blocking OpenAI call in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(_sync_call)

    async def judge_response(
        self,
        context_fmt_str: str,
        statement: str,
        correct_answer: str,
        response_student: str,
    ) -> float:
        if SPOILER:
            prompt = PROMPT_CRITIC_TEMPLATE_LIMITED_SPOILER.format(
                context_fmt_str=context_fmt_str,
                statement=statement,
                correct_answer=correct_answer,
                result=response_student,
            )
        else:
            prompt = PROMPT_CRITIC_TEMPLATE_SMALL_TEACHER_v2.format(
                context_fmt_str=context_fmt_str,
                statement=statement,
                correct_answer=correct_answer,
                result=response_student,
            )
            # prompt = PROMPT_CRITIQUE_GRPO.format(
            #     context_fmt_str=context_fmt_str,
            #     statement=statement,
            #     correct_answer=correct_answer,
            #     result=response_student,
            # )
        response = await self.req_model_vllm(prompt)
        return response

    async def test_network(self):
        response = await self.req_model_vllm("Hello, who are you?")
        print(response)


class HAVORAgent(FnCallAgent):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 10000,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
        max_context_length: int | None = None,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        # top_k: int = 50,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        # Get model max context length from tokenizer if available, otherwise use max_total_tokens
        if max_context_length is not None:
            self.max_context_length = max_context_length
        elif hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length > 0:
            self.max_context_length = tokenizer.model_max_length
        else:
            # Fallback: use a safe default that's typically larger than max_total_tokens
            self.max_context_length = max(max_total_tokens, 65536)
        self.temperature = temperature
        self.top_p = top_p
        # self.top_k = top_k
        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.teacher_model = TeacherModel(
            model_name=TEACHER_MODEL_NAME,
            model_url=TEACHER_MODEL_URL,
        )

    async def call_server(
        self,
        client: ArealOpenAI,
        messages: list[dict],
        # original_messages: list[dict] | None = None,
        max_attempts: int = 10,
    ) -> str:
        """
        Call the LLM server with messages.

        Args:
            client: The ArealOpenAI client
            messages: Messages to send to LLM (may be pruned/modified)
            original_messages: Original messages before pruning (if None, uses messages)
            max_attempts: Maximum number of retry attempts

        Returns:
            Tuple of (completion, content) or (None, None) on failure
        """
        # If original_messages is provided, use it for caching; otherwise use messages
        messages_for_llm = messages
        # messages_for_cache = original_messages if original_messages is not None else messages

        attempts = 0
        while attempts < max_attempts:
            try:
                completion = await client.chat.completions.create(
                    messages=messages_for_llm,
                    temperature=1.0,
                    # stop=["\n<tool_response>", "<tool_response>"],
                    stop=["<|im_end|>"],
                    max_completion_tokens=self.max_tokens_per_turn,
                    max_tokens=self.max_total_tokens,
                )
                content = completion.choices[0].message.content
                assert content, "Error: LLM response is empty."

                return completion, content
            except RuntimeError as e:
                logger.warning(
                    f"RuntimeError during LLM call_server at attempt {attempts}"
                )
                attempts += 1
                continue
        logger.error(
            f"Failed to get response from LLM after {max_attempts} attempts. "
            f"Returning None to end trajectory."
        )
        return None, None



    async def run_agent(
        self, data, client: ArealOpenAI, save_path: str | None = None
    ) -> List[List[Message]]:

        extra_info = data["extra_info"]
        prompt = extra_info["prompt"]
        labels = json.loads(extra_info["hallucination_labels"])
        messages = [
            {"role": "user", "content": prompt},
        ]
        stats = dict(
            turns=0,
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        # session_id = str(uuid.uuid4())
        need_feedback = random.random() < PROB_FEEDBACK
        while num_llm_calls_available > 0:
            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, content = await self.call_server(client, messages)
            completions.append(completion)
            messages.append({"role": "assistant", "content": content})

            termination, pred_hallu = parse_result(content)
            result_tmp = {
                "messages": messages,
                "termination": 'running',
                "completions": completions,  # final completion
                "stats": stats,
                "extra_info": extra_info,
            }
            score = await compute_score(result_tmp, to_disk=False, only_last=True)
            if num_llm_calls_available > 0:
                if termination == "SUCCESS":
                    # 只关注0分的情况，引入反馈机制
                    # if (len(pred_hallu) == 0 and len(labels) != 0) or (
                    #     len(pred_hallu) != 0 and len(labels) == 0
                    # ):
                    # score 小于0.5的情况，引入反馈机制
                    # is_feedback = np.random.rand() < 0.6
                    if score < THRESH_FEEDBACK and need_feedback:
                        context_fmt_str = (
                            extra_info["prompt"].rsplit("Below is a", 1)[0].strip()
                        )

                        response_teacher = await self.teacher_model.judge_response(
                            context_fmt_str=context_fmt_str,
                            statement=extra_info["response"],
                            correct_answer=extra_info["hallucination_labels"],
                            response_student=content,
                        )
                        if SHOW_GUIDANCE_ONLY:
                            response_teacher = re.findall(r"<guidance>(.*?)</guidance>", response_teacher, re.DOTALL)
                            if not len(response_teacher):
                                logger.warning("No guidance found in response_teacher")
                                break
                            response_teacher = response_teacher[0].strip()
                        elif SHOW_ANALYSIS_ONLY:
                            response_teacher = re.findall(r"<analysis>(.*?)</analysis>", response_teacher, re.DOTALL)
                            if not len(response_teacher):
                                logger.warning("No analysis found in response_teacher")
                                break
                            response_teacher = response_teacher[0].strip()
                        # else:
                        #     # pass
                        #     analysis = re.findall(r"<analysis>(.*?)</analysis>", response_teacher, re.DOTALL)
                        #     guidance = re.findall(r"<guidance>(.*?)</guidance>", response_teacher, re.DOTALL)
                        #     if len(analysis) and len(guidance):
                        #         response_teacher = analysis[0].strip() + "\n" + guidance[0].strip()
                        #     elif len(analysis):
                        #         logger.warning("No guidance found in response_teacher, use the analysis")
                        #         response_teacher = analysis[0].strip()
                        #     elif len(guidance):
                        #         logger.warning("No analysis found in response_teacher, use the guidance")
                        #         response_teacher = guidance[0].strip()
                        #     else:
                        #         logger.warning("No analysis or guidance found in response_teacher, break")
                        #         break
                            
                        messages.append({"role": "user", "content": response_teacher})
                    # 离散的情况，直接break
                    else:
                        break
                # 解析出问题
                elif termination == "ERROR: NUMBER OF HALLUCINATION LIST NOT EQUAL TO 1":
                    break
                elif termination == 'ERROR: THINK TOKEN "</think>" COUNT NOT EQUAL TO 1':
                    break
                elif termination == "ERROR: JSONDecodeError":
                    break
                else:
                    messages.append({"role": "user", "content": termination})
        termination = "result"

        result = {
            "messages": messages,
            "termination": termination,
            "completions": completions,  # final completion
            "stats": stats,
            "extra_info": extra_info,
        }
        return result

    async def compute_reward(self, result: Dict):
        reward = await compute_score(result)
        return reward

    async def make_trajectory(
        self,
        data: Dict[str, str],
        client: ArealOpenAI,
        save_path: str | None = None,
    ) -> Dict:
        result = await self.run_agent(data, client, save_path=save_path)
        reward = await self.compute_reward(result)
        completions = result["completions"]
        last_completion = completions[-1]
        client.set_reward(last_completion.id, reward)
        stats = result["stats"]
        return stats
