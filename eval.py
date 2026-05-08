import os
import pandas as pd
import traceback
import json, collections
from argparse import ArgumentParser
from openai import OpenAI
from datetime import datetime
import json_repair
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import time

## GRPO
MODEL_URL = "http://localhost:8000/v1"
MODEL_NAME = "SOPO_4B"

# TASK_TYPE_TO_INSTRUCTION = {
#     "QA": "answer",
#     "Data2txt": "article",
#     "Summary": "summary",
# }

TASK_TYPE_TO_INSTRUCTION = {
    "QA": "summary",
    "Data2txt": "summary",
    "Summary": "summary",
}


def req_model_vllm(query):
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=MODEL_URL,
        )
        temperature = 0
        chat_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": query}],
            max_tokens=8192,
            temperature=temperature,
            top_p=0.95,
            # presence_penalty=1.2,
            extra_body={
                "stop": ["<|im_end|>"],
                "top_k": -1,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                },
            },
        )

    except Exception as e:
        traceback.print_exc()
        return str(e)
    return chat_response.choices[0].message.content


# 计算span
def compute_multi_spans(gold_spans, pred_spans):
    """
    Compute the F1 score for multiple spans.
    :param gold_spans: A list of ground truth spans, each represented as a (start, end) tuple
    :param pred_spans: A list of predicted spans, each represented as a (start, end) tuple
    :return: The F1 score
    """
    gold_sets = [set(range(int(start), int(end))) for start, end in gold_spans]
    pred_sets = [set(range(int(start), int(end))) for start, end in pred_spans]

    gold_union = set.union(*gold_sets)
    pred_union = set.union(*pred_sets)

    tp = len(gold_union.intersection(pred_union))
    pred_len = len(pred_union)
    gold_len = len(gold_union)

    return tp, pred_len, gold_len


def find_start_end_index(response, output, strict=False):
    """
    Get the (start, end) index for span in the response string
    :param str: LLM response string
    :param sub_str: predicted hallucination spans
    :return: (start, end), where str[start:end] is sub_str
    """
    response = response.lower()
    # if not isinstance(output, str):
    #     sub_str = json.dumps(output, ensure_ascii=False)
    # else:
    sub_str = output
    sub_str = sub_str.lower().strip()
    start = response.find(sub_str)
    if start == -1:
        if len(sub_str) >= 10 and not strict:
            # Allow matching 10 characters at the beginning and end, with a total length ratio between [0.7, 1.3].
            match_length = 10
            prefix_match = response.find(sub_str[:match_length])
            suffix_match = response.find(sub_str[-match_length:])
            new_start, new_end = prefix_match, suffix_match + match_length

            if (
                prefix_match != -1
                and suffix_match != -1
                and 0.7 < (new_end - new_start) / len(sub_str) < 1.3
            ):
                return new_start, new_end
            else:
                sub_str = " ".join(sub_str.split())
                start = response.find(sub_str)
                if start == -1:
                    return -1, -1
    end = start + len(sub_str)
    return (start, end)


def cal_metrics(response, labels, output):
    """
    Get the span level and case level results
    :param sample: LLM response string
    :param sub_str: predicted hallucination spans
    :return: match_size, prediction, groud_truth, pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt
    """
    output_list = output
    # try:
    #     output_list = literal_eval(output)
    # except:
    #     # if fail to parse, we set it as empty
    #     output_list = []

    predict_span = []

    for output in output_list:
        if not isinstance(output, str):
            continue
        tup = find_start_end_index(response, output, strict=False)
        if tup != (-1, -1):
            predict_span.append(tup)

    ground_truth_span = []
    for label in labels:
        tup = (label["start"], label["end"])
        ground_truth_span.append(tup)

    if len(predict_span) == 0:
        predict_span.append((0, 0))
    if len(ground_truth_span) == 0:
        ground_truth_span.append((0, 0))

    match_size, prediction, groud_truth = compute_multi_spans(
        ground_truth_span, predict_span
    )

    # case_level
    pos_true_cnt, neg_true_cnt, neg_false_cnt, pos_false_cnt = 0, 0, 0, 0

    if output_list != [] and labels != []:
        pos_true_cnt += 1
    elif output_list == [] and labels != []:
        neg_true_cnt += 1
    elif output_list == [] and labels == []:
        neg_false_cnt += 1
    elif output_list != [] and labels == []:
        pos_false_cnt += 1
    else:
        raise Exception("Error! output: {} labels: {}".format(output, labels))
    return (
        match_size,
        prediction,
        groud_truth,
        pos_true_cnt,
        neg_true_cnt,
        neg_false_cnt,
        pos_false_cnt,
    )


def print_metrics(task_cur, record):
    """
    print span level and case level metrics
    :param task_cur: task type
    :param record: record dictionary containing metrics
    :return: structured metrics dictionary
    """
    print("\ntask_type: {}".format(task_cur))

    record["recall" + task_cur] = (
        record["match_size_" + task_cur] / record["groud_truth_" + task_cur]
        if record["groud_truth_" + task_cur]
        else 0
    )
    record["precision" + task_cur] = (
        record["match_size_" + task_cur] / record["prediction_" + task_cur]
        if record["prediction_" + task_cur]
        else 0
    )
    record["f1" + task_cur] = (
        2
        * record["recall" + task_cur]
        * record["precision" + task_cur]
        / (record["recall" + task_cur] + record["precision" + task_cur])
        if (record["recall" + task_cur] + record["precision" + task_cur]) > 0
        else 0
    )
    char_res = "span_level:\nprecision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
        record["precision" + task_cur],
        record["recall" + task_cur],
        record["f1" + task_cur],
    )

    record["case_recall_" + task_cur] = (
        record["pos_true_cnt_" + task_cur]
        / (record["pos_true_cnt_" + task_cur] + record["neg_true_cnt_" + task_cur])
        if (record["pos_true_cnt_" + task_cur] + record["neg_true_cnt_" + task_cur]) > 0
        else 0
    )
    record["case_precision_" + task_cur] = (
        record["pos_true_cnt_" + task_cur]
        / (record["pos_true_cnt_" + task_cur] + record["pos_false_cnt_" + task_cur])
        if (record["pos_true_cnt_" + task_cur] + record["pos_false_cnt_" + task_cur])
        > 0
        else 0
    )
    record["f1_" + task_cur] = (
        2
        * record["case_recall_" + task_cur]
        * record["case_precision_" + task_cur]
        / (record["case_recall_" + task_cur] + record["case_precision_" + task_cur])
        if (record["case_recall_" + task_cur] + record["case_precision_" + task_cur])
        > 0
        else 0
    )
    case_res = "case_level:\nprecision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
        record["case_precision_" + task_cur],
        record["case_recall_" + task_cur],
        record["f1_" + task_cur],
    )

    print(case_res)
    print(char_res)

    # 返回结构化的指标字典
    metrics_dict = {
        "task_type": task_cur,
        "span_level": {
            "precision": round(record["precision" + task_cur], 4),
            "recall": round(record["recall" + task_cur], 4),
            "f1": round(record["f1" + task_cur], 4),
            "match_size": record["match_size_" + task_cur],
            "prediction": record["prediction_" + task_cur],
            "ground_truth": record["groud_truth_" + task_cur],
        },
        "case_level": {
            "precision": round(record["case_precision_" + task_cur], 4),
            "recall": round(record["case_recall_" + task_cur], 4),
            "f1": round(record["f1_" + task_cur], 4),
            "pos_true_cnt": record["pos_true_cnt_" + task_cur],
            "neg_true_cnt": record["neg_true_cnt_" + task_cur],
            "neg_false_cnt": record["neg_false_cnt_" + task_cur],
            "pos_false_cnt": record["pos_false_cnt_" + task_cur],
        },
    }
    return metrics_dict


def main(
    args,
    record,
    chosen_task=["QA", "Data2txt", "Summary"],
):
    chosen_task.append("All")

    for i, sample in enumerate(results):
        response, labels, output, task = (
            sample["response"],
            sample["labels"],
            sample["output"],
            sample["task_type"],
        )

        res = cal_metrics(response, labels, output)
        (
            temp_tp,
            temp_pred_len,
            temp_gold_len,
            pos_true_cnt,
            neg_true_cnt,
            neg_false_cnt,
            pos_false_cnt,
        ) = res

        # calculate for difference tasks
        for task_cur in chosen_task:
            if task_cur == task or task_cur == "All":
                record["match_size_" + task_cur] += temp_tp
                record["prediction_" + task_cur] += temp_pred_len
                record["groud_truth_" + task_cur] += temp_gold_len

                record["pos_true_cnt_" + task_cur] += pos_true_cnt
                record["neg_true_cnt_" + task_cur] += neg_true_cnt
                record["neg_false_cnt_" + task_cur] += neg_false_cnt
                record["pos_false_cnt_" + task_cur] += pos_false_cnt

    # 收集所有任务的指标
    all_metrics = []
    for i, task_cur in enumerate(chosen_task):
        metrics_dict = print_metrics(task_cur, record)
        all_metrics.append(metrics_dict)

    print("chosen task: {}".format(chosen_task))
    print("filepath: {}".format(args.filepath))
    print("test_data_size: {}".format(len(results)))

    return all_metrics


def parse_output(output_str):
    # reasoning
    output = re.findall(r"""{"hallucination list": \[.*?\]}""", output_str)
    if not len(output):
        return []
    output = output[-1]
    output = json_repair.loads(output)
    # output_parsed = []
    # if isinstance(output, list):
    #     for o in output:
    #         if 'span' in o and o.get('label') != 'supported':
    #             output_parsed.append(o['span'])
    # elif isinstance(output, str):
    #     return output
    if "hallucination list" in output:
        return output["hallucination list"]
    elif isinstance(output, list):
        return output
    else:
        print("ERROR: hallucination list not found in output")
        return []


def trans_instruction(ins_ori, task_word):
    if "Your task" not in ins_ori:
        print("error")
    ins_ori = ins_ori.rsplit("Your task", 1)[0]
    ins = ins_ori.strip() + "\n" + extra_sft_prompt.replace("<|task_word|>", task_word)
    return ins


def trans_instruction_reasoning(ins_ori, task_word):
    if "Your task" not in ins_ori:
        print("error")
    ins_ori = ins_ori.rsplit("Your task", 1)[0]
    ins = (
        ins_ori.strip()
        + "\n"
        + extra_sft_prompt_reasoning.replace("<|task_word|>", task_word)
    )
    return ins


extra_sft_prompt_reasoning = """Your task is to determine whether the <|task_word|> contains hallucinations. First, provide reasoning with the following format:
## Step 1: < your first reasoning step >
## Step 2: < your next reasoning step >
...(add as many steps as needed)
Then, compile the labeled hallucinated spans into a JSON dict, with a key hallucination list and its value is a list of hallucinated spans. If there are potential hallucinations, the output should be in the following JSON format: {"hallucination list": [hallucination span1, hallucination span2, ...]}. Otherwise, leave the value as an empty list as follows: {"hallucination list": []}."""

extra_sft_prompt = """Your task is to determine whether the <|task_word|> contains hallucinations. Then, compile the labeled hallucinated spans into a JSON dict, with a key hallucination list and its value is a list of hallucinated spans. If there are potential hallucinations, the output should be in the following JSON format: {'hallucination list': [hallucination span1, hallucination span2, ...]}. Otherwise, leave the value as an empty list as follows: {'hallucination list': []}."""

# extra_sft_prompt_reasoning = """Your task is to determine whether the summary contains hallucinations. First, provide reasoning with the following format:
# ## Step 1: < your first reasoning step >
# ## Step 2: < your next reasoning step >
# ...(add as many steps as needed)
# Then, compile the labeled hallucinated spans into a JSON dict, with a key hallucination list and its value is a list of hallucinated spans. If there are potential hallucinations, the output should be in the following JSON format: {"hallucination list": [hallucination span1, hallucination span2, ...]}. Otherwise, leave the value as an empty list as follows: {"hallucination list": []}."""

# extra_sft_prompt = """Your task is to determine whether the summary contains hallucinations. Then, compile the labeled hallucinated spans into a JSON dict, with a key hallucination list and its value is a list of hallucinated spans. If there are potential hallucinations, the output should be in the following JSON format: {'hallucination list': [hallucination span1, hallucination span2, ...]}. Otherwise, leave the value as an empty list as follows: {'hallucination list': []}."""


def process_single_item(d, reasoning=False):
    """处理单个数据项：转换指令、请求模型、解析输出"""
    task_type = d["task_type"]
    task_word = TASK_TYPE_TO_INSTRUCTION[task_type]
    try:
        if reasoning:
            instruction = trans_instruction_reasoning(d["input_str"], task_word)
        else:
            instruction = trans_instruction(d["input_str"], task_word)
        t_req0 = time.perf_counter()
        output = req_model_vllm(instruction)
        request_latency_sec = time.perf_counter() - t_req0
        output_parsed = parse_output(output)
        res = {
            "id": d["id"],
            "response": d["output"],
            "labels": json.loads(d["hallucination_labels"]),
            "output": output_parsed,
            "output_raw": output,
            "task_type": d["task_type"],
            "request_latency_sec": round(request_latency_sec, 6),
        }
        return res
    except Exception as e:
        print(f"Error processing item {d.get('id', 'unknown')}: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filepath", type=str, default="")
    parser.add_argument("--chosen_task", type=str, default="QA,Data2txt,Summary")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=32,
        help="Number of threads for parallel processing",
    )

    # 先解析参数以获取 num_threads
    args = parser.parse_args()

    reasoning = True
    inference_latency_meta = None
    if not len(args.filepath):
        ### INFERENCE
        dp = "/data/dongjiawei/data/seek/hallu_validator/data_fact_checking/ragtruth/ragtruth_processed/test.parquet"
        data = pd.read_parquet(dp)
        data_fmt = data.to_dict("records")
        results = []

        # 使用多线程并行处理
        total_items = len(data_fmt)
        parallel_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_item, d, reasoning): d for d in data_fmt
            }

            # 收集结果，显示进度条
            with tqdm(total=total_items, desc="Processing items") as pbar:
                for future in as_completed(future_to_item):
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    pbar.update(1)
        parallel_wall_sec = time.perf_counter() - parallel_t0
        latency_wall_share_sec = parallel_wall_sec / total_items if total_items else 0.0

        # 按原始顺序排序（如果需要保持顺序）
        results.sort(key=lambda x: x["id"])
        inference_latency_meta = {
            "parallel_wall_sec": round(parallel_wall_sec, 6),
            "latency_wall_share_sec_per_sample": round(latency_wall_share_sec, 6),
            "num_threads": args.num_threads,
        }
        share = inference_latency_meta["latency_wall_share_sec_per_sample"]
        for res in results:
            res["latency_wall_share_sec"] = share
        _date = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("test_result", exist_ok=True)
        output_path = f"test_result/{MODEL_NAME}_ragtruth_test_{_date}_{args.num_threads}_threads.jsonl"

        with open(output_path, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")

    ## COMPUTE METRICS
    args.chosen_task = args.chosen_task.split(",")
    if not len(args.filepath):
        args.filepath = output_path
    else:
        results = []
        with open(args.filepath, "r") as f:
            for line in f:
                results.append(json.loads(line))

    output_path_metric = args.filepath.replace(".jsonl", "_metric.json")
    assert output_path_metric != args.filepath

    print("loading data from: {}".format(args.filepath.split("/")[-1]))

    record = collections.defaultdict(int)

    all_metrics = main(
        args,
        record,
        chosen_task=args.chosen_task,
    )

    latencies = [
        r["request_latency_sec"]
        for r in results
        if isinstance(r.get("request_latency_sec"), (int, float))
    ]
    latency_block = {}
    if latencies:
        latency_block = {
            "request_latency_sec_mean": round(sum(latencies) / len(latencies), 6),
            "request_latency_sec_min": round(min(latencies), 6),
            "request_latency_sec_max": round(max(latencies), 6),
            "request_latency_sec_sum": round(sum(latencies), 6),
        }
        if inference_latency_meta is not None:
            latency_block["parallel_wall_sec"] = inference_latency_meta[
                "parallel_wall_sec"
            ]
            latency_block["latency_wall_share_sec_per_sample"] = inference_latency_meta[
                "latency_wall_share_sec_per_sample"
            ]
            latency_block["num_threads"] = inference_latency_meta["num_threads"]
        else:
            r0 = results[0] if results else {}
            if r0.get("latency_wall_share_sec") is not None:
                latency_block["latency_wall_share_sec_per_sample"] = r0[
                    "latency_wall_share_sec"
                ]
                latency_block["parallel_wall_sec"] = round(
                    r0["latency_wall_share_sec"] * len(results), 6
                )

    # 保存指标到JSON文件
    metrics_output = {
        "filepath": args.filepath,
        "chosen_task": args.chosen_task,
        "test_data_size": len(results),
        "model_name": MODEL_NAME,
        "metrics": all_metrics,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    if latency_block:
        metrics_output["latency"] = latency_block
    result_qa = [x for x in all_metrics if x["task_type"] == "QA"][0]
    result_data2txt = [x for x in all_metrics if x["task_type"] == "Data2txt"][0]
    result_summary = [x for x in all_metrics if x["task_type"] == "Summary"][0]
    result_all = [x for x in all_metrics if x["task_type"] == "All"][0]

    excel_cp_str = []
    for task_result in [result_summary, result_qa, result_data2txt, result_all]:
        for k in ["f1", "precision", "recall"]:
            excel_cp_str.append(str(round(task_result["span_level"][k] * 100, 2)))
    excel_cp_str = ",".join(excel_cp_str)
    print(excel_cp_str)

    with open(output_path_metric, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)

    print(f"\nMetrics saved to: {output_path_metric}")


# print(data.head())
