import re
import json
import traceback

def parse_result(content: str) -> bool:
    """验证输出格式"""
    try:
        think_token_count = content.count("</think>")
        assert think_token_count == 1, 'THINK TOKEN "</think>" COUNT NOT EQUAL TO 1'
        output_str = content.rsplit("</think>", 1)[-1]
        output_str = output_str.replace("<|im_end|>", "").strip()

        ## 抽取"{'hallucination list': []}"类似的内容，只抽取最后一组
        output_str = re.findall(r"""{"hallucination list": \[.*?\]}""", output_str)
        assert len(output_str) == 1, f"NUMBER OF HALLUCINATION LIST NOT EQUAL TO 1"
        output_str = output_str[0]

        # 使用Pydantic验证
        output = json.loads(output_str, strict=False)
        assert isinstance(output, dict), f"OUTPUT IS NOT A DICT"
        assert "hallucination list" in output, f"HALLUCINATION LIST NOT FOUND"
        assert isinstance(
            output["hallucination list"], list
        ), f"HALLUCINATION LIST IS NOT A LIST"
        return "SUCCESS", output["hallucination list"]

    except json.JSONDecodeError as e:
        return f"ERROR: JSONDecodeError", None
    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {str(e)}", None


def print_metrics(task_cur, record):
        """
        print span level and case level metrics
        :param task_cur: task type
        :param record: record dictionary containing metrics
        :return: structured metrics dictionary
        """
        # print("\ntask_type: {}".format(task_cur))

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

        # print(case_res)
        # print(char_res)
        
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
            }
        }
        return metrics_dict


def compute_metrics(
        results,
        record,
        chosen_task=["QA", "Data2txt", "Summary", "All"],
    ):
        # chosen_task.append("All")

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
        f1_overall = 0.0
        for i, task_cur in enumerate(chosen_task):
            metrics_dict = print_metrics(task_cur, record)
            all_metrics.append(metrics_dict)
            if task_cur == "All":
                f1_overall = metrics_dict["span_level"]["f1"]
        # print("chosen task: {}".format(chosen_task))
        # print("filepath: {}".format(args.filepath))
        # print("test_data_size: {}".format(len(results)))

        return f1_overall, all_metrics


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

