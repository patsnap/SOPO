## start: 2026-03-25 13:27
## end : 2026-03-26 11:32

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import glob
import json
import os
import re

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# START_TIME = datetime.strptime("2026-03-25 13:27", "%Y-%m-%d %H:%M")
# END_TIME = datetime.strptime("2026-03-26 11:32", "%Y-%m-%d %H:%M")

# START_TIME = datetime.strptime("2026-01-01 00:00", "%Y-%m-%d %H:%M")
# END_TIME = datetime.strptime("2026-01-03 00:00", "%Y-%m-%d %H:%M")

START_TIME = datetime.strptime("2026-04-02 10:29", "%Y-%m-%d %H:%M")
END_TIME = datetime.strptime("2026-04-03 05:20", "%Y-%m-%d %H:%M")

TOKENIZER_PATH = "/mnt/cfs-rd/models/qwen3/Qwen3-4B-Thinking-2507/"

TIME_DIR_PATTERN = re.compile(r"^\d{4}(?:_\d{2}){3,5}$")
TIME_DIR_FORMATS = {
    4: "%Y_%m_%d_%H",
    5: "%Y_%m_%d_%H_%M",
    6: "%Y_%m_%d_%H_%M_%S",
}


def parse_time_dir(dirname: str) -> datetime | None:
    if not TIME_DIR_PATTERN.fullmatch(dirname):
        return None
    n_parts = dirname.count("_") + 1
    fmt = TIME_DIR_FORMATS.get(n_parts)
    if fmt is None:
        return None
    try:
        return datetime.strptime(dirname, fmt)
    except ValueError:
        return None


def path_in_range(fp: str) -> bool:
    for part in fp.split(os.sep):
        dt = parse_time_dir(part)
        if dt is None:
            continue
        if START_TIME <= dt <= END_TIME:
            return True
    return False


# files = glob.glob("/mnt/cfs-rd/dongjiawei/projects/rl/AReaL_20251111_040/havor_debug_rebuttal/havor_debug/2026_01_0*/**/*.json", recursive=True)
files = glob.glob("/mnt/cfs-rd/dongjiawei/projects/rl/AReaL_20251111_040/havor_debug/2026_04_0*/**/*.json", recursive=True)
# files = glob.glob(
#     "/mnt/cfs-rd/dongjiawei/projects/rl/AReaL_20251111_040/havor_debug/2026_03_*/**/*.json",
#     recursive=True,
# )
files = [fp for fp in files if path_in_range(fp)]

print(len(files))

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


def count_tokens(text: str) -> int:
    if not isinstance(text, str):
        text = str(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_json_with_fallback(fp: str) -> tuple[dict | None, bool]:
    with open(fp, "r") as f:
        text = f.read()

    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        # Some files may contain a valid first JSON object followed by extra data.
        # In that case, parse the first object and ignore the trailing part.
        try:
            obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
            return obj, True
        except json.JSONDecodeError:
            return None, False


def process_file(fp: str) -> tuple[list[int], int | None, bool, bool]:
    d, used_fallback = load_json_with_fallback(fp)
    if d is None:
        return [], None, True, False

    if "result" not in d or "messages" not in d["result"]:
        return [], None, True, used_fallback

    msgs = d["result"]["messages"]
    msgs_users = [msg for msg in msgs if msg["role"] == "user"]
    if not msgs_users:
        return [], None, False, used_fallback

    msgs_teacher = msgs_users[1:] if len(msgs_users) > 1 else []
    teacher_tokens = [count_tokens(msg["content"]) for msg in msgs_teacher]
    user_tokens = count_tokens(msgs_users[0]["content"])
    return teacher_tokens, user_tokens, False, used_fallback


n_token_teacher = []
n_token_user = []
skipped_files = 0
fallback_files = 0

max_workers = min(32, (os.cpu_count() or 1) * 4)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for teacher_tokens, user_tokens, is_skipped, used_fallback in tqdm(
        executor.map(process_file, files), total=len(files)
    ):
        if used_fallback:
            fallback_files += 1
        if is_skipped:
            skipped_files += 1
            continue

        n_token_teacher.extend(teacher_tokens)
        if user_tokens is not None:
            n_token_user.append(user_tokens)

print(np.mean(n_token_teacher))
print(np.mean(n_token_user))
print(f"fallback parsed files: {fallback_files}")
print(f"skipped invalid files: {skipped_files}")