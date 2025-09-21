import argparse
import ast
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import uuid
from functools import partial
from logging.handlers import RotatingFileHandler

from tqdm import tqdm
import tiktoken
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_operation import load_input_file
from utils.oai import count_tokens, get_model_prediction, load_openai_client

tokenizer = None


def init_tokenizer(tokenizer_id):
    global tokenizer
    if not tokenizer_id or tokenizer_id.lower() == "auto":
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_id)
        return
    except Exception:
        pass
    try:
        tokenizer = tiktoken.encoding_for_model(tokenizer_id)
        return
    except Exception:
        pass
    print(
        f"[tokenizer] Unable to resolve '{tokenizer_id}', falling back to cl100k_base."
    )
    tokenizer = tiktoken.get_encoding("cl100k_base")



def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_file, maxBytes=2000000, backupCount=10)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def remove_line_numbers(content):
    # Remove line numbers from the file content
    return re.sub(r"^\d+\s", "", content, flags=re.MULTILINE)


def load_file_content(file_path, repo_structure, decoding="utf-8"):
    path_parts = file_path.split("/")
    if len(path_parts) == 1:
        path_parts.insert(0, "")
    file_content = repo_structure
    for part in path_parts:
        if part in file_content:
            file_content = file_content[part]
        else:
            return ""
    if isinstance(file_content, dict) and "text" in file_content:
        text_lines = [
            line.encode("ISO-8859-1").decode(decoding) for line in file_content["text"]
        ]
        return "\n".join(text_lines)
    return ""


def remove_empty_lines(code):
    lines = code.splitlines()
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)


def check_syntax(code):
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def check_code_differ_by_just_empty_lines(code, prev_code):
    normalized_code1 = remove_empty_lines(code)
    normalized_code2 = remove_empty_lines(prev_code)
    return normalized_code1 == normalized_code2


def lint_code(repo_playground, temp_name, code, prev_code=""):
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"
    os.makedirs(repo_playground)

    with open(f"{repo_playground}/{temp_name}", "w") as f:
        f.write(prev_code)

    fatal = "E9,F821,F823,F831,F406,F407,F701,F702,F704,F706"
    o = subprocess.run(
        f"flake8 --select={fatal} --isolated {repo_playground}/{temp_name}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    prev_errors = set()
    if s != "":
        for error in s.split(f"{repo_playground}/{temp_name}:")[1:]:
            num_free_error = ":".join(error.split(":")[2:]).strip()
            prev_errors.add(num_free_error)

    with open(f"{repo_playground}/{temp_name}", "w") as f:
        f.write(code)

    o = subprocess.run(
        f"flake8 --select={fatal} --isolated {repo_playground}/{temp_name}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    errors = set()
    if s != "":
        for error in s.split(f"{repo_playground}/{temp_name}:")[1:]:
            num_free_error = ":".join(error.split(":")[2:]).strip()
            errors.add(num_free_error)

    if len(errors - prev_errors) > 0:
        return False, prev_errors, errors

    return True, set(), set()


def fake_git_repo(repo_playground, file_path, old_content, new_content):
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"
    os.makedirs(repo_playground)

    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    subprocess.run(
        f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
    )

    with open(f"{repo_playground}/{file_path}", "w") as f:
        f.write(old_content)

    subprocess.run(
        f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
        shell=True,
    )

    with open(f"{repo_playground}/{file_path}", "w") as f:
        f.write(new_content)

    o = subprocess.run(
        f"cd {repo_playground} && git diff {file_path}", shell=True, capture_output=True
    )

    s = o.stdout.decode("utf-8")

    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s


def evaluate_task_code_editing(json_output, repo_structure_path, instance_id):
    try:
        output = json.loads(json_output)
        edited_code = output["edited code"]
    except Exception as e:
        # logger.error(f"Error in parsing json output for task code editing: {e}")
        print(f"Error in parsing json output for task code editing: {e}")
        return "", ""
    try:
        repo_structure = load_input_file(
            os.path.join(repo_structure_path, instance_id + ".json")
        )["structure"]
        git_diffs = ""
        raw_git_diffs = ""
        lint_success = False

        for file in edited_code:
            # file_path = file["file path"]
            file_path = file["file"]
            code_snippet_to_be_modified = file["code snippet to be modified"]
            edited_code_snippet = file["edited code snippet"]

            code_snippet_to_be_modified = remove_line_numbers(
                code_snippet_to_be_modified
            ).rstrip()
            file_content = load_file_content(file_path, repo_structure)

            if (
                code_snippet_to_be_modified
                and code_snippet_to_be_modified in file_content
            ) or file_content == "":
                if file_content:
                    new_content = file_content.replace(
                        code_snippet_to_be_modified, edited_code_snippet
                    )
                else:  # new file
                    new_content = edited_code_snippet

                git_diff = fake_git_repo(
                    "playground", file_path, file_content, new_content
                )
                git_diff = "\n" + git_diff.replace("\ No newline at end of file\n", "")

                syntax_success = check_syntax(new_content)
                lint_success, prev_errors, errors = lint_code(
                    "playground", "test.py", new_content, file_content
                )

                differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                    new_content, file_content
                )

                print(lint_success, prev_errors, errors, differ_by_empty_lines)

                if syntax_success and not differ_by_empty_lines:
                    git_diffs += git_diff
                else:
                    git_diffs += ""  # no need to evaluate
                raw_git_diffs += git_diff
            else:
                raw_git_diffs += ""  # no need to evaluate
                git_diffs += ""  # no need to evaluate

        return git_diffs, raw_git_diffs
    except Exception as e:

        print(f"Error in evaluating task code editing for instance {instance_id}: {e}")
        return "", ""


def process_api_inference(
    api_key,
    base_url,
    input_file,
    output_file,
    repo_structure_path,
    max_tokens,
    name,
    if_with_reason,
):
    dataset = load_input_file(input_file)
    global tokenizer

    if os.path.exists(output_file):
        processed_instance_ids = set()
        with open(output_file, "r") as f:
            for line in f:
                result = json.loads(line)
                processed_instance_ids.add(result["instance_id"])
    else:
        processed_results = []
        processed_instance_ids = set()

    results = []
    client = load_openai_client(api_key, base_url)
    over_limit_count = 0

    for entry in tqdm(dataset, desc="Processing API inference"):
        instance_id = entry["instance_id"]
        if instance_id in processed_instance_ids:
            continue

        user_input = [entry["messages"][0]]
        if if_with_reason:
            json_input = json.loads(user_input[0]["content"])
            print("using string reason")
            json_input["output control"] = {
                "type": "object",
                "properties": {
                    "reasoning process": {"type": "string"},
                    "edited code": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "code snippet to be modified": {"type": "string"},
                                "edited code snippet": {"type": "string"},
                            },
                        },
                    },
                },
            }
            json_input["input"][
                "task"
            ] = "In this task, you will be provided with a software development issue from a real-world GitHub repository, along with the full content of relevant code files for modification. Your objective is to carefully analyze and understand the issue in the context of the provided files, explain your reasoning process for addressing it, and identify the exact file paths and original code snippets that require modification. Based on this analysis, you will propose new code snippets to replace the identified ones to effectively resolve the issue."
            user_input[0]["content"] = json.dumps(json_input)

        if count_tokens(tokenizer, user_input[0]["content"]) > max_tokens:
            git_diffs = ""
            raw_git_diffs = ""
            over_limit_count += 1
            result = {
                "instance_id": instance_id,
                "user_input": user_input,
                "issue": json.loads(user_input[0]["content"])["input"]["issue"],
                "expected_output": entry["messages"][-1]["content"],
                "prediction": None,
                "git_diffs": git_diffs,
                "model_patch": raw_git_diffs,
                "model_name_or_path": name,
            }
        else:
            prediction = get_model_prediction(client, user_input)
            git_diffs, raw_git_diffs = evaluate_task_code_editing(
                prediction, repo_structure_path, instance_id
            )
            print("prediction", prediction)
            result = {
                "instance_id": instance_id,
                "user_input": user_input,
                "issue": json.loads(user_input[0]["content"])["input"]["issue"],
                "expected_output": entry["messages"][-1]["content"],
                "prediction": prediction,
                "git_diffs": git_diffs,
                "model_patch": raw_git_diffs,
                "model_name_or_path": name,
            }

        results.append(result)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Number of instances over token limit: {over_limit_count}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate task code editing.")
    parser.add_argument(
        "--api_key",
        type=str,
        default="token-abc123",
        help="API key for OpenAI.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI API.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/code_edit/SWE-Fixer-res.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--repo_structure_path",
        type=str,
        default="./swe_bench_test_code_structure",
        help="Path to the repository structure file.",
    )
    parser.add_argument("--max_tokens", default=32768*2, type=int, help="max length")
    parser.add_argument(
        "--name",
        type=str,
        default="SWE-Fixer",
        help="Log file path.",
    )
    parser.add_argument(
        "--if_with_reason",
        type=bool,
        default=True,
        help="If with reason.",
    )
    args = parser.parse_args()
    init_tokenizer(args.tokenizer_path)
    # Set up logging
    global logger
    logger = setup_logger(args.log_file)

    process_api_inference(**vars(args))


if __name__ == "__main__":
    main()
