import argparse
import copy
import json
import os
from functools import partial
from multiprocessing import Pool

from pipeline.prompt import (
    BM25_RETRIEVAL_ORACLE_FILE_OUTPUT_CONTROL_NO_REASONING,
    BM25_RETRIEVAL_TASK_NO_REASONING,
    EDITING_LEVEL_OUTPUT_CONTROL_WITH_REASONING,
    EDITING_LEVEL_TASK_ONLY_FILE_CONTENT_WITH_REASONING,
    EDITING_SYS_PROMPT,
)
from tqdm import tqdm
import tiktoken

from evaluation.bm25_eval import evaluate_task_file_level_retrieval
from evaluation.code_edit import evaluate_task_code_editing
from utils.code_structure_operation import join_lines_with_numbers
from utils.file_operation import load_input_file, save_results
from utils.oai import count_tokens, get_model_prediction, load_openai_client

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_TOKENS = 32768 * 2
RETRY_TEMP = 0.7

tokenizer = None


def init_tokenizer(tokenizer_id):
    """Initialise the global tokenizer/encoder using tiktoken.

    The identifier can be a tiktoken encoding name (e.g. ``cl100k_base``)
    or a known model name supported by ``encoding_for_model``. If resolution
    fails we fall back to ``cl100k_base`` so that downstream logic still has
    a best-effort estimator.
    """

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


def validate_json(func, *args, max_attempts=5, force_temp=False, **kwargs):
    attempts = 0
    while attempts < max_attempts:
        try:
            if attempts == 0 and not force_temp:
                kwargs["temp"] = 0
            result = func(*args, **kwargs)
            json.loads(result)
            return result
        except json.JSONDecodeError:
            print(
                f"Invalid JSON format, attempt {attempts + 1}/{max_attempts}. Regenerating..."
            )
            attempts += 1

    raise ValueError("Exceeded maximum attempts to generate valid JSON.")


def gen_editing_data(
    instance_id_class_retrieval,
    code_structure_path,
    instance_file,
    file_retrieval_res,
):
    instance_id, class_retrieval = instance_id_class_retrieval
    try:
        code_structure = load_input_file(
            os.path.join(code_structure_path, instance_id + ".json")
        )["structure"]
        issue = instance_file[instance_id]["problem_statement"]
        all_file_path = file_retrieval_res[instance_id]
        file_content_with_line_number = {}
        file_structure_dict = {}
        files_to_be_modified = []
        for file_path in all_file_path:
            file_structure = copy.deepcopy(code_structure)
            path = file_path.split("/")
            if len(path) == 1:
                path.insert(0, "")
            for subpath in path:
                if subpath in file_structure:
                    file_structure = file_structure[subpath]
                else:
                    print("Error in file_structure of %s" % subpath)
                    continue
            source_code = file_structure.get("text", [])
            file_content = "\n".join(source_code)
            file_structure_dict[file_path] = file_structure
            file_content_with_line_number[file_path] = join_lines_with_numbers(
                source_code
            )
            files_to_be_modified.append(
                {
                    "file": file_path,
                    "file content": file_content_with_line_number[file_path],
                }
            )

        input_dict = {
            "input": {
                "issue": issue,
                "files to be modified": files_to_be_modified,
                "task": EDITING_LEVEL_TASK_ONLY_FILE_CONTENT_WITH_REASONING,
            },
        }
        input_dict["output control"] = EDITING_LEVEL_OUTPUT_CONTROL_WITH_REASONING
        return {"instance_id": instance_id, "input": input_dict}
    except Exception as e:
        print(f"Error processing instance {instance_id}: {e}")
        return None


def retrieve_files(
    client,
    input_file,
    save_dataset_file,
    output_file,
    post_process=False,
):
    global tokenizer
    if post_process:
        print("Running Post-Process for bm25 result")
    dataset = load_input_file(input_file)
    invalid_json_count = 0
    output_data = {
        "results": [],
        "invalid_json_count": invalid_json_count,
    }
    data_length = len(dataset)
    save_dataset = []
    return_data = {}
    for entry in tqdm(dataset, total=len(dataset)):
        user_input = ""
        expected_output = {}
        for message in entry["messages"]:
            if message["role"] == "user":
                user_input = message["content"]
            elif message["role"] == "assistant":
                try:
                    expected_output = json.loads(message["content"])
                except json.JSONDecodeError:
                    expected_output = {}
                    invalid_json_count += 1

        user_input = json.loads(user_input)
        user_input["input"]["task"] = BM25_RETRIEVAL_TASK_NO_REASONING
        user_input["output control"] = (
            BM25_RETRIEVAL_ORACLE_FILE_OUTPUT_CONTROL_NO_REASONING
        )
        if post_process:
            while count_tokens(tokenizer, json.dumps(user_input)) > MAX_TOKENS:
                print(
                    "Exceeding token limit, removing last file for instance {}".format(
                        entry["instance_id"]
                    )
                )
                del user_input["input"]["retrieved files documentation"][-1]
            del user_input["input"]["retrieved files documentation"][-1]
        save_dataset.append({"input": user_input, "instance_id": entry["instance_id"]})
        user_input = json.dumps(user_input)
        if not post_process:
            prediction = get_model_prediction(client, user_input)
        else:
            try:
                prediction = validate_json(
                    get_model_prediction,
                    client,
                    user_input,
                    max_attempts=5,
                    temp=RETRY_TEMP,
                )
            except Exception as e:
                print(f"Error in post-processing {entry['instance_id']}: {e}")
                prediction = "error processing"
        precision, recall, retrieval_files = evaluate_task_file_level_retrieval(
            prediction, expected_output
        )

        output_data["results"].append(
            {
                "expected_output": expected_output,
                "prediction": prediction,
                "precision": precision,
                "recall": recall,
                "instance_id": entry["instance_id"],
                "retrieval_files": retrieval_files,
            }
        )

        save_results(output_file, output_data)
        save_results(save_dataset_file, save_dataset)
        return_data[entry["instance_id"]] = retrieval_files

    the_whole_precision = (
        sum(result["precision"] for result in output_data["results"]) / data_length
    )
    the_whole_recall = (
        sum(result["recall"] for result in output_data["results"]) / data_length
    )
    print(f"Overall Precision: {the_whole_precision}")
    print(f"Overall Recall: {the_whole_recall}")
    print(f"Invalid JSON count: {invalid_json_count}")

    output_data["Overall_Precision"] = the_whole_precision
    output_data["Overall_Recall"] = the_whole_recall
    output_data["invalid_json_count"] = invalid_json_count
    output_data["length_of_results"] = len(output_data["results"])
    save_results(output_file, output_data)

    print(
        "Inputs, outputs, predictions, precision, and recall have been saved to results.json."
    )
    return return_data


def trim_file_content_to_fit(
    entry, max_tokens, buffer_tokens=4500, line_estimation_threshold=10
):
    new_entry = copy.deepcopy(entry)
    file_content_key = "file content"
    files_to_modify = new_entry["input"]["input"]["files to be modified"]

    line_token_estimates = {}
    line_estimation_threshold = int(
        count_tokens(tokenizer,files_to_modify[0]["file content"])
        / len(files_to_modify[0]["file content"].splitlines())
    )

    while count_tokens(tokenizer, json.dumps(new_entry["input"])) > (max_tokens - buffer_tokens):
        changes_made = False  
        current_tokens = count_tokens(tokenizer,json.dumps(new_entry["input"]))
        excess_tokens = current_tokens - (max_tokens - buffer_tokens)

        for file_data in new_entry["input"]["input"]["files to be modified"]:
            lines = file_data[file_content_key].splitlines()
            if lines:
                lines_to_remove = max(
                    1, int(excess_tokens / line_estimation_threshold / 2)
                )

                if len(lines) > lines_to_remove:
                    lines = lines[lines_to_remove:]
                    changes_made = True
                elif len(lines) > 1:
                    lines = lines[:-lines_to_remove]
                    changes_made = True
                else:
                    lines = []

                file_data[file_content_key] = "\n".join(lines)

        if not changes_made:
            break

    return new_entry


def code_edit(
    client,
    instance_file,
    code_structure_path,
    bm25_res_path,
    dataset_save_path,
    output_file,
    model_name,
    post_process=False,
):
    input_data_record = []
    if post_process:
        print("Running post-processing for code editing")
    file_retrieval_res = {}
    retrieval_files = load_input_file(bm25_res_path)["results"]
    for item in retrieval_files:
        try:
            prediction = json.loads(item["prediction"])

            file_retrieval_res[item["instance_id"]] = prediction["files for editing"]
        except:
            print(item["instance_id"])

    with Pool() as pool:
        process_instance_with_params = partial(
            gen_editing_data,
            code_structure_path=code_structure_path,
            instance_file=instance_file,
            file_retrieval_res=file_retrieval_res,
        )
        results = list(
            tqdm(
                pool.imap(process_instance_with_params, file_retrieval_res.items()),
                total=len(file_retrieval_res),
                desc="Processing instances",
            )
        )
        input_data_record = [result for result in results if result is not None]
    print("gen input data record for editing")
    save_results(dataset_save_path, input_data_record)
    input_data_record = load_input_file(dataset_save_path)
    print(f"Save Editing Input File to {dataset_save_path}")
    processed_instance_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                processed_instance_ids.add(data["instance_id"])
        print(f"Already processed {len(processed_instance_ids)} instances")
    over_limit_count = 0
    results = []
    git_diffs = ""
    raw_git_diffs = ""
    prediction = "No prediction"
    for entry in tqdm(input_data_record, desc="Processing API inference"):
        instance_id = entry["instance_id"]
        input_text = json.dumps(entry["input"])
        if instance_id in processed_instance_ids:
            continue

        if count_tokens(tokenizer, input_text) > MAX_TOKENS and not post_process:
            git_diffs = ""
            raw_git_diffs = ""
            over_limit_count += 1
            result = {
                "instance_id": instance_id,
                "prediction": None,
                "git_diffs": git_diffs,
                "model_patch": raw_git_diffs,
                "model_name_or_path": model_name,
            }
        else:
            attempt = 0
            valid_prediction = False
            if not post_process:
                prediction = get_model_prediction(
                    client, input_text, EDITING_SYS_PROMPT
                )
                git_diffs, raw_git_diffs = evaluate_task_code_editing(
                    json_output=prediction,
                    repo_structure_path=code_structure_path,
                    instance_id=instance_id,
                )
            else:
                if count_tokens(tokenizer, input_text) > MAX_TOKENS:
                    entry = trim_file_content_to_fit(entry, MAX_TOKENS)
                    print("trim file content to fit")
                    input_text = json.dumps(entry["input"])  
                max_attempts = 5
                attempt = 0
                valid_prediction = False
                force_temp = False
                while attempt < max_attempts:
                    if attempt >= 1:
                        print("********************************")
                        print("Force temp: ", RETRY_TEMP)
                        force_temp = True
                    try:
                        
                        prediction = validate_json(
                            get_model_prediction,
                            client,
                            input_text,
                            EDITING_SYS_PROMPT,
                            max_attempts=5,
                            force_temp=force_temp,
                            temp=RETRY_TEMP,
                        )
                    except:
                        print(f"Invalid JSON for instance {instance_id}")
                        prediction = "error processing"
                        break  

                    
                    git_diffs, raw_git_diffs = evaluate_task_code_editing(
                        json_output=prediction,
                        repo_structure_path=code_structure_path,
                        instance_id=instance_id,
                    )

                    
                    if git_diffs == raw_git_diffs and git_diffs != "":
                        valid_prediction = True  
                        break
                    else:
                        print(
                            f"Invalid git diffs for instance {instance_id}. Retrying prediction..."
                        )

                    attempt += 1
            if not post_process:
                result = {
                    "instance_id": instance_id,
                    "prediction": prediction,
                    "git_diffs": git_diffs,
                    "model_patch": raw_git_diffs,
                    "model_name_or_path": model_name,
                }
            else:
                result = {
                    "instance_id": instance_id,
                    "prediction": prediction,
                    "git_diffs": git_diffs,
                    "model_patch": raw_git_diffs,
                    "model_name_or_path": model_name,
                    "valid_prediction": valid_prediction,
                    "attempt": attempt,
                }

        results.append(result)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")


def pipline(
    client,
    model_name,
    instance_file_path,
    code_structure_path,
    bm25_retrieval_files,
    bm25_input_dataset,
    bm25_results_file,
    editing_dataset_path,
    editing_result_path,
    post_process,
    run_retrieval,
    run_editing,
):
    if post_process:
        print("Running post-processing...")
    instance_file_list = load_input_file(instance_file_path)
    instance_file_dict = {}
    for data in instance_file_list:
        instance_file_dict[data["instance_id"]] = data
    if run_retrieval:
        retrieve_files(
            client=client,
            input_file=bm25_retrieval_files,
            save_dataset_file=bm25_input_dataset,
            output_file=bm25_results_file,
            post_process=post_process,
        )
    if run_editing:
        code_edit(
            client=client,
            model_name=model_name,
            instance_file=instance_file_dict,
            code_structure_path=code_structure_path,
            bm25_res_path=bm25_results_file,
            dataset_save_path=editing_dataset_path,
            output_file=editing_result_path,
            post_process=post_process,
        )


def main():
    parser = argparse.ArgumentParser(description="Pipeline for processing instances.")
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
        "--model_name",
        type=str,
        default="SWE-Fixer",
        help="The model name.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--instance_file_path",
        required=True,
        help="Path to the instance file.",
    )
    parser.add_argument(
        "--code_structure_path",
        required=True,
        help="Path to the code structure files.",
    )
    parser.add_argument(
        "--bm25_retrieval_files",
        required=True,
        help="Path to the retrieval training files.",
    )
    parser.add_argument(
        "--bm25_input_dataset",
        required=True,
        help="Path to the retrieval input dataset.",
    )
    parser.add_argument(
        "--bm25_results_file", required=True, help="Path to save BM25 results."
    )
    parser.add_argument("--editing_dataset_path", required=True,help="Path to save editing dataset.")
    parser.add_argument("--editing_result_path", required=True,help="Path to save editing results.")

    parser.add_argument(
        "--post_process",
        action="store_true",
        help="Use post processing for all task.",
    )
    parser.add_argument(
        "--run_retrieval",
        action="store_true",
        help="run retrieval task.",
    )
    parser.add_argument(
        "--run_editing", action="store_true", help="run code editing task."
    )

    args = parser.parse_args()
    init_tokenizer(args.tokenizer_path)
    client = load_openai_client(args.api_key, args.base_url)
    pipline(
        client=client,
        model_name=args.model_name,
        instance_file_path=args.instance_file_path,
        code_structure_path=args.code_structure_path,
        bm25_retrieval_files=args.bm25_retrieval_files,
        bm25_input_dataset=args.bm25_input_dataset,
        bm25_results_file=args.bm25_results_file,
        editing_dataset_path=args.editing_dataset_path,
        editing_result_path=args.editing_result_path,
        post_process=args.post_process,
        run_retrieval=args.run_retrieval,
        run_editing=args.run_editing,
    )


if __name__ == "__main__":
    main()
