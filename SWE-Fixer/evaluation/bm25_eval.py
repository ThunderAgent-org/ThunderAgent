import argparse
import json
import os

from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_operation import load_input_file, save_results
from utils.oai import get_model_prediction, load_openai_client


def evaluate_task_file_level_retrieval(json_output, gold_files):
    retrieved_files = []
    try:
        output = json.loads(json_output)
        retrieved_files = output["files for editing"]
    except json.JSONDecodeError:
        print("Error in parsing json output for task file level retrieval!")

    gold_files = gold_files["files for editing"]

    precision, recall = 0, 0
    if retrieved_files:
        precision = len(set(retrieved_files).intersection(gold_files)) / len(
            retrieved_files
        )
        recall = len(set(retrieved_files).intersection(gold_files)) / len(gold_files)

    return precision, recall, retrieved_files


def add_suffix_to_filename(file_path, suffix):
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}-{suffix}{ext}"
    new_file_path = os.path.join(directory, new_filename)
    return new_file_path


def main(
    api_key, base_url, input_file, output_file
):

    client = load_openai_client(api_key, base_url)
    dataset = load_input_file(input_file)
    invalid_json_count = 0

    output_data = {
        "results": [],
        "invalid_json_count": invalid_json_count,
    }

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

        prediction = get_model_prediction(client, user_input)
        precision, recall, _ = evaluate_task_file_level_retrieval(
            prediction, expected_output
        )

        output_data["results"].append(
            {
                # "user_input": json.loads(user_input),\
                "instance_id": entry["instance_id"],
                "issue": json.loads(user_input)["input"]["issue"],
                "expected_output": expected_output,
                "prediction": prediction,
                "precision": precision,
                "recall": recall,
            }
        )

        save_results(output_file, output_data)

    print(f"Invalid JSON count: {invalid_json_count}")
    precision_for_whole_dataset = sum(
        result["precision"] for result in output_data["results"]
    ) / len(dataset)
    recall_for_whole_dataset = sum(
        result["recall"] for result in output_data["results"]
    ) / len(dataset)
    print(f"Overall Precision: {precision_for_whole_dataset}")
    print(f"Overall Recall: {recall_for_whole_dataset}")
    output_data["invalid_json_count"] = invalid_json_count
    output_data["Overall_Precision"] = precision_for_whole_dataset
    output_data["Overall_Recall"] = recall_for_whole_dataset
    output_data["length_of_results"] = len(dataset)
    save_results(output_file, output_data)

    print(
        "Inputs, outputs, predictions, precision, and recall have been saved to results.json."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate task file level retrieval.")
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
        "--input_file",
        type=str,
        default="",
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/retrieval.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--if_gold", type=str, default="False", help="Output number setting."
    )
    args = parser.parse_args()
    main(
        args.api_key,
        args.base_url,
        args.input_file,
        args.output_file,
        args.if_gold,
    )
