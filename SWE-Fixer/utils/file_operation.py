import os
import json
from argparse import ArgumentTypeError
from pathlib import Path


def load_input_file(instance_file):
    if os.path.isdir(instance_file):
        file_list = []
        for file in os.listdir(instance_file):
            if file.endswith(".jsonl"):
                file_list.append(os.path.join(instance_file, file))
        instances = []
        for file in file_list:
            with open(file, "r", encoding="utf-8") as file:
                instances.extend([json.loads(line) for line in file])
    elif os.path.isfile(instance_file):
        if instance_file.endswith(".jsonl"):
            with open(instance_file, "r", encoding="utf-8") as file:
                instances = [json.loads(line) for line in file]
        elif instance_file.endswith(".json"):
            with open(instance_file, "r", encoding="utf-8") as file:
                instances = json.load(file)
        else:
            raise ValueError("Invalid input file format. Only .jsonl and .json files are supported.")
    return instances

def save_results(output_file, output_data):
    if output_file.endswith(".jsonl"):
        if os.path.exists(output_file):
            os.remove(output_file)
        for item in output_data:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)