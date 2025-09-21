import ast

import unidiff
def construct_hunk_code(hunk_code, add_line_numbers=False):
    code = []
    for line_no, line in hunk_code:
        if add_line_numbers:
            # line already has a newline character
            code.append(f"{line_no} {line}")
        else:
            code.append(line)
    code = "".join(code)

    return code


def join_lines_with_numbers(lines):
    return "\n".join(f"{i + 1} {line}" for i, line in enumerate(lines))


def construct_classes_funcs_content(
    file_structure, classes, funcs, add_line_numbers=False
):
    classes_content = []
    funcs_content = []

    for cls in file_structure["classes"]:
        if cls["name"] in classes:
            start_line = cls["start_line"]
            hunk_code = [
                (start_line + i, line + "\n") for i, line in enumerate(cls["text"])
            ]
            classes_content.append(
                {
                    "name": cls["name"],
                    "content": construct_hunk_code(hunk_code, add_line_numbers),
                }
            )

    for func in file_structure["functions"]:
        if func["name"] in funcs:
            start_line = func["start_line"]
            hunk_code = [
                (start_line + i, line + "\n") for i, line in enumerate(func["text"])
            ]
            funcs_content.append(
                {
                    "name": func["name"],
                    "content": construct_hunk_code(hunk_code, add_line_numbers),
                }
            )

    return {"classes": classes_content, "functions": funcs_content}


def extract_detailed_documentation(
    file_path, file_content
):
    """Extract detailed documentation from a Python file.

    Args:
    - file_path: str, the path to the Python file
    - file_content: str, the content of the Python file

    Returns:
    - file_documentation: dict, a dictionary containing the extracted documentation
    """
    file_documentation = {
        "file_path": file_path,
        "module_docstring": None,
        "classes": [],
        "functions": [],
    }

    def get_signature(node):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            return f"{node.name}({', '.join(args)})"
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Handle cases where base class is specified with a module, e.g., module.BaseClass
                    base_name = []
                    while isinstance(base, ast.Attribute):
                        base_name.insert(0, base.attr)
                        base = base.value
                    if isinstance(base, ast.Name):
                        base_name.insert(0, base.id)
                    bases.append(".".join(base_name))
            return f"{node.name}({', '.join(bases)})"
        return node.name

    def get_code_snippet(node, source_code, num_lines=5):
        """Extract a code snippet for the given node."""
        start_line = node.lineno - 1
        end_line = node.end_lineno
        # print(start_line, end_line)
        if end_line - start_line + 1 <= 2 * num_lines:
            return "\n".join(source_code[start_line:end_line])
        else:
            start_code = source_code[start_line : start_line + num_lines]
            end_code = source_code[end_line - num_lines : end_line]
            return "\n".join(start_code + ["..."] + end_code)

    node = ast.parse(file_content)
    source_code = file_content.split("\n")

    # Extract module-level docstring
    file_documentation["module_docstring"] = ast.get_docstring(node)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_info = {
                "name": get_signature(child),
                "docstring": ast.get_docstring(child),
                "methods": [],
            }
            for grand_child in ast.iter_child_nodes(child):
                if isinstance(grand_child, ast.FunctionDef):
                    # method_info = {
                    #     "name": get_signature(grand_child),
                    #     "docstring": ast.get_docstring(grand_child) # or get_code_snippet(grand_child, source_code)
                    # }
                    # class_info["methods"].append(method_info)
                    class_info["methods"].append(get_signature(grand_child))
            file_documentation["classes"].append(class_info)
        elif isinstance(child, ast.FunctionDef):
            function_info = {
                "name": get_signature(child),
                # "docstring": ast.get_docstring(child) or get_code_snippet(child, source_code)
                "content": get_code_snippet(child, source_code),
            }
            file_documentation["functions"].append(function_info)

    return file_documentation


def parse_patch_and_get_codes(patch):
    patch = unidiff.PatchSet(patch)

    for patch_file in patch:
        before_code = []
        after_code = []
        file_path = patch_file.path

        for hunk in patch_file:
            before_hunk_code = []
            lines = hunk.source_lines()
            for line in lines:
                before_hunk_code.append((line.source_line_no, line.value))
            before_code.append(before_hunk_code)

            lines = hunk.target_lines()
            after_hunk_code = []
            for line in lines:
                after_hunk_code.append((line.target_line_no, line.value))
            after_code.append(after_hunk_code)

        yield file_path, before_code, after_code



def parse_patch_and_get_files(patch):
    patch = unidiff.PatchSet(patch)
    file_paths = []
    for patch_file in patch:
        file_path = patch_file.path
        file_paths.append(file_path)
    return file_paths


def parse_patch_and_get_line_nums(patch):
    patch = unidiff.PatchSet(patch)

    for patch_file in patch:
        line_nums = []
        file_path = patch_file.path

        for hunk in patch_file:
            lines = hunk.source_lines()
            for line in lines:
                line_nums.append(line.source_line_no)

        yield file_path, line_nums
        
def get_gold_classes_funcs(file_path, file_content, line_nums):
    node = ast.parse(file_content)
    line_nums = set(line_nums)

    classes = []
    funcs = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            start_line = child.lineno
            end_line = child.end_lineno
            if set(range(start_line, end_line + 1)).intersection(line_nums):
                classes.append(child.name)

        if isinstance(child, ast.FunctionDef):
            start_line = child.lineno
            end_line = child.end_lineno
            if set(range(start_line, end_line + 1)).intersection(line_nums):
                funcs.append(child.name)

    return {
        "file_path": file_path,
        "classes to be modified": classes,
        "functions to be modified": funcs,
    }
