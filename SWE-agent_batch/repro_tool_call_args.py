from sweagent.agent import models

# Construct history with tool call arguments as dict to mimic model output
history = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": {"path": "file.txt", "patch": "diff"},
                },
            }
        ],
    }
]

messages = models._history_to_openai_messages(history)

arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
print(type(arguments))
print(arguments)
