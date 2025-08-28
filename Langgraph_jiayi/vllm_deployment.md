# vLLM deployment

```
vllm serve MODEL_NAME --tensor-parallel-size 2
```

Note: the Model name should be identical to the name used in the ChatOpenAI function

```
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",
    model="MODEL_NAME",
    temperature=0, 
)
```

