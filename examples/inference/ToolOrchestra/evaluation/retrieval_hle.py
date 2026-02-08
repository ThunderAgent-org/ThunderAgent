# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import asyncio
import threading
import time
import queue
from dataclasses import dataclass
from typing import List, Optional, Any
import argparse
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tavily import TavilyClient

# HuggingFace Hub fast-download mode (`HF_HUB_ENABLE_HF_TRANSFER=1`) requires
# `hf_transfer`. If the env var is enabled but the package isn't installed,
# HuggingFace raises a ValueError during model/tokenizer downloads. To make
# `retrieval_hle.py` robust for smoke tests / fresh envs, auto-disable it.
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") in {"1", "true", "True"}:
    try:
        import hf_transfer  # type: ignore[unused-ignore]  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print(
            "[retrieval_hle] WARNING: HF_HUB_ENABLE_HF_TRANSFER=1 but 'hf_transfer' is not installed; "
            "disabling fast downloads. Install with `pip install hf_transfer` to re-enable.",
            flush=True,
        )

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=16,
        cache_dir='cache/hugggingface'
    )
    return corpus

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    if model_path in ['Qwen/Qwen3-Embedding-8B']:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        # Prefer FlashAttention2 when available, but fall back gracefully if the
        # environment doesn't have `flash_attn` installed (common in fresh/CPU-only envs).
        try:
            model = AutoModel.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
            ).cuda()
        except Exception as e:
            print(
                f"[retrieval_hle] WARNING: flash_attention_2 unavailable ({type(e).__name__}: {e}); "
                "falling back to SDPA/eager attention.",
                flush=True,
            )
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    torch_dtype=torch.float16,
                ).cuda()
            except Exception:
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                ).cuda()
    else:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        model.cuda()
        if use_fp16:
            model = model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask=None,
        pooling_method="mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in
                              query_list]

        if 'qwen' in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a search query, retrieve relevant passages\nQuery:{query}' for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs.to(self.model.device)

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        elif 'qwen' in self.model_name.lower():
            output = self.model(**inputs)
            embeddings = last_token_pool(output.last_hidden_state, inputs['attention_mask'])
            query_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(None,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, eid: str = None):
        return self._batch_search(query_list, num, return_score, eid)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print(f"[retrieval_hle] Loading FAISS index: {self.index_path}", flush=True)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            try:
                print(f"[retrieval_hle] Moving FAISS index to GPU(s) (visible_gpus={faiss.get_num_gpus()}) ...", flush=True)
            except Exception:
                print("[retrieval_hle] Moving FAISS index to GPU(s) ...", flush=True)
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            print("[retrieval_hle] FAISS index moved to GPU(s).", flush=True)

        print(f"[retrieval_hle] Loading corpus: {self.corpus_path}", flush=True)
        self.corpus = load_corpus(self.corpus_path)
        try:
            print(f"[retrieval_hle] Corpus loaded (size={len(self.corpus)})", flush=True)
        except Exception:
            print("[retrieval_hle] Corpus loaded.", flush=True)
        print(f"[retrieval_hle] Loading embedding model: {config.retrieval_model_path}", flush=True)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        print("[retrieval_hle] Embedding model loaded.", flush=True)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        print(f"[retrieval_hle] Loading example_id_file: {config.example_id_file}", flush=True)
        with open(config.example_id_file) as f:
            self.example_ids = json.load(f)
        # Speed up membership checks in retrieval filtering.
        # `example_ids` is expected to be {eid: [doc_id, ...]}.
        try:
            for _eid, _ids in list(self.example_ids.items()):
                if isinstance(_ids, list):
                    self.example_ids[_eid] = set(_ids)
        except Exception:
            pass
        try:
            print(f"[retrieval_hle] example_ids loaded (keys={len(self.example_ids)})", flush=True)
        except Exception:
            print("[retrieval_hle] example_ids loaded.", flush=True)

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def raw_batch_search(self, query_list: List[str], num: int) -> tuple[Any, Any]:
        """
        Batched GPU retrieval without any post-filtering or doc loading.
        Returns (scores, idxs) as returned by FAISS, shaped [B, num].
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        batch_emb = self.encoder.encode(query_list)
        batch_scores, batch_idxs = self.index.search(batch_emb, k=int(num))
        return batch_scores, batch_idxs

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, eid: str = None):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: ', disable=True):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num: (i + 1) * num] for i in range(len(batch_idxs))]

            updated_batch_results = []
            updated_scores = []
            for one_batch_results,one_batch_scores in zip(batch_results,batch_scores):
                cur_batch_results = []
                cur_batch_scores = []
                for r,s in zip(one_batch_results,one_batch_scores):
                    if int(r['id']) in self.example_ids[eid]:
                        cur_batch_results.append(r)
                        cur_batch_scores.append(s)
                updated_batch_results.append(cur_batch_results)
                updated_scores.append(cur_batch_scores)

            results.extend(updated_batch_results)
            scores.extend(updated_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results,updated_batch_results,updated_scores

        if return_score:
            return results, scores
        else:
            return results


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
            self,
            retrieval_method: str = "bm25",
            retrieval_topk: int = 10,
            index_path: str = "./index/bm25",
            corpus_path: str = "./data/corpus.jsonl",
            dataset_path: str = "./data",
            data_split: str = "train",
            faiss_gpu: bool = True,
            retrieval_model_path: str = "./model",
            retrieval_pooling_method: str = "mean",
            retrieval_query_max_length: int = 256,
            retrieval_use_fp16: bool = False,
            retrieval_batch_size: int = 128,
            new_cache_dir: str = None,
            example_id_file: str = None,
            tavily_key: str = None
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.new_cache_dir = new_cache_dir
        self.example_id_file = example_id_file
        self.tavily_key = tavily_key


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    eid: str = None
    new_cache_dir: str = None

app = FastAPI()
_RETRIEVE_LOCK = threading.Lock()

_BATCH_QUEUE: Optional["BatchQueue"] = None


@dataclass(frozen=True)
class _PendingRetrieve:
    query: str
    eid: Optional[str]
    topk: int
    return_scores: bool
    loop: asyncio.AbstractEventLoop
    fut: asyncio.Future


class BatchQueue:
    """
    Thread-safe queue that groups independent /retrieve requests into a micro-batch.

    Flush policy:
    - flush immediately when batch reaches max_batch_size
    - otherwise flush when the oldest item has waited >= max_wait_ms
    """

    def __init__(self, max_batch_size: int = 256, max_wait_ms: float = 5.0):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be > 0")
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self._q: "queue.Queue[_PendingRetrieve]" = queue.Queue()

    def put(self, item: _PendingRetrieve) -> None:
        self._q.put(item)

    def get_batch(self) -> list[_PendingRetrieve]:
        first = self._q.get()  # blocks until at least one request arrives
        batch: list[_PendingRetrieve] = [first]
        t0 = time.perf_counter()
        while len(batch) < self.max_batch_size:
            remaining = self.max_wait_s - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            try:
                batch.append(self._q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch


def _safe_set_future_result(fut: asyncio.Future, value: Any) -> None:
    try:
        if fut.done():
            return
        fut.set_result(value)
    except Exception:
        # If the client disconnected/cancelled, ignore.
        return


def _safe_set_future_exception(fut: asyncio.Future, exc: BaseException) -> None:
    try:
        if fut.done():
            return
        fut.set_exception(exc)
    except Exception:
        return


def _tavily_fallback_append(resp0: list[Any], query: str, eid: str) -> None:
    """
    Best-effort Tavily search/extract fallback. Appends extracted content into resp0.
    Mirrors the previous synchronous endpoint behavior.
    """
    if not getattr(config, "tavily_key", None):
        return

    tavily_client = TavilyClient(config.tavily_key)
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=20,
            chunks_per_source=5,
        )
    except Exception:
        return

    try:
        cache_dir = os.path.join(config.new_cache_dir, eid)
        os.makedirs(cache_dir, exist_ok=True)
        search_idx = 0
        while os.path.isfile(os.path.join(cache_dir, f"search_{search_idx}.json")):
            search_idx += 1
        with open(os.path.join(cache_dir, f"search_{search_idx}.json"), "w") as f:
            json.dump(response, f, indent=2)
    except Exception:
        # Cache is best-effort.
        search_idx = 0

    def extract_web(extract_argument):
        try:
            extraction = tavily_client.extract(
                urls=[extract_argument["url"]],
                extract_depth="advanced",
                format="text",
            )
        except Exception:
            return None
        try:
            cache_dir = os.path.join(config.new_cache_dir, eid)
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"extraction_{search_idx}_{extract_argument['extract_id']}.json"), "w") as f:
                json.dump(extraction, f, indent=2)
        except Exception:
            pass
        extract_argument["raw_extraction"] = extraction
        return extract_argument

    extraction_arguments = []
    for extract_id, r in enumerate(response.get("results", [])):
        extraction_arguments.append(
            [extract_web, {"extract_id": extract_id, "url": r.get("url"), "score": r.get("score")}]
        )

    all_extraction_results = []
    for argument in extraction_arguments:
        if not argument[1].get("url"):
            continue
        all_extraction_results.append(argument[0](argument[1]))

    extraction_results = []
    for extraction_return in all_extraction_results:
        if not extraction_return:
            continue
        extract_content = ""
        try:
            for one_extraction_result in extraction_return["raw_extraction"].get("results", []):
                extract_content += one_extraction_result.get("raw_content", "") + "\n\n"
        except Exception:
            continue
        if len(extract_content.strip()) > 100:
            extraction_results.append([extract_content, extraction_return.get("score", 0.0)])

    if len(extraction_results) > 1:
        extraction_results = sorted(extraction_results, key=lambda x: x[1], reverse=True)

    for new_doc_id, new_search in enumerate(extraction_results):
        if not (isinstance(new_search, list) and isinstance(new_search[0], str)):
            continue
        # Keep legacy shape: {"document": {"content": ...}, "score": -N}
        resp0.append({"document": {"content": new_search[0]}, "score": -new_doc_id - 1, "source": "tavily"})


def _format_one_response(
    *,
    query: str,
    eid: Optional[str],
    topk: int,
    return_scores: bool,
    scores_row: Any,
    idxs_row: Any,
) -> list[Any]:
    # NOTE: Response shape must remain: a list with exactly one element (per-request),
    # i.e. results[0] is the list of hits.
    resp0: list[Any] = []

    allowed_ids = None
    if eid is not None:
        allowed_ids = retriever.example_ids.get(eid)
    if allowed_ids is None:
        allowed_ids = set()

    # Iterate in FAISS rank order.
    for idx, score in zip(idxs_row, scores_row):
        try:
            score_f = float(score)
        except Exception:
            continue
        if score_f <= 0.1:
            continue
        try:
            doc = retriever.corpus[int(idx)]
        except Exception:
            continue
        try:
            doc_id = int(doc.get("id"))
        except Exception:
            continue
        if doc_id not in allowed_ids:
            continue
        content = doc.get("content") if isinstance(doc, dict) else None
        if content is None and isinstance(doc, dict):
            content = doc.get("contents")
        if not isinstance(content, str) or len(content) <= 100:
            continue
        if return_scores:
            resp0.append({"document": doc, "score": score_f, "source": "local"})
        else:
            resp0.append(doc)
        if len(resp0) >= topk:
            break

    # Optional web fallback via Tavily (legacy behavior).
    if len(resp0) < 3 and getattr(config, "tavily_key", None) and isinstance(eid, str) and eid:
        _tavily_fallback_append(resp0, query=query, eid=eid)

    return [resp0]


def _batch_worker_main(*, batch_queue: BatchQueue, k: int = 100) -> None:
    """
    Background worker: micro-batches independent /retrieve requests into a single
    GPU encode + FAISS search call, then post-processes per-request (eid filter).
    """
    while True:
        batch = batch_queue.get_batch()
        if not batch:
            continue

        try:
            queries = [b.query for b in batch]
            # GPU path (single encode + single search)
            batch_scores, batch_idxs = retriever.raw_batch_search(queries, num=int(k))

            # Per-request post-processing (eid filtering + formatting)
            for i, b in enumerate(batch):
                resp = _format_one_response(
                    query=b.query,
                    eid=b.eid,
                    topk=b.topk,
                    return_scores=b.return_scores,
                    scores_row=batch_scores[i],
                    idxs_row=batch_idxs[i],
                )
                b.loop.call_soon_threadsafe(_safe_set_future_result, b.fut, resp)
        except Exception as e:
            # Fail the whole batch; clients will see 500s (but server stays alive).
            for b in batch:
                b.loop.call_soon_threadsafe(_safe_set_future_exception, b.fut, e)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    assert len(request.queries) == 1, "We now assume single query search"
    topk = int(request.topk or config.retrieval_topk)

    q = _BATCH_QUEUE
    if q is None:
        raise RuntimeError("Retrieval micro-batching queue is not initialized")

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    q.put(
        _PendingRetrieve(
            query=request.queries[0],
            eid=request.eid,
            topk=topk,
            return_scores=bool(request.return_scores),
            loop=loop,
            fut=fut,
        )
    )
    return await fut


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--new_cache_dir', type=str, default='cache/hle')
parser.add_argument('--example_id_file', type=str, default='examples.json')
parser.add_argument('--tavily_key', type=str, default="")
parser.add_argument('--port', type=int)
parser.add_argument(
    "--faiss_k",
    type=int,
    default=int(os.environ.get("RETRIEVAL_FAISS_K", "1000")),
    help="FAISS k for candidate retrieval before eid filtering (default: 1000 or $RETRIEVAL_FAISS_K).",
)
parser.add_argument(
    "--max_batch_size",
    type=int,
    default=int(os.environ.get("RETRIEVAL_MAX_BATCH_SIZE", "256")),
    help="Micro-batch max size (default: 256 or $RETRIEVAL_MAX_BATCH_SIZE).",
)
parser.add_argument(
    "--max_wait_ms",
    type=float,
    default=float(os.environ.get("RETRIEVAL_MAX_WAIT_MS", "5")),
    help="Micro-batch flush timeout in ms (default: 5 or $RETRIEVAL_MAX_WAIT_MS).",
)
args = parser.parse_args()

index_dir = os.environ.get("INDEX_DIR")
if not index_dir:
    raise SystemExit(
        "ERROR: INDEX_DIR env var is not set. It must point to a directory containing eval.index and eval.jsonl."
    )

config = Config(
    retrieval_method='qwen',  # or "dense"
    index_path=os.path.join(index_dir, 'eval.index'),
    corpus_path=os.path.join(index_dir, 'eval.jsonl'),
    retrieval_topk=5,
    faiss_gpu=True,
    retrieval_model_path='Qwen/Qwen3-Embedding-8B',
    retrieval_pooling_method="mean",
    retrieval_query_max_length=32768,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
    new_cache_dir=args.new_cache_dir,
    example_id_file=args.example_id_file,
    tavily_key=args.tavily_key
)

print(
    "[retrieval_hle] Config:\n"
    f"  INDEX_DIR={os.environ.get('INDEX_DIR')}\n"
    f"  index_path={config.index_path}\n"
    f"  corpus_path={config.corpus_path}\n"
    f"  retrieval_model_path={config.retrieval_model_path}\n"
    f"  faiss_gpu={config.faiss_gpu}\n"
    f"  port={args.port}\n"
    f"  new_cache_dir={args.new_cache_dir}\n"
    f"  example_id_file={args.example_id_file}\n"
    f"  tavily_key_set={bool(args.tavily_key)}",
    flush=True,
)

retriever = DenseRetriever(config)

_BATCH_QUEUE = BatchQueue(max_batch_size=int(args.max_batch_size), max_wait_ms=float(args.max_wait_ms))
_BATCH_WORKER = threading.Thread(
    target=_batch_worker_main,
    kwargs={"batch_queue": _BATCH_QUEUE, "k": int(args.faiss_k)},
    daemon=True,
)
_BATCH_WORKER.start()
print(
    "[retrieval_hle] Micro-batching enabled "
    f"(max_batch_size={_BATCH_QUEUE.max_batch_size}, max_wait_ms={_BATCH_QUEUE.max_wait_s * 1000.0:.1f}, k={int(args.faiss_k)}).",
    flush=True,
)

print(f"[retrieval_hle] Starting uvicorn on 0.0.0.0:{args.port}", flush=True)
uvicorn.run(app, host="0.0.0.0", port=args.port)

# 
