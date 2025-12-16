#!/usr/bin/env python3
"""
vLLM 代理服务器 - 拦截并记录所有发送到 vLLM 的请求

用法：
    python tools/vllm_proxy_server.py --proxy-port 9000 --vllm-url http://localhost:8000

然后修改配置文件中的 api_base：
    model:
        type: vllm
        api_base: http://localhost:9000/v1  # 指向代理服务器
"""

import argparse
import json
import time
from pathlib import Path
from typing import AsyncIterator

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse


class VLLMProxy:
    """vLLM 代理服务器"""
    
    def __init__(self, vllm_url: str, log_dir: Path, verbose: bool = False):
        self.vllm_url = vllm_url.rstrip("/")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.request_count = 0
        
        # 创建日志文件
        self.request_log_file = self.log_dir / "requests.jsonl"
        self.response_log_file = self.log_dir / "responses.jsonl"
        
        print(f"✅ vLLM 代理服务器初始化完成")
        print(f"   📝 日志目录: {self.log_dir}")
        print(f"   🎯 vLLM 服务器: {self.vllm_url}")
    
    def _log_to_file(self, filepath: Path, data: dict):
        """写入日志文件"""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def _print_request(self, request_data: dict):
        """打印请求信息"""
        if not self.verbose:
            return
        
        print(f"\n{'='*80}")
        print(f"[请求 #{self.request_count}]")
        print(f"模型: {request_data.get('model')}")
        print(f"Stream: {request_data.get('stream', False)}")
        print(f"消息数: {len(request_data.get('messages', []))}")
        
        # 打印最后一条消息（通常是用户的问题）
        messages = request_data.get('messages', [])
        if messages:
            last_msg = messages[-1]
            content = last_msg.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"最后消息: {last_msg.get('role')} -> {content}")
        print(f"{'='*80}\n")
    
    def _print_response(self, response_data: dict, duration: float):
        """打印响应信息"""
        if not self.verbose:
            return
        
        print(f"\n{'='*80}")
        print(f"[响应 #{self.request_count}]")
        print(f"耗时: {duration:.2f}秒")
        
        if "choices" in response_data:
            for i, choice in enumerate(response_data["choices"]):
                message = choice.get("message", {})
                content = message.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"选择 {i}: {content}")
        
        if "usage" in response_data:
            usage = response_data["usage"]
            print(f"Token 使用: prompt={usage.get('prompt_tokens')}, "
                  f"completion={usage.get('completion_tokens')}, "
                  f"total={usage.get('total_tokens')}")
        print(f"{'='*80}\n")
    
    async def proxy_request(self, request: Request) -> Response:
        """代理请求到 vLLM 服务器"""
        self.request_count += 1
        start_time = time.time()
        
        # 读取请求体
        body = await request.body()
        request_data = json.loads(body) if body else {}
        
        # 记录请求
        log_entry = {
            "timestamp": start_time,
            "request_id": self.request_count,
            "method": request.method,
            "path": request.url.path,
            "request": request_data,
        }
        self._log_to_file(self.request_log_file, log_entry)
        self._print_request(request_data)
        
        # 构造转发的 URL
        target_url = f"{self.vllm_url}{request.url.path}"
        
        # 判断是否是 stream 模式
        is_stream = request_data.get("stream", False)
        
        if is_stream:
            # Stream 模式：需要特殊处理
            return await self._proxy_stream_request(
                target_url, request, request_data, start_time
            )
        else:
            # 非 Stream 模式：直接转发
            return await self._proxy_normal_request(
                target_url, request, request_data, start_time
            )
    
    async def _proxy_normal_request(
        self, target_url: str, request: Request, request_data: dict, start_time: float
    ) -> Response:
        """代理非 stream 请求"""
        async with aiohttp.ClientSession() as session:
            # 转发请求
            async with session.post(
                target_url,
                json=request_data,
                headers=dict(request.headers),
            ) as resp:
                response_body = await resp.read()
                response_data = json.loads(response_body)
                
                duration = time.time() - start_time
                
                # 记录响应
                log_entry = {
                    "timestamp": time.time(),
                    "request_id": self.request_count,
                    "duration": duration,
                    "response": response_data,
                }
                self._log_to_file(self.response_log_file, log_entry)
                self._print_response(response_data, duration)
                
                # 返回响应
                return Response(
                    content=response_body,
                    status_code=resp.status,
                    headers=dict(resp.headers),
                )
    
    async def _proxy_stream_request(
        self, target_url: str, request: Request, request_data: dict, start_time: float
    ) -> StreamingResponse:
        """代理 stream 请求"""
        
        async def stream_generator() -> AsyncIterator[bytes]:
            """生成器：逐块转发并记录"""
            all_chunks = []
            accumulated_text = {}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    target_url,
                    json=request_data,
                    headers=dict(request.headers),
                ) as resp:
                    # 逐块读取和转发
                    async for chunk in resp.content:
                        if not chunk:
                            continue
                        
                        # 记录 chunk
                        all_chunks.append(chunk)
                        
                        # 解析 SSE 格式
                        try:
                            # vLLM 使用 SSE 格式：data: {...}\n\n
                            chunk_str = chunk.decode('utf-8')
                            for line in chunk_str.split('\n'):
                                if line.startswith('data: '):
                                    data_str = line[6:]  # 去掉 "data: "
                                    if data_str.strip() == '[DONE]':
                                        continue
                                    
                                    chunk_data = json.loads(data_str)
                                    choices = chunk_data.get("choices", [])
                                    
                                    for choice in choices:
                                        idx = choice.get("index", 0)
                                        delta = choice.get("delta", {})
                                        content = delta.get("content")
                                        if content:
                                            accumulated_text[idx] = accumulated_text.get(idx, "") + content
                        except Exception:
                            pass  # 解析失败就跳过
                        
                        # 转发原始 chunk
                        yield chunk
            
            # Stream 结束，记录完整响应
            duration = time.time() - start_time
            log_entry = {
                "timestamp": time.time(),
                "request_id": self.request_count,
                "duration": duration,
                "stream": True,
                "response": {
                    "num_chunks": len(all_chunks),
                    "accumulated_text": accumulated_text,
                },
            }
            self._log_to_file(self.response_log_file, log_entry)
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"[Stream 响应完成 #{self.request_count}]")
                print(f"耗时: {duration:.2f}秒")
                print(f"总块数: {len(all_chunks)}")
                for idx, text in accumulated_text.items():
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"文本 {idx}: {preview}")
                print(f"{'='*80}\n")
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )


def create_app(vllm_url: str, log_dir: Path, verbose: bool = False) -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(title="vLLM Proxy Server")
    proxy = VLLMProxy(vllm_url, log_dir, verbose)
    
    @app.get("/")
    async def root():
        return {
            "service": "vLLM Proxy Server",
            "vllm_url": proxy.vllm_url,
            "request_count": proxy.request_count,
        }
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_all(request: Request, path: str):
        """代理所有请求"""
        return await proxy.proxy_request(request)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="vLLM 代理服务器")
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=9000,
        help="代理服务器监听端口（默认: 9000）",
    )
    parser.add_argument(
        "--proxy-host",
        type=str,
        default="0.0.0.0",
        help="代理服务器监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM 服务器地址（默认: http://localhost:8000）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./proxy_logs",
        help="日志保存目录（默认: ./proxy_logs）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式",
    )
    
    args = parser.parse_args()
    
    print(f"""
{'='*80}
🚀 启动 vLLM 代理服务器
{'='*80}
监听地址: {args.proxy_host}:{args.proxy_port}
vLLM 地址: {args.vllm_url}
日志目录: {args.log_dir}
详细输出: {args.verbose}
{'='*80}

配置 SWE-agent 使用代理服务器：
    model:
        api_base: http://localhost:{args.proxy_port}/v1

{'='*80}
    """)
    
    app = create_app(args.vllm_url, Path(args.log_dir), args.verbose)
    
    uvicorn.run(
        app,
        host=args.proxy_host,
        port=args.proxy_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

