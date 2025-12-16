#!/usr/bin/env python3
"""
ThunderReact OpenAI-compatible API Server
完全兼容 vLLM 的启动方式，自动转发所有请求到真实的 vLLM 服务器
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import signal
import socket
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse


class ThunderReactProxy:
    """vLLM 请求代理和记录器"""
    
    def __init__(self, vllm_port: int, log_dir: Path, verbose: bool = False):
        self.vllm_url = f"http://localhost:{vllm_port}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.request_count = 0
        
        # 日志文件
        self.request_log_file = self.log_dir / "requests.jsonl"
        self.response_log_file = self.log_dir / "responses.jsonl"
        
        print(f"✅ ThunderReact 代理服务器初始化完成")
        print(f"   📝 日志目录: {self.log_dir}")
        print(f"   🎯 转发目标: {self.vllm_url}")
    
    def _log_to_file(self, filepath: Path, data: dict):
        """写入日志"""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def _print_request(self, request_data: dict):
        """打印请求信息"""
        if not self.verbose:
            return
        
        print(f"\n{'='*80}")
        print(f"[ThunderReact 请求 #{self.request_count}]")
        print(f"模型: {request_data.get('model')}")
        print(f"Stream: {request_data.get('stream', False)}")
        print(f"消息数: {len(request_data.get('messages', []))}")
        print(f"{'='*80}\n")
    
    def _print_response(self, response_data: dict, duration: float):
        """打印响应信息"""
        if not self.verbose:
            return
        
        print(f"\n{'='*80}")
        print(f"[ThunderReact 响应 #{self.request_count}] 耗时: {duration:.2f}秒")
        if "usage" in response_data:
            usage = response_data["usage"]
            print(f"Token: prompt={usage.get('prompt_tokens')}, "
                  f"completion={usage.get('completion_tokens')}")
        print(f"{'='*80}\n")
    
    async def proxy_request(self, request: Request) -> Response:
        """代理请求到 vLLM"""
        self.request_count += 1
        start_time = time.time()
        
        # 读取请求
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
        
        # 转发 URL
        target_url = f"{self.vllm_url}{request.url.path}"
        
        # 判断是否 stream
        is_stream = request_data.get("stream", False)
        
        if is_stream:
            return await self._proxy_stream(target_url, request, request_data, start_time)
        else:
            return await self._proxy_normal(target_url, request, request_data, start_time)
    
    async def _proxy_normal(
        self, target_url: str, request: Request, request_data: dict, start_time: float
    ) -> Response:
        """代理普通请求"""
        async with aiohttp.ClientSession() as session:
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
                
                return Response(
                    content=response_body,
                    status_code=resp.status,
                    headers=dict(resp.headers),
                )
    
    async def _proxy_stream(
        self, target_url: str, request: Request, request_data: dict, start_time: float
    ) -> StreamingResponse:
        """代理 stream 请求"""
        
        async def stream_generator() -> AsyncIterator[bytes]:
            all_chunks = []
            accumulated_text = {}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    target_url,
                    json=request_data,
                    headers=dict(request.headers),
                ) as resp:
                    async for chunk in resp.content:
                        if not chunk:
                            continue
                        
                        all_chunks.append(chunk)
                        
                        # 解析 SSE
                        try:
                            chunk_str = chunk.decode('utf-8')
                            for line in chunk_str.split('\n'):
                                if line.startswith('data: '):
                                    data_str = line[6:]
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
                            pass
                        
                        # 转发原始 chunk
                        yield chunk
            
            # 记录完整响应
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
                print(f"\n[ThunderReact Stream #{self.request_count}] 完成，耗时 {duration:.2f}秒\n")
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )


def create_app(vllm_port: int, log_dir: Path, verbose: bool) -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(title="ThunderReact OpenAI-compatible API")
    proxy = ThunderReactProxy(vllm_port, log_dir, verbose)
    
    @app.get("/")
    async def root():
        return {
            "service": "ThunderReact Proxy",
            "vllm_url": proxy.vllm_url,
            "request_count": proxy.request_count,
        }
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    # 捕获所有其他路由并转发
    @app.get("/{path:path}")
    async def proxy_get(request: Request, path: str):
        return await proxy.proxy_request(request)
    
    @app.post("/{path:path}")
    async def proxy_post(request: Request, path: str):
        return await proxy.proxy_request(request)
    
    @app.put("/{path:path}")
    async def proxy_put(request: Request, path: str):
        return await proxy.proxy_request(request)
    
    @app.delete("/{path:path}")
    async def proxy_delete(request: Request, path: str):
        return await proxy.proxy_request(request)
    
    @app.patch("/{path:path}")
    async def proxy_patch(request: Request, path: str):
        return await proxy.proxy_request(request)
    
    return app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ThunderReact OpenAI-compatible API Server (vLLM Proxy)",
        add_help=False,  # 禁用默认 help，避免和 vLLM 参数冲突
    )
    
    # ThunderReact 特有参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="代理服务器监听地址")
    parser.add_argument("--port", type=int, default=9000, help="代理服务器监听端口")
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="vLLM 服务器端口（默认: 8000）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./thunderreact_logs",
        help="日志保存目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式",
    )
    parser.add_argument(
        "--auto-start-vllm",
        action="store_true",
        help="自动在内部启动 vLLM 服务器",
    )
    parser.add_argument(
        "--help",
        action="store_true",
        help="显示帮助信息",
    )
    
    # 解析已知参数，保留未知参数
    args, unknown = parser.parse_known_args()
    
    if args.help:
        parser.print_help()
        print("\n所有其他参数将传递给 vLLM（如果 --auto-start-vllm 启用）")
        sys.exit(0)
    
    # 保存所有原始参数和 vLLM 参数
    args.vllm_args = unknown
    args.all_args = sys.argv[1:]  # 保存所有原始参数
    
    return args


def build_vllm_command(args) -> list[str]:
    """构建 vLLM 启动命令"""
    # 过滤掉 ThunderReact 特有的参数
    thunderreact_args = {
        '--vllm-port', '--log-dir', '--verbose', '--auto-start-vllm'
    }
    
    vllm_cmd = ['python', '-m', 'vllm.entrypoints.openai.api_server']
    
    # 添加所有参数，但排除 ThunderReact 特有参数
    i = 0
    all_args = args.all_args
    while i < len(all_args):
        arg = all_args[i]
        
        # 跳过 ThunderReact 参数
        if arg in thunderreact_args:
            i += 1
            # 如果下一个参数不是以 -- 开头，说明是这个参数的值，也跳过
            if i < len(all_args) and not all_args[i].startswith('--'):
                i += 1
            continue
        
        # 处理 --port，改成 --vllm-port 指定的端口
        if arg == '--port':
            vllm_cmd.append('--port')
            vllm_cmd.append(str(args.vllm_port))
            i += 1
            # 跳过原来的端口值
            if i < len(all_args) and not all_args[i].startswith('--'):
                i += 1
            continue
        
        # 其他参数直接添加
        vllm_cmd.append(arg)
        i += 1
    
    # 如果没有指定 --port，添加默认端口
    if '--port' not in vllm_cmd:
        vllm_cmd.extend(['--port', str(args.vllm_port)])
    
    return vllm_cmd


def wait_for_vllm_ready(port: int, timeout: int = 300) -> bool:
    """等待 vLLM 服务器启动完成"""
    print(f"⏳ 等待 vLLM 服务器在端口 {port} 启动...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # 尝试连接端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                # 端口开放，尝试访问 /v1/models
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ vLLM 服务器已就绪！")
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n❌ vLLM 服务器在 {timeout} 秒内未能启动")
    return False


def main():
    args = parse_args()
    
    vllm_process: Optional[subprocess.Popen] = None
    
    try:
        if args.auto_start_vllm:
            # 自动启动 vLLM
            print(f"""
{'='*80}
🚀 ThunderReact - 自动启动模式
{'='*80}
代理服务器: {args.host}:{args.port}
vLLM 端口: {args.vllm_port}
日志目录: {args.log_dir}
{'='*80}
            """)
            
            # 构建 vLLM 命令
            vllm_cmd = build_vllm_command(args)
            print(f"🔧 启动 vLLM 服务器...")
            if args.verbose:
                print(f"   命令: {' '.join(vllm_cmd)}")
            
            # 启动 vLLM 进程
            vllm_process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE if not args.verbose else None,
                stderr=subprocess.PIPE if not args.verbose else None,
            )
            
            # 等待 vLLM 启动
            if not wait_for_vllm_ready(args.vllm_port):
                if vllm_process:
                    vllm_process.terminate()
                    vllm_process.wait()
                print("❌ vLLM 启动失败")
                sys.exit(1)
        else:
            # 手动模式
            print(f"""
{'='*80}
🚀 ThunderReact OpenAI-compatible API Server
{'='*80}
监听地址: {args.host}:{args.port}
转发目标: http://localhost:{args.vllm_port}
日志目录: {args.log_dir}
详细模式: {args.verbose}
{'='*80}

⚠️  请确保 vLLM 服务器已在端口 {args.vllm_port} 上运行！

使用方式：
    1. 先启动 vLLM 服务器在端口 {args.vllm_port}
    2. 本服务器会自动转发所有请求
    3. 查看日志: {args.log_dir}/requests.jsonl 和 responses.jsonl

{'='*80}
            """)
        
        # 启动代理服务器
        print(f"🌐 启动代理服务器在 {args.host}:{args.port}")
        app = create_app(args.vllm_port, Path(args.log_dir), args.verbose)
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    
    except KeyboardInterrupt:
        print("\n\n👋 收到停止信号，正在关闭...")
    
    finally:
        # 清理：关闭 vLLM 进程
        if vllm_process:
            print("🛑 正在停止 vLLM 服务器...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️  vLLM 未能正常停止，强制终止...")
                vllm_process.kill()
                vllm_process.wait()
            print("✅ vLLM 服务器已停止")
        
        print("✅ ThunderReact 已完全停止")
        sys.exit(0)


if __name__ == "__main__":
    main()

