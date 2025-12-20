#!/usr/bin/env python3
"""
Minimal HTTP proxy - Pure forwarding with logging
"""
import sys
import json
import time
import argparse
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# Import system profiler
try:
    from system_profiler import SystemProfiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    SystemProfiler = None


class SimpleProxyHandler(BaseHTTPRequestHandler):
    """Simple HTTP proxy handler"""
    
    def log_request(self, code='-', size='-'):
        """Override default logging"""
        pass  # Don't print default logs
    
    def do_GET(self):
        self.proxy_request('GET')
    
    def do_POST(self):
        self.proxy_request('POST')
    
    def do_PUT(self):
        self.proxy_request('PUT')
    
    def do_DELETE(self):
        self.proxy_request('DELETE')
    
    def proxy_request(self, method):
        """Forward request to vLLM"""
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''
        
        # Build target URL
        target_url = f"http://localhost:{self.server.vllm_port}{self.path}"
        
        # Extract instance_id and is_terminated from request body
        instance_id = None
        is_terminated = False
        if body:
            try:
                body_json = json.loads(body.decode('utf-8'))
                # Check for instance metadata in various possible locations
                if "extra_body" in body_json:
                    instance_id = body_json["extra_body"].get("instance_id")
                    is_terminated = body_json["extra_body"].get("is_terminated", False)
                elif "metadata" in body_json:
                    instance_id = body_json["metadata"].get("instance_id")
                    is_terminated = body_json["metadata"].get("is_terminated", False)
                
                # Update instance state
                if instance_id:
                    self.server.instance_states[instance_id] = {
                        "is_terminated": is_terminated,
                        "last_seen": time.time()
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Log request (only log to file, not console)
        request_data = {
            "timestamp": time.time(),
            "method": method,
            "path": self.path,
            "body": body.decode('utf-8', errors='ignore') if body else "",
            "instance_id": instance_id,
            "is_terminated": is_terminated,
        }
        self.server.log_request(request_data)
        
        # Forward request
        try:
            req = urllib.request.Request(
                target_url,
                data=body if body else None,
                headers=dict(self.headers),
                method=method
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                response_body = response.read()
                
                # Inject instance metadata into response if present
                modified_response_body = response_body
                if instance_id:
                    try:
                        response_json = json.loads(response_body.decode('utf-8'))
                        # Add instance metadata to response
                        if "metadata" not in response_json:
                            response_json["metadata"] = {}
                        response_json["metadata"]["instance_id"] = instance_id
                        response_json["metadata"]["is_terminated"] = self.server.instance_states.get(
                            instance_id, {}
                        ).get("is_terminated", False)
                        modified_response_body = json.dumps(response_json).encode('utf-8')
                    except (json.JSONDecodeError, KeyError):
                        # If response is not JSON or doesn't have expected structure, keep original
                        pass
                
                # Log response
                response_data = {
                    "timestamp": time.time(),
                    "status": response.status,
                    "body": modified_response_body.decode('utf-8', errors='ignore'),
                    "instance_id": instance_id,
                }
                self.server.log_response(response_data)
                
                # Return response
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() != 'content-length':  # Recalculate content-length
                        self.send_header(key, value)
                self.send_header('Content-Length', str(len(modified_response_body)))
                self.end_headers()
                self.wfile.write(modified_response_body)
                
        except urllib.error.HTTPError as e:
            # Forward error response
            error_body = e.read()
            self.send_response(e.code)
            for key, value in e.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)
            
        except Exception as e:
            # Internal error
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


class ProxyServer(HTTPServer):
    """Proxy server"""
    
    def __init__(self, server_address, handler_class, vllm_port, log_dir, verbose, enable_logging):
        super().__init__(server_address, handler_class)
        self.vllm_port = vllm_port
        self.log_dir = log_dir
        self.verbose = verbose
        self.enable_logging = enable_logging
        
        # Track instance states: {instance_id: {"is_terminated": bool, "last_seen": timestamp}}
        self.instance_states = {}
        
        # Only open log files if logging is enabled
        if self.enable_logging:
            self.request_log = open(f"{log_dir}/requests.jsonl", "a")
            self.response_log = open(f"{log_dir}/responses.jsonl", "a")
        else:
            self.request_log = None
            self.response_log = None
    
    def log_request(self, data):
        if self.enable_logging and self.request_log:
            self.request_log.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.request_log.flush()
    
    def log_response(self, data):
        if self.enable_logging and self.response_log:
            self.response_log.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.response_log.flush()
    
    def __del__(self):
        if hasattr(self, 'request_log') and self.request_log:
            self.request_log.close()
        if hasattr(self, 'response_log') and self.response_log:
            self.response_log.close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--log-dir", default="./thunderreact_logs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--auto-start-vllm", action="store_true")
    parser.add_argument("--enable-logging", action="store_true", 
                       help="Enable request/response logging to files")
    
    # System profiling arguments
    parser.add_argument("--enable-system-profiling", action="store_true",
                       help="Enable system resource profiling (GPU/CPU/Memory/Disk)")
    parser.add_argument("--profiling-output", default="system_metrics.json",
                       help="System profiling output file (default: system_metrics.json)")
    parser.add_argument("--profiling-interval", type=float, default=0.5,
                       help="System profiling sampling interval in seconds (default: 0.5)")
    
    args, unknown = parser.parse_known_args()
    
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    
    vllm_process = None
    profiler = None
    
    try:
        # Start system profiler if enabled
        if args.enable_system_profiling:
            if not PROFILER_AVAILABLE:
                print("Warning: System profiler not available (missing dependencies)")
                print("  Install: pip install psutil pynvml")
            else:
                import os.path
                profiling_output = os.path.join(args.log_dir, args.profiling_output)
                profiler = SystemProfiler(
                    output_file=profiling_output,
                    sample_interval=args.profiling_interval,
                    verbose=args.verbose
                )
                profiler.start()
                print(f"[System Profiling] Started")
                print(f"  Output: {profiling_output}")
                print(f"  Interval: {args.profiling_interval}s")
        
        if args.auto_start_vllm:
            # Start vLLM - filter out ThunderReact-specific arguments
            vllm_cmd = ['python', '-m', 'vllm.entrypoints.openai.api_server']
            
            # ThunderReact-specific arguments that should NOT be passed to vLLM
            thunder_args = {
                '--auto-start-vllm',
                '--vllm-port', str(args.vllm_port),
                '--log-dir', args.log_dir,
                '--verbose',
                '--enable-logging',  # ThunderReact-only
                '--enable-system-profiling',  # ThunderReact-only
                '--profiling-output', args.profiling_output,
                '--profiling-interval', str(args.profiling_interval),
            }
            
            skip_next = False
            for i, arg in enumerate(sys.argv[1:]):
                if skip_next:
                    skip_next = False
                    continue
                
                if arg in thunder_args:
                    # Check if next arg is the value for this flag
                    if arg in ['--vllm-port', '--log-dir', '--profiling-output', '--profiling-interval']:
                        skip_next = True
                    continue
                
                if arg == '--port':
                    vllm_cmd.extend(['--port', str(args.vllm_port)])
                    skip_next = True
                    continue
                
                vllm_cmd.append(arg)
            
            print(f"[vLLM] Starting: {' '.join(vllm_cmd)}")
            vllm_process = subprocess.Popen(vllm_cmd)
            print(f"[vLLM] Waiting to start...")
            time.sleep(10)  # Simple wait
        
        # Start proxy
        print(f"[Proxy] Starting: localhost:{args.port} -> localhost:{args.vllm_port}")
        if args.enable_logging:
            print(f"[Logging] Enabled: {args.log_dir}/")
        else:
            print(f"[Logging] Disabled (use --enable-logging to enable)")
        
        server = ProxyServer(
            ('0.0.0.0', args.port),
            SimpleProxyHandler,
            args.vllm_port,
            args.log_dir,
            args.verbose,
            args.enable_logging
        )
        
        print("[Proxy] Ready - Press Ctrl+C to stop")
        server.serve_forever()
    
    except KeyboardInterrupt:
        print("\n[Proxy] Stopping...")
    finally:
        # Stop system profiler
        if profiler:
            print("[System Profiling] Stopping...")
            profiler.stop()
            summary = profiler.get_summary()
            print(f"  Collected {summary.get('total_samples', 0)} samples over {summary.get('duration_seconds', 0):.1f}s")
            if 'gpu' in summary:
                print(f"  GPU avg: {summary['gpu']['avg_utilization']:.1f}% util, {summary['gpu']['avg_mem_utilization']:.1f}% mem")
            if 'cpu' in summary:
                print(f"  CPU avg: {summary['cpu']['avg_percent']:.1f}%")
            if 'memory' in summary:
                print(f"  Memory avg: {summary['memory']['avg_percent']:.1f}%")
        
        if vllm_process:
            print("[vLLM] Stopping...")
            vllm_process.terminate()
            vllm_process.wait()


if __name__ == "__main__":
    main()

