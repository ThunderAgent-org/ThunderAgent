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
        
        # Log request (only log to file, not console)
        
        request_data = {
            "timestamp": time.time(),
            "method": method,
            "path": self.path,
            "body": body.decode('utf-8', errors='ignore') if body else "",
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
                
                # Log response
                response_data = {
                    "timestamp": time.time(),
                    "status": response.status,
                    "body": response_body.decode('utf-8', errors='ignore'),
                }
                self.server.log_response(response_data)
                
                # Return response
                self.send_response(response.status)
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)
                
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
    
    args, unknown = parser.parse_known_args()
    
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    
    vllm_process = None
    
    try:
        if args.auto_start_vllm:
            # Start vLLM
            vllm_cmd = ['python', '-m', 'vllm.entrypoints.openai.api_server']
            for arg in sys.argv[1:]:
                if arg not in ['--auto-start-vllm', '--vllm-port', str(args.vllm_port), 
                              '--log-dir', args.log_dir, '--verbose']:
                    if arg == '--port':
                        vllm_cmd.extend(['--port', str(args.vllm_port)])
                        # Skip next arg (port value)
                        continue
                    vllm_cmd.append(arg)
            
            print(f"🚀 Starting vLLM: {' '.join(vllm_cmd)}")
            vllm_process = subprocess.Popen(vllm_cmd)
            print(f"⏳ Waiting for vLLM to start...")
            time.sleep(10)  # Simple wait
        
        # Start proxy
        print(f"🌐 Starting proxy: localhost:{args.port} -> localhost:{args.vllm_port}")
        if args.enable_logging:
            print(f"📝 Logging enabled: {args.log_dir}/")
        else:
            print(f"📝 Logging disabled (use --enable-logging to enable)")
        
        server = ProxyServer(
            ('0.0.0.0', args.port),
            SimpleProxyHandler,
            args.vllm_port,
            args.log_dir,
            args.verbose,
            args.enable_logging
        )
        
        server.serve_forever()
    
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    finally:
        if vllm_process:
            print("🛑 Stopping vLLM...")
            vllm_process.terminate()
            vllm_process.wait()


if __name__ == "__main__":
    main()

