"""Qualify an official LiteLLM image using only a disposable rootless database."""
import base64
import http.cookiejar
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


def main(image):
    if os.getuid() == 0:
        raise RuntimeError('Run as tester')
    sensitive = []
    def docker(*args):
        result = subprocess.run(['docker', *args], capture_output=True, timeout=300)
        if result.returncode:
            error = result.stderr.decode(errors='replace')
            for value in sensitive:
                error = error.replace(value, '<synthetic-credential>')
            raise RuntimeError('Synthetic Docker ' + args[0] + ' failed: ' + error[-900:])
        return (result.stdout + (result.stderr if args[0] == 'logs' else b'')).decode().strip()
    if 'rootless' not in docker('info', '--format', '{{json .SecurityOptions}}'):
        raise RuntimeError('Rootless Docker required')
    name = 'one-litellm-' + secrets.token_hex(4)
    master = 'sk-' + secrets.token_urlsafe(32)
    password = 'Synthetic-Only-' + secrets.token_urlsafe(24)
    sensitive.extend((master, password))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='one-litellm-') as temporary:
        config = Path(temporary) / 'config.json'
        config.write_text(json.dumps({'model_list': [{'model_name':'synthetic-approved', 'litellm_params': {
            'model':'openai/synthetic', 'api_base':'http://127.0.0.1:1/v1', 'api_key':'synthetic-only'}}],
            'litellm_settings': {'telemetry':False, 'turn_off_message_logging':True},
            'general_settings': {'store_model_in_db':False, 'store_prompts_in_spend_logs':False}}))
        config.chmod(0o644)
        def request(path, body=None, key=None):
            req = urllib.request.Request('http://127.0.0.1:' + str(port) + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json', **({'Authorization':'Bearer ' + key} if key else {})})
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as response:
                return response.code, {}
        def ready():
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                try:
                    if request('/health/liveliness')[0] == 200: return
                except OSError: pass
                state = json.loads(docker('inspect', '--format', '{{json .State}}', name+'-app'))
                if not state['Running']:
                    break
                time.sleep(1)
            state = json.loads(docker('inspect', '--format', '{{json .State}}', name+'-app'))
            logs = docker('logs', '--tail', '40', name+'-app')
            for value in sensitive:
                logs = logs.replace(value, '<synthetic-credential>')
            print(json.dumps({'running':state['Running'],'exit':state['ExitCode'],'oom':state['OOMKilled']}))
            print(logs[-5000:])
            raise RuntimeError('Official proxy readiness deadline exceeded')
        docker('network', 'create', '--internal', name)
        try:
            docker('run','-d','--name',name+'-pg','--network',name,'--network-alias','postgres',
                '--memory','768m','-e','POSTGRES_USER=litellm','-e','POSTGRES_DB=litellm',
                '-e','POSTGRES_PASSWORD='+password,
                'postgres@sha256:3645570cccdfa447589da9f57dd740faa29b30938e861289a5574b6ca6b03826')
            for _ in range(90):
                try:
                    docker('exec',name+'-pg','pg_isready','-U','litellm'); break
                except RuntimeError: time.sleep(1)
            else: raise RuntimeError('Disposable database not ready')
            docker('run','-d','--name',name+'-app','--network',name,'--memory','1536m',
                '-p','127.0.0.1:'+str(port)+':4000',
                '--mount','type=bind,src='+str(config)+',dst=/one-config.json,readonly',
                '-e','DATABASE_URL=postgresql://litellm:'+password+'@postgres:5432/litellm',
                '-e','LITELLM_MASTER_KEY='+master,'-e','LITELLM_SALT_KEY=synthetic-fixed-salt-00000000000000',
                '-e','LITELLM_TELEMETRY=False','-e','UI_USERNAME=zalazium@zalazium.de',
                '-e','UI_PASSWORD='+password,'-e','LITELLM_LOG=ERROR',
                image,'--config','/one-config.json','--port','4000')
            ready()
            assert request('/key/generate', {'models':['synthetic-approved']})[0] in (401,403)
            status,key = request('/key/generate', {'models':['synthetic-approved'],'max_budget':0.1,
                'rpm_limit':2,'tpm_limit':1000,'max_parallel_requests':1},master)
            assert status == 200 and key['key'].startswith('sk-')
            token = key['key']
            assert request('/v1/models',key=token)[0] == 200
            assert request('/v1/chat/completions',{'model':'unknown','messages':[{'role':'user','content':'synthetic'}]},token)[0] in (400,401,403)
            docker('stop','--time','60',name+'-app')
            assert docker('inspect','--format','{{.State.ExitCode}}',name+'-app') == '0'
            docker('start',name+'-app'); ready()
            assert request('/v1/models',key=token)[0] == 200
            status,record = request('/key/info?key='+urllib.parse.quote(token),key=master)
            assert status == 200 and record['info']['max_budget'] == 0.1 and record['info']['rpm_limit'] == 2
            docker('exec',name+'-pg','pg_amcheck','-U','litellm','--all','--install-missing')
            docker('stop','--time','60',name+'-app')
            assert docker('inspect','--format','{{.State.ExitCode}}',name+'-app') == '0'
            print('PASS official image: fresh/repeated migration, authentication, scoped key, model denial, budgets, persistence and database integrity')
        finally:
            for suffix in ('-app','-pg'):
                subprocess.run(['docker','rm','-f',name+suffix],capture_output=True)
            subprocess.run(['docker','network','rm',name],capture_output=True)


if __name__ == '__main__':
    main(sys.argv[1])
