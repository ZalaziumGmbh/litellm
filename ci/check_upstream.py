"""Raise a reviewable CI status for unqualified upstream releases; never promote."""
import json
from pathlib import Path
import urllib.request

record = json.loads(Path('one-upstream.json').read_text())
assert record['repository'] in ('BerriAI/litellm', 'Mintplex-Labs/anything-llm')
request = urllib.request.Request('https://api.github.com/repos/' + record['repository'] + '/releases/latest',
                                 headers={'Accept': 'application/vnd.github+json'})
with urllib.request.urlopen(request, timeout=30) as response:
    latest = json.load(response)
assert not latest['draft'] and not latest['prerelease']
print('Qualified upstream: ' + record['version'])
print('Current stable upstream: ' + latest['tag_name'])
print('Review: ' + latest['html_url'])
if latest['tag_name'] != record['version']:
    raise SystemExit('Upgrade review required: qualify migrations, workflows, backup and rollback before promoting. Current channel preserved.')
print('Qualified upstream is current; application channel unchanged.')
