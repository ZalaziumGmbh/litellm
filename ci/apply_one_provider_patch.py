"""Apply the reviewed speech fixes to the exact official LiteLLM base package."""
import hashlib
import importlib.util
import json
from pathlib import Path


def apply(root, manifest):
    prepared = []
    for name, patch in manifest.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Patch leaves the package')
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != patch['before_sha256']:
            raise ValueError('Unreviewed upstream source: ' + name)
        updated = original.decode()
        for edit in patch['edits']:
            if not edit['before'] or updated.count(edit['before']) != 1:
                raise ValueError('Ambiguous source patch: ' + name)
            updated = updated.replace(edit['before'], edit['after'], 1)
        if hashlib.sha256(updated.encode()).hexdigest() != patch['after_sha256']:
            raise ValueError('Unexpected patched source: ' + name)
        compile(updated, str(path), 'exec')
        prepared.append((path, updated))
    for path, updated in prepared:
        path.write_text(updated)


if __name__ == '__main__':
    root = Path(importlib.util.find_spec('litellm').origin).parent
    apply(root, json.loads(Path(__file__).with_name('one-provider-patch.json').read_text()))
    print('Reviewed speech adapters and explicit-region guard applied to pinned upstream')
