from pathlib import Path
import gzip
import hashlib

ROOT = Path(__file__).resolve().parent
FILES = [{'path': 'datasets/dianping/rating.txt', 'sha256': 'e2ce92c6610e5464fef2258b8f06a79f4a14285f1f879e9ba4b67b4797e65717'}, {'path': 'datasets/dianping/processed_data/dianping_rating.csv', 'sha256': '39572021c66b20b347c58d2c8a289ff2992f48f52580d0b8cf6e3bb0853eb73c'}]

for entry in FILES:
    target = ROOT / entry['path']
    archive = target.with_name(target.name + '.gz')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Existing file does not match: {target}')
        print(f'Already restored: {target.relative_to(ROOT)}')
        continue
    data = gzip.decompress(archive.read_bytes())
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise RuntimeError(f'Checksum mismatch: {archive}')
    target.write_bytes(data)
    print(f'Restored: {target.relative_to(ROOT)}')
