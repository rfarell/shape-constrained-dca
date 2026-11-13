import json
import tarfile

import pytest
from joint_production.artifacts import capture_run, sha256_file, write_json


def test_uncommitted_source_is_preserved_and_run_is_immutable(tmp_path):
    root = tmp_path / 'repo'
    (root / 'src').mkdir(parents=True)
    source = root / 'src/model.py'
    source.write_text('answer = 42\n')
    config = root / 'config.toml'
    config.write_text('seed = 42\n')
    data = root / 'data.csv'
    data.write_text('immutable raw\n')
    split = root / 'split.json'
    split.write_text('{}\n')
    out = root / 'build/run1'
    manifest = capture_run(root, out, config=config, data=data, split=split, seeds=[42])
    assert manifest['git_commit'] is None
    assert manifest['git_dirty']
    assert manifest['source_hashes']['src/model.py'] == sha256_file(source)
    source.write_text('answer = 99\n')
    with tarfile.open(out / 'source.tar.gz') as archive:
        assert archive.extractfile('src/model.py').read() == b'answer = 42\n'
    before = (out / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        capture_run(root, out, config=config, data=data, split=split, seeds=[42])
    assert (out / 'manifest.json').read_bytes() == before
    assert json.loads(before)['data_sha256'] == sha256_file(data)


def test_json_outputs_reject_nonfinite_numbers_and_overwrites(tmp_path):
    output = tmp_path / 'result.json'
    write_json(output, {'metric': 1})
    with pytest.raises(FileExistsError):
        write_json(output, {'metric': 2})
    with pytest.raises(ValueError):
        write_json(tmp_path / 'bad.json', {'metric': float('nan')})
    assert not (tmp_path / 'bad.json').exists()
