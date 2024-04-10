
# type: ignore
from hashlib import blake2b
import argparse
import json
from pathlib import Path


def hash_from_creation_log():
    parser = argparse.ArgumentParser(prog="ANI Hash from creation log")
    parser.add_argument("--path")
    args = parser.parse_args()
    path = Path(args.path).resolve()
    with open(path / "creation_log.json", "r") as f:
        config = json.load(f)

    splits = config["splits"]
    md5s = config["source_md5s"]
    folds = config["folds"]
    shuffle_seed = config["shuffle_seed"]
    include_properties = config["include_properties"]
    return _calculate_hash(folds, splits, md5s, shuffle_seed, include_properties)


def _calculate_hash(folds, splits, md5s, shuffle_seed, include_properties):
    hasher = blake2b(digest_size=10)
    hash_components = md5s
    splits_keys = sorted(splits.keys()) if splits is not None else []
    splits_hash = (
        splits_keys + [str(hash(splits[k])) for k in splits_keys]
        if splits is not None
        else [str(0)]
    )
    hash_components.extend(splits_hash + [str(folds) if folds is not None else str(0)])
    hash_components.extend(
        sorted(include_properties) if include_properties is not None else [str(0)]
    )
    hash_components.append(str(shuffle_seed))
    hash_components = bytes("".join(hash_components), "utf-8")
    hasher.update(hash_components)
    digest = hasher.hexdigest()
    return digest
