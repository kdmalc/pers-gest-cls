"""
Generate the committed episode files + MANIFEST.json (docs/specs/SPEC_episode_cache.md).

    python system/episodes/make_episodes.py            # write missing files
    python system/episodes/make_episodes.py --check    # verify committed files == regenerated
    python system/episodes/make_episodes.py --force    # overwrite files whose hash differs

Refuses to overwrite a file whose hash differs unless --force is given.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episodes import episode_cache as ec  # noqa: E402


def all_targets():
    t = [("dev", None)]
    for k in range(ec.kfold_split.N_FOLDS):
        t += [("test", k), ("val", k)]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    directory = ec.EPISODE_DIR
    manifest_path = directory / ec.MANIFEST_NAME
    old = json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else {}
    new = dict(old)
    bad = []
    for split, fold in all_targets():
        name = ec.file_name(split, fold)
        obj = ec.generate(split, fold)
        sha = ec.canonical_sha256(obj)
        if args.check:
            if old.get(name) != sha:
                bad.append(name)
            continue
        if name in old and old[name] != sha and not args.force:
            raise SystemExit(f"{name}: hash would change {old[name]} -> {sha}; pass --force")
        ec.write(obj, directory / name)
        new[name] = sha
        print(f"{name}  episodes={len(obj['episodes'])}  sha256={sha[:12]}")
    if args.check:
        if bad:
            raise SystemExit(f"MISMATCH vs manifest: {bad}")
        print(f"OK: {len(old)} files regenerate to the manifest hashes")
        return
    manifest = {"schema_version": ec.SCHEMA_VERSION, "protocol": ec.PROTOCOL,
                "base_seed": ec.BASE_SEED, "files": dict(sorted(new.items()))}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
