# EUVIGIL_COMPARE_DISK_CACHE_V1
import functools
import hashlib
import json
import os
import threading
import fcntl
from pathlib import Path

CACHE_DIR = Path(
    "/root/ema/export/raw_stats_cache/_compare_json_v1"
)
CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

LOCAL_LOCK = threading.RLock()

def _key(scope, code, target_scope, target_code):
    return (
        str(scope).strip().lower(),
        str(code).strip().upper(),
        str(target_scope).strip().lower(),
        str(target_code).strip().upper()
    )

def _fingerprint(engine, scope, code):
    # Toujours utiliser le périmètre physique résolu.
    _, _, root = engine._scope_path(scope, code)

    h = hashlib.sha256()

    for name in (
        "icsr",
        "scope_links",
        "scope_substances",
        "reactions"
    ):
        source = root / name

        if not source.exists():
            raise FileNotFoundError(source)

        files = (
            sorted(source.rglob("*.parquet"))
            if source.is_dir()
            else [source]
        )

        if not files:
            raise RuntimeError(f"Source vide : {source}")

        h.update(name.encode())

        for file in files:
            st = file.stat()
            h.update(str(file.relative_to(root)).encode())
            h.update(
                f":{st.st_size}:{st.st_mtime_ns}".encode()
            )

    # EUVIGIL_COMPARE_CODE_FINGERPRINT_V2
    # Invalidation si le calcul comparatif change.
    import inspect
    code = inspect.unwrap(engine.compare).__code__
    # EUVIGIL_COMPARE_STABLE_FINGERPRINT_V3
    # Sérialisation stable du code, sans adresses mémoire.
    import marshal
    h.update(b"EUVIGIL_COMPARE_CODE_V3")
    h.update(marshal.dumps(code))

    return h.hexdigest()

def install(engine):
    original = engine.compare

    @functools.wraps(original)
    def cached_compare(
        scope, code, target_scope, target_code
    ):
        args = _key(
            scope, code, target_scope, target_code
        )

        try:
            fingerprint = _fingerprint(
                engine, args[0], args[1]
            )
        except Exception:
            # Ne jamais casser une comparaison à cause du cache.
            return original(*args)

        key = hashlib.sha256(
            json.dumps(args).encode()
        ).hexdigest()

        path = CACHE_DIR / f"{key}.json"
        lock_path = CACHE_DIR / f"{key}.lock"

        def read_cache():
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)

                if (
                    payload.get("version") == 1
                    and payload.get("arguments") == list(args)
                    and payload.get("fingerprint") == fingerprint
                    and isinstance(payload.get("result"), dict)
                ):
                    return payload["result"]
            except (OSError, ValueError, TypeError):
                pass

            return None

        result = read_cache()
        if result is not None:
            return result

        try:
            with LOCAL_LOCK:
                with open(lock_path, "a+b") as lock:
                    os.chmod(lock_path, 0o600)
                    fcntl.flock(lock, fcntl.LOCK_EX)

                    result = read_cache()
                    if result is not None:
                        return result

                    result = original(*args)

                    # Si les sources ont changé pendant le calcul,
                    # ne pas enregistrer de résultat périmé.
                    if _fingerprint(
                        engine, args[0], args[1]
                    ) != fingerprint:
                        return result

                    payload = {
                        "version": 1,
                        "arguments": list(args),
                        "fingerprint": fingerprint,
                        "result": result
                    }

                    tmp = path.with_name(
                        path.name + f".{os.getpid()}.tmp"
                    )

                    try:
                        with tmp.open(
                            "w", encoding="utf-8"
                        ) as f:
                            json.dump(
                                payload,
                                f,
                                ensure_ascii=False,
                                allow_nan=True
                            )
                            f.flush()
                            os.fsync(f.fileno())

                        os.chmod(tmp, 0o600)
                        os.replace(tmp, path)

                    finally:
                        tmp.unlink(missing_ok=True)

                    return result

        except (OSError, TypeError, ValueError):
            # Le stockage est facultatif.
            return original(*args)

    engine.compare = cached_compare
    print(
        "[compare-cache] Cache persistant installé",
        flush=True
    )
