import copy, os, sys, threading, time
SKIP = {"available", "compare"}  # EUVIGIL_COMPARE_NO_RAM_MEMO_V1

def _trim(name, result, keep=50):
    # La page n'affiche que le top 10 de "literature" (130 000 entrées, 33 Mo).
    try:
        dims = result.get("dimensions") if isinstance(result, dict) else None
        if isinstance(dims, dict) and isinstance(dims.get("literature"), list):
            dims["literature"] = sorted(dims["literature"], key=lambda x: -(x.get("count") or 0))[:keep]
    except Exception:
        pass
    return result

def _wrap(engine):
    memo, inflight, lock = {}, {}, threading.Lock()
    def make(fn, name):
        def wrapped(*args, **kwargs):
            key = (name, repr(args), repr(sorted(kwargs.items())))
            with lock:
                if key in memo:
                    return copy.deepcopy(memo[key])
                event = inflight.get(key)
                owner = event is None
                if owner:
                    event = inflight[key] = threading.Event()
            if not owner:
                event.wait()
                with lock:
                    if key in memo:
                        return copy.deepcopy(memo[key])
                return fn(*args, **kwargs)
            try:
                result = _trim(name, fn(*args, **kwargs))
                with lock:
                    memo[key] = result
                return copy.deepcopy(result)
            finally:
                with lock:
                    inflight.pop(key, None)
                event.set()
        return wrapped
    names = []
    for name in dir(engine):
        if name.startswith("_") or name in SKIP:
            continue
        fn = getattr(engine, name, None)
        if callable(fn):
            setattr(engine, name, make(fn, name)); names.append(name)
    print("[memo] " + ", ".join(names), file=sys.stderr, flush=True)

def _port():
    a = sys.argv
    for i, x in enumerate(a):
        if x == "--port" and i + 1 < len(a): return a[i + 1]
        if x.startswith("--port="): return x.split("=", 1)[1]
    return ""

def _warmup(engine):
    time.sleep(2)
    jobs = []
    for scope in ("l1", "l2"):
        d = engine.root / scope
        if d.is_dir():
            jobs += [(scope, p.name) for p in sorted(d.iterdir()) if p.is_dir() and "." not in p.name]
    t0 = time.time()
    for n, (scope, code) in enumerate(jobs, 1):
        try:
            engine.analyse(scope, code)
            import urllib.request
            for _ in range(10):
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{_port()}/api/analyse-ranking?scope={scope}&code={code}&kind=reactions&limit=100', timeout=300).read()
                    break
                except Exception:
                    time.sleep(3)
        except Exception as e:
            print(f"[warmup] {scope}:{code} échec : {e}", file=sys.stderr, flush=True)
        if n % 10 == 0 or n == len(jobs):
            print(f"[warmup] {n}/{len(jobs)} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

def install(engine):
    _wrap(engine)
    warm = os.environ.get("EUVIGIL_WARMUP") or ("1" if _port() == "8094" else "0")
    if warm == "1":
        threading.Thread(target=_warmup, args=(engine,), daemon=True).start()
