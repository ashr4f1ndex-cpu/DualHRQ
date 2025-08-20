import ast, sys, pathlib
bad = ["requests.", "urllib.request", "aiohttp"]
for path in pathlib.Path("lab_v10/src").rglob("*.py"):
    src = path.read_text(encoding="utf-8", errors="ignore")
    try: tree = ast.parse(src, filename=str(path))
    except Exception: continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s = ast.get_source_segment(src, node) or ""
            if any(b in s for b in bad):
                print(f"WARNING import-time IO risk: {path}: {s.strip()}")