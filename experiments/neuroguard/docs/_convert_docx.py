# -*- coding: utf-8 -*-
"""DAH2026_docs 내 .md → .docx 일괄 변환 (pandoc via pypandoc)."""
import pypandoc, os, glob

base = os.path.dirname(os.path.abspath(__file__))
files = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(base, "*.md")))

ok = 0
for f in files:
    md = os.path.join(base, f + ".md")
    dx = os.path.join(base, f + ".docx")
    if os.path.exists(md):
        pypandoc.convert_file(md, "docx", format="gfm", outputfile=dx)
        print(f"OK: {os.path.getsize(dx):,} bytes -> {f}.docx")
        ok += 1
    else:
        print(f"MISSING: {f}.md")
print(f"DONE: {ok}/{len(files)} converted")
