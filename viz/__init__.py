"""viz/ — figure rendering helpers for FOSAE outputs (SPEC §I12).

Per V6 / C12: each generated figure shows ONE concept. The sibling
`<name>.caption.md` (What / Why / How to read) was retracted -- it created
hundreds of stray files -- and is written only under `VIZ_CAPTIONS=1`. The
callers still pass the caption text, so setting that variable restores it.
See viz/io.py.
"""
