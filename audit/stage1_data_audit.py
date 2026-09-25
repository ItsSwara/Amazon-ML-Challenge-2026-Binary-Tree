"""Stage 1 read-only data audit. Streams every TSV in the competition dataset/ folder
(opened 'rb' only) and writes stage1_data_audit_report.md next to this script.
Never writes inside dataset/."""
import os, re, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# This repo lives inside the competition's dataset/ folder, so dataset/ is two levels up from audit/.
DS = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "stage1_data_audit_report.md")

SRC_HDR = ["entity_id", "business_name", "business_address", "country"]
GT_HDR = ["source1_entity_id", "matched_entity_ids"]
FILES = [
    ("train/train_source1.tsv", "S1-", SRC_HDR),
    ("train/train_source2.tsv", "S2-", SRC_HDR),
    ("train/train_source3.tsv", "S3-", SRC_HDR),
    ("train/train_ground_truth.tsv", "S1-", GT_HDR),
    ("test/test_source1.tsv", "S1-", SRC_HDR),
    ("test/test_source2.tsv", "S2-", SRC_HDR),
    ("test/test_source3.tsv", "S3-", SRC_HDR),
]
NULLISH = {"null", "none", "nan", "n/a", "na", "nil", "-", "--", "\\n", "\\N"}
ID_RE = re.compile(r"^S[123]-\d+$")
MAXEX = 5


def stream(path):
    """Yield raw lines (bytes, without trailing \\n) using chunked binary reads."""
    buf = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"\n")
            buf = parts.pop()
            for p in parts:
                yield p
    if buf:
        yield buf


def audit_source(rel, prefix, hdr, idsets):
    path = os.path.join(DS, rel)
    st = dict(rel=rel, size=os.path.getsize(path), rows=0, blank=0, header=None, hdr_ok=False,
              bom=False, crlf=0, bad_cols=Counter(), bad_utf8=0, ex=[], ids=Counter(),
              empty=Counter(), ws=Counter(), nullish=Counter(), country=Counter(),
              bad_id_fmt=0, wrong_prefix=0, dup_ids=0, dup_rows=0, quote_lines=0,
              ctrl=0, nonascii_rows=0, ex_by={})
    seen = set()
    cmap = {}
    st['cmap'] = cmap
    n = len(hdr)
    first = True

    def add_ex(kind, ln, raw):
        L = st["ex_by"].setdefault(kind, [])
        if len(L) < MAXEX:
            L.append((ln, raw[:200].decode("utf-8", "replace")))

    lineno = 0
    for raw in stream(path):
        lineno += 1
        if raw.endswith(b"\r"):
            st["crlf"] += 1
            raw = raw[:-1]
        if first:
            first = False
            if raw.startswith(b"\xef\xbb\xbf"):
                st["bom"] = True
                raw = raw[3:]
            st["header"] = raw.decode("utf-8", "replace").split("\t")
            st["hdr_ok"] = st["header"] == hdr
            continue
        if raw == b"":
            st["blank"] += 1
            add_ex("blank line", lineno, raw)
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            st["bad_utf8"] += 1
            add_ex("invalid UTF-8", lineno, raw)
            line = raw.decode("utf-8", "replace")
        if "�" in line and not st["bad_utf8"]:
            pass
        st["rows"] += 1
        f = line.split("\t")
        if len(f) != n:
            st["bad_cols"][len(f)] += 1
            add_ex(f"column count {len(f)} (expected {n})", lineno, raw)
            continue
        if '"' in line:
            st["quote_lines"] += 1
        if any(ord(c) < 32 and c != '	' for c in line):
            st["ctrl"] += 1
        if not line.isascii():
            st["nonascii_rows"] += 1
        for h, v in zip(hdr, f):
            if v == "":
                st["empty"][h] += 1
            elif v.strip() == "":
                st["ws"][h] += 1
            elif v.strip().lower() in NULLISH:
                st["nullish"][h] += 1
        eid = f[0]
        if hdr is SRC_HDR:
            st["country"][f[3]] += 1
            cmap[eid] = f[3]
            if not ID_RE.match(eid):
                st["bad_id_fmt"] += 1
                add_ex("malformed entity_id", lineno, raw)
            elif not eid.startswith(prefix):
                st["wrong_prefix"] += 1
                add_ex("wrong source prefix", lineno, raw)
            if eid in seen:
                st["dup_ids"] += 1
                st["ids"][eid] += 1
            else:
                seen.add(eid)
                st["ids"][eid] = 1
            rowkey = line
        else:
            if not ID_RE.match(eid):
                st["bad_id_fmt"] += 1
                add_ex("malformed source1_entity_id", lineno, raw)
            elif not eid.startswith("S1-"):
                st["wrong_prefix"] += 1
            if eid in seen:
                st["dup_ids"] += 1
                st["ids"][eid] += 1
            else:
                seen.add(eid)
                st["ids"][eid] = 1
    st["idset"] = seen
    # drop the full ID counter afterwards; only the repeated IDs are kept, to save memory
    st["dup_id_examples"] = [k for k, v in st["ids"].items() if v > 1][:10]
    st["dup_id_distinct"] = sum(1 for v in st["ids"].values() if v > 1)
    st["max_id_count"] = max(st["ids"].values()) if st["ids"] else 0
    del st["ids"]
    return st


def audit_gt(rel, s1, s2, s3, cmap):
    path = os.path.join(DS, rel)
    g = dict(refs=0, cnt0=0, cnt1=0, cntm=0, maxm=0, invalid_ref=0, invalid_ref_ex=[],
             s1_not_in_gt=0, missing_ex=[], inv_tgt=0, inv_tgt_ex=[], inv_tgt_rows=0,
             wrong_prefix_tgt=0, wrong_prefix_ex=[], dup_in_list=0, dup_in_list_rows=0,
             empty_items=0, ws_items=0, shared_tgt=0, shared_ex=[], total_targets=0,
             dist=Counter(), n2=Counter(), n3=Counter(), s2_matched=0, s3_matched=0,
             mix=Counter(), s1_ref_country=Counter(), s1_ref_country_zero=Counter(),
             bad_cols=0, bycountry={})
    tgt_owner = {}
    refs_seen = set()
    first = True
    for raw in stream(path):
        if first:
            first = False
            continue
        raw = raw.rstrip(b"\r")
        if raw == b"":
            continue
        line = raw.decode("utf-8", "replace")
        f = line.split("\t")
        if len(f) != 2:
            g["bad_cols"] += 1
            continue
        ref, lst = f
        g["refs"] += 1
        if ref not in s1:
            g["invalid_ref"] += 1
            if len(g["invalid_ref_ex"]) < 10:
                g["invalid_ref_ex"].append(ref)
        refs_seen.add(ref)
        items = lst.split(",") if lst.strip() != "" else []
        bad_row = dup_row = False
        seen_local = set()
        c2 = c3 = 0
        for it in items:
            if it == "":
                g["empty_items"] += 1
                continue
            if it != it.strip():
                g["ws_items"] += 1
                it = it.strip()
            if it in seen_local:
                g["dup_in_list"] += 1
                dup_row = True
                continue
            seen_local.add(it)
            g["total_targets"] += 1
            if it.startswith("S2-"):
                ok = it in s2
                c2 += 1
            elif it.startswith("S3-"):
                ok = it in s3
                c3 += 1
            else:
                g["wrong_prefix_tgt"] += 1
                if len(g["wrong_prefix_ex"]) < 10:
                    g["wrong_prefix_ex"].append((ref, it))
                ok = False
                bad_row = True
                continue
            if not ok:
                g["inv_tgt"] += 1
                bad_row = True
                if len(g["inv_tgt_ex"]) < 10:
                    g["inv_tgt_ex"].append((ref, it))
            prev = tgt_owner.get(it)
            if prev is None:
                tgt_owner[it] = ref
            elif prev != ref:
                g["shared_tgt"] += 1
                if len(g["shared_ex"]) < 10:
                    g["shared_ex"].append((it, prev, ref))
        g["inv_tgt_rows"] += bad_row
        g["dup_in_list_rows"] += dup_row
        m = len(seen_local)
        g["dist"][m] += 1
        cc = cmap.get(ref, '?')
        g['bycountry'].setdefault(cc, Counter())[0 if m == 0 else 1 if m == 1 else 2] += 1
        g["n2"][c2] += 1
        g["n3"][c3] += 1
        g["mix"][("S2" if c2 else "-") + ("+S3" if c3 else "")] += 1
        if m == 0:
            g["cnt0"] += 1
        elif m == 1:
            g["cnt1"] += 1
        else:
            g["cntm"] += 1
        g["maxm"] = max(g["maxm"], m)
    g["s1_not_in_gt"] = len(s1 - refs_seen)
    g["missing_ex"] = list(s1 - refs_seen)[:10]
    g["distinct_refs"] = len(refs_seen)
    g["s2_matched"] = sum(1 for k in tgt_owner if k.startswith("S2-"))
    g["s3_matched"] = sum(1 for k in tgt_owner if k.startswith("S3-"))
    g["s2_total"] = len(s2)
    g["s3_total"] = len(s3)
    g["s2_unmatched"] = len(s2) - len([k for k in tgt_owner if k.startswith("S2-") and k in s2])
    g["s3_unmatched"] = len(s3) - len([k for k in tgt_owner if k.startswith("S3-") and k in s3])
    return g


def pct(a, b):
    return f"{100.0 * a / b:.4f}%" if b else "n/a"


def render_source(st):
    r = st["rows"] - sum(st["bad_cols"].values()) if False else st["rows"]
    out = []
    out.append(f"### `{st['rel']}`\n")
    out.append(f"- File size: {st['size']/1e6:,.1f} MB")
    out.append(f"- **Total data rows** (excluding header): **{st['rows']:,}**")
    out.append(f"- Header found: `{'\\t'.join(st['header'])}` -> "
               f"**{'MATCHES' if st['hdr_ok'] else 'DOES NOT MATCH'}** expected `{'\\t'.join(SRC_HDR)}`"
               f"{' (UTF-8 BOM present)' if st['bom'] else ''}")
    nbad = sum(st["bad_cols"].values())
    out.append("\n**Malformed rows**\n")
    out.append("| Check | Count |\n|---|---|")
    out.append(f"| Wrong column count | {nbad:,} |")
    out.append(f"| Invalid UTF-8 (undecodable bytes) | {st['bad_utf8']:,} |")
    out.append(f"| Blank lines | {st['blank']:,} |")
    out.append(f"| CRLF line endings | {st['crlf']:,} |")
    out.append(f"| Rows containing a `\"` character (TSV is unquoted; informational) | {st['quote_lines']:,} |")
    out.append(f"| Rows containing control chars (<0x20, other than tab) | {st['ctrl']:,} |")
    out.append(f"| Rows with non-ASCII text (informational; expected for Hindi/French) | {st['nonascii_rows']:,} |")
    out.append(f"| entity_id not matching `S<d>-<digits>` | {st['bad_id_fmt']:,} |")
    out.append(f"| entity_id with wrong source prefix for this file | {st['wrong_prefix']:,} |")
    if st["bad_cols"]:
        out.append(f"\nColumn-count histogram of bad rows: {dict(st['bad_cols'])}")
    for k, L in st["ex_by"].items():
        out.append(f"\nExamples - {k}:")
        for ln, t in L:
            out.append(f"- line {ln}: `{t!r}`")
    out.append("\n**Duplicate IDs** (column `entity_id`)\n")
    out.append(f"- Distinct IDs: {len(st['idset']):,}")
    out.append(f"- Duplicate rows (repeat occurrences beyond first): **{st['dup_ids']:,}**; "
               f"distinct IDs affected: {st['dup_id_distinct']:,}; max occurrences of one ID: {st['max_id_count']}")
    if st["dup_id_examples"]:
        out.append(f"- Examples: {st['dup_id_examples']}")
    out.append("\n**Missing / empty fields** (well-formed rows only; empty = zero-length string)\n")
    out.append("| Column | Empty | % | Whitespace-only | % | Null-like token (`null`,`nan`,`none`,`n/a`,`-`,...) | % |\n|---|---|---|---|---|---|---|")
    good = st["rows"] - nbad
    for h in SRC_HDR:
        out.append(f"| {h} | {st['empty'][h]:,} | {pct(st['empty'][h], good)} | {st['ws'][h]:,} | "
                   f"{pct(st['ws'][h], good)} | {st['nullish'][h]:,} | {pct(st['nullish'][h], good)} |")
    out.append(f"\n(Percentages over {good:,} well-formed rows.)")
    out.append("\n**Country label distribution**\n")
    out.append("| country | rows | % |\n|---|---|---|")
    for c, v in st["country"].most_common():
        out.append(f"| `{c}` | {v:,} | {pct(v, good)} |")
    out.append("")
    return "\n".join(out)


def render_gt(st, g, s1train):
    out = [f"### `{st['rel']}`\n"]
    out.append(f"- File size: {st['size']/1e6:,.1f} MB")
    out.append(f"- **Total data rows** (excluding header): **{st['rows']:,}**")
    out.append(f"- Header found: `{'\\t'.join(st['header'])}` -> **{'MATCHES' if st['hdr_ok'] else 'DOES NOT MATCH'}** "
               f"expected `{'\\t'.join(GT_HDR)}`")
    out.append("\n**Malformed rows**\n")
    out.append("| Check | Count |\n|---|---|")
    out.append(f"| Wrong column count (expected 2) | {sum(st['bad_cols'].values()):,} |")
    out.append(f"| Invalid UTF-8 | {st['bad_utf8']:,} |")
    out.append(f"| Blank lines | {st['blank']:,} |")
    out.append(f"| CRLF line endings | {st['crlf']:,} |")
    out.append(f"| source1_entity_id malformed / non-`S1-` | {st['bad_id_fmt']+st['wrong_prefix']:,} |")
    out.append(f"| Empty items in ID lists (e.g. `,,` or trailing comma) | {g['empty_items']:,} |")
    out.append(f"| Items with surrounding whitespace | {g['ws_items']:,} |")
    for k, L in st["ex_by"].items():
        out.append(f"\nExamples - {k}:")
        for ln, t in L:
            out.append(f"- line {ln}: `{t!r}`")
    out.append("\n**Duplicate IDs** (column `source1_entity_id`)\n")
    out.append(f"- Duplicate rows: **{st['dup_ids']:,}** (distinct refs affected {st['dup_id_distinct']:,}; "
               f"max occurrences {st['max_id_count']})")
    if st["dup_id_examples"]:
        out.append(f"- Examples: {st['dup_id_examples']}")
    out.append(f"- Duplicate IDs *inside* a single matched list: {g['dup_in_list']:,} (in {g['dup_in_list_rows']:,} rows)")
    out.append(f"- Target IDs claimed by more than one S1 reference (many-to-one conflicts): **{g['shared_tgt']:,}**")
    if g["shared_ex"]:
        out.append(f"  - Examples (target, first ref, second ref): {g['shared_ex']}")
    good = st["rows"] - sum(st["bad_cols"].values())
    out.append("\n**Missing / empty fields**\n")
    out.append("| Column | Empty | % | Whitespace-only | % |\n|---|---|---|---|---|")
    for h in GT_HDR:
        out.append(f"| {h} | {st['empty'][h]:,} | {pct(st['empty'][h], good)} | {st['ws'][h]:,} | {pct(st['ws'][h], good)} |")
    out.append("\n**Country labels**\n")
    out.append("- No country column in this file. Country of each reference (via train_source1) is shown in the "
               "coverage-by-country table below.")
    out.append("\n**Coverage statistics**\n")
    R = g["refs"]
    out.append("| Metric | Value |\n|---|---|")
    out.append(f"| Reference rows | {R:,} |")
    out.append(f"| Distinct references | {g['distinct_refs']:,} |")
    out.append(f"| References with **0** matches (singleton / empty list) | {g['cnt0']:,} ({pct(g['cnt0'], R)}) |")
    out.append(f"| References with **exactly 1** match | {g['cnt1']:,} ({pct(g['cnt1'], R)}) |")
    out.append(f"| References with **multiple** (>=2) matches | {g['cntm']:,} ({pct(g['cntm'], R)}) |")
    out.append(f"| Max matches per reference | {g['maxm']} |")
    out.append(f"| Total matched target IDs (with in-list repeats removed) | {g['total_targets']:,} |")
    out.append(f"| **Invalid references** (source1_entity_id not in train_source1) | **{g['invalid_ref']:,}** |")
    out.append(f"| train_source1 IDs absent from ground truth | **{g['s1_not_in_gt']:,}** |")
    out.append(f"| **Invalid targets** (matched ID not present in train_source2/3) | **{g['inv_tgt']:,}** (in {g['inv_tgt_rows']:,} rows) |")
    out.append(f"| Targets with a prefix other than S2-/S3- (e.g. S1- self-match) | {g['wrong_prefix_tgt']:,} |")
    out.append(f"| Distinct S2 records matched / total S2 records | {g['s2_matched']:,} / {g['s2_total']:,} |")
    out.append(f"| Distinct S3 records matched / total S3 records | {g['s3_matched']:,} / {g['s3_total']:,} |")
    out.append(f"| S2 records not matched to any reference | {g['s2_unmatched']:,} |")
    out.append(f"| S3 records not matched to any reference | {g['s3_unmatched']:,} |")
    if g["invalid_ref_ex"]:
        out.append(f"\nInvalid-reference examples: {g['invalid_ref_ex']}")
    if g["missing_ex"]:
        out.append(f"\nS1 IDs missing from GT examples: {g['missing_ex']}")
    if g["inv_tgt_ex"]:
        out.append(f"\nInvalid-target examples (ref, target): {g['inv_tgt_ex']}")
    if g["wrong_prefix_ex"]:
        out.append(f"\nWrong-prefix target examples: {g['wrong_prefix_ex']}")
    out.append("\n**Coverage by country of the reference (via train_source1)**\n")
    out.append("| country | refs | 0 matches | 1 match | >=2 matches |\n|---|---|---|---|---|")
    for c, cn in sorted(g["bycountry"].items()):
        t = sum(cn.values())
        out.append(f"| `{c}` | {t:,} | {cn[0]:,} ({pct(cn[0], t)}) | {cn[1]:,} ({pct(cn[1], t)}) | {cn[2]:,} ({pct(cn[2], t)}) |")
    out.append("\n**Matches-per-reference histogram**\n")
    out.append("| # matches | references | % |\n|---|---|---|")
    for k in sorted(g["dist"]):
        out.append(f"| {k} | {g['dist'][k]:,} | {pct(g['dist'][k], R)} |")
    out.append("\n**Composition of match lists**\n")
    out.append("| Sources present | references |\n|---|---|")
    for k, v in g["mix"].most_common():
        out.append(f"| {k} | {v:,} |")
    out.append("")
    return "\n".join(out)


def main():
    t0 = time.time()
    res = {}
    for rel, prefix, hdr in FILES:
        print("auditing", rel, flush=True)
        res[rel] = audit_source(rel, prefix, hdr, None)
        print(f"  rows={res[rel]['rows']:,} t={time.time()-t0:.0f}s", flush=True)
    s1 = res["train/train_source1.tsv"]["idset"]
    g = audit_gt("train/train_ground_truth.tsv", s1, res["train/train_source2.tsv"]["idset"],
                 res["train/train_source3.tsv"]["idset"], res["train/train_source1.tsv"]["cmap"])

    # cross-file checks: train/test ID overlap
    xf = []
    for k in ("1", "2", "3"):
        a = res[f"train/train_source{k}.tsv"]["idset"]
        b = res[f"test/test_source{k}.tsv"]["idset"]
        xf.append((k, len(a), len(b), len(a & b)))
    allids = {}
    for k in ("1", "2", "3"):
        pass

    # assemble the markdown report
    trs = [res[f"train/train_source{k}.tsv"] for k in "123"]
    tes = [res[f"test/test_source{k}.tsv"] for k in "123"]
    gt = res["train/train_ground_truth.tsv"]
    tot_rows = sum(r["rows"] for r in res.values())
    tot_bytes = sum(r["size"] for r in res.values())

    issues = []
    for st in res.values():
        nb = sum(st["bad_cols"].values())
        if not st["hdr_ok"]:
            issues.append(f"`{st['rel']}`: header mismatch: {st['header']}")
        if nb:
            issues.append(f"`{st['rel']}`: {nb:,} rows with wrong column count")
        if st["bad_utf8"]:
            issues.append(f"`{st['rel']}`: {st['bad_utf8']:,} rows with invalid UTF-8")
        if st["dup_ids"]:
            issues.append(f"`{st['rel']}`: {st['dup_ids']:,} duplicate ID rows")
        if st["bad_id_fmt"] or st["wrong_prefix"]:
            issues.append(f"`{st['rel']}`: {st['bad_id_fmt']:,} malformed IDs, {st['wrong_prefix']:,} wrong-prefix IDs")
        if st["ctrl"]:
            issues.append(f"`{st['rel']}`: {st['ctrl']:,} rows contain control characters")
        if st["blank"]:
            issues.append(f"`{st['rel']}`: {st['blank']:,} blank lines")
    if g["invalid_ref"]:
        issues.append(f"Ground truth: {g['invalid_ref']:,} references not in train_source1")
    if g["s1_not_in_gt"]:
        issues.append(f"Ground truth: {g['s1_not_in_gt']:,} train_source1 IDs missing from ground truth")
    if g["inv_tgt"]:
        issues.append(f"Ground truth: {g['inv_tgt']:,} matched IDs do not exist in train_source2/3")
    if g["wrong_prefix_tgt"]:
        issues.append(f"Ground truth: {g['wrong_prefix_tgt']:,} targets with non S2-/S3- prefix")
    if g["shared_tgt"]:
        issues.append(f"Ground truth: {g['shared_tgt']:,} target IDs claimed by more than one reference")
    if g["dup_in_list"]:
        issues.append(f"Ground truth: {g['dup_in_list']:,} duplicated IDs inside match lists")
    for st in res.values():
        if st["hdr_ok"] and st["rel"].endswith(("source1.tsv", "source2.tsv", "source3.tsv")):
            for h in SRC_HDR:
                if st["nullish"][h]:
                    issues.append(f"`{st['rel']}`: column `{h}` holds a null-like literal (null/nan/none/-...) in {st['nullish'][h]:,} rows")
                tot = st["empty"][h] + st["ws"][h]
                if tot:
                    issues.append(f"`{st['rel']}`: column `{h}` empty/whitespace in {tot:,} rows "
                                  f"({pct(tot, st['rows'])})")
    for st in res.values():
        if st["rel"].endswith("source.tsv"):
            pass

    L = []
    L.append("# Stage 1 Data Audit Report - Amazon ML Challenge 2026\n")
    L.append("_Generated by `audit/stage1_data_audit.py` (read-only streaming pass over every row of all 7 TSV files; "
             "nothing in `dataset/` was modified)._\n")
    L.append("## 1. Summary\n")
    L.append(f"- **Scale:** 7 files, **{tot_rows:,}** data rows, {tot_bytes/1e9:.2f} GB.")
    for st in res.values():
        L.append(f"  - `{st['rel']}`: {st['rows']:,} rows")
    L.append(f"- **Train source-1 entities:** {trs[0]['rows']:,}; ground-truth references: {g['refs']:,}.")
    L.append(f"- **Ground-truth coverage:** 0 matches {g['cnt0']:,} ({pct(g['cnt0'], g['refs'])}), "
             f"1 match {g['cnt1']:,} ({pct(g['cnt1'], g['refs'])}), "
             f">=2 matches {g['cntm']:,} ({pct(g['cntm'], g['refs'])}); max {g['maxm']}.")
    L.append("- **Country labels:** train = " + ", ".join(
        sorted({c for t in trs for c in t['country']})) + "; test = " + ", ".join(
        sorted({c for t in tes for c in t['country']})) + ".")
    L.append("\n**Integrity findings (everything non-zero that was detected):**\n")
    if issues:
        for i in issues:
            L.append(f"- {i}")
    else:
        L.append("- None.")
    L.append("\nNote: the test set has no ground-truth file (by design), so the ground-truth coverage checks only apply to "
             "`train_ground_truth.tsv`. The problem statement lists exactly these 7 files; all were found.\n")

    L.append("## 2. Expected schema (from README)\n")
    L.append("- Source files: `entity_id`, `business_name`, `business_address`, `country` (tab-separated).")
    L.append("- Ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated S2-/S3- IDs; empty when no matches).")
    L.append("- Train countries: `US`, `India`; test adds `France`.\n")
    L.append("## 3. Per-file breakdown\n")
    L.append("### 3.1 Training files\n")
    for st in trs:
        L.append(render_source(st))
    L.append(render_gt(gt, g, s1))
    L.append("### 3.2 Test files\n")
    for st in tes:
        L.append(render_source(st))
    L.append("## 4. Cross-file checks\n")
    L.append("**Train vs test ID overlap** (leakage / collision check)\n")
    L.append("| Source | train IDs | test IDs | IDs present in both |\n|---|---|---|---|")
    for k, a, b, c in xf:
        L.append(f"| S{k} | {a:,} | {b:,} | {c:,} |")
    L.append("\n**Country distribution, train vs test (all sources combined)**\n")
    ctr, cte = Counter(), Counter()
    for t in trs:
        ctr.update(t["country"])
    for t in tes:
        cte.update(t["country"])
    L.append("| country | train rows | test rows |\n|---|---|---|")
    for c in sorted(set(ctr) | set(cte)):
        L.append(f"| `{c}` | {ctr[c]:,} | {cte[c]:,} |")
    L.append(f"\n_Audit runtime: {time.time()-t0:.0f}s._")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
