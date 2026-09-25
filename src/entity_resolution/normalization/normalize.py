"""Identical normalization for every source file (S1/S2/S3, train and test).

Output columns (see docs/schemas/normalized_columns.md):
    entity_id, name_norm, legal_suffix, address_norm, country

Missing values are true NaN, never imputed. The ground-truth file is not
normalized here.

Run:  python src/entity_resolution/normalization/normalize.py --source mock
      python src/entity_resolution/normalization/normalize.py --source real
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from unidecode import unidecode  # noqa: E402

from entity_resolution.config import REPO_ROOT, get_dataset_dir, get_mock_dir  # noqa: E402

CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F-\x9F]")
# Same placeholder set as audit/stage1_data_audit.py (plus the empty string).
NULL_TOKENS = {"", "null", "nan", "none", "n/a", "na", "nil", "-", "--", "\\n"}

# Legal-form token -> canonical form. Only trailing tokens of a name are extracted.
LEGAL_SUFFIXES = {
    "corp": "corp", "corporation": "corp",
    "inc": "inc", "incorporated": "inc",
    "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt",
    "public": "pub", "pub": "pub",
    "co": "co", "company": "co",
    "llc": "llc", "lc": "lc", "llp": "llp", "lp": "lp", "lllp": "lllp",
    "pllc": "pllc", "pc": "pc", "plc": "plc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "sa": "sa", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "snc": "snc", "sci": "sci",
    "pty": "pty", "pte": "pte", "srl": "srl", "spa": "spa", "ltda": "ltda", "opc": "opc",
}

# "private"/"public" alone are ordinary words, so they only count as a suffix directly
# before "limited"/"ltd" ("private limited", "public ltd").
_QUALIFIED = r"(?:private|pvt|public|pub)\s+(?:limited|ltd)"
_PLAIN = sorted((t for t in LEGAL_SUFFIXES if t not in {"private", "public"}), key=len, reverse=True)
SUFFIX_RE = re.compile(
    r"^(?P<name>.+?)(?P<suffix>(?:\s+(?:%s|%s))+)$" % (_QUALIFIED, "|".join(map(re.escape, _PLAIN)))
)

OUTPUT_COLUMNS = ["entity_id", "name_norm", "legal_suffix", "address_norm", "country"]

# A UTF-8 lead byte read as a Latin-1 character, then 1-3 continuation bytes (0x80-0xBF).
MOJIBAKE_RUN_RE = re.compile(r"[Â-ô][\u0080-¿]{1,3}")


def _repair_run(match: re.Match) -> str:
    """Decode one mojibake run, or return it unchanged.

    The data was title-cased after being mangled, which turned lead bytes like
    0xE2 ('â') into 0xC2 ('Â'), so when the run does not decode as-is the
    lowercased lead is tried too. Longest decode wins; trailing chars are kept.
    """
    run = match.group()
    for n in range(min(len(run), 4), 1, -1):
        for lead in (run[0], run[0].lower()):
            try:
                return (lead + run[1:n]).encode("latin-1").decode("utf-8") + run[n:]
            except UnicodeError:
                continue
    return run


def repair_mojibake(s: str) -> str:
    """Undo UTF-8 text that was mis-decoded as Latin-1 (e.g. 'PrÃ©sident' -> 'Président').

    First tries the whole string (latin-1 encode, utf-8 decode). If that raises, as it does for
    strings that mix mojibake with legitimate characters, only the mojibake runs are repaired.
    Anything that does not decode is left unchanged; this never raises.
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return MOJIBAKE_RUN_RE.sub(_repair_run, s)


def normalize_text(s: pd.Series) -> pd.Series:
    """Null-coerce, strip control chars, transliterate, lowercase, strip punctuation.

    Works on object dtype because the dotted-initials regex needs lookarounds.
    """
    s = s.astype(object)
    null = s.isna() | s.str.strip().str.lower().isin(NULL_TOKENS)
    s = s.mask(null)

    # Mojibake repair must precede the control-char strip: its continuation bytes are C1 controls.
    suspect = s.notna() & s.str.contains(MOJIBAKE_RUN_RE, regex=True).fillna(False).astype(bool)
    s = s.mask(suspect, s.where(suspect).map(repair_mojibake, na_action="ignore"))

    s = s.str.replace(CONTROL_CHARS_RE, " ", regex=True)

    non_ascii = s.notna() & ~s.str.isascii().fillna(True).astype(bool)
    s = s.mask(non_ascii, s.where(non_ascii).map(unidecode, na_action="ignore"))

    s = s.str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace("'", "", regex=False)
    s = s.str.replace(r"(?<=\b[a-z0-9])\.(?=[a-z0-9]\b)", "", regex=True)  # l.l.c. -> llc
    s = s.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return s.mask(s == "")


def extract_legal_suffix(names: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split trailing legal-form tokens off normalized names.

    Returns (name_without_suffix, canonical_suffix); suffix is NaN when none found.
    A name that consists only of a suffix token is left intact.
    """
    parts = names.str.extract(SUFFIX_RE)
    matched = parts["name"].notna()
    canonical = parts["suffix"].map(
        lambda x: " ".join(LEGAL_SUFFIXES[t] for t in x.split()), na_action="ignore"
    )
    return names.where(~matched, parts["name"]), canonical.astype(object).mask(~matched)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize one source-file DataFrame (raw columns, all str) into the output schema."""
    name_norm, legal_suffix = extract_legal_suffix(normalize_text(df["business_name"]))
    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "name_norm": name_norm.to_numpy(),
            "legal_suffix": legal_suffix.to_numpy(),
            "address_norm": normalize_text(df["business_address"]).to_numpy(),
            "country": df["country"].str.strip().to_numpy(),
        }
    )
    return out.astype({c: "str" for c in OUTPUT_COLUMNS})


def read_raw(path: Path, chunksize: int | None = None):
    """Read a raw source TSV with no pandas-side NaN handling or quoting."""
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8", chunksize=chunksize,
    )


def check_invariants(out: pd.DataFrame, n_in: int) -> None:
    """Assert the schema contract from docs/schemas/normalized_columns.md."""
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == n_in, "row count changed"
    assert out["entity_id"].notna().all() and out["country"].notna().all()
    assert out["entity_id"].is_unique
    clean = r"[a-z0-9]+(?: [a-z0-9]+)*"
    for col in ("name_norm", "legal_suffix", "address_norm"):
        vals = out[col].dropna().astype(object)
        bad = ~vals.str.fullmatch(clean)
        assert not bad.any(), f"{col}: {int(bad.sum())} values not [a-z0-9 ] / trimmed, e.g. {vals[bad].iloc[0]!r}"
    assert out.loc[out["name_norm"].isna(), "legal_suffix"].isna().all()


def normalize_file(src: Path, dst: Path, chunksize: int = 500_000) -> dict:
    """Stream src through normalize_df into a Parquet file; return summary stats."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(c, pa.string()) for c in OUTPUT_COLUMNS])
    rows = 0
    nan = dict.fromkeys(OUTPUT_COLUMNS, 0)
    with pq.ParquetWriter(dst, schema) as writer:
        for chunk in read_raw(src, chunksize=chunksize):
            out = normalize_df(chunk)
            check_invariants(out, len(chunk))
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            rows += len(out)
            for c in OUTPUT_COLUMNS:
                nan[c] += int(out[c].isna().sum())
    return {"rows": rows, "nan": nan}


def source_files(base: Path, mock: bool) -> list[Path]:
    if mock:
        return sorted(base.glob("*_source[123].tsv"))
    return sorted((base / "train").glob("train_source[123].tsv")) + sorted(
        (base / "test").glob("test_source[123].tsv")
    )


def show_examples(src: Path, n: int = 8) -> None:
    raw = next(read_raw(src, chunksize=20_000))
    out = normalize_df(raw)
    pick = out[out["legal_suffix"].notna() | out["address_norm"].isna()].index[:n]
    for i in pick:
        print(f"    raw : {raw.loc[i, 'business_name']!r} | {raw.loc[i, 'business_address']!r}")
        print(f"    norm: name={out.loc[i, 'name_norm']!r} suffix={out.loc[i, 'legal_suffix']!r} "
              f"addr={out.loc[i, 'address_norm']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["mock", "real"], default="mock")
    parser.add_argument("--out-dir", type=Path, help="default: data/processed/normalized[_mock]")
    parser.add_argument("--examples", action="store_true", help="print before/after samples")
    args = parser.parse_args()

    mock = args.source == "mock"
    base = get_mock_dir() if mock else get_dataset_dir()
    out_dir = args.out_dir or REPO_ROOT / "data" / "processed" / ("normalized_mock" if mock else "normalized")

    for src in source_files(base, mock):
        t0 = time.time()
        stats = normalize_file(src, out_dir / f"{src.stem}.parquet")
        n = stats["rows"]
        nan_txt = ", ".join(f"{c}={v:,} ({v / n:.2%})" for c, v in stats["nan"].items() if v)
        print(f"{src.name:22s} {n:>10,} rows  {time.time() - t0:6.1f}s  NaN: {nan_txt or 'none'}")
        if args.examples:
            show_examples(src)
    print(f"invariants OK; wrote {out_dir}")


if __name__ == "__main__":
    main()
