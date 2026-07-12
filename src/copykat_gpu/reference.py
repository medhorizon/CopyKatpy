"""Gene-coordinate reference handling."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import pandas as pd
_REQUIRED_COLUMNS = {"chromosome", "start", "end"}

def _normalise_chromosome(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.replace("^chr", "", regex=True, case=False)
    return normalized.replace({"X": "23", "Y": "24", "MT": "25", "M": "25"}).astype(int)

def load_gene_coordinates(reference: str | Path | pd.DataFrame, gene_id_type: Literal["symbol", "ensembl"] = "symbol") -> pd.DataFrame:
    """Load a validated gene-coordinate table."""
    if isinstance(reference, pd.DataFrame):
        table = reference.copy()
    else:
        path = Path(reference)
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    candidates = ("gene", "symbol", "hgnc_symbol") if gene_id_type == "symbol" else ("ensembl_id", "ensembl_gene_id", "gene")
    gene_column = next((name for name in candidates if name in table.columns), None)
    if gene_column is None:
        raise ValueError(f"Reference needs one of {candidates}; got {list(table.columns)}.")
    missing = _REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"Reference is missing coordinate columns: {sorted(missing)}.")
    result = table.loc[:, [gene_column, "chromosome", "start", "end"]].copy()
    result.columns = ["gene", "chromosome", "start", "end"]
    result["gene"] = result["gene"].astype(str)
    result["chromosome"] = _normalise_chromosome(result["chromosome"])
    result[["start", "end"]] = result[["start", "end"]].apply(pd.to_numeric, errors="coerce")
    result = result.dropna().drop_duplicates("gene")
    result = result.loc[result["chromosome"].between(1, 24)]
    result[["start", "end"]] = result[["start", "end"]].astype(int)
    return result.sort_values(["chromosome", "start", "end", "gene"], kind="stable").reset_index(drop=True)
