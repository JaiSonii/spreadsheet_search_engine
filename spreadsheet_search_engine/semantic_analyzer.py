"""
Semantic Analyzer (High-Priority Improvements Only)
- Reuses existing concept recognition approach (simple synonyms placeholder)
- Enrich embeddings with new metadata:
  - Units (percentage, currency, date)
  - Magnitude bin
  - Dependency summary (count, cross-sheet presence, function tags)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer


class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Minimal concept map (extend later if needed)
        self.concept_map = {
            "revenue": ["sales", "income", "turnover", "receipts"],
            "cost": ["expense", "expenditure", "spending", "cogs"],
            "profit": ["earnings", "net income", "ebitda"],
            "margin": ["markup", "spread", "profit margin"],
            "efficiency": ["productivity", "roi", "roa", "roe", "turnover"],
            "growth": ["increase", "expansion", "yoy", "qoq", "cagr"],
            "budget": ["plan", "forecast", "projection", "target"],
            "actual": ["realized", "achieved", "current"],
            "variance": ["difference", "deviation", "gap"],
            "ratio": ["proportion", "percentage", "rate", "metric"],
        }

    def analyze(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(parsed_data)
        cells = out.get("cells", [])
        for cell in cells:
            # concepts
            cell["concepts"] = self._recognize_concepts(cell)

            # simple importance (headers + formulas)
            cell["importance"] = self._importance(cell)

            # embedding
            cell["embedding"] = self._generate_embedding(cell, out)

        return out

    def _recognize_concepts(self, cell: Dict[str, Any]) -> List[str]:
        concepts: List[str] = []
        text = f"{cell.get('header','')} {cell.get('value','')}".lower()
        for concept, synonyms in self.concept_map.items():
            if concept in text:
                concepts.append(concept)
                continue
            for s in synonyms:
                if s in text:
                    concepts.append(concept)
                    break
        # include sheet concepts if available
        for sc in cell.get("sheet_concepts", []) or []:
            concepts.append(sc)
        # normalize
        seen = set()
        uniq = []
        for c in concepts:
            c = c.lower()
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _importance(self, cell: Dict[str, Any]) -> float:
        score = 0.0
        if cell.get("row") == 1 and (cell.get("data_type") == "text" or cell.get("header")):
            score += 0.3
        if cell.get("formula"):
            score += 0.4
            # small bump for more tags
            t = cell.get("formula_type") or []
            score += min(len(t) * 0.05, 0.2)
        if cell.get("concepts"):
            score += min(0.05 * len(cell["concepts"]), 0.2)
        if cell.get("header"):
            if any(kw in cell["header"].lower() for kw in ["total", "summary", "revenue", "profit", "margin", "variance", "yoy"]):
                score += 0.2
        return min(score, 1.0)

    def _dependency_summary(self, cell: Dict[str, Any]) -> str:
        deps: List[str] = cell.get("dependencies") or []
        if not deps:
            return ""
        cross_sheet = any("!" in d and not d.startswith(f"{cell.get('sheet','')}!") for d in deps)
        fn_tags = ", ".join(cell.get("formula_type") or [])
        parts = []
        parts.append(f"Dependencies: {len(deps)} precedent reference(s).")
        if cross_sheet:
            parts.append("Includes cross-sheet references.")
        if fn_tags:
            parts.append(f"Function tags: {fn_tags}.")
        return " ".join(parts)

    def _units_summary(self, cell: Dict[str, Any]) -> str:
        units = cell.get("units") or {}
        parts: List[str] = []
        if units.get("is_percentage"):
            parts.append("Unit: percentage.")
        if units.get("currency_code") or units.get("currency_symbol"):
            cur = units.get("currency_code") or units.get("currency_symbol")
            parts.append(f"Currency: {cur}.")
        if units.get("is_date"):
            parts.append("Type: date/time.")
        if units.get("magnitude_bin"):
            parts.append(f"Magnitude: ~{units['magnitude_bin']}.")
        return " ".join(parts)

    def _row_context(self, cell: Dict[str, Any], full_data: Dict[str, Any]) -> str:
        row = cell.get("row")
        col = cell.get("column")
        sheet = cell.get("sheet")
        out: List[str] = []
        for c in full_data.get("cells", []):
            if c.get("sheet") != sheet:
                continue
            if c.get("row") == row and c.get("column") != col and c.get("header"):
                val = c.get("value", "")
                if val is None or str(val).strip() == "":
                    continue
                out.append(f"{c['header']}: {val}")
            if len(out) >= 5:
                break
        return "; ".join(out)

    def _generate_embedding(self, cell: Dict[str, Any], full_data: Dict[str, Any]) -> List[float]:
        parts: List[str] = []

        # Identity
        header = cell.get("header")
        if header:
            parts.append(f"Column: {header}.")
        if cell.get("formula"):
            parts.append("Content: This cell is a formula.")
        elif cell.get("value") is not None:
            parts.append(f"Content: Value '{cell['value']}'.")

        # Units and magnitude
        u = self._units_summary(cell)
        if u:
            parts.append(u)

        # Concepts
        if cell.get("concepts"):
            parts.append(f"Business Concepts: {', '.join(cell['concepts'])}.")

        # Row context
        rc = self._row_context(cell, full_data)
        if rc:
            parts.append(f"Row Context: [{rc}].")

        # Location
        sheet = cell.get("sheet")
        if sheet:
            parts.append(f"Location: '{sheet}' sheet.")

        # Dependency summary
        ds = self._dependency_summary(cell)
        if ds:
            parts.append(ds)

        text = " ".join(parts) or "empty cell"
        return self.model.encode(text).tolist()
