# agent.py — thin "agent" wrapper around drive_ingest + harmonizer

from __future__ import annotations
import os, io, datetime as _dt
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import harmonizer as hz
import drive_ingest as din


@dataclass
class AgentState:
    last_disease_query: Optional[str] = None
    last_plan: Optional[Dict[str, Any]] = None
    last_run_id: Optional[str] = None
    last_out: Optional[Dict[str, Any]] = None
    last_multi: Optional[Dict[str, Any]] = None
    base_out_dir: str = "out"


class HarmonizationAgent:
    """
    Very small 'agent' that exposes high-level actions:
      - search_disease_on_drive
      - run_harmonization_from_plan
      - summarize_last_run
    You can call these directly OR route natural language into them.
    """
    def __init__(self, state: Optional[AgentState] = None):
        self.state = state or AgentState()

    # ---------- core tools / actions ----------

    def search_disease_on_drive(
        self,
        sa_json_bytes: bytes,
        deg_root_link_or_id: str,
        disease_query: str,
    ) -> str:
        """
        Uses DriveClient + make_ingest_plan to find datasets for a disease.
        """
        self.state.last_disease_query = disease_query.strip()

        drv = din.DriveClient.from_service_account_bytes(sa_json_bytes)
        plan = din.make_ingest_plan(drv, deg_root_link_or_id, disease_query=disease_query.strip())
        self.state.last_plan = plan

        mode = plan.get("mode")
        if mode == "none":
            return f"❌ No datasets found for query: '{disease_query}'. Reason: {plan.get('reason','')}"
        elif mode == "single":
            s = plan["single"]
            return (
                f"✅ Found **1 dataset** for '{disease_query}':\n"
                f"- Counts: `{s['counts_name']}`\n"
                f"- Meta: `{s['meta_name']}`\n"
                f"Use: `run harmonization` to process it."
            )
        elif mode == "multi_dataset":
            ds = plan["datasets"]
            preview = "\n".join(
                f"- {d['label']}   [counts: {d['counts_name']} | meta: {d['meta_name']}]"
                for d in ds[:10]
            )
            extra = "" if len(ds) <= 10 else f"\n… and {len(ds) - 10} more."
            return (
                f"✅ Found **{len(ds)} datasets** for '{disease_query}'.\n"
                f"{preview}{extra}\n\n"
                "Use: `run harmonization` to run multi-dataset harmonization + meta-analysis."
            )
        else:
            return f"⚠️ Unexpected ingest mode: {mode}"

    def run_harmonization_from_plan(
        self,
        pca_topk_features: int = 5000,
        make_nonlinear: bool = True,
        combine_minoverlap_genes: int = 3000,
    ) -> str:
        """
        Run harmonizer based on the last ingest plan.
        """
        plan = self.state.last_plan
        if not plan:
            return "❌ No ingest plan found. First say something like: `search diabetes on drive`."

        out_root = os.path.join(
            self.state.base_out_dir,
            _dt.datetime.now().strftime("agent_%Y%m%d_%H%M%S"),
        )

        mode = plan.get("mode")
        if mode == "single":
            s = plan["single"]
            kwargs = {
                "single_expression_file": s["counts"],
                "single_expression_name_hint": s["counts_name"],
                "metadata_file": s["meta"],
                "metadata_name_hint": s["meta_name"],
                "metadata_id_cols": ["sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"],
                "metadata_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "metadata_batch_col": None,
                "out_root": out_root,
                "pca_topk_features": int(pca_topk_features),
                "make_nonlinear": make_nonlinear,
            }
            out = hz.run_pipeline(**kwargs)
            self.state.last_run_id = os.path.basename(out_root)
            self.state.last_out = out
            self.state.last_multi = None

            return (
                f"🚀 Ran harmonization for **1 dataset**.\n"
                f"- Outdir: `{out['outdir']}`\n"
                f"- Figures in: `{out['figdir']}`\n"
                f"- Zip bundle: `{out['zip']}`"
            )

        elif mode == "multi_dataset":
            ds = plan["datasets"]
            datasets_arg = []
            for d in ds:
                datasets_arg.append({
                    "geo": d["label"],
                    "counts": d["counts"],
                    "counts_name": d["counts_name"],
                    "meta": d["meta"],
                    "meta_name": d["meta_name"],
                    "meta_id_cols": ["sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"],
                    "meta_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                    "meta_batch_col": None,
                })
            kwargs_multi = {
                "datasets": datasets_arg,
                "attempt_combine": True,
                "combine_minoverlap_genes": int(combine_minoverlap_genes),
                "out_root": out_root,
                "pca_topk_features": int(pca_topk_features),
                "make_nonlinear": make_nonlinear,
            }
            multi_out = hz.run_pipeline_multi(**kwargs_multi)
            self.state.last_run_id = os.path.basename(out_root)
            self.state.last_out = (multi_out.get("combined") or next(iter(multi_out["runs"].values())))
            self.state.last_multi = multi_out

            decision = multi_out.get("combine_decision", {}) or {}
            combined = decision.get("combined", False)
            overlap = decision.get("overlap_genes", 0)
            n_ds = len(multi_out.get("runs", {}))

            return (
                f"🚀 Ran harmonization for **{n_ds} datasets**.\n"
                f"- Combined run created: {'Yes' if combined else 'No'}\n"
                f"- Overlapping genes: {overlap}\n"
                f"- Meta-analysis dir: `{multi_out.get('meta_dir', '')}`"
            )
        else:
            return f"⚠️ Cannot run harmonization: unsupported mode '{mode}'."

    def summarize_last_run(self) -> str:
        """
        Simple text summary of last run based on report.json/meta-analysis.
        """
        out = self.state.last_out
        if not out:
            return "ℹ️ No completed harmonization found yet."

        # read report.json if present
        import json
        report = {}
        try:
            with open(out["report_json"], "r") as fh:
                report = json.load(fh)
        except Exception:
            pass

        qc = report.get("qc", {})
        shp = report.get("shapes", {})
        lines = [
            f"Run ID: {self.state.last_run_id}",
            f"Genes: {shp.get('genes','?')}, Samples: {shp.get('samples','?')}",
            f"Platform: {qc.get('platform','?')}",
            f"Harmonization mode: {qc.get('harmonization_mode','?')}",
            f"Zero fraction: {qc.get('zero_fraction', float('nan')):.2f}"
              if isinstance(qc.get('zero_fraction'), (int,float)) else "Zero fraction: ?",
        ]

        # optional multi-dataset meta summary
        if self.state.last_multi and self.state.last_multi.get("meta_dir"):
            meta_dir = self.state.last_multi["meta_dir"]
            summary_csv = os.path.join(meta_dir, "final_analysis_summary.csv")
            try:
                import pandas as pd
                m = pd.read_csv(summary_csv, index_col=0)
                top_genes = ", ".join(list(m.index[:5]))
                lines.append(f"Top meta-analysis genes (first 5): {top_genes}")
            except Exception:
                pass

        return "\n".join(lines)

    # ---------- OPTIONAL: naive NL router (no external LLM) ----------

    def handle_command(
        self,
        text: str,
        *,
        sa_json_bytes: Optional[bytes] = None,
        deg_root_link_or_id: Optional[str] = None,
    ) -> str:
        """
        Simple keyword-based router:
          - 'search' + 'drive' or 'disease' => search_disease_on_drive
          - 'run' or 'harmoniz'            => run_harmonization_from_plan
          - 'summary' or 'summarize'       => summarize_last_run
        You can replace this with an LLM that chooses which tool to call.
        """
        t = text.lower().strip()

        if ("search" in t or "find" in t) and ("drive" in t or "disease" in t):
            if not sa_json_bytes or not deg_root_link_or_id:
                return "❌ To search Drive I need the service-account JSON and deg_data folder link/ID."
            # extract disease query: very naive ("search <stuff> on drive")
            # You can make this smarter or pass disease separately from UI.
            q = t
            for k in ["search", "find", "on drive", "in drive", "disease"]:
                q = q.replace(k, "")
            q = q.strip()
            if not q:
                q = self.state.last_disease_query or ""
            if not q:
                return "Please specify a disease, e.g. `search diabetes on drive`."
            return self.search_disease_on_drive(sa_json_bytes, deg_root_link_or_id, q)

        if "run" in t or "harmoniz" in t:
            return self.run_harmonization_from_plan()

        if "summary" in t or "summarize" in t or "report" in t:
            return self.summarize_last_run()

        return (
            "I didn't recognize that command.\n"
            "You can say for example:\n"
            "• `search diabetes on drive`\n"
            "• `run harmonization`\n"
            "• `show summary`"
        )
