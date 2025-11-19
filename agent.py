# agent.py — simple orchestration agent for Drive search + harmonization

import os
import io
import datetime as _dt
import json
from typing import Optional, Dict, Any

import drive_ingest as din
import harmonizer as hz


class HarmonizationAgent:
    """
    Very lightweight "agent" that:
      - Searches Google Drive deg_data for disease keywords
      - Stores the ingest plan
      - Runs harmonization using harmonizer.run_pipeline / run_pipeline_multi
      - Summarizes the last run
    """

    def __init__(self):
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_result: Optional[Dict[str, Any]] = None  # {"out":..., "multi":..., "run_id":...}

    # ------------------------------
    # Public entrypoint
    # ------------------------------
    def handle_command(
        self,
        text: str,
        *,
        sa_json_bytes: Optional[bytes] = None,
        deg_root_link_or_id: Optional[str] = None,
        out_root_base: str = "out/agent_runs",
        pca_topk: int = 5000,
        do_nonlinear: bool = True,
        combine_thresh: int = 3000,
    ) -> Dict[str, Any]:
        """
        Returns a dict:
          {
            "reply": <str>,
            "out": <single-run out dict or None>,
            "multi": <multi-run dict or None>,
            "run_id": <str or None>,
          }
        """

        cmd = (text or "").strip()
        cmd_lower = cmd.lower()

        # 1) SEARCH DRIVE
        if "search" in cmd_lower and "drive" in cmd_lower:
            return self._cmd_search_drive(
                cmd=cmd,
                sa_json_bytes=sa_json_bytes,
                deg_root_link_or_id=deg_root_link_or_id,
            )

        # 2) RUN HARMONIZATION
        if "run" in cmd_lower and "harmonization" in cmd_lower:
            return self._cmd_run_harmonization(
                out_root_base=out_root_base,
                pca_topk=pca_topk,
                do_nonlinear=do_nonlinear,
                combine_thresh=combine_thresh,
            )

        # 3) SUMMARY
        if "summary" in cmd_lower or "report" in cmd_lower:
            return self._cmd_summary()

        # 4) HELP / UNKNOWN
        return {
            "reply": (
                "I didn't fully understand that.\n\n"
                "You can try commands like:\n"
                "- `search diabetes on drive`\n"
                "- `run harmonization`\n"
                "- `show summary`"
            ),
            "out": None,
            "multi": None,
            "run_id": None,
        }

    # ------------------------------
    # Internal: search drive
    # ------------------------------
    def _cmd_search_drive(
        self,
        cmd: str,
        *,
        sa_json_bytes: Optional[bytes],
        deg_root_link_or_id: Optional[str],
    ) -> Dict[str, Any]:
        if not sa_json_bytes:
            return {
                "reply": "Please upload a **Service Account JSON** for Drive access.",
                "out": None,
                "multi": None,
                "run_id": None,
            }
        if not deg_root_link_or_id:
            return {
                "reply": "Please provide the **deg_data folder link or ID** for Drive search.",
                "out": None,
                "multi": None,
                "run_id": None,
            }

        # crude extraction: everything between 'search' and 'on drive'
        text = cmd.strip()
        lower = text.lower()
        disease_query = ""
        try:
            start = lower.index("search") + len("search")
            end = lower.index("on drive") if "on drive" in lower else len(lower)
            disease_query = text[start:end].strip(" :,-")
        except Exception:
            # fallback: remove keywords and use remaining
            disease_query = (
                lower.replace("search", "")
                .replace("on drive", "")
                .strip(" :,-")
            )

        try:
            drv = din.DriveClient.from_service_account_bytes(sa_json_bytes)
            plan = din.make_ingest_plan(
                drv,
                deg_root_link_or_id,
                disease_query=disease_query or None,
            )
        except Exception as e:
            return {
                "reply": f"Drive search failed: {e}",
                "out": None,
                "multi": None,
                "run_id": None,
            }

        self.last_plan = plan

        mode = plan.get("mode")
        if mode == "none":
            return {
                "reply": plan.get("reason", "No (counts, meta) pairs found under deg_data with that query."),
                "out": None,
                "multi": None,
                "run_id": None,
            }

        if mode == "single":
            label = plan["single"].get("label", "dataset")
            disease = plan.get("disease")
            prep = plan.get("prep_path")
            extra = []
            if disease:
                extra.append(f"disease: **{disease}**")
            if prep:
                extra.append(f"prep path: `{prep}`")
            extra_str = "  \n".join(extra) if extra else ""
            reply = f"Found **1 dataset** for query `{disease_query or '(all)'}`: **{label}**"
            if extra_str:
                reply += "\n\n" + extra_str
            reply += "\n\nYou can now say: `run harmonization`."
            return {
                "reply": reply,
                "out": None,
                "multi": None,
                "run_id": None,
            }

        if mode == "multi_dataset":
            ds = plan.get("datasets", [])
            disease = plan.get("disease")
            reply = f"Found **{len(ds)} datasets** for query `{disease_query or '(all)'}`."
            if disease:
                reply += f"\n\nDisease folder: **{disease}**"
            reply += (
                "\n\nYou can now say: `run harmonization` "
                "to perform multi-dataset harmonization + meta-analysis."
            )
            return {
                "reply": reply,
                "out": None,
                "multi": None,
                "run_id": None,
            }

        return {
            "reply": f"Got ingest plan with mode `{mode}`, but I don't know how to handle it yet.",
            "out": None,
            "multi": None,
            "run_id": None,
        }

    # ------------------------------
    # Internal: run harmonization
    # ------------------------------
    def _cmd_run_harmonization(
        self,
        *,
        out_root_base: str,
        pca_topk: int,
        do_nonlinear: bool,
        combine_thresh: int,
    ) -> Dict[str, Any]:
        if not self.last_plan:
            return {
                "reply": "I don't have a Drive plan yet. First ask me to `search <disease> on drive`.",
                "out": None,
                "multi": None,
                "run_id": None,
            }

        mode = self.last_plan.get("mode")
        os.makedirs(out_root_base, exist_ok=True)
        run_id = _dt.datetime.now().strftime("agent_%Y%m%d_%H%M%S")
        out_root = os.path.join(out_root_base, run_id)

        out = None
        multi_out = None

        try:
            if mode == "single":
                single = self.last_plan["single"]
                kwargs = {
                    "single_expression_file": single["counts"],
                    "single_expression_name_hint": single["counts_name"],
                    "metadata_file": single["meta"],
                    "metadata_name_hint": single["meta_name"],
                    "metadata_id_cols": [
                        "sample", "Sample", "Id", "ID", "id", "CleanID",
                        "sample_id", "Sample_ID", "SampleID", "bare_id"
                    ],
                    "metadata_group_cols": [
                        "group", "Group", "condition", "Condition",
                        "phenotype", "Phenotype"
                    ],
                    "metadata_batch_col": None,
                    "out_root": out_root,
                    "pca_topk_features": int(pca_topk),
                    "make_nonlinear": bool(do_nonlinear),
                }
                out = hz.run_pipeline(**kwargs)

            elif mode == "multi_dataset":
                ds = self.last_plan.get("datasets", [])
                datasets_arg = []
                for d in ds:
                    datasets_arg.append({
                        "geo": d["label"],
                        "counts": d["counts"],
                        "counts_name": d["counts_name"],
                        "meta": d["meta"],
                        "meta_name": d["meta_name"],
                        "meta_id_cols": [
                            "sample", "Sample", "Id", "ID", "id", "CleanID",
                            "sample_id", "Sample_ID", "SampleID", "bare_id"
                        ],
                        "meta_group_cols": [
                            "group", "Group", "condition", "Condition",
                            "phenotype", "Phenotype"
                        ],
                        "meta_batch_col": None,
                    })
                kwargs_multi = {
                    "datasets": datasets_arg,
                    "attempt_combine": True,
                    "combine_minoverlap_genes": int(combine_thresh),
                    "out_root": out_root,
                    "pca_topk_features": int(pca_topk),
                    "make_nonlinear": bool(do_nonlinear),
                }
                multi_out = hz.run_pipeline_multi(**kwargs_multi)
                out = multi_out.get("combined") or next(iter(multi_out["runs"].values()))

            else:
                return {
                    "reply": f"Plan has unsupported mode `{mode}` for harmonization.",
                    "out": None,
                    "multi": None,
                    "run_id": None,
                }

        except Exception as e:
            return {
                "reply": f"Harmonization failed: {e}",
                "out": None,
                "multi": None,
                "run_id": None,
            }

        self.last_result = {"out": out, "multi": multi_out, "run_id": run_id}

        # Build a short reply
        shapes = {}
        try:
            with open(out["report_json"], "r") as fh:
                rep = json.load(fh)
            shapes = rep.get("shapes", {})
        except Exception:
            pass

        n_samples = shapes.get("samples", "unknown")
        n_genes = shapes.get("genes", "unknown")
        if mode == "single":
            reply = (
                f"✅ Harmonization complete for **1 dataset**.\n"
                f"- Run ID: `{run_id}`\n"
                f"- Samples: `{n_samples}`\n"
                f"- Genes: `{n_genes}`\n\n"
                "You can inspect the results in the main tabs."
            )
        else:
            n_ds = len(multi_out.get("runs", {})) if multi_out else 0
            reply = (
                f"✅ Multi-dataset harmonization complete.\n"
                f"- Run ID: `{run_id}`\n"
                f"- Datasets: `{n_ds}`\n"
                f"- Combined samples: `{n_samples}`\n"
                f"- Genes: `{n_genes}`\n\n"
                "You can inspect the results in the main tabs, including the **Multi-dataset Summary**."
            )

        return {
            "reply": reply,
            "out": out,
            "multi": multi_out,
            "run_id": run_id,
        }

    # ------------------------------
    # Internal: summary of last run
    # ------------------------------
    def _cmd_summary(self) -> Dict[str, Any]:
        if not self.last_result:
            return {
                "reply": "No harmonization run yet. Ask me to `run harmonization` first.",
                "out": None,
                "multi": None,
                "run_id": None,
            }

        out = self.last_result["out"]
        multi = self.last_result["multi"]
        run_id = self.last_result["run_id"]

        summary_lines = [f"📊 **Summary for run `{run_id}`**"]

        # main report
        try:
            with open(out["report_json"], "r") as fh:
                rep = json.load(fh)
            qc = rep.get("qc", {})
            shapes = rep.get("shapes", {})

            summary_lines.append(
                f"- Samples: `{shapes.get('samples', 'unknown')}`, "
                f"Genes: `{shapes.get('genes', 'unknown')}`"
            )
            summary_lines.append(
                f"- Platform: `{qc.get('platform', 'unknown')}`"
            )
            summary_lines.append(
                f"- Harmonization mode: `{qc.get('harmonization_mode', 'unknown')}`"
            )
            summary_lines.append(
                f"- Zero fraction: `{qc.get('zero_fraction', 'NA')}`"
            )
            if "silhouette_batch" in qc:
                summary_lines.append(
                    f"- Silhouette (batch): `{qc.get('silhouette_batch')}` (lower is better)"
                )
        except Exception:
            summary_lines.append("- Could not read report.json for more details.")

        # meta-analysis if multi
        if multi and multi.get("meta_dir"):
            meta_dir = multi["meta_dir"]
            meta_csv = os.path.join(meta_dir, "meta_analysis_results.csv")
            if os.path.exists(meta_csv):
                try:
                    import pandas as pd
                    mdf = pd.read_csv(meta_csv, index_col=0)
                    sig = mdf[mdf["q_meta"] < 0.10]
                    summary_lines.append(
                        f"- Meta-analysis significant genes (FDR < 0.1): `{sig.shape[0]}`"
                    )
                except Exception:
                    pass

        reply = "\n".join(summary_lines)
        return {
            "reply": reply,
            "out": None,    # don't override Streamlit state
            "multi": None,
            "run_id": None,
        }
