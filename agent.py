# agent.py — LLM-aware harmonization agent for Drive + harmonizer

import os
import io
import json
import datetime as _dt

import drive_ingest as din
import harmonizer as hz

# Optional: use OpenAI LLM for intent routing if OPENAI_API_KEY is set.
try:
    from openai import OpenAI
    _OPENAI_CLIENT = OpenAI()
except Exception:  # openai not installed or misconfigured
    _OPENAI_CLIENT = None


class HarmonizationAgent:
    """
    Agent that can:
    - search <disease> on drive
    - run harmonization
    - show summary

    It can also use an LLM (if available) to understand more free-form prompts like:
      "diabetes disease search and then harmonized"
      "please run the analysis for lupus and show me the summary"
    """

    def __init__(self):
        # Agent memory (per Streamlit session – the app holds the object)
        self.last_plan = None       # latest drive ingest plan
        self.last_disease = None
        self.last_out = None        # representative "out" dict
        self.last_multi = None      # full multi-dataset result (if any)
        self.last_run_id = None

    # -------------------------
    # Public entry point
    # -------------------------
    def handle_command(self, text, sa_json_bytes=None, deg_root_link_or_id=None):
        """
        Main entry. Returns a dict:
        {
          "reply": <human-readable text>,
          "out": <harmonizer out or None>,
          "multi": <multi_out or None>,
          "run_id": <run_id or None>,
        }
        """
        text = (text or "").strip()
        if not text:
            return self._resp("I didn’t get anything to do. Try something like: `search diabetes on drive`.")

        # Step 1: route to intent
        intent = self._route_intent(text)

        action = intent.get("action")
        disease = intent.get("disease")  # may be None
        # If user specifically mentioned a disease, use that
        if disease:
            self.last_disease = disease

        # Step 2: execute
        if action == "search":
            return self._do_search(disease, sa_json_bytes, deg_root_link_or_id)
        elif action == "run":
            return self._do_run(sa_json_bytes, deg_root_link_or_id)
        elif action == "summary":
            return self._do_summary()
        elif action == "help":
            return self._help()
        else:
            # Fallback if routing failed
            return self._help(extra="I couldn’t confidently parse that request as search / run / summary.")

    # -------------------------
    # Intent routing
    # -------------------------
    def _route_intent(self, text: str) -> dict:
        """
        Decide what the user wants: search / run / summary / help.
        Try some heuristics first, then optionally use an LLM for fuzzy cases.
        Returns: {"action": "search|run|summary|help", "disease": <str or None>}
        """
        low = text.lower()

        # 1) Simple heuristics cover 90% of use cases (fast, no LLM)
        if any(w in low for w in ["search", "find", "scan", "look for"]):
            return {
                "action": "search",
                "disease": self._extract_disease_from_text(low),
            }

        if any(w in low for w in ["run harmonization", "run analysis", "run the analysis", "harmonize", "harmonised", "harmonized"]):
            return {"action": "run", "disease": self.last_disease}

        if any(w in low for w in ["show summary", "summary", "overview", "results summary"]):
            return {"action": "summary", "disease": self.last_disease}

        if low in {"help", "?", "what can you do"}:
            return {"action": "help", "disease": None}

        # 2) If no obvious match: optionally ask the LLM
        if _OPENAI_CLIENT is not None:
            try:
                return self._llm_route(text)
            except Exception:
                # If LLM fails for any reason, gracefully fall back to help
                return {"action": "help", "disease": None}

        # No LLM available: best-effort guess
        # Single word that could be a disease (like "diabetes") -> search
        if len(text.split()) == 1:
            return {"action": "search", "disease": text.strip()}

        return {"action": "help", "disease": None}

    def _llm_route(self, text: str) -> dict:
        """
        Use OpenAI to interpret the user command into an action + disease.
        Requires OPENAI_API_KEY and openai package.
        """
        system = (
            "You are an intent router for a gene-expression harmonization agent. "
            "You MUST reply with a single-line JSON object ONLY, no extra text.\n\n"
            "Valid actions: 'search', 'run', 'summary', 'help'.\n"
            "- 'search': user wants to search Drive for disease datasets.\n"
            "- 'run': user wants to run harmonization/meta-analysis using the last search.\n"
            "- 'summary': user wants a summary of the last completed run.\n"
            "- 'help': user is asking what you can do or unclear.\n\n"
            "If they mention a disease (e.g. diabetes, lupus, covid, leukemia), "
            "extract it as the 'disease' field (string). Otherwise, disease=null.\n\n"
            "Example outputs:\n"
            '{"action":"search","disease":"diabetes"}\n'
            '{"action":"run","disease":null}\n'
            '{"action":"summary","disease":null}\n'
            '{"action":"help","disease":null}'
        )

        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            # If the model added explanation, try to find the JSON substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw[start:end + 1])
            else:
                raise ValueError("Could not parse LLM routing JSON")

        # sanity checks
        action = parsed.get("action", "help")
        disease = parsed.get("disease")
        if not isinstance(disease, (str, type(None))):
            disease = None
        return {"action": action, "disease": disease}

    @staticmethod
    def _extract_disease_from_text(low: str) -> str | None:
        """
        Very light heuristic: if user wrote `search diabetes on drive`,
        return 'diabetes'. Not perfect but works for common cases.
        """
        # Strip obvious keywords
        for kw in ["search", "find", "scan", "look for", "on drive"]:
            low = low.replace(kw, " ")
        candidate = low.strip(" .,;")
        # Avoid empty / super generic
        if not candidate or len(candidate.split()) > 6:
            return None
        return candidate.strip() or None

    # -------------------------
    # Actions
    # -------------------------
    def _do_search(self, disease, sa_json_bytes, deg_root_link_or_id):
        if not sa_json_bytes:
            return self._resp(
                "To search Drive I need a **Service Account JSON** in the Agent tab.\n"
                "Please upload it and try again (e.g. `search diabetes on drive`)."
            )
        if not deg_root_link_or_id:
            return self._resp(
                "I also need the **deg_data folder link or ID** in the Agent tab.\n"
                "Paste the folder link and try again (e.g. `search diabetes on drive`)."
            )

        disease = (disease or self.last_disease or "").strip()
        if not disease:
            return self._resp(
                "What disease should I search for? Try e.g. `search diabetes on drive`."
            )

        drv = din.DriveClient.from_service_account_bytes(sa_json_bytes)
        plan = din.make_ingest_plan(
            drv,
            deg_root_link_or_id.strip(),
            disease_query=disease,
        )

        self.last_plan = plan
        self.last_disease = disease

        mode = plan.get("mode", "none")
        if mode == "none":
            reason = plan.get("reason", "No content found.")
            return self._resp(
                f"I looked for datasets for **{disease}** but found nothing.\n\nReason: {reason}"
            )

        if mode == "multi_dataset":
            n_ds = len(plan.get("datasets", []))
            reply = (
                f"✅ Found **{n_ds}** datasets for disease **{disease}**.\n\n"
                "You can now say: `run harmonization` to perform multi-dataset harmonization + meta-analysis."
            )
        elif mode == "single":
            reply = (
                f"✅ Found a **single dataset** for **{disease}**.\n\n"
                "You can now say: `run harmonization` to process it."
            )
        else:
            reply = (
                f"✅ Found datasets for **{disease}** (mode: `{mode}`).\n\n"
                "You can now say: `run harmonization`."
            )

        return {
            "reply": reply,
            "out": None,
            "multi": None,
            "run_id": None,
        }

    def _do_run(self, sa_json_bytes, deg_root_link_or_id):
        if not self.last_plan:
            return self._resp(
                "I don’t have a search plan yet.\n\n"
                "First say something like `search diabetes on drive`, then `run harmonization`."
            )

        plan = self.last_plan
        mode = plan.get("mode")
        if mode == "none" or not mode:
            return self._resp(
                "The last search didn’t produce a usable ingest plan. "
                "Try a different disease query (e.g. `search leukemia on drive`)."
            )

        disease = self.last_disease or plan.get("disease", "disease")
        run_id = _dt.datetime.now().strftime("agent_%Y%m%d_%H%M%S")
        out_root = os.path.join("out", "agent_runs", run_id)

        # ---- SINGLE ----
        if mode == "single":
            single = plan["single"]
            kwargs = {
                "single_expression_file": single["counts"],
                "single_expression_name_hint": single["counts_name"],
                "metadata_file": single["meta"],
                "metadata_name_hint": single["meta_name"],
                "metadata_id_cols": [
                    "sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"
                ],
                "metadata_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "metadata_batch_col": None,
                "out_root": out_root,
                "pca_topk_features": 5000,
                "make_nonlinear": True,
            }
            out = hz.run_pipeline(**kwargs)
            multi_out = None

            reply = (
                f"✅ Harmonization complete for **{disease}** (single dataset).\n"
                f"- Run ID: `{run_id}`\n"
                f"- Outdir: `{out['outdir']}`\n\n"
                "You can inspect the results in the main tabs (Overview, QC, DE & GSEA, etc.)."
            )

        # ---- MULTI-FILES-ONE-META ----
        elif mode == "multi_files_one_meta":
            groups = plan["groups"]
            meta = plan["meta"]
            kwargs = {
                "group_to_file": {k: v[0] for k, v in groups.items()},
                "metadata_file": meta,
                "metadata_name_hint": plan["meta_name"],
                "metadata_id_cols": [
                    "sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"
                ],
                "metadata_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                "metadata_batch_col": None,
                "out_root": out_root,
                "pca_topk_features": 5000,
                "make_nonlinear": True,
            }
            out = hz.run_pipeline(**kwargs)
            multi_out = None

            reply = (
                f"✅ Harmonization complete for **{disease}** (multi-files-one-metadata).\n"
                f"- Run ID: `{run_id}`\n"
                f"- Outdir: `{out['outdir']}`\n\n"
                "You can inspect the results in the main tabs."
            )

        # ---- MULTI-DATASET ----
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
                    "meta_id_cols": [
                        "sample","Sample","Id","ID","id","CleanID","sample_id","Sample_ID","SampleID","bare_id"
                    ],
                    "meta_group_cols": ["group","Group","condition","Condition","phenotype","Phenotype"],
                    "meta_batch_col": None,
                })

            kwargs_multi = {
                "datasets": datasets_arg,
                "attempt_combine": True,
                "combine_minoverlap_genes": 3000,
                "out_root": out_root,
                "pca_topk_features": 5000,
                "make_nonlinear": True,
            }
            multi_out = hz.run_pipeline_multi(**kwargs_multi)
            # Representative out: combined if present, else first run
            if multi_out.get("combined"):
                out = multi_out["combined"]
            else:
                out = next(iter(multi_out["runs"].values()))

            # build small summary if report is available
            report = {}
            try:
                with open(out["report_json"], "r") as fh:
                    report = json.load(fh)
            except Exception:
                pass
            shp = report.get("shapes", {})
            qc = report.get("qc", {})
            samples = shp.get("samples", "—")
            genes = shp.get("genes", "—")
            zero_frac = qc.get("zero_fraction", None)
            platform = qc.get("platform", "unknown")

            reply = (
                f"✅ Multi-dataset harmonization complete for **{disease}**.\n"
                f"- Run ID: `{run_id}`\n"
                f"- Datasets: {len(multi_out.get('runs', {}))}\n"
                f"- Samples: {samples}\n"
                f"- Genes: {genes}\n"
                f"- Platform: {platform}\n"
            )
            dec = multi_out.get("combine_decision", {}) or {}
            if dec:
                reply += f"- Combined run: {'Yes' if dec.get('combined') else 'No'} (overlap genes: {dec.get('overlap_genes', 0)})\n"

            reply += (
                "\nYou can inspect the results in the main UI tabs, including **Multi-dataset Summary**, "
                "**Presenter Mode**, and **Comparisons**."
            )

        else:
            return self._resp(f"Unexpected ingest mode: `{mode}`. I can’t run harmonization for that.")

        # Store state
        self.last_run_id = run_id
        self.last_out = out
        self.last_multi = multi_out

        return {
            "reply": reply,
            "out": out,
            "multi": multi_out,
            "run_id": run_id,
        }

    def _do_summary(self):
        if not self.last_out or not self.last_run_id:
            return self._resp(
                "I don’t have a completed run to summarize yet.\n\n"
                "First run something (e.g. `search diabetes on drive` then `run harmonization`)."
            )

        out = self.last_out
        report = {}
        try:
            with open(out["report_json"], "r") as fh:
                report = json.load(fh)
        except Exception:
            pass

        shp = report.get("shapes", {})
        qc = report.get("qc", {})
        samples = shp.get("samples", "—")
        genes = shp.get("genes", "—")
        platform = qc.get("platform", "unknown")
        mode = qc.get("harmonization_mode", "unknown")
        zf = qc.get("zero_fraction", None)
        sil = qc.get("silhouette_batch", None)

        reply = (
            f"📊 Summary for run `{self.last_run_id}`\n"
            f"- Samples: {samples}, Genes: {genes}\n"
            f"- Platform: {platform}\n"
            f"- Harmonization mode: {mode}\n"
        )
        if isinstance(zf, (int, float)):
            reply += f"- Zero fraction: {zf:.3f}\n"
        if isinstance(sil, (int, float)):
            reply += f"- Silhouette (batch): {sil:.3f} (lower is better)\n"

        reply += "\nYou can switch datasets and inspect figures in the main tabs."

        return {
            "reply": reply,
            "out": None,   # don’t overwrite current out; UI already has it
            "multi": None,
            "run_id": self.last_run_id,
        }

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _resp(msg: str):
        return {"reply": msg, "out": None, "multi": None, "run_id": None}

    def _help(self, extra: str | None = None):
        msg = ""
        if extra:
            msg += extra + "\n\n"

        msg += (
            "Here’s what I can do for you:\n"
            "- **Search Drive** for disease datasets: `search diabetes on drive`, `find lupus`\n"
            "- **Run harmonization/meta-analysis** on the last search: `run harmonization`, `run analysis`\n"
            "- **Summarize** the latest run: `show summary`, `overview`\n\n"
            "Typical workflow:\n"
            "1. In the Agent tab, upload your Service Account JSON and set the deg_data folder.\n"
            "2. Ask me: `search <disease> on drive`.\n"
            "3. Then: `run harmonization`.\n"
            "4. Finally: `show summary` or explore the main tabs (Overview, QC, DE & GSEA, Multi-dataset Summary, etc.)."
        )
        return self._resp(msg)
