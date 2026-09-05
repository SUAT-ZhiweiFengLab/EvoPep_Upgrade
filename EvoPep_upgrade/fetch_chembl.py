"""
Fetch real peptide bioactivity data from ChEMBL for the 7 EvoPep targets
(manuscript section 2.6: ChEMBL bioactivity data as the training corpus).

Outputs:
  data/chembl_targets.json   - resolved ChEMBL target IDs
  data/chembl_peptides.json  - per-target peptide pXC50 records
  data/peptide_corpus.txt    - peptide sequences (MLM pretraining corpus)
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 7 targets from the manuscript (GLP-1R primary + 6 AD-related off-targets)
# name -> (ChEMBL search query, UniProt accession for validation)

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def get_json(url, retries=4, timeout=40):
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "evopep-research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            print(f"  [retry {i + 1}/{retries}] {e}", flush=True)
            time.sleep(2 * (i + 1))
    return None


TARGET_SEARCH_NAMES = {
    "GLP-1R":  ("glucagon-like peptide 1 receptor", "P43220"),
    "GRIN2A":  ("GRIN2A", "Q12879"),
    "HDAC6":   ("histone deacetylase 6", "Q9UBN7"),
    "CHRNA7":  ("cholinergic receptor nicotinic alpha 7", "P36544"),
    "SIGMAR1": ("sigma non-opioid intracellular receptor 1", "Q99720"),
    "PI3K":    ("phosphatidylinositol 3-kinase catalytic subunit alpha", "P42336"),
    "MEK1":    ("mitogen-activated protein kinase kinase 1", "Q02750"),
}


def resolve_target(name, accession):
    """Resolve via text search, then validate the UniProt accession matches."""
    q = urllib.parse.quote(name)
    d = get_json(f"{BASE}/target/search.json?q={q}&limit=25")
    if not d:
        return None, None
    for t in d.get("targets", []):
        accs = [c.get("accession") for c in t.get("target_components", [])]
        if accession in accs:
            return t["target_chembl_id"], t["pref_name"]
    # fallback: accept a single-protein hit with matching description
    for t in d.get("targets", []):
        comps = t.get("target_components", [])
        if len(comps) == 1 and accession in [c.get("accession") for c in comps]:
            return t["target_chembl_id"], t["pref_name"]
    return None, None



def fetch_activities(tid, cap=4000):
    """Fetch pXC50 records (Ki/Kd/IC50) for a target, paginated."""
    recs, offset = [], 0
    while offset < cap:
        url = (
            f"{BASE}/activity.json?target_chembl_id={tid}"
            f"&pchembl_value__isnotnull=on&standard_type__in=(Ki,Kd,IC50)"
            f"&limit=100&offset={offset}"
            f"&only=molecule_chembl_id,pchembl_value,standard_type,assay_type"
        )
        d = get_json(url)
        if not d or not d.get("activities"):
            break
        recs.extend(d["activities"])
        if len(d["activities"]) < 100:
            break
        offset += 100
        time.sleep(0.12)
    return recs[:cap]


def fetch_molecule_components(mols):
    """Batch-fetch component IDs for molecules (full record includes molecule_components)."""
    mol2comp = {}
    for i in range(0, len(mols), 25):
        batch = mols[i : i + 25]
        url = f"{BASE}/molecule.json?molecule_chembl_id__in={','.join(batch)}"
        d = get_json(url)
        if not d:
            continue
        for m in d.get("molecules", []):
            comps = [c.get("component_id") for c in m.get("molecule_components", [])]
            mol2comp[m["molecule_chembl_id"]] = comps
        time.sleep(0.12)
    return mol2comp


def fetch_component_sequences(comp_ids):
    """Batch-fetch amino-acid sequences for components."""
    seq = {}
    ids = sorted(set(c for c in comp_ids if c))
    for i in range(0, len(ids), 30):
        batch = ids[i : i + 30]
        url = f"{BASE}/component.json?component_id__in={','.join(str(c) for c in batch)}&only=component_id,sequence"
        d = get_json(url)
        if not d or not d.get("components"):
            for cid in batch:  # fallback: individual fetch
                d1 = get_json(f"{BASE}/component/{cid}.json?only=component_id,sequence")
                if d1:
                    c = d1.get("component", d1)
                    if c and c.get("sequence"):
                        seq[c["component_id"]] = c["sequence"]
                time.sleep(0.1)
            continue
        for c in d["components"]:
            seq[c["component_id"]] = c.get("sequence") or ""
        time.sleep(0.12)
    return seq


def main():
    dataset = {}
    corpus = set()

    for name, (query, acc) in TARGET_SEARCH_NAMES.items():
        print(f"== {name} ({acc}) ==", flush=True)
        tid, pname = resolve_target(query, acc)
        print(f"  ChEMBL target: {tid} {pname}", flush=True)
        if not tid:
            dataset[name] = {"accession": acc, "target_chembl_id": None, "records": []}
            continue

        acts = fetch_activities(tid)
        print(f"  raw activity records: {len(acts)}", flush=True)
        mols = sorted({a["molecule_chembl_id"] for a in acts})
        print(f"  unique molecules: {len(mols)}, fetching components...", flush=True)
        mol2comp = fetch_molecule_components(mols)
        comp2seq = fetch_component_sequences([c for cs in mol2comp.values() for c in cs])
        print(f"  components with sequence: {len(comp2seq)}", flush=True)

        agg = {}
        n_pep_obs = 0
        for a in acts:
            mid = a["molecule_chembl_id"]
            comps = mol2comp.get(mid, [])
            seqs = [comp2seq.get(c, "") for c in comps]
            seqs = [s for s in seqs if s and 8 <= len(s) <= 60 and AA_RE.match(s)]
            if not seqs:
                continue
            s = max(seqs, key=len)
            agg.setdefault(s, []).append(float(a["pchembl_value"]))
            corpus.add(s)
            n_pep_obs += 1

        records = [
            {"sequence": s, "pchembl_mean": sum(v) / len(v), "pchembl_max": max(v), "n_obs": len(v)}
            for s, v in sorted(agg.items())
        ]
        dataset[name] = {
            "accession": acc,
            "target_chembl_id": tid,
            "pref_name": pname,
            "n_peptide_observations": n_pep_obs,
            "records": records,
        }
        print(f"  unique peptide sequences with affinity: {len(records)}", flush=True)

        # incremental cache: persist after every target so network hiccups lose nothing
        with open(os.path.join(DATA_DIR, "chembl_peptides.json"), "w") as f:
            json.dump(dataset, f, indent=1)
        with open(corpus_path := os.path.join(DATA_DIR, "peptide_corpus.txt"), "w") as f:
            f.write("\n".join(sorted(corpus)) + "\n")

    with open(os.path.join(DATA_DIR, "chembl_targets.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in dataset.items()}, f, indent=1)
    with open(os.path.join(DATA_DIR, "chembl_peptides.json"), "w") as f:
        json.dump(dataset, f, indent=1)

    corpus_path = os.path.join(DATA_DIR, "peptide_corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(sorted(corpus)) + "\n")
    print(f"target-derived peptide corpus: {len(corpus)} sequences -> {corpus_path}", flush=True)

    # supplement MLM corpus with general ChEMBL biologic peptides if sparse
    if len(corpus) < 3000:
        print("Supplementing MLM corpus with ChEMBL biological molecules (8..60 aa)...", flush=True)
        extra, offset, seen = 0, 0, set(corpus)
        while offset < 20000 and len(corpus) < 5000:
            url = (
                f"{BASE}/molecule.json?molecule_type=(Biologic)&limit=100&offset={offset}"
                "&only=molecule_chembl_id,molecule_components"
            )
            d = get_json(url)
            if not d or not d.get("molecules"):
                break
            mols = [m["molecule_chembl_id"] for m in d["molecules"]]
            mol2comp = fetch_molecule_components(mols)
            comp2seq = fetch_component_sequences([c for cs in mol2comp.values() for c in cs])
            for comps in mol2comp.values():
                for c in comps:
                    s = comp2seq.get(c, "")
                    if s and 8 <= len(s) <= 60 and AA_RE.match(s) and s not in seen:
                        seen.add(s)
                        corpus.add(s)
                        extra += 1
            offset += 100
            time.sleep(0.1)
        with open(corpus_path, "w") as f:
            f.write("\n".join(sorted(corpus)) + "\n")
        print(f"final MLM corpus: {len(corpus)} sequences (+{extra} supplementary)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
