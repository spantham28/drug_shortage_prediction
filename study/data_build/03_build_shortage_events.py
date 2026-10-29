"""
Build one row per ASHP shortage event with start/end dates, route, therapeutic
category, and supplier information.

Inputs
  ashp_drug_shortages_with_ndc.csv   ASHP/University of Utah shortage listings with affected NDCs
  ndc_product_directory.csv          FDA NDC directory (route, dosage form, pharmacologic class)

Output
  data/interim/shortage_events.csv

Dating rules
  start = ASHP "Created Date"
  end   = ASHP "Revision Date" for resolved / discontinued / no-commercial-product listings
          (ASHP revises a listing when it is resolved), or the extract date for current shortages.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
INTERIM = ROOT / "data" / "interim"
EXTRACT_DATE = pd.Timestamp("2025-08-19")  # date the ASHP listing was scraped

PARENTERAL_WORDS = r"injection|injectable|infusion|vial|syringe|for injection|intravenous|\biv\b|bag|premix|ampul|carpuject|pca|large volume|irrigation|dialysis|parenteral|cardioplegi|intrathecal|epidural"
NONPARENTERAL_WORDS = r"tablet|capsule|oral|ophthalmic|otic|topical|cream|ointment|gel|patch|inhal|nasal|rectal|suppositor|lozenge|powder for suspension|chewable|elixir|syrup"
PARENTERAL_ROUTES = {"INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "INTRA-ARTERIAL", "EPIDURAL", "INTRATHECAL",
                     "INFILTRATION", "PERINEURAL", "INTRACAUDAL", "INTRAPERITONEAL", "IRRIGATION", "INTRAVESICAL",
                     "INTRACARDIAC", "INTRAOSSEOUS", "INTRAVITREAL", "INTRA-ARTICULAR", "INTRADERMAL", "INTRACAVERNOUS"}

# Category keyword lists (matched against lower-cased generic name). Order = priority.
CATEGORIES = {
    "oncology": [
        "carboplatin", "cisplatin", "oxaliplatin", "fluorouracil", "methotrexate", "vincristine", "vinblastine",
        "vinorelbine", "cytarabine", "doxorubicin", "daunorubicin", "epirubicin", "idarubicin", "mitoxantrone",
        "etoposide", "bleomycin", "dacarbazine", "cyclophosphamide", "ifosfamide", "mesna", "leucovorin",
        "levoleucovorin", "gemcitabine", "paclitaxel", "docetaxel", "irinotecan", "topotecan", "azacitidine",
        "decitabine", "bendamustine", "busulfan", "carmustine", "melphalan", "mitomycin", "fludarabine",
        "cladribine", "thiotepa", "bcg", "arsenic trioxide", "bortezomib", "pemetrexed", "dexrazoxane",
        "asparaginase", "capecitabine", "hydroxyurea", "mercaptopurine", "tamoxifen", "anastrozole", "letrozole",
        "nelarabine", "clofarabine", "streptozocin", "dactinomycin", "romidepsin", "eribulin", "fulvestrant",
        "leuprolide", "octreotide", "goserelin", "temozolomide", "lomustine", "procarbazine", "chlorambucil",
        "imatinib", "tretinoin", "aminolevulinic", "palonosetron", "fosaprepitant", "rasburicase", "filgrastim",
        "pegfilgrastim", "zoledronic", "pamidronate",
    ],
    "anti_infective": [
        "cef", "penicillin", "ampicillin", "amoxicillin", "nafcillin", "oxacillin", "piperacillin", "ticarcillin",
        "meropenem", "imipenem", "ertapenem", "doripenem", "vancomycin", "gentamicin", "tobramycin", "amikacin",
        "streptomycin", "azithromycin", "erythromycin", "clindamycin", "metronidazole", "doxycycline",
        "minocycline", "tigecycline", "aztreonam", "acyclovir", "ganciclovir", "foscarnet", "fluconazole",
        "voriconazole", "amphotericin", "micafungin", "caspofungin", "sulfamethoxazole", "trimethoprim",
        "colistimethate", "polymyxin", "bacitracin", "rifampin", "isoniazid", "ethambutol", "pyrazinamide",
        "linezolid", "daptomycin", "levofloxacin", "ciprofloxacin", "moxifloxacin", "chloramphenicol",
        "pentamidine", "quinidine gluconate", "artesunate", "nitrofurantoin", "oseltamivir", "remdesivir",
        "vaccine", "toxoid", "tuberculin", "bicillin", "dalbavancin", "oritavancin", "telavancin", "quinupristin",
    ],
    "perioperative": [
        "lidocaine", "bupivacaine", "ropivacaine", "chloroprocaine", "mepivacaine", "tetracaine", "procaine",
        "propofol", "etomidate", "ketamine", "methohexital", "midazolam", "dexmedetomidine", "fentanyl",
        "sufentanil", "remifentanil", "alfentanil", "hydromorphone", "morphine", "meperidine", "rocuronium",
        "vecuronium", "succinylcholine", "cisatracurium", "atracurium", "pancuronium", "neostigmine",
        "glycopyrrolate", "sugammadex", "ondansetron", "ephedrine", "sevoflurane", "isoflurane", "desflurane",
        "ketorolac", "acetaminophen injection", "prochlorperazine", "promethazine", "droperidol", "metoclopramide",
        "famotidine injection", "nalbuphine", "butorphanol", "buprenorphine", "methadone injection",
        "oxytocin", "methylergonovine", "carboprost", "terbutaline", "tranexamic", "aminocaproic", "hyaluronidase",
        "indocyanine", "methylene blue", "isosulfan", "fluorescein", "tropicamide", "phenylephrine and ketorolac",
    ],
    "critical_care": [
        "norepinephrine", "epinephrine", "vasopressin", "dopamine", "dobutamine", "phenylephrine", "milrinone",
        "isoproterenol", "nitroglycerin", "nitroprusside", "nicardipine", "clevidipine", "esmolol", "labetalol",
        "metoprolol injection", "diltiazem", "amiodarone", "adenosine", "atropine", "procainamide", "lidocaine in dextrose",
        "digoxin", "naloxone", "flumazenil", "alteplase", "tenecteplase", "heparin", "enoxaparin", "argatroban",
        "bivalirudin", "protamine", "phytonadione", "vitamin k", "lorazepam", "diazepam", "phenytoin",
        "fosphenytoin", "levetiracetam", "valproate", "phenobarbital", "mannitol", "hydroxocobalamin",
        "physostigmine", "fomepizole", "glucagon", "deferoxamine", "digoxin immune", "crotalidae", "antivenin",
        "pralidoxime", "sodium bicarbonate", "dextrose 50", "50% dextrose", "calcium chloride", "hydralazine",
        "enalaprilat", "furosemide", "bumetanide", "chlorothiazide", "methylprednisolone", "hydrocortisone sodium",
        "dexamethasone", "insulin", "thiamine", "cyanide", "sodium nitrite", "sodium thiosulfate", "ethanolamine",
        "verapamil", "propranolol", "haloperidol", "chlorpromazine", "ziprasidone", "olanzapine injection",
        "cosyntropin", "desmopressin", "acetazolamide", "dantrolene", "conivaptan", "angiotensin", "immune globulin", "albumin", "plasma protein",
        "factor", "prothrombin complex", "aminophylline", "rho(d)", "tetanus", "rabies", "botulism",
    ],
    "fluids_nutrition_electrolytes": [
        "sodium chloride", "dextrose", "lactated ringer", "ringer", "sterile water", "water for injection",
        "amino acid", "lipid", "fat emulsion", "intralipid", "trace element", "multivitamin", "multiple vitamin",
        "infuvite", "m.v.i", "zinc", "selenium", "copper", "manganese", "chromium", "cysteine", "potassium",
        "magnesium", "calcium gluconate", "sodium phosphate", "sodium acetate", "ammonium", "tromethamine",
        "plasma-lyte", "normosol", "electrolyte", "ascorbic acid", "folic acid injection", "cyanocobalamin",
        "pyridoxine", "levocarnitine", "dialysis", "cardioplegi", "hetastarch", "dextran", "irrigation",
        "bacteriostatic", "heparin lock", "flush",
    ],
}

ONCOLOGY_CLASSES = r"Alkylating|Platinum-based|Nucleoside Metabolic Inhibitor|Folate Analog Metabolic Inhibitor|Vinca Alkaloid|Anthracycline|Topoisomerase Inhibitor|Microtubule Inhibitor|Kinase Inhibitor|Antineoplastic|Hormonal Antineoplastic"
ANTIINFECTIVE_CLASSES = r"Cephalosporin|Penicillin|Aminoglycoside|Carbapenem|Macrolide|Lincosamide|Glycopeptide|Tetracycline|Antifungal|Antiviral|Antibacterial|Quinolone|Sulfonamide|Nitroimidazole|Polymyxin|Monobactam|Vaccine"


def load_ndc_directory():
    n = pd.read_csv(REPO / "ndc_product_directory.csv", dtype=str, encoding="utf-8-sig")
    parts = n["PRODUCTNDC"].str.split("-", expand=True)
    n["product9"] = parts[0].str.zfill(5) + parts[1].str.zfill(4)
    return n[["product9", "ROUTENAME", "DOSAGEFORMNAME", "PHARM_CLASSES", "LABELERNAME"]]


def classify(name, routes, classes, parenteral):
    low = name.lower()
    classes = classes if isinstance(classes, str) else ""
    for cat, words in CATEGORIES.items():
        if any(w in low for w in words):
            return cat
    if re.search(ONCOLOGY_CLASSES, classes):
        return "oncology"
    if re.search(ANTIINFECTIVE_CLASSES, classes):
        return "anti_infective"
    return "other_parenteral" if parenteral else "non_parenteral"


def main():
    a = pd.read_csv(REPO / "ashp_drug_shortages_with_ndc.csv")
    a = a.rename(columns={"Generic Name": "name", "Shortage Status": "status", "Revision Date": "revised",
                          "Created Date": "created", "NDC Codes": "ndcs"})
    a["start"] = pd.to_datetime(a["created"])
    a["revised"] = pd.to_datetime(a["revised"])
    a["end"] = np.where(a["status"].eq("Current Shortages"), EXTRACT_DATE, a["revised"])
    a["end"] = pd.to_datetime(a["end"])
    a["shortage_id"] = np.arange(len(a))

    ndc_dir = load_ndc_directory()
    e = a[["shortage_id", "ndcs"]].assign(ndc=a["ndcs"].fillna("").str.split(";")).explode("ndc")
    e["ndc"] = e["ndc"].str.strip()
    e = e[e["ndc"].str.len() > 0]
    e["ndc11"] = e["ndc"].str.replace("-", "")
    e["labeler"] = e["ndc11"].str[:5]
    e["product9"] = e["ndc11"].str[:9]
    e = e.merge(ndc_dir, on="product9", how="left")
    e.drop(columns=["ndcs"]).to_csv(INTERIM / "shortage_ndcs.csv", index=False)

    info = e.groupby("shortage_id").agg(
        n_ndcs=("ndc11", "nunique"),
        n_labelers=("labeler", "nunique"),
        routes=("ROUTENAME", lambda s: ";".join(sorted(set(";".join(s.dropna()).split("; "))))),
        classes=("PHARM_CLASSES", lambda s: ";".join(s.dropna().unique())),
    ).reset_index()
    a = a.merge(info, on="shortage_id", how="left")
    a["n_ndcs"] = a["n_ndcs"].fillna(0).astype(int)
    a["n_labelers"] = a["n_labelers"].fillna(0).astype(int)

    name_parenteral = a["name"].str.lower().str.contains(PARENTERAL_WORDS, regex=True)
    name_nonparenteral = a["name"].str.lower().str.contains(NONPARENTERAL_WORDS, regex=True)
    route_parenteral = a["routes"].fillna("").apply(
        lambda r: any(x.strip() in PARENTERAL_ROUTES for x in re.split(r"[;,]", r)))
    a["parenteral"] = ((name_parenteral | route_parenteral) & ~(name_nonparenteral & ~name_parenteral)).astype(int)
    a["category"] = [classify(n, r, c, p) for n, r, c, p in
                     zip(a["name"], a["routes"], a["classes"], a["parenteral"])]
    # Non-parenteral listings stay in the outpatient bucket even when the molecule matches a hospital category
    a.loc[(a["parenteral"] == 0) & (a["category"] != "non_parenteral"), "category"] = "non_parenteral"
    a["limited_supply"] = (a["n_labelers"].between(1, 2)).astype(int)
    a["duration_days"] = (a["end"] - a["start"]).dt.days.clip(lower=1)

    cols = ["shortage_id", "name", "status", "start", "end", "duration_days", "parenteral", "category",
            "n_ndcs", "n_labelers", "limited_supply", "routes"]
    a[cols].to_csv(INTERIM / "shortage_events.csv", index=False)
    print(a["category"].value_counts())
    print(a.groupby(a["start"].dt.year).size())


if __name__ == "__main__":
    main()
