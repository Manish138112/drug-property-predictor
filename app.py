import os
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

st.set_page_config(page_title="Drug Property Predictor", page_icon="💊", layout="wide")
st.title("💊 Drug Property Predictor")
st.caption("Solubility, Lipinski, BCS aur toxicity. Sirf learning ke liye.")

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
ESOL_FILE = "delaney-processed.csv"
TOX_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX_FILE = "tox21.csv.gz"
TOX_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
             "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
FEATURES = ["Mol. Weight", "LogP", "TPSA", "H-bond donors", "H-bond acceptors",
            "Rotatable bonds", "Aromatic rings", "Fraction sp3", "Heavy atoms"]


def get_desc(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    d = []
    d.append(Descriptors.MolWt(m))
    d.append(Descriptors.MolLogP(m))
    d.append(Descriptors.TPSA(m))
    d.append(Descriptors.NumHDonors(m))
    d.append(Descriptors.NumHAcceptors(m))
    d.append(Descriptors.NumRotatableBonds(m))
    d.append(rdMolDescriptors.CalcNumAromaticRings(m))
    d.append(rdMolDescriptors.CalcFractionCSP3(m))
    d.append(m.GetNumHeavyAtoms())
    return d


def read_data(url, fname):
    if os.path.exists(fname):
        try:
            return pd.read_csv(fname)
        except Exception:
            pass
    df = pd.read_csv(url)
    try:
        if fname.endswith(".gz"):
            df.to_csv(fname, index=False, compression="gzip")
        else:
            df.to_csv(fname, index=False)
    except Exception:
        pass
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    df = read_data(ESOL_URL, ESOL_FILE)
    X = []
    y = []
    for smi, val in zip(df["smiles"], df["measured log solubility in mols per litre"]):
        d = get_desc(smi)
        if d is not None:
            X.append(d)
            y.append(val)
    X = np.array(X)
    y = np.array(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    chk = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    chk.fit(X_tr, y_tr)
    p = chk.predict(X_te)
    metrics = {
        "r2": r2_score(y_te, p),
        "rmse": float(np.sqrt(mean_squared_error(y_te, p))),
        "n": len(X),
    }
    final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final.fit(X, y)
    return final, metrics


@st.cache_resource(show_spinner=False)
def load_tox(task):
    t = read_data(TOX_URL, TOX_FILE)[["smiles", task]].dropna()
    X = []
    y = []
    for smi, val in zip(t["smiles"], t[task]):
        d = get_desc(smi)
        if d is not None:
            X.append(d)
            y.append(int(val))
    X = np.array(X)
    y = np.array(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    return clf, auc


def predict(model, d):
    arr = np.array([d])
    preds = np.array([t.predict(arr)[0] for t in model.estimators_])
    return float(preds.mean()), float(preds.std())


def lipinski(d):
    rules = {}
    rules["MW <= 500"] = d[0] <= 500
    rules["LogP <= 5"] = d[1] <= 5
    rules["H-bond donors <= 5"] = d[3] <= 5
    rules["H-bond acceptors <= 10"] = d[4] <= 10
    return rules


def sol_class(p):
    if p > -2:
        return "Soluble"
    if p > -4:
        return "Moderately soluble"
    return "Poorly soluble"


def bcs_class(logS, logp, tpsa):
    hs = logS > -3
    hp = (logp > 0) and (tpsa < 140)
    if hs and hp:
        return "Class I: high solubility, high permeability"
    if (not hs) and hp:
        return "Class II: low solubility, high permeability"
    if hs and (not hp):
        return "Class III: high solubility, low permeability"
    return "Class IV: low solubility, low permeability"


def fetch_smiles():
    name = st.session_state.get("name_in", "").strip()
    if not name:
        return
    try:
        import pubchempy as pcp
        res = pcp.get_compounds(name, "name")
        if res:
            c = res[0]
            smi = getattr(c, "isomeric_smiles", None)
            if not smi:
                smi = getattr(c, "canonical_smiles", None)
            st.session_state["smiles"] = smi
            st.session_state["msg"] = "SMILES mil gaya: " + name
        else:
            st.session_state["msg"] = "PubChem pe ye naam nahi mila."
    except Exception as e:
        st.session_state["msg"] = "PubChem error: " + str(e)


if "smiles" not in st.session_state:
    st.session_state["smiles"] = "CC(=O)OC1=CC=CC=C1C(=O)O"

with st.spinner("Model train ho raha hai (pehli baar 1-2 minute)..."):
    try:
        model, metrics = load_model()
    except Exception as e:
        st.error("Dataset load nahi hua: " + str(e))
        st.info("Internet check karo, ya delaney-processed.csv download karke "
                "drug_predictor folder mein rakh do.")
        st.stop()

st.sidebar.title("Model info")
st.sidebar.write("Random Forest, ESOL ({} molecules)".format(metrics["n"]))
st.sidebar.metric("R2 (test)", "{:.2f}".format(metrics["r2"]))
st.sidebar.metric("RMSE (test)", "{:.2f}".format(metrics["rmse"]))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single molecule", "Batch (CSV)", "Model explain", "Toxicity & BCS"])

# ---------- TAB 1 ----------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Drug ka naam (English, jaise ibuprofen)", key="name_in")
        st.button("PubChem se SMILES lao", on_click=fetch_smiles)
        if "msg" in st.session_state:
            st.info(st.session_state["msg"])
        st.text_input("SMILES", key="smiles")
        go = st.button("Predict", type="primary")

    if go:
        smi = st.session_state["smiles"].strip()
        d = get_desc(smi) if smi else None
        if d is None:
            st.error("Invalid SMILES.")
        else:
            mean, std = predict(model, d)
            with c2:
                st.subheader("Predicted logS: {:.2f} (± {:.2f})".format(mean, std))
                lab = sol_class(mean)
                if lab == "Soluble":
                    st.success(lab)
                elif lab == "Moderately soluble":
                    st.warning(lab)
                else:
                    st.error(lab)
                st.image(Draw.MolToImage(Chem.MolFromSmiles(smi), size=(300, 300)))
            st.markdown("#### Lipinski Rule of 5")
            rules = lipinski(d)
            st.table(pd.DataFrame({
                "Rule": list(rules.keys()),
                "Pass?": ["Yes" if v else "No" for v in rules.values()],
            }))
            fails = sum(not v for v in rules.values())
            if fails <= 1:
                st.success("Drug-like")
            else:
                st.error("{} rules fail".format(fails))
            st.markdown("#### Descriptors")
            st.table(pd.DataFrame({
                "Descriptor": FEATURES,
                "Value": [round(x, 2) for x in d],
            }))

# ---------- TAB 2 ----------
with tab2:
    st.write("CSV upload karo. **smiles** column zaruri hai (optional: **name**).")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        data.columns = [c.lower() for c in data.columns]
        if "smiles" not in data.columns:
            st.error("'smiles' column nahi mila.")
        else:
            rows = []
            for i, r in data.iterrows():
                d = get_desc(r["smiles"])
                if "name" in data.columns:
                    nm = r["name"]
                else:
                    nm = "mol_{}".format(i + 1)
                if d is None:
                    rows.append({"name": nm, "smiles": r["smiles"], "pred_logS": None,
                                 "class": "invalid", "lipinski_fails": None, "BCS": None})
                else:
                    m_, s_ = predict(model, d)
                    rows.append({
                        "name": nm,
                        "smiles": r["smiles"],
                        "pred_logS": round(m_, 2),
                        "class": sol_class(m_),
                        "lipinski_fails": sum(not v for v in lipinski(d).values()),
                        "BCS": bcs_class(m_, d[1], d[2]).split(":")[0],
                    })
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button("Results CSV download",
                               out.to_csv(index=False).encode("utf-8"),
                               "batch_predictions.csv", "text/csv")

# ---------- TAB 3 ----------
with tab3:
    st.markdown("#### Model ne kis descriptor pe zyada bharosa kiya?")
    imp = pd.DataFrame({"Descriptor": FEATURES,
                        "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).set_index("Descriptor")
    st.bar_chart(imp)
    st.write("LogP sabse important aana chahiye: lipophilic molecule = kam water solubility.")

# ---------- TAB 4 ----------
with tab4:
    cur = st.session_state["smiles"].strip()
    st.code(cur if cur else "(khali)")
    d4 = get_desc(cur) if cur else None
    if d4 is None:
        st.error("Pehle Tab 1 mein valid SMILES daalo.")
    else:
        m4, s4 = predict(model, d4)
        st.info("{} | predicted logS = {:.2f}".format(bcs_class(m4, d4[1], d4[2]), m4))
        st.caption("BCS yahan sirf rough andaza hai.")
        task = st.selectbox("Tox21 assay", TOX_TASKS, index=10)
        if st.button("Toxicity check karo", type="primary"):
            with st.spinner("Pehli baar 1-2 minute lagega..."):
                clf, auc = load_tox(task)
            prob = float(clf.predict_proba(np.array([d4]))[0][1])
            st.subheader("{} active probability: {:.0%}".format(task, prob))
            st.progress(prob)
            st.caption("Test AUC: {:.2f}. Cell-based assay hai, insaan mein asli toxicity nahi.".format(auc))