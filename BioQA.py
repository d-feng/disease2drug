import os
import sys
import json
import logging
import warnings
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Silence OWL parser and logging warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pronto").setLevel(logging.CRITICAL)
logging.getLogger("rdflib").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)

# Suppress parser warnings by redirecting stderr
class NullWriter:
    def write(self, s): pass
    def flush(self): pass
sys.stderr = NullWriter()

# Set OnToma cache directory
os.environ["ONTOLOGY_INDEX_CACHE_DIR"] = os.path.join(Path.home(), ".ontoma_cache")

# Core matching function
def match_disease_to_phenotype_llm(disease_term: str, tsv_path="db/clinicalAnnotations/clinical_annotations.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")
    df.dropna(subset=["Phenotype(s)"], inplace=True)
    phenotypes = list(df["Phenotype(s)"].unique())

    joined_phenos = "\n".join(f"- {p}" for p in phenotypes)
    prompt = (
        f"You are a biomedical assistant. A user is searching for a disease: '{disease_term}'.\n"
        f"From the list below, identify the phenotype that most closely matches this disease:\n\n"
        f"{joined_phenos}\n\n"
        f"Only return the matching phenotype string exactly as it appears above. Do not invent or modify anything."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        matched = response.choices[0].message.content.strip()

        similarity_scores = [(p, SequenceMatcher(None, matched, p).ratio()) for p in phenotypes]
        threshold = 0.75
        top_matches = [p for p, score in similarity_scores if score >= threshold]

        all_drugs = set()
        table_rows = []

        if top_matches:
            for match in top_matches:
                matching_rows = df[df["Phenotype(s)"].str.contains(match, na=False)]
                for _, row in matching_rows.iterrows():
                    drugs = str(row["Drug(s)"]).split(';') if pd.notna(row["Drug(s)"]) else []
                    for d in drugs:
                        table_rows.append({"Drug": d.strip(), "Phenotype": row["Phenotype(s)"]})
                        all_drugs.add(d.strip())
            return ", ".join(top_matches), sorted(all_drugs), pd.DataFrame(table_rows)
        else:
            for _, row in df.iterrows():
                pheno_list = str(row["Phenotype(s)"]).split(';')
                for p in pheno_list:
                    score = SequenceMatcher(None, matched, p.strip()).ratio()
                    if score >= threshold:
                        drug_list = str(row["Drug(s)"]).split(';') if pd.notna(row["Drug(s)"]) else []
                        for d in drug_list:
                            table_rows.append({"Drug": d.strip(), "Phenotype": row["Phenotype(s)"]})
                            all_drugs.add(d.strip())
            return matched, sorted(all_drugs) if all_drugs else (None, []), pd.DataFrame(table_rows)

    except Exception as e:
        return None, [], pd.DataFrame()

# --- Streamlit UI ---
st.set_page_config(page_title="Disease2Drug | Find Drugs from Disease Term", layout="centered")

st.title("💊 Disease2Drug")
st.markdown("### *Find the drugs for any disease term.*")
st.markdown(
    """
    Disease2Drug helps you explore drug–phenotype associations using natural language input.  
    Powered by LLMs and curated datasets from **PharmGKB**.
    """
)

user_input = st.text_input("🔍 Enter a disease name", "asthma")

if user_input:
    label, drugs, df_table = match_disease_to_phenotype_llm(user_input)
    if drugs:
        st.success(f"🎯 Phenotype matched: **{label}**")
        st.subheader("💊 Matched Drugs:")
        for d in drugs:
            st.markdown(f"- {d}")

        if not df_table.empty:
            st.subheader("📊 Drug–Phenotype Mapping Table")
            st.dataframe(df_table.reset_index(drop=True))
    else:
        st.warning("⚠️ No drugs found for the entered disease term.")

st.markdown(
    """
    ---
    #### 📘 About this App
    - 🔗 Data source: [PharmGKB](https://www.pharmgkb.org/)
    - 📄 License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
    - 🧠 Model: LLM-matched phenotype search (OpenAI GPT-4o)
    """
)
