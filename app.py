import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 앱 제목
st.title("🧬 단백질 구조 시각화기")

# UniProt ID 입력 받기
protein_id = st.text_input("🔎 단백질 ID (예: P04406)", "P04406")

# UniProt API에서 데이터 가져오기
def get_uniprot_data(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# 단백질 피처 시각화
def draw_protein_features(data):
    sequence = data.get("sequence", {}).get("value", "")
    protein_length = len(sequence)
    features = data.get("features", [])

    # 색상 정의
    colors = {
        "Domain": "skyblue",
        "Region": "orange",
        "Signal peptide": "lightgreen",
        "Topological domain": "plum",
        "Transmembrane": "tomato",
        "Repeat": "gold"
    }

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, protein_length)
    ax.set_ylim(0, 10)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"{data['proteinDescription']['recommendedName']['fullName']['value']} ({data['primaryAccession']})")

    # 기본 단백질 바
    ax.add_patch(patches.Rectangle((0, 4), protein_length, 2, color="lightgray", edgecolor="black"))

    # 피처별로 시각화
    for feat in features:
        try:
            start = int(feat["location"]["start"]["value"])
            end = int(feat["location"]["end"]["value"])
            width = end - start
            label = feat.get("description", feat["type"])
            color = colors.get(feat["type"], "gray")

            ax.add_patch(patches.Rectangle((start, 4), width, 2, color=color, edgecolor='black'))
            ax.text(start + width/2, 6.5, label, ha='center', va='bottom', fontsize=8, rotation=30)
        except:
            continue

    st.pyplot(fig)

# 동작 로직
if protein_id:
    data = get_uniprot_data(protein_id)
    if data:
        st.success("✅ 단백질 정보를 성공적으로 가져왔습니다.")
        draw_protein_features(data)
    else:
        st.error("❌ 단백질 정보를 불러올 수 없습니다. UniProt ID를 확인해주세요.")
