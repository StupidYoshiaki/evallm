import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

# --- データ読み込み（変更なし） ---
@st.cache_data
def load_data_from_path(filepath: Path):
    """
    指定されたファイルパスからJSONLファイルを読み込み、
    ネストされた評価結果をフラットなDataFrameに変換する。
    """
    try:
        with filepath.open('r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        
        processed_records = []
        for r in records:
            qc = r.get("question_clarity", {})
            r["clarity_score"] = qc.get("score")
            r["clarity_reason"] = qc.get("reason")
            
            cg = r.get("context_grounding", {})
            r["grounding_score"] = cg.get("score")
            r["grounding_reason"] = cg.get("reason")
            
            processed_records.append(r)

        df = pd.DataFrame(processed_records)
        
        df['clarity_score'] = pd.to_numeric(df['clarity_score'], errors='coerce')
        df['grounding_score'] = pd.to_numeric(df['grounding_score'], errors='coerce')
        
        df['page_id'] = df['id'].apply(lambda x: x.rsplit('q', 1)[0] if isinstance(x, str) else None)
        return df
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()

# --- UIコンポーネント（変更なし） ---
def display_score_badge(score, label):
    """スコアとラベル（基準名）に応じてバッジを表示"""
    if pd.isna(score):
        badge_color = "#808080" # 灰色
        score_text = "N/A"
    elif score == 1:
        badge_color = "#28a745" # 緑
        score_text = "Good"
    else: # score == 0
        badge_color = "#dc3545" # 赤
        score_text = "Bad"
    
    st.markdown(
        f'<span style="background-color: {badge_color}; color: white; padding: 3px 10px; border-radius: 15px; margin-right: 5px;">{label}: {score_text}</span>', 
        unsafe_allow_html=True
    )

# --- 全体サマリー表示関数（変更なし） ---
def display_overall_summary(df):
    st.header("📊 全体結果サマリー")
    
    total_items = len(df)
    st.metric(label="総データ数", value=f"{total_items} 件")
    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("基準1: 質問の明瞭性 (Clarity)")
        clarity_good = df[df["clarity_score"] == 1].shape[0]
        st.metric(label="Good率", value=f"{(clarity_good / total_items) * 100:.1f} %")
        st.bar_chart(df["clarity_score"].value_counts().sort_index().rename(index={1.0: "Good", 0.0: "Bad"}))

    with col2:
        st.subheader("基準2: 文脈への準拠度 (Grounding)")
        grounding_good = df[df["grounding_score"] == 1].shape[0]
        st.metric(label="Good率", value=f"{(grounding_good / total_items) * 100:.1f} %")
        st.bar_chart(df["grounding_score"].value_counts().sort_index().rename(index={1.0: "Good", 0.0: "Bad"}))

# --- ページ別詳細表示関数（ロジック修正） ---
def display_page_details(page_df):
    st.header("📊 ページサマリー")
    num_qa = len(page_df)
    
    clarity_good = page_df[page_df['clarity_score'] == 1].shape[0]
    grounding_good = page_df[page_df['grounding_score'] == 1].shape[0]
    clarity_rate = (clarity_good / num_qa) * 100 if num_qa > 0 else 0
    grounding_rate = (grounding_good / num_qa) * 100 if num_qa > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("QAペア総数", num_qa)
    c2.metric("Clarity Good率", f"{clarity_rate:.1f} %")
    c3.metric("Grounding Good率", f"{grounding_rate:.1f} %")
    st.divider()
    
    context = page_df.iloc[0]['context']
    st.header("📄 ページコンテキスト")
    st.text_area("Context", value=context, height=150, disabled=True)
    st.divider()

    st.header("個別の評価結果リスト")
    
    for _, item in page_df.iterrows():
        # ### 変更箇所 ###
        # 2つのスコアを取得
        clarity_score = item['clarity_score']
        grounding_score = item['grounding_score']
        
        # 両方のスコアが1の場合のバッジ
        if clarity_score == 1 and grounding_score == 1:
            badge = "✅ Passed"
        # それ以外の場合のバッジ
        else:
            c_badge = "👍" if clarity_score == 1 else "👎" if clarity_score == 0 else "❔"
            g_badge = "👍" if grounding_score == 1 else "👎" if grounding_score == 0 else "❔"
            badge = f"C:{c_badge} / G:{g_badge}"

        expander_label = f"**ID:** {item['id']} - {badge}"
        # ### ここまで ###
        
        with st.expander(expander_label):
            st.markdown("**評価結果**")
            b1, b2 = st.columns(2)
            with b1:
                display_score_badge(clarity_score, "Clarity")
            with b2:
                display_score_badge(grounding_score, "Grounding")
            
            st.markdown("**評価理由**")
            r1, r2 = st.columns(2)
            with r1:
                st.info(f"**Clarity:** {item.get('clarity_reason', 'N/A')}")
            with r2:
                st.info(f"**Grounding:** {item.get('grounding_reason', 'N/A')}")
            
            st.markdown("---")
            st.markdown(f"**❓ Question:**")
            st.markdown(f"> {item['question']}")
            st.markdown(f"**💡 Answer:**")
            st.success(item['answer'])

# --- メインアプリケーション（変更なし） ---
def main():
    st.set_page_config(layout="wide")
    st.title("🤖 LLM-as-a-Judge 結果ビューア")

    try:
        filepath = Path(sys.argv[1])
    except IndexError:
        st.error("エラー: コマンドラインでJSONLファイルのパスを指定してください。")
        st.info("例: streamlit run app.py /path/to/your/file.jsonl")
        return
    
    st.sidebar.info(f"パス名: \n{filepath}")

    if not filepath.is_file():
        st.error(f"エラー: 指定されたファイルが見つかりません: {filepath}")
        return

    df = load_data_from_path(filepath)
    if df.empty:
        return

    st.sidebar.header("表示選択")
    page_id_list = ["Overall"] + sorted(df['page_id'].unique().tolist())
    
    if 'page_index' not in st.session_state:
        st.session_state.page_index = 0

    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("⬅️ 前へ", use_container_width=True):
        st.session_state.page_index = max(0, st.session_state.page_index - 1)
    if col_next.button("次へ ➡️", use_container_width=True):
        st.session_state.page_index = min(len(page_id_list) - 1, st.session_state.page_index + 1)

    selected_page_id = st.sidebar.selectbox(
        "ページIDまたはOverallを選択",
        options=page_id_list,
        index=st.session_state.page_index,
    )
    st.session_state.page_index = page_id_list.index(selected_page_id)
    
    st.sidebar.divider()
    st.sidebar.info(f"表示中: `{selected_page_id}`")

    if selected_page_id == "Overall":
        display_overall_summary(df)
    else:
        page_df = df[df['page_id'] == selected_page_id].copy()
        display_page_details(page_df)

if __name__ == "__main__":
    main()
