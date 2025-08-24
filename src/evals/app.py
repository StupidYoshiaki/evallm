import streamlit as st
import pandas as pd
import json
import sys
import re
from pathlib import Path

# --- 評価スクリプトから移植した正規化関数 ---
def normalize_answer(answer: str) -> str:
    """
    回答文字列を正規化する（末尾の句点・ピリオドの削除、空白除去）。
    evaluate.pyのロジックと完全に一致させています。
    """
    if not isinstance(answer, str):
        return ""
    normalized = answer.strip()
    normalized = re.sub(r"[。\.]+$", "", normalized)
    return normalized

# --- データ読み込み（複数モデル対応 + 正しい正解率計算） ---
@st.cache_data
def load_all_data(results_dir: Path, judge_filepath: Path):
    """
    評価ファイルと、ディレクトリ内の各モデルの予測ファイルを読み込み、
    1つの統合されたDataFrameを生成する。
    """
    try:
        # 1. 評価基準となるjudgeファイルを読み込む
        with judge_filepath.open('r', encoding='utf-8') as f:
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
        
        base_df = pd.DataFrame(processed_records)
        if 'id' not in base_df.columns:
            st.error(f"{judge_filepath.name}に 'id' 列がありません。")
            return pd.DataFrame(), []

        base_df['page_id'] = base_df['id'].apply(lambda x: x.rsplit('q', 1)[0] if isinstance(x, str) else None)
        
        # 'is_contained' (基準3) の計算
        context_map = base_df.groupby('page_id')['context'].first().to_dict()
        page_contexts = base_df['page_id'].map(context_map)
        results = [
            (ans in ctx if isinstance(ans, str) and isinstance(ctx, str) else False)
            for ans, ctx in zip(base_df['answer'], page_contexts)
        ]
        base_df['is_contained'] = results

        # 2. 各モデルのprediction.jsonlを読み込んでマージ
        model_names = []
        for model_dir in results_dir.iterdir():
            pred_file = model_dir / "prediction.jsonl"
            if model_dir.is_dir() and pred_file.exists():
                model_name = model_dir.name
                model_names.append(model_name)
                pred_df = pd.read_json(pred_file, lines=True, dtype={'id': str})
                pred_df = pred_df[['id', 'answer']].rename(columns={'answer': f'answer_{model_name}'})
                base_df = pd.merge(base_df, pred_df, on='id', how='left')

        # 3. 各モデルの正解/不正解を判定（正規化を適用）
        correct_cols = []
        for model_name in model_names:
            correct_col = f'correct_{model_name}'
            answer_col = f'answer_{model_name}'
            # 正規化した上で回答を比較
            base_df[correct_col] = base_df.apply(
                lambda row: normalize_answer(row.get('answer')) == normalize_answer(row.get(answer_col)),
                axis=1
            )
            correct_cols.append(correct_col)

        # 4. 設問ごとの平均正解率を計算
        if correct_cols:
            base_df['avg_accuracy'] = base_df[correct_cols].mean(axis=1)
        else:
            base_df['avg_accuracy'] = 0.0
        
        # 共通の処理
        base_df['clarity_score'] = pd.to_numeric(base_df['clarity_score'], errors='coerce')
        base_df['grounding_score'] = pd.to_numeric(base_df['grounding_score'], errors='coerce')
        
        return base_df, sorted(model_names)
        
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        st.exception(e)
        return pd.DataFrame(), []

# --- UIコンポーネント ---
def display_score_badge(score, label):
    if pd.isna(score): color, text = "#808080", "N/A"
    elif score == 1: color, text = "#28a745", "Good"
    else: color, text = "#dc3545", "Bad"
    st.markdown(f'<span style="background-color: {color}; color: white; padding: 3px 10px; border-radius: 15px;">{label}: {text}</span>', unsafe_allow_html=True)

def display_boolean_badge(is_contained, label):
    if is_contained: color, text = "#28a745", "Contained"
    else: color, text = "#dc3545", "Not Contained"
    st.markdown(f'<span style="background-color: {color}; color: white; padding: 3px 10px; border-radius: 15px;">{label}: {text}</span>', unsafe_allow_html=True)

# --- 全体サマリー表示関数 ---
def display_overall_summary(df, model_names):
    st.header("📊 全体結果サマリー")
    total_items = len(df)
    if total_items == 0: return
        
    st.metric(label="総データ数", value=f"{total_items} 件")
    st.divider()

    st.subheader("LLM-as-a-Judge 評価基準")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("###### 基準1: 質問の明瞭性 (Clarity)")
        clarity_good = df["clarity_score"].eq(1).sum()
        st.metric(label="Good率", value=f"{(clarity_good / total_items) * 100:.1f} %")
    with col2:
        st.markdown("###### 基準2: 文脈への準拠度 (Grounding)")
        grounding_good = df["grounding_score"].eq(1).sum()
        st.metric(label="Good率", value=f"{(grounding_good / total_items) * 100:.1f} %")
    with col3:
        st.markdown("###### 基準3: 回答の文脈包含率 (Containment)")
        contained_good = df['is_contained'].sum()
        st.metric(label="包含率", value=f"{(contained_good / total_items) * 100:.1f} %")
    
    st.divider()

    st.header("🏆 モデル別 正解率ランキング")
    if not model_names:
        st.warning("比較対象のモデルが見つかりませんでした。")
    else:
        accuracies = {model: (df[f'correct_{model}'].sum() / total_items) * 100 for model in model_names}
        acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy (%)'])
        acc_df = acc_df.sort_values('Accuracy (%)', ascending=False)
        st.dataframe(acc_df.style.format("{:.2f}%").background_gradient(cmap='Greens'), use_container_width=True)

# --- ページ別詳細表示関数 ---
def display_page_details(page_df, model_names):
    context = page_df.iloc[0]['context']
    st.header("📄 ページコンテキスト")
    st.text_area("Context", value=context, height=150, disabled=True)
    st.divider()

    st.header("個別の評価結果リスト")
    
    for _, item in page_df.iterrows():
        # Judgeによる評価結果のサマリーバッジ
        avg_acc = item.get('avg_accuracy', 0) * 100
        c_score, g_score, is_cont = item['clarity_score'], item['grounding_score'], item['is_contained']
        if c_score == 1 and g_score == 1 and is_cont: badge = f"✅ All Passed, Avg Acc: {avg_acc:.1f}%"
        else:
            c_b = "👍" if c_score == 1 else "👎" if c_score == 0 else "❔"
            g_b = "👍" if g_score == 1 else "👎" if g_score == 0 else "❔"
            a_b = "👍" if is_cont else "👎"
            badge = f"Clarity:{c_b}, Grounding:{g_b}, Containment:{a_b}, Avg Acc:{avg_acc:.1f}%"

        with st.expander(f"**ID:** {item['id']}  —  **Judge Eval:** {badge}"):
            st.markdown("##### LLM-as-a-Judgeによる評価")
            b1, b2, b3 = st.columns(3)
            with b1: display_score_badge(c_score, "Clarity")
            with b2: display_score_badge(g_score, "Grounding")
            with b3: display_boolean_badge(is_cont, "Containment")

            ### ### 変更箇所: 評価理由の表示機能を追加 ### ###
            st.markdown("##### 評価理由")
            r1, r2 = st.columns(2)
            with r1:
                st.info(f"**Clarity:** {item.get('clarity_reason', 'N/A')}")
            with r2:
                st.info(f"**Grounding:** {item.get('grounding_reason', 'N/A')}")
            ### ### ここまで ### ###
            
            st.markdown(f"**❓ Question:**\n> {item['question']}")
            st.markdown(f"**💡 Judge Answer:**")
            st.success(f"{item['answer']}")
            
            st.markdown("---")
            
            st.metric(label="全モデル平均正解率", value=f"{avg_acc:.1f} %")

            st.markdown("##### 各モデルの回答と正誤")
            for model_name in model_names:
                model_answer = item.get(f'answer_{model_name}', "N/A")
                is_correct = item.get(f'correct_{model_name}', False)
                icon = "✅" if is_correct else "❌"
                st.markdown(f"`{icon}` **{model_name}:** \n> {model_answer}")

# --- メインアプリケーション（ページ送り機能などを復元） ---
def main():
    st.set_page_config(layout="wide")
    st.title("🤖 LLM Judge & Model Comparison Viewer")

    try:
        results_dir = Path(sys.argv[1])
        judge_filepath = Path(sys.argv[2])
    except IndexError:
        st.error("エラー: 2つのパスをコマンドラインで指定してください。")
        st.info("例: streamlit run app.py <結果ディレクトリへのパス> <評価ファイルへのパス>")
        return
    
    st.sidebar.info(f"結果ディレクトリ: \n`{results_dir}`")
    st.sidebar.info(f"評価ファイル: \n`{judge_filepath}`")

    if not results_dir.is_dir() or not judge_filepath.is_file():
        st.error(f"エラー: 指定されたパスが見つからないか、ディレクトリ/ファイルではありません。")
        return

    df, model_names = load_all_data(results_dir, judge_filepath)
    if df.empty: return

    st.sidebar.header("表示選択")
    # page_idがNoneのものを除外
    valid_page_ids = sorted([pid for pid in df['page_id'].unique() if pid is not None])
    page_id_list = ["Overall"] + valid_page_ids
    
    # セッションステートを利用したページ管理
    if 'page_index' not in st.session_state:
        st.session_state.page_index = 0

    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("⬅️ 前へ", use_container_width=True):
        st.session_state.page_index = max(0, st.session_state.page_index - 1)
        st.rerun()
    if col_next.button("次へ ➡️", use_container_width=True):
        st.session_state.page_index = min(len(page_id_list) - 1, st.session_state.page_index + 1)
        st.rerun()

    selected_page_id = st.sidebar.selectbox(
        "ページIDまたはOverallを選択",
        options=page_id_list,
        index=st.session_state.page_index,
        key="page_selector"
    )
    # selectboxの変更をsession_stateに反映
    if page_id_list.index(selected_page_id) != st.session_state.page_index:
        st.session_state.page_index = page_id_list.index(selected_page_id)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.info(f"表示中: `{selected_page_id}`")
    
    if selected_page_id == "Overall":
        display_overall_summary(df, model_names)
    else:
        page_df = df[df['page_id'] == selected_page_id].copy()
        display_page_details(page_df, model_names)

if __name__ == "__main__":
    main()
