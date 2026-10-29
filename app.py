import json

import joblib
import pandas as pd
import streamlit as st

from features import make_features, load_data, Maps

st.set_page_config(page_title="CSGO Analyst", page_icon="csgo_logo.png", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load("model.pkl")


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_results():
    comparison = pd.read_csv("model_comparison.csv")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return comparison, metrics


bundle = get_model()
data = get_data()
comparison, metrics = get_results()

col_logo, col_title = st.columns([1, 12], vertical_alignment="center")
with col_logo:
    st.image("csgo_logo.png", width=80)
with col_title:
    st.title("CSGO Analyst")

tab_pred, tab_dash, tab_model = st.tabs(["🔮 Dự đoán trận", "📊 Phân tích", "🧠 Về mô hình"])

# ======================= TAB 1: DỰ ĐOÁN =======================
with tab_pred:
    st.subheader("Nhập chỉ số cuối trận")

    c1, c2, c3 = st.columns(3)
    with c1:
        map_name = st.selectbox("Map", Maps + ["Other"])
        ping = st.slider("Ping", 0, 300, 75)
        points = st.slider("Points", 0, 100, 40)
    with c2:
        kills = st.slider("Kills", 0, 40, 15)
        deaths = st.slider("Deaths", 0, 30, 17)
        assists = st.slider("Assists", 0, 15, 4)
    with c3:
        mvps = st.slider("MVPs", 0, 10, 2)
        hs = st.slider("Headshot %", 0, 100, 25)

    row = pd.DataFrame([{
        "map": map_name, "ping": ping, "kills": kills, "assists": assists,
        "deaths": deaths, "mvps": mvps, "hs_percent": hs, "points": points,
    }])

    X_new = make_features(row)
    p_win = float(bundle["model"].predict_proba(X_new)[0][1])

    st.divider()
    left, right = st.columns([1, 1.4])

    with left:
        st.metric("Xác suất THẮNG", f"{p_win:.1%}")
        st.progress(p_win)
        if p_win > 0.6:
            st.success("Nghiêng về THẮNG")
        elif p_win < 0.4:
            st.error("Nghiêng về THUA")
        else:
            st.warning("Sát nút (40-60%), model không chắc chắn")
        st.caption(f"K/D của trận này: {kills / max(deaths, 1):.2f}")

    with right:
        st.markdown("**Yếu tố đang đẩy kết quả (ước lượng bằng Logistic Regression)**")
        ex = bundle["explainer"]
        z = ex.named_steps["scaler"].transform(X_new)[0]
        contrib = pd.Series(ex.named_steps["clf"].coef_[0] * z, index=bundle["features"])
        contrib = contrib[~contrib.index.str.startswith("map_")]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
        st.bar_chart(top, horizontal=True)
        st.caption("Dương = tăng khả năng thắng, âm = giảm. Đây là giải thích gần đúng; "
                   "model dự đoán chính là ensemble nhiều thuật toán.")

    st.info("⚠️ Đây là chỉ số cuối trận: app trả lời \"với số liệu thế này thì "
            "khả năng đã thắng bao nhiêu\", không phải dự đoán trước khi trận bắt đầu.")

# ======================= TAB 2: PHÂN TÍCH =======================
with tab_dash:
    d = data.copy()
    d["kd"] = d["kills"] / d["deaths"].clip(lower=1)

    a, b, c, e = st.columns(4)
    a.metric("Số trận (không tính Tie)", len(d))
    b.metric("Tỉ lệ thắng", f"{d['win'].mean():.1%}")
    c.metric("K/D trung bình", f"{d['kd'].mean():.2f}")
    e.metric("HS% trung bình", f"{d['hs_percent'].mean():.1f}%")

    # ---- Tỉ lệ thắng theo map ----
    st.markdown("#### Tỉ lệ thắng theo map")
    by_map = d.groupby("map").agg(so_tran=("win", "size"), ti_le_thang=("win", "mean"))
    by_map = by_map.sort_values("so_tran", ascending=False)

    min_n = st.slider("Chỉ hiện map có ít nhất N trận", 1, 50, 10)
    by_map = by_map[by_map["so_tran"] >= min_n]

    m1, m2 = st.columns([1.3, 1])
    m1.bar_chart(by_map["ti_le_thang"])
    m2.dataframe(by_map.style.format({"ti_le_thang": "{:.1%}"}), width="stretch")

    # ---- Phong độ theo thời gian ----
    st.markdown("#### Phong độ theo thời gian")
    freq = st.radio("Gom theo", ["Tháng", "Quý"], horizontal=True)
    rule = "MS" if freq == "Tháng" else "QS"

    t = d.set_index("date_dt").resample(rule).agg(
        ti_le_thang=("win", "mean"), kd=("kd", "mean"), so_tran=("win", "size"))
    t = t[t["so_tran"] >= 3]

    t1, t2 = st.columns(2)
    t1.line_chart(t["ti_le_thang"])
    t1.caption("Tỉ lệ thắng")
    t2.line_chart(t["kd"])
    t2.caption("K/D trung bình")

    # ---- Ping ảnh hưởng thế nào ----
    st.markdown("#### Ping ảnh hưởng thế nào?")
    d["nhom_ping"] = pd.cut(d["ping"], [-1, 50, 80, 110, 1000],
                            labels=["≤50", "51-80", "81-110", ">110"])
    pg = d.groupby("nhom_ping", observed=True).agg(
        so_tran=("win", "size"), ti_le_thang=("win", "mean"))

    p1, p2 = st.columns([1.3, 1])
    p1.bar_chart(pg["ti_le_thang"])
    p2.dataframe(pg.style.format({"ti_le_thang": "{:.1%}"}), width="stretch")

    # ---- Trận thắng và trận thua ----
    st.markdown("#### Trận thắng và trận thua khác nhau ra sao?")
    cols = ["kills", "deaths", "assists", "mvps", "hs_percent", "points", "ping"]
    cmp_ = d.groupby("result")[cols].mean().round(1)
    st.dataframe(cmp_, width="stretch")

# ======================= TAB 3: MÔ HÌNH =======================
with tab_model:
    st.subheader("So sánh thuật toán (cross-validation 5 fold × 3 lần)")

    show = comparison.copy()
    show["Accuracy"] = show.apply(
        lambda r: f"{r['Accuracy']:.1%} ± {r['Accuracy_std']:.1%}", axis=1)
    show = show.drop(columns="Accuracy_std")
    st.dataframe(show.style.format({"F1": "{:.3f}", "ROC-AUC": "{:.3f}"}),
                 width="stretch", hide_index=True)

    st.bar_chart(comparison.set_index("Model")["Accuracy"])
    st.caption("Các cột gần như bằng nhau: các thuật toán ngang nhau về accuracy.")

    st.markdown("#### Chấm trên tập test 20% (chưa từng thấy lúc học)")
    h1, h2, h3 = st.columns(3)
    h1.metric("Accuracy", f"{metrics['holdout_accuracy']:.1%}")
    h2.metric("F1", f"{metrics['holdout_f1']:.3f}")
    h3.metric("ROC-AUC", f"{metrics['holdout_auc']:.3f}")
    st.caption(f"Học trên {metrics['n_train']} trận, chấm trên {metrics['n_test']} trận.")

    st.markdown(
        "**Model đang dùng:** Ensemble (voting mềm) gồm Logistic Regression, SVC, "
        "Gradient Boosting và Random Forest, huấn luyện lại trên toàn bộ dữ liệu."
    )
    st.info(
        "**Lưu ý trung thực:** các thuật toán đều đạt khoảng 79-81%, chênh nhau nhỏ hơn "
        "độ dao động giữa các lần chia dữ liệu (±2%). Với khoảng 1000 trận và 7 chỉ số, "
        "đây gần như là giới hạn của bộ chỉ số hiện có. Muốn cao hơn cần thêm thông tin mới "
        "(đồng đội, đối thủ, vai trò, số round...), không phải thuật toán phức tạp hơn."
    )