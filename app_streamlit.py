"""
app_streamlit.py
واجهة Streamlit لتحميل CSV، ضبط معلمات GA، تشغيله، وعرض/تحميل النتائج.
"""

import io
import json
import time
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ga_feature_selection import SimpleGA, baseline_select_kbest, baseline_rfe, save_json

st.set_page_config(page_title="اختيار الميزات - GA", layout="wide")
st.title("اختيار الميزات باستخدام الخوارزمية الجينية (Genetic Algorithm)")

st.markdown("""
هذا التطبيق يسمح بتحميل ملف CSV (ميزات + عمود هدف)، تشغيل GA لاختيار الميزات، 
والقارنة مع طريقتي SelectKBest وRFE.
""")

with st.sidebar:
    st.header("معلمات GA (افتراضية معقولة)")
    pop_size = st.number_input("Population size", min_value=10, max_value=200, value=40, step=5)
    generations = st.number_input("Generations", min_value=5, max_value=300, value=30, step=5)
    cx_prob = st.slider("Crossover probability", 0.0, 1.0, 0.6)
    mut_prob = st.slider("Mutation probability", 0.0, 0.5, 0.02)
    elitism = st.number_input("Elitism (top-k preserved)", min_value=0, max_value=10, value=2)
    alpha = st.number_input("Alpha (penalty per fraction of features)", min_value=0.0, max_value=0.5, value=0.01, step=0.005)
    cv = st.number_input("CV folds", min_value=2, max_value=10, value=5)
    n_jobs = st.number_input("Estimator n_estimators (RF)", min_value=10, max_value=1000, value=150, step=10)

uploaded = st.file_uploader("ارفع ملف CSV (يحتوي ميزات وعمود الهدف)", type=["csv"])
example_btn = st.button("استخدم مثال (breast_cancer)")

if uploaded is not None or example_btn:
    if example_btn:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    else:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"فشل في قراءة CSV: {e}")
            st.stop()

    st.subheader("معاينة البيانات")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    target_col = st.selectbox("اختر عمود الهدف (Target)", options=cols, index=len(cols)-1)
    if st.button("تشغيل GA"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        estimator = RandomForestClassifier(n_estimators=int(n_jobs), random_state=42, n_jobs=-1)
        ga = SimpleGA(X, y,
                     pop_size=pop_size,
                     generations=generations,
                     cx_prob=cx_prob,
                     mut_prob=mut_prob,
                     elitism=elitism,
                     estimator=estimator,
                     alpha=alpha,
                     cv=cv,
                     random_state=42,
                     verbose=False)
        st.info("جارٍ تشغيل الخوارزمية... (يرجى الانتظار حتى اكتمال التنفيذ)")
        start = time.time()
        res = ga.run(save_history=True)
        end = time.time()
        st.success(f"انتهى التنفيذ. أفضل دقة متقاطعة: {res['best_raw_score']:.4f} — الوقت: {end-start:.1f}s")
        st.markdown("### الميزات المختارة")
        st.write(f"عدد الميزات: {res['n_selected']} / {res['n_total_features']}")
        st.write(res['selected_features'])

        # show history chart
        hist_df = pd.DataFrame(res['history'])
        hist_df = hist_df.set_index("generation")
        st.line_chart(hist_df[["best_raw_score", "mean_fitness"]])

        # compare baselines
        st.markdown("### المقارنة مع الأساليب التقليدية")
        k = min(10, X.shape[1])
        kbest = baseline_select_kbest(X, y, k=k, estimator=estimator, cv=cv)
        rfe = baseline_rfe(X, y, n_features_to_select=k, estimator=estimator, cv=cv)
        st.write(f"SelectKBest (k={k}) score: {kbest['score']:.4f} — features: {len(kbest['feature_names'])}")
        st.write(kbest['feature_names'])
        st.write(f"RFE (n={k}) score: {rfe['score']:.4f} — features: {len(rfe['feature_names'])}")
        st.write(rfe['feature_names'])

        # provide download of result JSON
        json_bytes = json.dumps(res, ensure_ascii=False, indent=2).encode('utf-8')
        st.download_button("تحميل نتيجة GA (JSON)", data=json_bytes, file_name="ga_result.json", mime="application/json")
else:
    st.info("ارفع ملف CSV أو اضغط 'استخدم مثال' لتجربة التطبيق.")
