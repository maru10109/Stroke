import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧠",
    layout="wide"
)

st.title("⚙️ 분류 모델 학습 및 평가")
st.markdown("로지스틱 회귀 모델과 의사결정트리 모델을 사용하여 뇌졸중 발생 여부를 예측합니다.")
st.divider()

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 우리말 이름과 실제 열 이름 매핑
feature_map = {
    '나이': 'age',
    '평균 혈당': 'avg_glucose_level',
    '체질량지수': 'bmi',
    '고혈압': 'hypertension',
    '심장병': 'heart_disease'
}
reverse_feature_map = {v: k for k, v in feature_map.items()}

df = load_data()

if not df.empty:
    st.subheader("1. 입력 속성 선택")
    st.markdown("모델 학습에 사용할 속성을 2개 이상 선택해 주세요.")
    
    all_features_kr = list(feature_map.keys())
    default_features_kr = ['나이', '평균 혈당', '고혈압', '심장병'] # 체질량지수(bmi) 제외
    
    selected_kr = st.multiselect(
        "입력 속성",
        options=all_features_kr,
        default=default_features_kr
    )
    
    if len(selected_kr) < 2:
        st.warning("입력 속성을 2개 이상 골라야 분석을 진행할 수 있습니다.")
        st.stop()
        
    selected_en = [feature_map[k] for k in selected_kr]

    # 번호(id) 순으로 정렬 후 인덱스 리셋
    df = df.sort_values(by='id').reset_index(drop=True)
    
    # 10명씩 묶어 앞 3명 테스트, 나머지 7명 훈련
    test_mask = (df.index % 10) < 3
    train_mask = ~test_mask
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # 체질량지수(bmi) 결측치 처리 (선택된 경우에만 훈련용 중앙값으로 채움)
    if 'bmi' in selected_en:
        train_bmi_median = df_train['bmi'].median()
        df_train['bmi'] = df_train['bmi'].fillna(train_bmi_median)
        df_test['bmi'] = df_test['bmi'].fillna(train_bmi_median)
        
    X_train = df_train[selected_en]
    y_train = df_train['stroke']
    X_test = df_test[selected_en]
    y_test = df_test['stroke']
    
    # 모델 학습을 위한 데이터 크기 맞추기 (Standard Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. 로지스틱 회귀 모델
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # 2. 의사결정트리 모델 (스케일링하지 않은 데이터로 학습하여 해석력 유지)
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 3. 기준 모델 (입력을 보지 않고 다수결로 찍는 모델)
    mode_val = y_train.mode()[0]
    
    def get_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
        
    lr_train_acc = get_accuracy(y_train, lr_model.predict(X_train_scaled))
    lr_test_acc = get_accuracy(y_test, lr_model.predict(X_test_scaled))
    
    dt_train_acc = get_accuracy(y_train, dt_model.predict(X_train))
    dt_test_acc = get_accuracy(y_test, dt_model.predict(X_test))
    
    base_train_acc = get_accuracy(y_train, np.full_like(y_train, mode_val))
    base_test_acc = get_accuracy(y_test, np.full_like(y_test, mode_val))

    st.divider()
    st.subheader("2. 모델 정확도 비교")
    st.markdown(f"총 {len(df):,}명 중 훈련용 {len(df_train):,}명, 테스트용 {len(df_test):,}명으로 나누어 평가했습니다.")
    
    def make_card_html(title, test_acc, train_acc):
        return f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; border: 1px solid #dee2e6; text-align: center; height: 100%;">
            <h5 style="margin-top: 0; color: #495057;">{title}</h5>
            <h2 style="margin: 10px 0; color: #1c7ed6;">{test_acc * 100:.1f}%</h2>
            <p style="margin-bottom: 0; font-size: 0.9rem; color: #6c757d;">
                훈련: {train_acc * 100:.1f}% &nbsp;|&nbsp; 테스트: {test_acc * 100:.1f}%
            </p>
        </div>
        """
        
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(make_card_html("입력을 하나도 보지 않고<br>훈련용에서 많은 쪽으로만 답하는 모델", base_test_acc, base_train_acc), unsafe_allow_html=True)
    with col_c2:
        st.markdown(make_card_html("로지스틱 회귀<br>(확률로 답하는 모델)", lr_test_acc, lr_train_acc), unsafe_allow_html=True)
    with col_c3:
        st.markdown(make_card_html("의사결정트리<br>(질문으로 답하는 모델)", dt_test_acc, dt_train_acc), unsafe_allow_html=True)

    st.divider()
    st.subheader("3. 결정 경계 시각화")
    st.markdown("선택한 속성 중 2개를 골라 모델이 뇌졸중 여부를 어떻게 가르는지 확인합니다.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        sel_x_kr = st.selectbox("가로축 속성", selected_kr, index=0)
    with col_y:
        y_default_idx = 1 if len(selected_kr) > 1 else 0
        sel_y_kr = st.selectbox("세로축 속성", selected_kr, index=y_default_idx)
        
    if sel_x_kr == sel_y_kr:
        st.warning("가로축과 세로축을 다르게 선택해 주세요.")
    else:
        x_col = feature_map[sel_x_kr]
        y_col = feature_map[sel_y_kr]
        
        x_min, x_max = X_test[x_col].min(), X_test[x_col].max()
        y_min, y_max = X_test[y_col].min(), X_test[y_col].max()
        
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        xx = np.linspace(x_min - x_margin, x_max + x_margin, 100)
        yy = np.linspace(y_min - y_margin, y_max + y_margin, 100)
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        
        grid_df = pd.DataFrame(index=range(10000), columns=selected_en)
        grid_df[x_col] = xx_grid.ravel()
        grid_df[y_col] = yy_grid.ravel()
        
        # 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
        fixed_text_parts = []
        for col in selected_en:
            if col not in [x_col, y_col]:
                median_val = X_test[col].median()
                grid_df[col] = median_val
                kr_name = reverse_feature_map[col]
                fixed_text_parts.append(f"{kr_name}={median_val:.1f}")
                
        if fixed_text_parts:
            st.info(f"나머지 속성 고정 값: {', '.join(fixed_text_parts)}")
            
        # 의사결정트리 예측 영역 (Unscaled Data)
        Z_dt = dt_model.predict(grid_df)
        Z_dt = Z_dt.reshape(xx_grid.shape)
        
        # 로지스틱 회귀 0.5 가름선 확률 (Scaled Data)
        grid_scaled = scaler.transform(grid_df)
        Z_lr_prob = lr_model.predict_proba(grid_scaled)[:, 1]
        Z_lr_prob = Z_lr_prob.reshape(xx_grid.shape)
        
        # 가름선이 화면 밖에 있는지 확인
        if Z_lr_prob.min() >= 0.5 or Z_lr_prob.max() <= 0.5:
            st.info("안내: 로지스틱 회귀 모델의 0.5 가름선이 현재 화면 범위 밖에 있습니다 (이 화면 영역에서는 어느 한쪽으로만 예측함).")

        fig = go.Figure()
        
        # 의사결정트리 분할 영역 (옅은 색 배경)
        fig.add_trace(go.Contour(
            x=xx, y=yy, z=Z_dt,
            colorscale=[[0, 'rgba(173, 216, 230, 0.3)'], [1, 'rgba(255, 182, 193, 0.3)']],
            showscale=False,
            hoverinfo='skip',
            name='의사결정트리 영역'
        ))
        
        # 로지스틱 회귀 가름선
        fig.add_trace(go.Contour(
            x=xx, y=yy, z=Z_lr_prob,
            type='contour',
            contours=dict(start=0.5, end=0.5, size=1, coloring='lines'),
            line=dict(color='black', width=3, dash='dash'),
            showscale=False,
            hoverinfo='skip',
            name='로지스틱 회귀 가름선'
        ))
        
        # 실제 데이터 산점도 (정상)
        df_test_0 = df_test[df_test['stroke'] == 0]
        fig.add_trace(go.Scatter(
            x=df_test_0[x_col], y=df_test_0[y_col],
            mode='markers',
            marker=dict(color='#3b82f6', size=7, line=dict(width=1, color='white')),
            name='정상 (실제)'
        ))
        
        # 실제 데이터 산점도 (뇌졸중)
        df_test_1 = df_test[df_test['stroke'] == 1]
        fig.add_trace(go.Scatter(
            x=df_test_1[x_col], y=df_test_1[y_col],
            mode='markers',
            marker=dict(color='#ef4444', size=7, line=dict(width=1, color='white')),
            name='뇌졸중 (실제)'
        ))
        
        fig.update_layout(
            height=600,
            title="결정 경계 시각화 (테스트 데이터)",
            xaxis_title=sel_x_kr,
            yaxis_title=sel_y_kr,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("4. 의사결정트리 질문 구조")
    
    def generate_dot_data(tree_model, feature_names_kr):
        tree = tree_model.tree_
        dot_lines = ['digraph Tree {']
        dot_lines.append('node [shape=box, style="rounded,filled", fontname="sans-serif", margin="0.2,0.1"];')
        dot_lines.append('edge [fontname="sans-serif"];')
        
        leaf_count = 0
        leaf_0_count = 0
        used_features = set()
        
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            val_0 = tree.value[i][0][0]
            val_1 = tree.value[i][0][1]
            stroke_ratio = (val_1 / n_samples) * 100 if n_samples > 0 else 0
            
            is_leaf = tree.feature[i] == -2
            
            if not is_leaf:
                feat_idx = tree.feature[i]
                fname = feature_names_kr[feat_idx]
                used_features.add(fname)
                threshold = tree.threshold[i]
                label = f"{fname} <= {threshold:.2f}\\n총 {n_samples}명\\n뇌졸중 {int(val_1)}명 ({stroke_ratio:.1f}%)"
                dot_lines.append(f'{i} [label="{label}", fillcolor="#f8f9fa"];')
                
                left_child = tree.children_left[i]
                right_child = tree.children_right[i]
                dot_lines.append(f'{i} -> {left_child} [label="예"];')
                dot_lines.append(f'{i} -> {right_child} [label="아니요"];')
            else:
                leaf_count += 1
                pred_class = 1 if val_1 >= val_0 else 0
                if pred_class == 0:
                    leaf_0_count += 1
                
                pred_str = "뇌졸중" if pred_class == 1 else "정상"
                fillcolor = "#ffcccc" if pred_class == 1 else "#cce6ff"
                label = f"예측: {pred_str}\\n총 {n_samples}명\\n뇌졸중 {int(val_1)}명 ({stroke_ratio:.1f}%)"
                dot_lines.append(f'{i} [label="{label}", fillcolor="{fillcolor}"];')
                
        dot_lines.append('}')
        return "\n".join(dot_lines), leaf_count, leaf_0_count, list(used_features)

    dot_source, leaf_total, leaf_normal, actually_used = generate_dot_data(dt_model, selected_kr)
    
    # Graphviz 차트 출력
    st.graphviz_chart(dot_source, use_container_width=True)
    
    # 하단 설명
    st.markdown(f"- **답을 내는 마디(끝 마디) 개수:** 총 {leaf_total}칸")
    st.markdown(f"- **그중 '정상(아님)'이라고 답하는 칸 수:** {leaf_normal}칸")
    if actually_used:
        st.markdown(f"- **이 나무가 실제로 질문에 사용한 속성:** {', '.join(actually_used)}")
    else:
        st.markdown("- **이 나무가 실제로 질문에 사용한 속성:** 없음 (질문을 던지지 못함)")

else:
    st.warning("데이터를 불러오지 못했습니다. 경로를 확인해 주세요.")
