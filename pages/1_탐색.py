import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 데이터 탐색",
    page_icon="🧠",
    layout="wide"
)

st.title("📊 데이터 탐색")
st.markdown("수집된 데이터를 다양한 관점에서 살펴보고 뇌졸중과의 연관성을 탐색합니다.")
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

df = load_data()

if not df.empty:
    st.subheader("1. 주요 수치형 변수의 분포")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.histogram(df, x='age', title='나이 분포', 
                               labels={'age': '나이', 'count': '사람 수'}, 
                               nbins=30, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_age, use_container_width=True)
        
    with col2:
        fig_glucose = px.histogram(df, x='avg_glucose_level', title='평균 혈당 분포', 
                                   labels={'avg_glucose_level': '평균 혈당', 'count': '사람 수'}, 
                                   nbins=30, color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_glucose, use_container_width=True)
        
    st.divider()

    st.subheader("2. 뇌졸중 여부에 따른 수치형 변수 비교")
    
    # 그래프를 그리기 위해 뇌졸중 여부를 문자열로 변환한 복사본 생성
    df_plot = df.copy()
    df_plot['stroke_label'] = df_plot['stroke'].map({0: '정상', 1: '뇌졸중'})
    
    col3, col4 = st.columns(2)
    with col3:
        fig_box_age = px.box(df_plot, x='stroke_label', y='age', color='stroke_label', 
                             title='뇌졸중 여부에 따른 나이 비교',
                             labels={'stroke_label': '뇌졸중 여부', 'age': '나이'})
        st.plotly_chart(fig_box_age, use_container_width=True)
        
    with col4:
        fig_box_glucose = px.box(df_plot, x='stroke_label', y='avg_glucose_level', color='stroke_label', 
                                 title='뇌졸중 여부에 따른 평균 혈당 비교',
                                 labels={'stroke_label': '뇌졸중 여부', 'avg_glucose_level': '평균 혈당'})
        st.plotly_chart(fig_box_glucose, use_container_width=True)

    # 두 그룹의 평균값 표 작성
    st.markdown("**두 그룹의 평균값 비교**")
    avg_table = df.groupby('stroke')[['age', 'avg_glucose_level']].mean().reset_index()
    avg_table['stroke'] = avg_table['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})
    avg_table.columns = ['뇌졸중 여부', '평균 나이', '평균 혈당']
    st.dataframe(avg_table, hide_index=True, use_container_width=True)
    
    st.divider()

    st.subheader("3. 기저질환에 따른 뇌졸중 발생 비율")
    
    # 고혈압에 따른 뇌졸중 비율 계산
    ht_ratio = df.groupby('hypertension')['stroke'].mean().reset_index()
    ht_ratio['stroke_ratio'] = ht_ratio['stroke'] * 100
    ht_ratio['hypertension_label'] = ht_ratio['hypertension'].map({0: '없음', 1: '있음'})
    
    # 심장병에 따른 뇌졸중 비율 계산
    hd_ratio = df.groupby('heart_disease')['stroke'].mean().reset_index()
    hd_ratio['stroke_ratio'] = hd_ratio['stroke'] * 100
    hd_ratio['heart_disease_label'] = hd_ratio['heart_disease'].map({0: '없음', 1: '있음'})
    
    col5, col6 = st.columns(2)
    with col5:
        fig_ht = px.bar(ht_ratio, x='hypertension_label', y='stroke_ratio', 
                        title='고혈압 여부에 따른 뇌졸중 발생 비율', text_auto='.2f',
                        labels={'hypertension_label': '고혈압 여부', 'stroke_ratio': '뇌졸중 비율 (%)'})
        st.plotly_chart(fig_ht, use_container_width=True)
        
    with col6:
        fig_hd = px.bar(hd_ratio, x='heart_disease_label', y='stroke_ratio', 
                        title='심장병 여부에 따른 뇌졸중 발생 비율', text_auto='.2f',
                        labels={'heart_disease_label': '심장병 여부', 'stroke_ratio': '뇌졸중 비율 (%)'})
        st.plotly_chart(fig_hd, use_container_width=True)

    st.divider()

    st.subheader("4. 체질량지수(BMI) 결측치 분석")
    st.markdown("BMI 데이터가 비어 있는 사람들의 뇌졸중 발생 비율이 전체 비율과 어떻게 다른지 확인합니다.")
    
    total_stroke_ratio = df['stroke'].mean() * 100
    missing_bmi_df = df[df['bmi'].isna()]
    missing_bmi_count = len(missing_bmi_df)
    
    missing_bmi_stroke_ratio = 0
    if missing_bmi_count > 0:
        missing_bmi_stroke_ratio = missing_bmi_df['stroke'].mean() * 100
        
    bmi_analysis_data = {
        "구분": ["전체 인원", "BMI 값이 비어 있는 인원"],
        "사람 수": [f"{len(df):,} 명", f"{missing_bmi_count:,} 명"],
        "뇌졸중 발생 비율 (%)": [round(total_stroke_ratio, 2), round(missing_bmi_stroke_ratio, 2)]
    }
    st.dataframe(pd.DataFrame(bmi_analysis_data), hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("5. 흡연 상태 분포")
    
    smoking_counts = df['smoking_status'].value_counts().reset_index()
    smoking_counts.columns = ['흡연 상태 (smoking_status)', '사람 수']
    
    col7, _ = st.columns([1, 1])
    with col7:
        st.dataframe(smoking_counts, hide_index=True, use_container_width=True)

else:
    st.warning("데이터를 불러오지 못했습니다. 경로를 확인해 주세요.")
