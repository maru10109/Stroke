import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split # Included as requested in the requirements

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    try:
        # Load the CSV file, handle potential errors gracefully
        df = pd.read_csv(url, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        # Return an empty dataframe with the expected columns if loading fails to prevent app crash
        columns = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                   'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
        return pd.DataFrame(columns=columns)

df = load_data()

st.title("🧠 뇌졸중 예측 실습실")
st.markdown("이 화면은 뇌졸중 예측에 사용되는 건강 및 생활 습관 데이터셋을 소개합니다.")
st.divider()

if not df.empty:
    total_people = len(df)
    total_columns = len(df.columns)
    stroke_cases = len(df[df['stroke'] == 1])
    stroke_ratio = (stroke_cases / total_people) * 100 if total_people > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="전체 사람 수", value=f"{total_people:,} 명")
    with col2:
        st.metric(label="데이터 열 개수", value=f"{total_columns} 개")
    with col3:
        st.metric(label="뇌졸중 발생 (stroke=1)", value=f"{stroke_cases:,} 명")
    with col4:
        st.metric(label="뇌졸중 발생 비율", value=f"{stroke_ratio:.2f} %")
    
    st.divider()

    st.subheader("📋 데이터 속성 안내 (데이터 사전)")
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    
    # Determine value types/examples for the table
    value_types = {
        'id': '고유 숫자',
        'gender': '문자열 (Male, Female 등)',
        'age': '숫자 (연령)',
        'hypertension': '0 또는 1 (없음/있음)',
        'heart_disease': '0 또는 1 (없음/있음)',
        'ever_married': '문자열 (Yes, No)',
        'work_type': '문자열 (Private, Govt_job 등)',
        'Residence_type': '문자열 (Urban, Rural)',
        'avg_glucose_level': '숫자 (연속형)',
        'bmi': '숫자 (연속형, 빈 값 포함)',
        'smoking_status': '문자열 (formerly smoked 등)',
        'stroke': '0 또는 1 (정상/뇌졸중)'
    }

    # Prepare data for the display table
    dict_data = {
        "열 이름": df.columns,
        "우리말 뜻": ["" for _ in df.columns], # Leave empty for the user to fill
        "값의 종류": [value_types.get(col, "알 수 없음") for col in df.columns],
        "빈 값 개수": [missing_values[col] for col in df.columns]
    }
    dict_df = pd.DataFrame(dict_data)
    
    # Use st.data_editor so the user can interactively type in the '우리말 뜻' column if they want, 
    # or just st.dataframe for a static view. Using dataframe as requested.
    st.dataframe(dict_df, use_container_width=True, hide_index=True)
    
    st.info("💡 **안내:** '우리말 뜻' 칸은 비어 있습니다. 교재를 참고하여 각 열이 의미하는 바를 확인해 보세요.")
    st.divider()

    st.subheader("👀 데이터 미리보기 (첫 5줄)")
    st.dataframe(df.head(5), use_container_width=True)

else:
    st.warning("데이터를 불러오지 못해 정보를 표시할 수 없습니다.")

st.divider()

st.subheader("📚 데이터 출처")
# A text area where the user can type or paste the source, or just a markdown placeholder.
st.markdown("""
> *(이곳에 교재에 있는 데이터 출처를 적어주세요)*
""")
# Alternatively, you can use st.text_input or st.text_area if you want it to be editable in the app UI:
# source_text = st.text_area("출처 입력", placeholder="교재에 있는 데이터 출처를 여기에 적어주세요...", height=100)
