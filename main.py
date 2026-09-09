import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIE_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


# =========================================================
# 제목
# =========================================================

st.title("🎬 영화 흥행 예측기")
st.markdown(
    """
    **박스오피스 데이터와 영화 정보를 이용해서 영화의 총 관객 수를 예측해 봅니다.**

    영화코드 순으로 영화를 정렬한 뒤, **10편마다 앞의 3편을 시험용**으로 떼어 놓고
    나머지 영화로 다중 회귀 모델을 학습합니다.
    """
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIE_URL, encoding="utf-8")

    return daily, movies


try:
    daily, movies = load_data()
except Exception as e:
    st.error("❌ 데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 컬럼 이름 정리
# =========================================================

daily = daily.copy()
movies = movies.copy()

# daily 데이터의 영화코드 → movieCd
if "영화코드" in daily.columns:
    daily["movieCd"] = daily["영화코드"].astype(str).str.strip()

if "날짜" in daily.columns:
    daily["날짜"] = pd.to_datetime(
        daily["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

# movieCd 통일
if "movieCd" in movies.columns:
    movies["movieCd"] = movies["movieCd"].astype(str).str.strip()


# =========================================================
# 숫자형 변환
# =========================================================

daily_numeric = [
    "순위",
    "일관객",
    "누적관객",
    "스크린수",
    "상영횟수"
]

for col in daily_numeric:
    if col in daily.columns:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

movie_numeric = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for col in movie_numeric:
    if col in movies.columns:
        movies[col] = pd.to_numeric(movies[col], errors="coerce")


# =========================================================
# 기준 기간
# =========================================================

valid_dates = daily["날짜"].dropna()

if len(valid_dates) > 0:
    start_date = valid_dates.min()
    end_date = valid_dates.max()
    period_text = (
        f"{start_date.strftime('%Y년 %m월 %d일')} ~ "
        f"{end_date.strftime('%Y년 %m월 %d일')}"
    )
else:
    period_text = "날짜 정보를 확인할 수 없습니다."


# =========================================================
# daily 데이터에서 영화별 파생변수 만들기
# =========================================================

if "movieCd" in daily.columns and len(daily) > 0:

    daily_group = daily.groupby("movieCd")

    daily_features = pd.DataFrame(index=daily_group.size().index)

    if "일관객" in daily.columns:
        daily_features["daily_max_audi"] = daily_group["일관객"].max()
        daily_features["daily_mean_audi"] = daily_group["일관객"].mean()
        daily_features["daily_sum_audi"] = daily_group["일관객"].sum()

    if "스크린수" in daily.columns:
        daily_features["max_screens"] = daily_group["스크린수"].max()
        daily_features["mean_screens"] = daily_group["스크린수"].mean()

    if "상영횟수" in daily.columns:
        daily_features["max_shows"] = daily_group["상영횟수"].max()
        daily_features["mean_shows"] = daily_group["상영횟수"].mean()
        daily_features["sum_shows"] = daily_group["상영횟수"].sum()

    if "순위" in daily.columns:
        daily_features["best_rank"] = daily_group["순위"].min()
        daily_features["mean_rank"] = daily_group["순위"].mean()

    if "날짜" in daily.columns:
        daily_features["observed_days"] = daily_group["날짜"].nunique()

    daily_features = daily_features.reset_index()

else:
    daily_features = pd.DataFrame(columns=["movieCd"])


# =========================================================
# 영화 정보 + 일일 데이터 결합
# =========================================================

df = movies.merge(
    daily_features,
    on="movieCd",
    how="left"
)


# =========================================================
# 영화코드 기준 정렬
# =========================================================

df["movieCd_sort"] = df["movieCd"].astype(str)

df = df.sort_values(
    by="movieCd_sort",
    ascending=True
).reset_index(drop=True)


# =========================================================
# 10편마다 앞의 3편 시험용
# =========================================================

df["data_type"] = "학습용"

for start in range(0, len(df), 10):
    end = min(start + 3, len(df))
    df.loc[start:end - 1, "data_type"] = "시험용"


train_df = df[df["data_type"] == "학습용"].copy()
test_df = df[df["data_type"] == "시험용"].copy()


# =========================================================
# 기본 정보 표시
# =========================================================

st.subheader("📊 데이터 정보")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("전체 영화", f"{len(df):,}편")

with info2:
    st.metric("학습에 사용한 영화", f"{len(train_df):,}편")

with info3:
    st.metric("점수를 잰 시험 영화", f"{len(test_df):,}편")

with info4:
    st.metric("기준 기간", period_text)


st.info(
    "💡 영화코드 순으로 정렬한 뒤, 10편씩 묶어서 각 묶음의 앞 3편을 시험용으로 사용했습니다."
)


# =========================================================
# 사용할 변수 선택
# =========================================================

st.subheader("⚙️ 예측에 사용할 변수 선택")

st.write(
    "체크한 변수만 다중 회귀 모델의 입력값으로 사용합니다. "
    "총 관객 수(`total_audi`)는 예측 대상이므로 자동으로 제외됩니다."
)


# 변수 설명
variable_info = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권에 머문 날",
    "daily_max_audi": "일일 관객 최대값",
    "daily_mean_audi": "일일 관객 평균",
    "daily_sum_audi": "일일 관객 합계",
    "max_screens": "최대 스크린수",
    "mean_screens": "평균 스크린수",
    "max_shows": "최대 상영횟수",
    "mean_shows": "평균 상영횟수",
    "sum_shows": "상영횟수 합계",
    "best_rank": "최고 순위",
    "mean_rank": "평균 순위",
    "observed_days": "관측된 날짜 수",
    "genre": "장르",
    "nation": "국가"
}


available_variables = [
    col for col in variable_info.keys()
    if col in df.columns
]


# 기본으로 선택할 변수
default_variables = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10",
    "genre",
    "nation"
]

selected_variables = []

cols = st.columns(3)

for i, col in enumerate(available_variables):

    default_value = col in default_variables

    with cols[i % 3]:

        if st.checkbox(
            variable_info[col],
            value=default_value,
            key=f"variable_{col}"
        ):
            selected_variables.append(col)


if len(selected_variables) == 0:
    st.warning("⚠️ 최소 1개 이상의 변수를 선택해 주세요.")
    st.stop()


st.success(
    f"✅ 현재 선택된 변수: {len(selected_variables)}개"
)


# =========================================================
# 학습 / 시험 데이터 준비
# =========================================================

target = "total_audi"

# target이 없는 경우
if target not in df.columns:
    st.error("❌ 영화 정보 표에서 total_audi 열을 찾을 수 없습니다.")
    st.stop()


train_model_df = train_df.dropna(subset=[target]).copy()
test_model_df = test_df.dropna(subset=[target]).copy()


if len(train_model_df) < 2:
    st.error("❌ 학습할 영화가 부족합니다.")
    st.stop()

if len(test_model_df) < 1:
    st.error("❌ 시험할 영화가 없습니다.")
    st.stop()


X_train = train_model_df[selected_variables].copy()
y_train = train_model_df[target].copy()

X_test = test_model_df[selected_variables].copy()
y_test = test_model_df[target].copy()


# =========================================================
# 숫자 / 문자 변수 구분
# =========================================================

numeric_features = [
    col for col in selected_variables
    if pd.api.types.is_numeric_dtype(df[col])
]

categorical_features = [
    col for col in selected_variables
    if col not in numeric_features
]


# =========================================================
# 전처리
# =========================================================

transformers = []


if numeric_features:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    )


if categorical_features:

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# =========================================================
# 다중 회귀 모델
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


# =========================================================
# 모델 학습
# =========================================================

try:
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

except Exception as e:
    st.error("❌ 모델을 학습하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 예측값 정리
# =========================================================

predictions = np.asarray(predictions)

# 회귀모델이 음수를 예측할 경우 관객 수로는 의미가 없으므로 1명으로 처리
predictions_for_metric = np.maximum(predictions, 1)


# =========================================================
# 평가 지표
# =========================================================

r2 = r2_score(
    y_test,
    predictions_for_metric
)

mae = mean_absolute_error(
    y_test,
    predictions_for_metric
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions_for_metric
    )
)


# 평균 절대 퍼센트 오차
nonzero_mask = y_test != 0

if nonzero_mask.sum() > 0:
    mape = (
        np.mean(
            np.abs(
                (
                    y_test[nonzero_mask]
                    - predictions_for_metric[nonzero_mask]
                )
                / y_test[nonzero_mask]
            )
        )
        * 100
    )
else:
    mape = np.nan


# =========================================================
# 평가 결과
# =========================================================

st.subheader("🎯 시험용 영화 예측 점수")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "R² 점수",
        f"{r2:.3f}"
    )

with metric2:
    st.metric(
        "MAE",
        f"{mae:,.0f}명"
    )

with metric3:
    st.metric(
        "RMSE",
        f"{rmse:,.0f}명"
    )

with metric4:
    if np.isfinite(mape):
        st.metric(
            "평균 절대 오차율",
            f"{mape:.1f}%"
        )
    else:
        st.metric(
            "평균 절대 오차율",
            "계산 불가"
        )


st.caption(
    "R²는 1에 가까울수록 예측력이 좋고, MAE·RMSE·오차율은 작을수록 실제 관객 수에 가깝다는 뜻입니다."
)


# =========================================================
# 시험용 영화 결과표
# =========================================================

result_df = test_model_df[
    ["movieCd", "movieNm", "total_audi"]
].copy()

result_df["예측 총 관객 수"] = predictions_for_metric

result_df["오차"] = (
    result_df["예측 총 관객 수"]
    - result_df["total_audi"]
)

result_df["절대 오차"] = (
    result_df["오차"]
    .abs()
)

result_df["오차율"] = np.where(
    result_df["total_audi"] != 0,
    result_df["절대 오차"]
    / result_df["total_audi"]
    * 100,
    np.nan
)

result_df = result_df.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객 수"
    }
)

result_df = result_df.sort_values(
    "영화코드"
).reset_index(drop=True)


# =========================================================
# 산점도
# =========================================================

st.subheader("📈 실제 관객 수 vs 예측 관객 수")

st.write(
    "가로축은 실제 총 관객 수, 세로축은 예측 총 관객 수입니다. "
    "점이 대각선 기준선에 가까울수록 예측이 실제와 비슷합니다."
)


plot_df = result_df.copy()

plot_df["예측 그래프 값"] = plot_df["예측 총 관객 수"].clip(lower=1)


# 1,000명 미만 예측
low_prediction_mask = (
    plot_df["예측 총 관객 수"] < 1000
)

low_prediction_count = int(
    low_prediction_mask.sum()
)

normal_df = plot_df[
    ~low_prediction_mask
].copy()

low_df = plot_df[
    low_prediction_mask
].copy()


fig = go.Figure()


# ---------------------------------------------------------
# 일반 예측값
# ---------------------------------------------------------

if len(normal_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal_df["실제 총 관객 수"],
            y=normal_df["예측 총 관객 수"],
            mode="markers",
            name="예측값",
            text=normal_df["영화명"],
            customdata=np.column_stack(
                [
                    normal_df["영화코드"],
                    normal_df["실제 총 관객 수"],
                    normal_df["예측 총 관객 수"],
                    normal_df["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 관객: %{customdata[1]:,.0f}명<br>"
                "예측 관객: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                size=10,
                opacity=0.75
            )
        )
    )


# ---------------------------------------------------------
# 1,000명 미만 예측값
# 그래프 바닥인 y=1000에 표시
# ---------------------------------------------------------

if len(low_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=low_df["실제 총 관객 수"],
            y=[1000] * len(low_df),
            mode="markers",
            name="예측 1,000명 미만",
            text=low_df["영화명"],
            customdata=np.column_stack(
                [
                    low_df["영화코드"],
                    low_df["실제 총 관객 수"],
                    low_df["예측 총 관객 수"],
                    low_df["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 관객: %{customdata[1]:,.0f}명<br>"
                "실제 예측값: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                size=12,
                symbol="diamond"
            )
        )
    )


# ---------------------------------------------------------
# 기준선
# ---------------------------------------------------------

all_values = pd.concat(
    [
        plot_df["실제 총 관객 수"],
        plot_df["예측 그래프 값"],
        pd.Series([1000])
    ]
)

positive_values = all_values[
    all_values > 0
]

if len(positive_values) > 0:

    line_min = positive_values.min()
    line_max = positive_values.max()

    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="실제 = 예측 기준선",
            line=dict(
                dash="dash",
                width=2
            ),
            hoverinfo="skip"
        )
    )


fig.update_xaxes(
    type="log",
    title="실제 총 관객 수",
    rangemode="tozero"
)

fig.update_yaxes(
    type="log",
    title="예측 총 관객 수",
    rangemode="tozero"
)

fig.update_layout(
    height=650,
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 1,000명 미만 예측 결과
# =========================================================

if low_prediction_count > 0:

    st.warning(
        f"⚠️ 예측 총 관객 수가 1,000명보다 작게 나온 영화는 "
        f"**{low_prediction_count}편**입니다."
    )

else:

    st.success(
        "🎉 예측 총 관객 수가 1,000명보다 작게 나온 영화는 없습니다."
    )


# =========================================================
# 오차 설명
# =========================================================

st.subheader("🔎 예측이 실제와 얼마나 달랐을까?")

st.write(
    "아래 표에서 각 시험용 영화의 실제 총 관객 수와 예측 총 관객 수, "
    "그리고 예측이 실제보다 얼마나 많거나 적었는지 확인할 수 있습니다."
)


display_result = result_df.copy()

display_result["실제 총 관객 수"] = (
    display_result["실제 총 관객 수"]
    .round(0)
    .astype(int)
)

display_result["예측 총 관객 수"] = (
    display_result["예측 총 관객 수"]
    .round(0)
    .astype(int)
)

display_result["오차"] = (
    display_result["오차"]
    .round(0)
    .astype(int)
)

display_result["절대 오차"] = (
    display_result["절대 오차"]
    .round(0)
    .astype(int)
)

display_result["오차율"] = (
    display_result["오차율"]
    .round(1)
)

display_result = display_result[
    [
        "영화코드",
        "영화명",
        "실제 총 관객 수",
        "예측 총 관객 수",
        "오차",
        "절대 오차",
        "오차율"
    ]
]


st.dataframe(
    display_result,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 선택 변수 및 데이터 구성 확인
# =========================================================

st.subheader("🧩 현재 모델 구성")

config1, config2 = st.columns(2)

with config1:

    st.write("**사용한 예측 변수**")

    for variable in selected_variables:
        st.write(
            f"- {variable_info.get(variable, variable)}"
        )


with config2:

    st.write("**데이터 분할 방법**")

    st.write(
        "영화코드 오름차순 정렬"
    )

    st.write(
        "10편마다 앞 3편 → 시험용"
    )

    st.write(
        "10편마다 나머지 → 학습용"
    )

    st.write(
        f"전체 {len(df):,}편 중 "
        f"학습 {len(train_df):,}편 / "
        f"시험 {len(test_df):,}편"
    )


# =========================================================
# 데이터 미리보기
# =========================================================

with st.expander("📋 결합된 영화 데이터 확인"):

    preview_columns = [
        col for col in [
            "movieCd",
            "movieNm",
            "openDt",
            "genre",
            "nation",
            "first_scrn",
            "first_show",
            "peak",
            "first_week_audi",
            "total_audi",
            "days_in_top10",
            "data_type"
        ]
        if col in df.columns
    ]

    st.dataframe(
        df[preview_columns],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 하단 설명
# =========================================================

st.markdown("---")

st.markdown(
    """
    ### 💡 이 앱에서 확인할 수 있는 것

    - 어떤 영화 정보가 총 관객 수 예측에 사용되는지 직접 선택할 수 있습니다.
    - 학습에 사용하지 않은 영화만으로 모델의 예측 성능을 평가합니다.
    - 실제 관객 수와 예측 관객 수가 얼마나 차이 나는지 확인할 수 있습니다.
    - 로그 스케일 산점도를 통해 관객 수가 매우 작은 영화부터 매우 큰 영화까지 함께 비교할 수 있습니다.
    - 영화코드 순서에 따라 10편마다 3편을 시험용으로 분리하므로, 임의로 시험 영화를 선택하는 방식과 다릅니다.
    """
)
```

### `requirements.txt`

버전 숫자 없이 이렇게만 넣으면 돼.

streamlit
pandas
numpy
scikit-learn
plotly

**참고:** 이 코드는 `total_audi(총 관객)` 자체는 입력 변수에서 제외하고, `first_week_audi`, `first_scrn`, `first_show`, `genre`, `nation` 같은 변수를 체크해서 선택하도록 해놨어. 특히 **시험용 영화는 학습에 전혀 사용하지 않은 영화**로 평가하도록 구성했어.
