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


# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# 제목
# ============================================================

st.title("🎬 영화 흥행 예측기")

st.markdown(
    """
    ### 영화의 정보를 이용해 총 관객 수를 예측해 봅니다!

    박스오피스 데이터와 영화 정보 데이터를 영화코드로 합친 뒤,
    **다중 회귀 모델**을 이용하여 영화의 총 관객 수를 예측합니다.
    """
)


# ============================================================
# 데이터 주소
# ============================================================

DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_daily.csv"
)

MOVIE_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/kobis_movies.csv"
)


# ============================================================
# 데이터 불러오기
# ============================================================

@st.cache_data
def load_data():

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8"
    )

    movies = pd.read_csv(
        MOVIE_URL,
        encoding="utf-8"
    )

    return daily, movies


try:

    daily, movies = load_data()

except Exception as e:

    st.error("❌ 데이터를 불러오지 못했습니다.")

    st.write(
        "인터넷 연결이나 GitHub의 CSV 파일을 확인해 주세요."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 데이터 복사
# ============================================================

daily = daily.copy()
movies = movies.copy()


# ============================================================
# 영화코드 통일
# ============================================================

if "영화코드" not in daily.columns:

    st.error(
        "❌ kobis_daily.csv에서 '영화코드' 열을 찾을 수 없습니다."
    )

    st.stop()


if "movieCd" not in movies.columns:

    st.error(
        "❌ kobis_movies.csv에서 'movieCd' 열을 찾을 수 없습니다."
    )

    st.stop()


daily["movieCd"] = (
    daily["영화코드"]
    .astype(str)
    .str.strip()
)

movies["movieCd"] = (
    movies["movieCd"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 날짜 변환
# ============================================================

if "날짜" in daily.columns:

    daily["날짜"] = pd.to_datetime(
        daily["날짜"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )


# ============================================================
# 숫자형 변환
# ============================================================

daily_numeric_columns = [
    "순위",
    "일관객",
    "누적관객",
    "스크린수",
    "상영횟수"
]

for column in daily_numeric_columns:

    if column in daily.columns:

        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce"
        )


movie_numeric_columns = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10"
]

for column in movie_numeric_columns:

    if column in movies.columns:

        movies[column] = pd.to_numeric(
            movies[column],
            errors="coerce"
        )


# ============================================================
# 기준 기간
# ============================================================

if "날짜" in daily.columns and daily["날짜"].notna().any():

    start_date = daily["날짜"].min()
    end_date = daily["날짜"].max()

    period_text = (
        f"{start_date.strftime('%Y년 %m월 %d일')} ~ "
        f"{end_date.strftime('%Y년 %m월 %d일')}"
    )

else:

    period_text = "날짜 확인 불가"


# ============================================================
# daily 데이터에서 영화별 첫 관측 정보 만들기
#
# 총 관객을 예측하면서 일관객 전체 합계 등을 입력값으로
# 사용하면 정답을 미리 알고 예측하는 문제가 생길 수 있기 때문에
# 전체 기간의 관객 합계 같은 변수를 입력값으로 사용하지 않는다.
# ============================================================

daily_features = pd.DataFrame()

if len(daily) > 0 and "movieCd" in daily.columns:

    temp_daily = daily.copy()

    if "날짜" in temp_daily.columns:

        temp_daily = temp_daily.sort_values(
            ["movieCd", "날짜"]
        )

    else:

        temp_daily = temp_daily.sort_values(
            ["movieCd"]
        )

    first_rows = (
        temp_daily
        .groupby("movieCd", as_index=False)
        .first()
    )

    daily_features = first_rows[
        [
            column
            for column in [
                "movieCd",
                "스크린수",
                "상영횟수"
            ]
            if column in first_rows.columns
        ]
    ].copy()

    daily_features = daily_features.rename(
        columns={
            "스크린수": "daily_first_screens",
            "상영횟수": "daily_first_shows"
        }
    )


# ============================================================
# 두 데이터 합치기
# ============================================================

df = movies.merge(
    daily_features,
    on="movieCd",
    how="left"
)


# ============================================================
# 영화코드 순 정렬
# ============================================================

df["movieCd_sort"] = (
    df["movieCd"]
    .astype(str)
)

df = df.sort_values(
    "movieCd_sort",
    ascending=True
).reset_index(drop=True)


# ============================================================
# 10편마다 앞의 3편을 시험용으로 분리
# ============================================================

df["구분"] = "학습용"

for start in range(
    0,
    len(df),
    10
):

    end = min(
        start + 3,
        len(df)
    )

    df.loc[
        start:end - 1,
        "구분"
    ] = "시험용"


train_df = df[
    df["구분"] == "학습용"
].copy()

test_df = df[
    df["구분"] == "시험용"
].copy()


# ============================================================
# 데이터 정보
# ============================================================

st.subheader("📊 데이터 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "전체 영화",
        f"{len(df):,}편"
    )

with col2:

    st.metric(
        "학습 영화",
        f"{len(train_df):,}편"
    )

with col3:

    st.metric(
        "시험 영화",
        f"{len(test_df):,}편"
    )

with col4:

    st.metric(
        "기준 기간",
        period_text
    )


st.info(
    "📌 영화코드 순으로 정렬한 후, 10편마다 앞의 3편을 시험용으로 사용하고 "
    "나머지 영화들을 학습용으로 사용합니다."
)


# ============================================================
# 예측 변수 목록
# ============================================================

variable_names = {

    "first_scrn":
        "첫 관측일 스크린수",

    "first_show":
        "첫 관측일 상영횟수",

    "peak":
        "성수기 개봉 여부",

    "first_week_audi":
        "첫 주 관객",

    "days_in_top10":
        "10위권에 있었던 날",

    "genre":
        "장르",

    "nation":
        "국가",

    "daily_first_screens":
        "박스오피스 첫 관측 스크린수",

    "daily_first_shows":
        "박스오피스 첫 관측 상영횟수"
}


available_variables = [
    column
    for column in variable_names
    if column in df.columns
]


# ============================================================
# 변수 선택
# ============================================================

st.subheader("⚙️ 예측에 사용할 변수 선택")

st.write(
    "체크한 변수만 회귀 모델의 입력값으로 사용됩니다."
)

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

variable_columns = st.columns(3)

for i, column in enumerate(
    available_variables
):

    with variable_columns[i % 3]:

        checked = st.checkbox(
            variable_names[column],
            value=(
                column in default_variables
            ),
            key=f"check_{column}"
        )

        if checked:

            selected_variables.append(column)


if len(selected_variables) == 0:

    st.warning(
        "⚠️ 최소 1개의 변수를 선택해 주세요."
    )

    st.stop()


st.success(
    "✅ 선택한 변수: "
    + ", ".join(
        variable_names[column]
        for column in selected_variables
    )
)


# ============================================================
# 학습 / 시험 데이터
# ============================================================

target = "total_audi"

if target not in df.columns:

    st.error(
        "❌ total_audi 열을 찾을 수 없습니다."
    )

    st.stop()


train_model_df = train_df.dropna(
    subset=[target]
).copy()

test_model_df = test_df.dropna(
    subset=[target]
).copy()


if len(train_model_df) < 2:

    st.error(
        "❌ 학습에 사용할 영화가 부족합니다."
    )

    st.stop()


if len(test_model_df) == 0:

    st.error(
        "❌ 시험에 사용할 영화가 없습니다."
    )

    st.stop()


X_train = train_model_df[
    selected_variables
].copy()

y_train = train_model_df[
    target
].copy()

X_test = test_model_df[
    selected_variables
].copy()

y_test = test_model_df[
    target
].copy()


# ============================================================
# 숫자 / 문자 변수 구분
# ============================================================

numeric_features = []

categorical_features = []

for column in selected_variables:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        numeric_features.append(
            column
        )

    else:

        categorical_features.append(
            column
        )


# ============================================================
# 전처리
# ============================================================

transformers = []


if len(numeric_features) > 0:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
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


if len(categorical_features) > 0:

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
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
    transformers=transformers
)


# ============================================================
# 다중 회귀 모델
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regression",
            LinearRegression()
        )
    ]
)


# ============================================================
# 모델 학습
# ============================================================

try:

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

except Exception as e:

    st.error(
        "❌ 모델 학습 중 오류가 발생했습니다."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 예측값이 음수가 되는 경우 처리
# ============================================================

predictions = np.asarray(
    predictions,
    dtype=float
)

predictions = np.maximum(
    predictions,
    1
)


# ============================================================
# 모델 평가
# ============================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# 평균 절대 오차율
non_zero = (
    y_test != 0
)

if non_zero.sum() > 0:

    mape = (
        np.mean(
            np.abs(
                (
                    y_test[non_zero]
                    - predictions[non_zero]
                )
                / y_test[non_zero]
            )
        )
        * 100
    )

else:

    mape = np.nan


# ============================================================
# 점수 표시
# ============================================================

st.subheader("🎯 시험용 영화 예측 점수")

score1, score2, score3, score4 = st.columns(4)

with score1:

    st.metric(
        "R² 점수",
        f"{r2:.3f}"
    )

with score2:

    st.metric(
        "MAE",
        f"{mae:,.0f}명"
    )

with score3:

    st.metric(
        "RMSE",
        f"{rmse:,.0f}명"
    )

with score4:

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
    "R²는 높을수록 좋고, MAE·RMSE·오차율은 낮을수록 좋습니다."
)


# ============================================================
# 결과 데이터프레임
# ============================================================

result_df = test_model_df[
    [
        "movieCd",
        "movieNm",
        "total_audi"
    ]
].copy()

result_df["예측 총 관객 수"] = predictions

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
).reset_index(
    drop=True
)


# ============================================================
# 산점도
# ============================================================

st.subheader(
    "📈 실제 총 관객 수 vs 예측 총 관객 수"
)

st.write(
    "가로축은 실제 총 관객 수, 세로축은 예측 총 관객 수입니다. "
    "점이 기준선에 가까울수록 예측이 실제와 비슷합니다."
)


plot_df = result_df.copy()

plot_df["그래프용 예측값"] = (
    plot_df["예측 총 관객 수"]
    .clip(lower=1)
)


# ============================================================
# 1,000명 미만 예측
# ============================================================

low_mask = (
    plot_df["예측 총 관객 수"]
    < 1000
)

low_df = plot_df[
    low_mask
].copy()

normal_df = plot_df[
    ~low_mask
].copy()

low_count = len(low_df)


# ============================================================
# 그래프
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# 일반 예측값
# ------------------------------------------------------------

if len(normal_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=normal_df[
                "실제 총 관객 수"
            ],
            y=normal_df[
                "예측 총 관객 수"
            ],
            mode="markers",
            name="예측값",
            text=normal_df["영화명"],
            customdata=np.column_stack(
                [
                    normal_df["영화코드"],
                    normal_df[
                        "실제 총 관객 수"
                    ],
                    normal_df[
                        "예측 총 관객 수"
                    ],
                    normal_df["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[1]:,.0f}명<br>"
                "예측 총 관객: %{customdata[2]:,.0f}명<br>"
                "오차: %{customdata[3]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                size=10,
                opacity=0.75
            )
        )
    )


# ------------------------------------------------------------
# 1,000명 미만 예측값
#
# 로그 그래프에서 너무 작은 값을 그대로 표시하면
# 그래프 바닥 아래로 내려가기 때문에 y=1,000에 표시
# ------------------------------------------------------------

if len(low_df) > 0:

    fig.add_trace(
        go.Scatter(
            x=low_df[
                "실제 총 관객 수"
            ],
            y=[
                1000
            ] * len(low_df),
            mode="markers",
            name="예측 1,000명 미만",
            text=low_df["영화명"],
            customdata=np.column_stack(
                [
                    low_df["영화코드"],
                    low_df[
                        "실제 총 관객 수"
                    ],
                    low_df[
                        "예측 총 관객 수"
                    ],
                    low_df["오차"]
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[1]:,.0f}명<br>"
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


# ============================================================
# 실제 = 예측 기준선
# ============================================================

all_values = pd.concat(
    [
        plot_df[
            "실제 총 관객 수"
        ],
        plot_df[
            "그래프용 예측값"
        ],
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
            x=[
                line_min,
                line_max
            ],
            y=[
                line_min,
                line_max
            ],
            mode="lines",
            name="실제 = 예측",
            line=dict(
                dash="dash",
                width=2
            ),
            hoverinfo="skip"
        )
    )


# ============================================================
# 로그 축
# ============================================================

fig.update_xaxes(
    type="log",
    title="실제 총 관객 수 (명)"
)

fig.update_yaxes(
    type="log",
    title="예측 총 관객 수 (명)",
    range=[
        3,
        np.log10(
            max(
                line_max,
                1000
            )
        ) + 0.2
    ]
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


# ============================================================
# 1,000명 미만 영화 수
# ============================================================

if low_count > 0:

    st.warning(
        f"⚠️ 예측 총 관객 수가 1,000명보다 작게 나온 영화는 "
        f"**{low_count}편**입니다."
    )

else:

    st.success(
        "🎉 예측 총 관객 수가 1,000명보다 작게 나온 영화는 없습니다."
    )


# ============================================================
# 시험용 영화별 결과
# ============================================================

st.subheader(
    "🔎 시험용 영화별 예측 결과"
)

st.write(
    "시험용 영화는 모델을 학습할 때 사용하지 않은 영화입니다."
)


display_df = result_df.copy()

display_df[
    "실제 총 관객 수"
] = (
    display_df[
        "실제 총 관객 수"
    ]
    .round()
    .astype(int)
)

display_df[
    "예측 총 관객 수"
] = (
    display_df[
        "예측 총 관객 수"
    ]
    .round()
    .astype(int)
)

display_df["오차"] = (
    display_df["오차"]
    .round()
    .astype(int)
)

display_df["절대 오차"] = (
    display_df["절대 오차"]
    .round()
    .astype(int)
)

display_df["오차율"] = (
    display_df["오차율"]
    .round(1)
)

display_df = display_df[
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
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 오차가 가장 큰 영화
# ============================================================

st.subheader(
    "🏆 예측 오차가 가장 큰 영화"
)

if len(result_df) > 0:

    largest_error = result_df.loc[
        result_df["절대 오차"].idxmax()
    ]

    a, b, c = st.columns(3)

    with a:

        st.write(
            "**영화명**"
        )

        st.write(
            largest_error["영화명"]
        )

    with b:

        st.write(
            "**실제 총 관객 수**"
        )

        st.write(
            f"{largest_error['실제 총 관객 수']:,.0f}명"
        )

    with c:

        st.write(
            "**예측 총 관객 수**"
        )

        st.write(
            f"{largest_error['예측 총 관객 수']:,.0f}명"
        )


# ============================================================
# 모델 구성
# ============================================================

st.subheader(
    "🧩 모델 구성"
)

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.write(
        "### 선택된 변수"
    )

    for column in selected_variables:

        st.write(
            "• "
            + variable_names[column]
        )


with model_col2:

    st.write(
        "### 데이터 분리 방법"
    )

    st.write(
        "1️⃣ 영화코드 순으로 정렬"
    )

    st.write(
        "2️⃣ 10편마다 앞의 3편을 시험용으로 분리"
    )

    st.write(
        "3️⃣ 나머지 영화를 학습용으로 사용"
    )

    st.write(
        f"📚 학습: {len(train_df):,}편"
    )

    st.write(
        f"🧪 시험: {len(test_df):,}편"
    )


# ============================================================
# 결합된 데이터 확인
# ============================================================

with st.expander(
    "📋 결합된 영화 데이터 보기"
):

    preview_columns = [
        column
        for column in [
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
            "daily_first_screens",
            "daily_first_shows",
            "구분"
        ]
        if column in df.columns
    ]

    st.dataframe(
        df[preview_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 하단 설명
# ============================================================

st.markdown("---")

st.markdown(
    """
    ### 💡 이 앱에서 확인할 수 있는 것

    🎬 영화 정보를 선택하여 총 관객 수를 예측할 수 있습니다.

    📚 학습에 사용한 영화와 시험에 사용한 영화를 분리했습니다.

    📈 실제 관객 수와 예측 관객 수를 로그 스케일로 비교합니다.

    🎯 대각선 기준선에 가까운 영화일수록 예측이 실제에 가깝습니다.

    ⚠️ 예측값이 1,000명보다 작은 영화는 그래프 바닥에 따로 표시합니다.
    """
)
