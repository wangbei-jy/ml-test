import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              AdaBoostRegressor, AdaBoostClassifier)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                    Lasso, Ridge)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.rcParams['font.size'] = 18  # 设置全局字体大小

# 定义机器学习模型字典及其超参数设置
models = {
    "线性回归": (LinearRegression(), {"fit_intercept": [True, False]}),
    "逻辑回归": (LogisticRegression(max_iter=200), {"C": [0.01, 0.1, 1.0, 10.0]}),
    "Lasso回归": (Lasso(), {"alpha": [0.1, 1.0, 10.0]}),
    "Ridge回归": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
    "随机森林回归": (RandomForestRegressor(), {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}),
    "随机森林分类": (RandomForestClassifier(), {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}),
    "支持向量回归": (SVR(), {"C": [0.1, 1.0, 10.0], "kernel": ['linear', 'rbf']}),
    "支持向量分类": (SVC(), {"C": [0.1, 1.0, 10.0], "kernel": ['linear', 'rbf']}),
    "K近邻回归": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 10]}),
    "K近邻分类": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 10]}),
    "决策树回归": (DecisionTreeRegressor(), {"max_depth": [None, 10, 20]}),
    "决策树分类": (DecisionTreeClassifier(), {"max_depth": [None, 10, 20]}),
    "梯度提升回归": (GradientBoostingRegressor(), {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]}),
    "梯度提升分类": (GradientBoostingClassifier(), {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]}),
    "AdaBoost回归": (AdaBoostRegressor(), {"n_estimators": [50, 100]}),
    "AdaBoost分类": (AdaBoostClassifier(), {"n_estimators": [50, 100]}),
    "高斯朴素贝叶斯": (GaussianNB(), {}),
    "XGBoost回归": (XGBRegressor(eval_metric='rmse'), {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}),
    "XGBoost分类": (XGBClassifier(eval_metric='mlogloss'), {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}),
}

# 设置Streamlit页面布局
st.set_page_config(layout="wide")

# 初始化 session_state 存储模型和状态
if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['trained'] = False
    st.session_state['df'] = None

# 上传训练数据文件
uploaded_file = st.sidebar.file_uploader("上传训练数据 (xlsx)", type=["xlsx"])
if uploaded_file is not None:
    # 读取Excel文件
    st.session_state['df'] = pd.read_excel(uploaded_file)
    st.write("上传的数据预览：")
    st.dataframe(st.session_state['df'].head())

# 选择特征列和标签列
if st.session_state['df'] is not None:
    columns = st.session_state['df'].columns.tolist()
    feature_columns = st.sidebar.multiselect("选择特征列", columns)
    target_column = st.sidebar.selectbox("选择标签列", columns)

    # 设置训练集和测试集的比例
    test_size = st.sidebar.slider("训练集和测试集的比例", 0.1, 0.9, 0.2)

    # 分割训练集和测试集
    if feature_columns and target_column:
        X = st.session_state['df'][feature_columns]
        y = st.session_state['df'][target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 模型选择及其超参数设置
        model_choice = st.sidebar.selectbox("选择模型", list(models.keys()))
        model, param_options = models.get(model_choice)

        # 超参数手动调节
        params = {}
        for param, options in param_options.items():
            params[param] = st.sidebar.selectbox(f"选择 {param}", options)

        # 训练模型
        if st.sidebar.button("训练模型"):
            try:
                model.set_params(**params)
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                st.session_state['trained'] = True
                st.success(f"{model_choice} 模型训练成功！")
            except Exception as e:
                st.error(f"模型训练失败: {e}")
                st.session_state['trained'] = False

# 显示结果
st.subheader("模型效果展示")
if st.session_state['trained']:
    y_train_pred = st.session_state['model'].predict(X_train)
    y_test_pred = st.session_state['model'].predict(X_test)

    # 绘制训练集和测试集的预测结果
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(y_train, y_train_pred, label='训练集预测', color='blue', alpha=0.6)
    ax.scatter(y_test, y_test_pred, label='测试集预测', color='orange', alpha=0.6)
    ax.set_xlabel("实际值")
    ax.set_ylabel("预测值")
    ax.set_title("模型效果展示")
    ax.legend()
    st.pyplot(fig)

    # 显示评估指标
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    st.write(f"均方误差: {mse:.2f}, R²: {r2:.2f}")

# 相关性分析结果
st.subheader("相关性分析结果")
if st.session_state['df'] is not None:
    corr_columns = st.sidebar.multiselect("选择相关性分析的列", columns)
    corr_method = st.sidebar.selectbox("选择相关性分析方法", ["Pearson", "Spearman", "Kendall"])

    if st.sidebar.button("生成相关性矩阵"):
        if len(corr_columns) > 1:
            corr_df = st.session_state['df'][corr_columns].corr(method=corr_method.lower())
            st.write("相关性热图：")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
            plt.title(f"{corr_method} 相关性热图")
            st.pyplot(fig)

# 散点图
st.subheader("特征散点图")
if st.session_state['df'] is not None:
    plot_columns = st.sidebar.multiselect("选择绘制散点图的列", columns)
    if st.sidebar.button("生成散点图"):
        if len(plot_columns) > 1:
            fig = sns.pairplot(st.session_state['df'][plot_columns], diag_kind='kde')
            st.pyplot(fig)

# 待预测数据
st.sidebar.subheader("待预测数据")
predict_file = st.sidebar.file_uploader("上传待预测数据 (xlsx)", type=["xlsx"])
if predict_file is not None:
    predict_df = pd.read_excel(predict_file)
    st.sidebar.write("待预测数据预览：")
    st.sidebar.dataframe(predict_df.head())
    predict_columns = st.sidebar.multiselect("选择特征列", predict_df.columns.tolist())
    if st.sidebar.button("预测"):
        if predict_columns and st.session_state['trained']:
            predictions = st.session_state['model'].predict(predict_df[predict_columns])
            # 将预测结果合并到待预测数据中
            predict_df['预测结果'] = predictions
            st.write("待预测数据与预测结果：")
            st.dataframe(predict_df)

# 提示用户操作
st.sidebar.write("请上传训练数据和待预测数据，然后选择特征和标签进行训练和预测。")
