from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["STATIC_FOLDER"] = "static"


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["STATIC_FOLDER"], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


def handle_missing_values(data, method):
    if method == "drop":
        data.dropna(inplace=True)
    elif method == "fill_mean":
        data.fillna(data.mean(numeric_only=True), inplace=True)
    elif method == "fill_median":
        data.fillna(data.median(numeric_only=True), inplace=True)
    return data


def handle_outliers(data):
    numeric_data = data.select_dtypes(include=["number"])  
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]


def scale_data(X, method):
    scaler = None
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()

    if scaler:
        numeric_cols = X.select_dtypes(include="number").columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X


def encode_non_numeric(X):
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
    for col in non_numeric_cols:
        X[col] = LabelEncoder().fit_transform(X[col])
    return X


def generate_charts(data, chart_types, scatter_x, scatter_y):
    plot_paths = []
    numeric_data = data.select_dtypes(include=["number"])

    try:
        for chart_type in chart_types:
            if chart_type == "correlation":
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
                plot_path = os.path.join(app.config["STATIC_FOLDER"], "correlation_heatmap.png")
                plt.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close()

            elif chart_type == "scatter" and scatter_x and scatter_y:
                plt.figure(figsize=(10, 8))
                plt.scatter(data[scatter_x], data[scatter_y])
                plt.xlabel(scatter_x)
                plt.ylabel(scatter_y)
                plot_path = os.path.join(app.config["STATIC_FOLDER"], f"scatter_{scatter_x}_{scatter_y}.png")
                plt.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close()

            elif chart_type == "boxplot":
                plt.figure(figsize=(10, 8))
                numeric_data.boxplot()
                plot_path = os.path.join(app.config["STATIC_FOLDER"], "boxplot.png")
                plt.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close()

            elif chart_type == "pairplot":
                plt.figure(figsize=(10, 8))
                sns.pairplot(numeric_data)
                plot_path = os.path.join(app.config["STATIC_FOLDER"], "pairplot.png")
                plt.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close()

    except Exception as e:
        plot_paths = None

    return plot_paths


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)


    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading CSV file: {e}", 500

 
    missing_values = request.form.get("missing_values", "none")
    handle_outliers_option = request.form.get("outliers", "remove")
    scale_option = request.form.get("scaling", "none")
    target_column = request.form.get("target")
    ml_model = request.form.get("ml_model", "logistic")
    chart_types = request.form.getlist("charts")  
    scatter_x = request.form.get("scatter_x")
    scatter_y = request.form.get("scatter_y")
    test_size = float(request.form.get("test_size", 0.2))
    random_state = int(request.form.get("random_state", 42))

    data = handle_missing_values(data, missing_values)


    if handle_outliers_option == "remove":
        data = handle_outliers(data)

  
    if target_column not in data.columns:
        return f"Target column '{target_column}' not found in dataset", 400

    X = data.drop(columns=[target_column])
    y = data[target_column]


    X = encode_non_numeric(X)


    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y)

   
    X = scale_data(X, scale_option)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = None
    if ml_model == "logistic":
        model = LogisticRegression()
    elif ml_model == "decision_tree":
        model = DecisionTreeClassifier()
    elif ml_model == "random_forest":
        model = RandomForestClassifier()
    elif ml_model == "svm":
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    plot_paths = generate_charts(data, chart_types, scatter_x, scatter_y)

    return render_template(
    "results.html",
    summary={
        "columns": list(data.columns),
        "shape": data.shape,
        "head": data.head().to_html(classes="table table-bordered"),
    },
    accuracy=accuracy_score(y_test, y_pred),
    classification_report=classification_report(y_test, y_pred, output_dict=True),
    plot_paths=plot_paths,
)


if __name__ == "__main__":
    app.run(debug=True)
