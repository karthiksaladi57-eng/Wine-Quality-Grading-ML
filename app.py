
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report
)

# 1. Load Wine Quality Dataset
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)

df = pd.read_csv(url, sep=";")

# 2. Data preprocessing
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Train Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)
dt_model.fit(X_train, y_train)

# 4. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 5. Model evaluation
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

def evaluate(model_name, actual, predicted):
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(actual, predicted), 4),
        "Precision": round(
            precision_score(actual, predicted,
                            average="weighted", zero_division=0), 4
        ),
        "Recall": round(
            recall_score(actual, predicted,
                         average="weighted", zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(actual, predicted,
                     average="weighted", zero_division=0), 4
        )
    }

results = pd.DataFrame([
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("Random Forest", y_test, rf_pred)
])

# 6. Prediction function
def predict_wine(
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
):
    values = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]], columns=X.columns)

    dt_quality = int(dt_model.predict(values)[0])
    rf_quality = int(rf_model.predict(values)[0])

    # Random Forest is used as the primary prediction
    if rf_quality >= 7:
        grade = "Excellent"
    elif rf_quality >= 6:
        grade = "Good"
    elif rf_quality >= 5:
        grade = "Average"
    else:
        grade = "Low Quality"

    return (
        f"Decision Tree Prediction: {dt_quality}/10\n"
        f"Random Forest Prediction: {rf_quality}/10\n"
        f"Final Grade: {grade}"
    )

# 7. Build model comparison chart
def comparison_chart():
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(
        x - width/2,
        results.iloc[0, 1:].values.astype(float),
        width,
        label="Decision Tree"
    )
    ax.bar(
        x + width/2,
        results.iloc[1, 1:].values.astype(float),
        width,
        label="Random Forest"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Machine Learning Model Comparison")
    ax.legend()
    plt.tight_layout()

    return fig

# 8. Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🍷 Wine Quality Grading System

    ### Decision Tree vs Random Forest
    Machine Learning and Predictive Analytics Mini Project
    """)

    gr.Markdown(
        "Enter the laboratory chemical properties of wine "
        "to predict its quality."
    )

    with gr.Tab("Wine Quality Prediction"):

        with gr.Row():
            with gr.Column():
                fixed = gr.Number(label="Fixed Acidity", value=7.4)
                volatile = gr.Number(label="Volatile Acidity", value=0.70)
                citric = gr.Number(label="Citric Acid", value=0.0)
                sugar = gr.Number(label="Residual Sugar", value=1.9)
                chloride = gr.Number(label="Chlorides", value=0.076)
                free = gr.Number(label="Free Sulfur Dioxide", value=11)

            with gr.Column():
                total = gr.Number(label="Total Sulfur Dioxide", value=34)
                density = gr.Number(label="Density", value=0.9978)
                ph = gr.Number(label="pH", value=3.51)
                sulphates = gr.Number(label="Sulphates", value=0.56)
                alcohol = gr.Number(label="Alcohol", value=9.4)

        predict_btn = gr.Button("Predict Wine Quality", variant="primary")

        output = gr.Textbox(
            label="Prediction Result",
            lines=4
        )

        predict_btn.click(
            predict_wine,
            inputs=[
                fixed, volatile, citric, sugar, chloride,
                free, total, density, ph, sulphates, alcohol
            ],
            outputs=output
        )

    with gr.Tab("Model Performance"):

        gr.Markdown("### Decision Tree vs Random Forest")

        gr.Dataframe(
            value=results,
            label="Actual Test Performance",
            interactive=False
        )

        gr.Plot(value=comparison_chart())

        gr.Markdown(
            "Performance metrics are calculated using the held-out "
            "test dataset."
        )

    with gr.Tab("Dataset Analysis"):

        gr.Markdown("### Wine Quality Dataset Preview")
        gr.Dataframe(value=df.head(10), interactive=False)

        gr.Markdown("### Dataset Statistics")
        gr.Dataframe(value=df.describe(), interactive=False)

        gr.Markdown("### Wine Quality Distribution")
        gr.BarPlot(
            value=df["quality"].value_counts().sort_index()
            .rename_axis("Quality").reset_index(name="Count"),
            x="Quality",
            y="Count",
            title="Wine Quality Distribution"
        )

# 9. Launch application
app.launch(share=True)
