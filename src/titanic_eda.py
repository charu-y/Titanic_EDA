import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(path="data/train.csv"):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    df["Age"] = df["Age"].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna("S")
    df = df.drop("Cabin", axis = 1)

    return df

def run_eda(df):
    print("\nOverall survival rate:")
    print(df["Survived"].value_counts(normalize=True)*100)

    print("\nSurvival rate by gender:")
    print(df.groupby("Sex")["Survived"].mean()*100)

    print("\nSurvival rate by class:")
    print(df.groupby("Pclass")["Survived"].mean()*100)

    print("\nSurvival rate by class and gender:")
    print(df.groupby(["Pclass", "Sex"])["Survived"].mean()*100)

    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0,12,18,35,60,100],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
    )
    print("\nSurvival rate by age group:")
    print(df.groupby("AgeGroup", observed=True)["Survived"].mean()*100)

def plot_survival_by_class_gender(df, out_path="outputs/survival_by_class_gender.png"):
    sns.catplot(x="Pclass", y="Survived", hue="Sex", data=df, kind="bar")
    plt.title("Survival Rate by Class and Gender")
    plt.ylabel("Surival Rate")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def train_model(df):
    df_model = df.copy()
    df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})
    df_model["Embarked"] = df_model["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    x = df_model[features]
    y = df_model["Survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    importance = pd.DataFrame({
        "Feature" : x.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    print("\nFeature Importance (Logistic Regression Coefficients):")
    print(importance)

    return model

def main():
    df = load_data()
    df = clean_data(df)
    run_eda(df)
    plot_survival_by_class_gender(df)
    train_model(df)

if __name__ == "__main__":
    main()