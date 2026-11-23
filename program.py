import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from core import FEATURES, TARGET, load_data, sections, train

# The CSV contains 90 SIMULATED records created for a school project.
data = load_data()
corr = data.corr(numeric_only=True)
r = train(data)
S = sections(data, corr, r)


def show(title):
    print(f"\n==============================\n {title}\n==============================")
    print(S[title])


def get_value(feature):
    low, high = data[feature].min(), data[feature].max()
    while True:
        try:
            value = float(input(f"{feature} (between {low:.1f} and {high:.1f}): "))
        except ValueError:
            print("Please enter a number.")
            continue
        if low <= value <= high:
            return value
        print("Out of range, the model is only reliable inside the data's range.")


for title in list(S)[:5]:
    show(title)

print("\n==============================\n SALES PREDICTION\n==============================")
if input("Press Enter for the example scenario, or type 'c' for custom input: ").strip().lower() == "c":
    values = [get_value(f) for f in FEATURES]
else:
    values = [3500, 8, 70, 5]
scenario = pd.DataFrame([values], columns=FEATURES)
print(scenario.iloc[0].to_string())
print("Predicted Sales =", round(r["model"].predict(scenario)[0], 2))
print("This is an estimate from simulated data, not a dependable real-world forecast.")

show("LIMITATIONS")

# Plots: each feature vs Sales, correlation heatmap, coefficient bars.
for f in FEATURES:
    plt.figure()
    plt.scatter(data[f], data[TARGET])
    plt.xlabel(f), plt.ylabel(TARGET), plt.title(f"{f} vs {TARGET}")
    plt.tight_layout()
    plt.show()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(FEATURES, r["std_coef"])
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Business Factors"), plt.ylabel("Regression Coefficient")
plt.title("Regression Coefficients of Business Factors")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()