import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import dice_ml

st.set_page_config(page_title="WESOnline Student Stress Screening", page_icon="🎓", layout="wide")
st.title("🎓 Student Stress Screening")
st.write("Answer the questions below, then press **Estimate Stress Level**.")

# ---------- 1. Settings ----------
NAMES = {
    "Exam_Frequency": "Exam frequency",
    "Assignment_Load": "Assignment load",
    "Sleep_Hours": "Sleep",
    "Social_Media_Use": "Social media use",
    "Screen_Time": "Total screen time",
    "Family_Support": "Family support",
    "Anxiety_Level": "Anxiety level",
    "Physical_Exercise": "Physical exercise",
}
UNITS = {
    "Exam_Frequency": "(1-9)",
    "Assignment_Load": "(1-9)",
    "Sleep_Hours": "h/night",
    "Social_Media_Use": "h/day",
    "Screen_Time": "h/day",
    "Family_Support": "(1-9)",
    "Anxiety_Level": "(1-9)",
}
COLORS = {"Low": "#2E8B57", "Medium": "#E0A100", "High": "#C0392B"}

# Habits a student can realistically change
HABITS = ["Sleep_Hours", "Social_Media_Use", "Screen_Time", "Physical_Exercise"]
LIMITS = {"Sleep_Hours": [4, 9], "Social_Media_Use": [0, 7], "Screen_Time": [1, 11]}


# ---------- 2. Load the trained model ----------
@st.cache_resource
def load_model():
    return joblib.load("stress_model.joblib")


saved = load_model()
model = saved["model"]                    # the whole pipeline: scaling + Logistic Regression
prep = model.named_steps["prep"]          # the scaling / encoding step
clf = model.named_steps["model"]          # the Logistic Regression itself
classes = list(clf.classes_)              # ['High', 'Low', 'Medium']


# ---------- 3. Helper functions ----------
def ask(feature):
    """A slider that starts at the average answer."""
    low, high = saved["ranges"][feature]
    average = int(round(saved["averages"][feature]))
    return st.slider(f"{NAMES[feature]} {UNITS[feature]}", low, high, average)


def explain(student):
    """How much each answer pushed the result towards High (positive) or away from it (negative).
    For Logistic Regression this is exact: weight x (student's value - average value)."""
    values = prep.transform(student)[0]
    weights = clf.coef_[classes.index("High")] - clf.coef_[classes.index("Low")]
    pushes = weights * (values - saved["scaled_averages"])
    result = pd.Series(pushes, index=saved["scaled_names"])
    result.index = result.index.str.replace("_Yes", "", regex=False)
    return result


def explain_in_words(feature, value, push):
    """One readable sentence about one answer."""
    name = NAMES[feature]
    direction = "raises" if push > 0 else "lowers"

    if feature == "Physical_Exercise":
        return f"**{name}:** answering *{value}* {direction} the stress estimate."

    average = saved["averages"][feature]
    comparison = "above" if value > average else "below"
    answer = f"A rating of {value}" if UNITS[feature] == "(1-9)" else f"{value} {UNITS[feature]}"
    return f"**{name}:** {answer} is {comparison} the average of {average:.1f}, which {direction} the stress estimate."


@st.cache_resource
def load_dice():
    """DiCE finds 'what would have to change' answers."""
    numeric = [c for c in saved["features"] if c != "Physical_Exercise"]
    allowed = {}
    for column in numeric:
        allowed[column] = saved["ranges"][column]
    allowed["Physical_Exercise"] = ["No", "Yes"]
    data = dice_ml.Data(features=allowed, outcome_name="Stress_Level", continuous_features=numeric)
    return dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="random")


def what_if(student, level):
    """Up to 3 different small habit changes that would lower the level by one step."""
    target = {"High": "Medium", "Medium": "Low"}.get(level)
    if target is None:
        return pd.DataFrame()

    try:
        result = load_dice().generate_counterfactuals(
            student,
            total_CFs=6,                              # ask for 6 ideas; some are filtered out below
            desired_class=classes.index(target),
            features_to_vary=HABITS,
            permitted_range=LIMITS,
            random_seed=1,                            # same student, same ideas every time
            verbose=False,
        )
        found = result.cf_examples_list[0].final_cfs_df
    except Exception:                                 # DiCE raises an error when nothing works
        return pd.DataFrame()

    chance_before = model.predict_proba(student)[0][classes.index(level)]
    options = []
    for i in range(len(found)):
        new = found.iloc[[i]][saved["features"]]
        if new["Screen_Time"].iloc[0] < new["Social_Media_Use"].iloc[0]:
            continue                                  # screen time below social media time is not realistic

        changes = {}
        for habit in HABITS:
            old_value, new_value = student[habit].iloc[0], new[habit].iloc[0]
            if old_value != new_value:
                changes[habit] = (old_value, new_value)

        options.append({
            "count": len(changes),
            "changes": changes,
            "level": model.predict(new)[0],
            "before": chance_before,
            "after": model.predict_proba(new)[0][classes.index(level)],
        })

    if not options:
        return pd.DataFrame()
    return pd.DataFrame(options).sort_values("count", kind="stable").head(3)   # fewest changes first


def describe_change(habit, old, new):
    if habit == "Physical_Exercise":
        return f"{NAMES[habit]}: {old} to {new}"
    return f"{NAMES[habit]}: {int(old)} to {int(new)} {UNITS[habit]}"


# ---------- 4. Sidebar ----------
st.sidebar.header("About WESOnline")
st.sidebar.write("WESOnline estimates a student's stress level (Low, Medium or High) from academic, emotional "
                 "and lifestyle answers, and shows why.")
st.sidebar.warning("This is a screening aid, not a medical or psychological diagnosis. Students experiencing "
                   "serious stress should speak to a qualified counsellor.")

# ---------- 5. Questions ----------
answers = {}
left, right = st.columns(2)

with left:
    st.subheader("Academic Factors")
    answers["Exam_Frequency"] = ask("Exam_Frequency")
    answers["Assignment_Load"] = ask("Assignment_Load")
    st.subheader("Mind and Support")
    answers["Anxiety_Level"] = ask("Anxiety_Level")
    answers["Family_Support"] = ask("Family_Support")

with right:
    st.subheader("Daily Habits")
    answers["Sleep_Hours"] = ask("Sleep_Hours")
    answers["Social_Media_Use"] = ask("Social_Media_Use")
    answers["Screen_Time"] = ask("Screen_Time")
    answers["Physical_Exercise"] = st.radio("Do you do physical exercise?", ["Yes", "No"], horizontal=True)

student = pd.DataFrame([answers])[saved["features"]]     # one row, in the order the model expects

# Only a warning: about 1 in 4 students in the training data answer this way
if answers["Screen_Time"] < answers["Social_Media_Use"]:
    st.info("Your total screen time is lower than your social media time. Please double-check these two answers.")

# ---------- 6. Result ----------
if st.button("Estimate Stress Level", type="primary"):
    prediction = model.predict(student)[0]
    probabilities = pd.Series(model.predict_proba(student)[0], index=classes)[["Low", "Medium", "High"]]

    st.divider()
    result_col, why_col = st.columns([1, 1.4])

    with result_col:
        st.subheader("Stress Level Result")
        st.markdown(
            f"<div style='background:{COLORS[prediction]};color:white;padding:18px;border-radius:12px;"
            f"text-align:center;font-size:26px;font-weight:bold'>{prediction} Stress</div>",
            unsafe_allow_html=True,
        )
        st.caption("Model confidence for each level")
        for level, probability in probabilities.items():
            st.progress(float(probability), text=f"{level}: {probability:.0%}")

        if prediction == "High":
            st.error("Consider speaking with a school counsellor or student support team.")
        elif prediction == "Medium":
            st.info("Some factors may be contributing to your stress. See the explanation for what stands out.")
        else:
            st.success("Your answers indicate a low estimated stress level.")

    with why_col:
        st.subheader("Why This Result?")
        contributions = explain(student).sort_values(key=abs)

        bar_colors = []
        for push in contributions:
            bar_colors.append("#C0392B" if push > 0 else "#2E75B6")     # red raises, blue lowers

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(contributions.index.str.replace("_", " "), contributions.values, color=bar_colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("← lowers stress estimate      raises stress estimate →")
        ax.set_title("Effect of each answer compared with the average student", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # The same explanation in plain words (the 5 biggest effects)
    st.subheader("Understanding Your Result")
    shown = 0
    for feature in contributions.abs().sort_values(ascending=False).index:
        if abs(contributions[feature]) >= 0.5 and shown < 5:
            st.markdown("- " + explain_in_words(feature, answers[feature], contributions[feature]))
            shown += 1
    if shown == 0:
        st.write("Your answers are all close to the average student, so no single factor stands out.")

    st.caption("Red bars raise the stress estimate and blue bars lower it, relative to the average student in the "
               "training data. This shows what the model uses, not what causes stress.")

    # What-if suggestions
    st.divider()
    st.subheader("What Could Change This Result?")

    if prediction == "Low":
        st.success("Your estimated stress level is already Low. Keep up the habits that help.")
    else:
        options = what_if(student, prediction)
        if options.empty:
            st.info("No small change in sleep, social media, screen time or exercise is enough to lower the "
                    "estimate. Workload, anxiety and family support (not changed here) may be the main drivers.")
        for number, (_, option) in enumerate(options.iterrows(), start=1):
            changes = []
            for habit, (old, new) in option["changes"].items():
                changes.append(describe_change(habit, old, new))
            st.markdown(f"**Option {number}:** " + "; ".join(changes))
            st.write(f"The model would then predict **{option['level']} stress**. Its chance of {prediction} "
                     f"would fall from {option['before']:.0%} to {option['after']:.0%}.")