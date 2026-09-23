import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox


# ---------------------------------------------------
# STEP 1: Create Dataset
# ---------------------------------------------------

data = {
    "Study_Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
        4, 5, 6, 7, 8, 1, 2, 4, 6, 7,
        3, 5, 8, 9, 2, 4, 6, 7, 9, 10
    ],

    "Attendance": [
        45, 50, 55, 60, 65, 70, 75, 80, 48, 58,
        62, 68, 72, 78, 85, 40, 52, 63, 76, 82,
        57, 67, 90, 92, 49, 61, 73, 79, 88, 95
    ],

    "Previous_Marks": [
        35, 40, 45, 48, 50, 55, 60, 65, 38, 44,
        49, 52, 57, 62, 68, 30, 42, 47, 59, 64,
        43, 54, 70, 75, 36, 46, 58, 63, 72, 80
    ],

    "Result": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 0, 0, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("dataset.csv", index=False)


# ---------------------------------------------------
# STEP 2: Prepare Data
# ---------------------------------------------------

X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Result"]


# ---------------------------------------------------
# STEP 3: Train Machine Learning Model
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 4: Check Accuracy
# ---------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")


# ---------------------------------------------------
# STEP 5: Prediction Function
# ---------------------------------------------------

def predict_result():

    try:

        study_hours = float(study_entry.get())
        attendance = float(attendance_entry.get())
        previous_marks = float(marks_entry.get())

        # Validation
        if study_hours < 0:
            messagebox.showerror(
                "Error",
                "Study hours cannot be negative!"
            )
            return

        if attendance < 0 or attendance > 100:
            messagebox.showerror(
                "Error",
                "Attendance must be between 0 and 100!"
            )
            return

        if previous_marks < 0 or previous_marks > 100:
            messagebox.showerror(
                "Error",
                "Marks must be between 0 and 100!"
            )
            return

        # Prediction
        prediction = model.predict([
            [study_hours, attendance, previous_marks]
        ])

        probability = model.predict_proba([
            [study_hours, attendance, previous_marks]
        ])

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:

            result_label.config(
                text="Result: PASS",
                fg="green"
            )

        else:

            result_label.config(
                text="Result: FAIL",
                fg="red"
            )

        confidence_label.config(
            text=f"Prediction Confidence: {confidence:.2f}%"
        )

    except ValueError:

        messagebox.showerror(
            "Error",
            "Please enter valid numbers!"
        )


# ---------------------------------------------------
# STEP 6: GUI
# ---------------------------------------------------

window = tk.Tk()

window.title("Student Result Prediction")

window.geometry("500x500")

window.resizable(False, False)


# Heading
title_label = tk.Label(
    window,
    text="Student Result Prediction",
    font=("Arial", 22, "bold")
)

title_label.pack(pady=25)


# Study Hours
study_label = tk.Label(
    window,
    text="Study Hours:",
    font=("Arial", 13)
)

study_label.pack()

study_entry = tk.Entry(
    window,
    font=("Arial", 13),
    width=25
)

study_entry.pack(pady=8)


# Attendance
attendance_label = tk.Label(
    window,
    text="Attendance (%):",
    font=("Arial", 13)
)

attendance_label.pack()

attendance_entry = tk.Entry(
    window,
    font=("Arial", 13),
    width=25
)

attendance_entry.pack(pady=8)


# Previous Marks
marks_label = tk.Label(
    window,
    text="Previous Marks (%):",
    font=("Arial", 13)
)

marks_label.pack()

marks_entry = tk.Entry(
    window,
    font=("Arial", 13),
    width=25
)

marks_entry.pack(pady=8)


# Prediction Button
predict_button = tk.Button(
    window,
    text="Predict Result",
    font=("Arial", 14, "bold"),
    command=predict_result
)

predict_button.pack(pady=25)


# Result
result_label = tk.Label(
    window,
    text="Result: ",
    font=("Arial", 18, "bold")
)

result_label.pack(pady=10)


# Confidence
confidence_label = tk.Label(
    window,
    text="Prediction Confidence: ",
    font=("Arial", 12)
)

confidence_label.pack(pady=10)


window.mainloop()