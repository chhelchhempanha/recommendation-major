import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

console = Console()

df = pd.read_csv('major_observation.csv')

df = df.apply(lambda col: col.map({'yes': 1, 'no': 0}) if col.dtype == 'object' and col.name != 'Major' else col)

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

questions = [
    "Do you like analytical tasks?",
    "Do you consider yourself creative?",
    "Do you enjoy scientific work?",
    "Do you prefer working in a team?",
    "Do you enjoy working with numbers?",
    "Do you like solving problems logically?",
    "Are you interested in helping others?",
    "Do you prefer working alone?",
    "Do you enjoy writing essays or reports?",
    "Do you like to take the lead in group projects?",
    "Are you comfortable with public speaking?",
    "Do you enjoy learning new languages?",
    "Do you like hands-on work (e.g., building or creating things)?",
    "Are you detail-oriented?",
    "Do you enjoy exploring new ideas and concepts?",
    "Do you like structured environments?",
    "Do you enjoy working with technology?",
    "Do you enjoy outdoor activities?",
    "Do you like working under pressure?",
    "Do you enjoy debating or discussing topics?"
]

def welcome_message():
    panel = Panel("Welcome to the Major Recommendation System!", style="bold yellow", width=80, expand=False)
    console.print(Align.center(panel))

def get_user_responses():
    responses = []
    for question in questions:
        while True:
            console.print(f"{question} ['y'/'n' or 'yes'/'no']: ", style="bold cyan", end="")
            answer = input().lower()
            if answer in ["yes", "y"]:
                responses.append(1)
                break
            elif answer in ["no", "n"]:
                responses.append(0)
                break
            else:
                console.print("Invalid input. Please enter 'y'/'n' or 'yes'/'no'.", style="bold red")
    return responses

def predict_major(responses):
    responses_df = pd.DataFrame([responses], columns=X.columns)
    predicted_major = clf.predict(responses_df)[0]
    console.print(f"Based on your answers, we recommend the major: [bold green]{predicted_major}[/bold green]")

def main():
    welcome_message()

    while True:
        console.print("Would you like to get a major recommendation? ['y'/'n' or 'yes'/'no']: ", style="bold cyan", end="")
        start = input().lower()
        if start in ["yes", "y"]:
            break
        elif start in ["no", "n"]:
            console.print("Thank you! Goodbye!", style="bold green")
            return
        else:
            console.print("Invalid input. Please enter 'y'/'n' or 'yes'/'no'.", style="bold red")

    while True:
        responses = get_user_responses()
        console.print("Processing your responses...\n", style="bold cyan")
        predict_major(responses)
        again = input("Would you like to get another recommendation? (y/n): ").lower()
        if again not in ["y", "yes"]:
            console.print("Thank you for using the Major Recommendation System! Goodbye!", style="bold green")
            break

if __name__ == "__main__":
    main()
