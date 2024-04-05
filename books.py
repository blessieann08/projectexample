import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

book_data = pd.read_csv('recommended_books - Sheet1.csv')

label_encoder = LabelEncoder()

book_data['Gender'] = label_encoder.fit_transform(book_data['Gender'])

X = book_data[['Age', 'Gender']]
y = book_data['Books']

classifier = DecisionTreeClassifier()
classifier.fit(X, y)

app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        gender = request.form["gender"]

        gender_encoded = label_encoder.transform([gender])[0]

        input_data = [[age, gender_encoded]]

        recommended_book = classifier.predict(input_data)[0]

        image_path = book_data[book_data['Books'] == recommended_book]['Images'].values[0]

        return render_template("success.html", recommended_book=recommended_book, image_path=image_path)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
