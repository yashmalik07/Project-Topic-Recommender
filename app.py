from flask import Flask, render_template, request
from recommender import ProjectTopicRecommender

app = Flask(__name__)

# Create recommender object
recommender = ProjectTopicRecommender()


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        interests = request.form.get("interests", "")
        skills = request.form.get("skills", "")
        experience_level = request.form.get(
            "experience_level", ""
        )

        recommendations = recommender.recommend_projects(
            interests=interests,
            skills=skills,
            experience_level=experience_level,
            top_n=5
        )

    return render_template(
        "index.html",
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
