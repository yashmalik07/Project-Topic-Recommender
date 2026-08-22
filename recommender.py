import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProjectTopicRecommender:

    def __init__(self):

        # Load project dataset
        self.projects = pd.read_csv("projects.csv")

        # Preprocess project data
        self.projects["combined_text"] = (
            self.projects["title"].fillna("") + " " +
            self.projects["domain"].fillna("") + " " +
            self.projects["tech_stack"].fillna("") + " " +
            self.projects["difficulty"].fillna("")
        )

        self.projects["clean_text"] = (
            self.projects["combined_text"]
            .apply(self.preprocess_text)
        )

        # Create TF-IDF model
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        # Convert project topics into TF-IDF vectors
        self.topic_vectors = self.vectorizer.fit_transform(
            self.projects["clean_text"]
        )


    def preprocess_text(self, text):

        """
        Clean and preprocess text.
        """

        # Convert to lowercase
        text = text.lower()

        # Remove special characters
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text


    def recommend_projects(
        self,
        interests,
        skills,
        experience_level=None,
        top_n=5
    ):

        """
        Generate project recommendations
        using TF-IDF and Cosine Similarity.
        """

        # Create student profile
        student_profile = (
            interests + " " +
            skills
        )

        # Add experience level if provided
        if experience_level:
            student_profile += " " + experience_level

        # Preprocess student profile
        student_profile = self.preprocess_text(
            student_profile
        )

        # Convert student profile into TF-IDF vector
        student_vector = self.vectorizer.transform(
            [student_profile]
        )

        # Calculate cosine similarity
        similarities = cosine_similarity(
            student_vector,
            self.topic_vectors
        )[0]

        # Add similarity score
        self.projects["similarity_score"] = similarities

        # Sort projects by similarity
        recommendations = self.projects.sort_values(
            by="similarity_score",
            ascending=False
        ).head(top_n)

        # Prepare result
        results = []

        for _, project in recommendations.iterrows():

            results.append({

                "topic_id": project["topic_id"],

                "title": project["title"],

                "domain": project["domain"],

                "tech_stack": project["tech_stack"],

                "difficulty": project["difficulty"],

                "duration": project["duration"],

                "similarity_score": round(
                    project["similarity_score"],
                    3
                )

            })

        return results


    def filter_by_difficulty(
        self,
        recommendations,
        difficulty
    ):

        """
        Filter recommendations by difficulty.
        """

        return [
            project
            for project in recommendations
            if project["difficulty"].lower()
            == difficulty.lower()
        ]


    def filter_by_domain(
        self,
        recommendations,
        domain
    ):

        """
        Filter recommendations by domain.
        """

        return [
            project
            for project in recommendations
            if project["domain"].lower()
            == domain.lower()
        ]


# -----------------------------------------
# Example Usage
# -----------------------------------------

if __name__ == "__main__":

    recommender = ProjectTopicRecommender()

    recommendations = recommender.recommend_projects(

        interests="Healthcare NLP Chatbot",

        skills="Python NLP Transformers",

        experience_level="Intermediate",

        top_n=5

    )

    print("\n🎯 Recommended Project Topics")
    print("=" * 60)

    for index, project in enumerate(
        recommendations,
        start=1
    ):

        print(f"\n{index}. {project['title']}")

        print(
            f"   📂 Domain: "
            f"{project['domain']}"
        )

        print(
            f"   ⚡ Difficulty: "
            f"{project['difficulty']}"
        )

        print(
            f"   ⏱️ Duration: "
            f"{project['duration']}"
        )

        print(
            f"   🔧 Tech Stack: "
            f"{project['tech_stack']}"
        )

        print(
            f"   📊 Match Score: "
            f"{project['similarity_score'] * 100:.2f}%"
        )
