import numpy as np
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class KNNElectiveRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.knn_model = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='cosine')
        self.courses_df = None

    def load_sample_data(self):
        """Loads dataset with more detailed courses"""
        self.courses_df = pd.DataFrame([
            {
                "course_id": "CS101",
                "title": "Data Structures",
                "description": "This course covers fundamental data structures, including arrays, linked lists, stacks, queues, trees, and graphs. Students will learn efficient sorting and searching algorithms, hash tables, and dynamic programming. The course emphasizes time and space complexity analysis, preparing students for coding interviews and real-world problem-solving.",
                "keywords": ["data structures", "algorithms", "trees", "graphs", "sorting", "searching", "complexity analysis", "dynamic programming"]
            },
            {
                "course_id": "CS102",
                "title": "Database Management",
                "description": "Learn the foundations of relational database systems, covering SQL, normalization, indexing, transaction management, and distributed databases. Topics include query optimization, NoSQL alternatives, and data modeling. This course is crucial for backend developers, data engineers, and software architects.",
                "keywords": ["database", "SQL", "PostgreSQL", "MySQL", "MongoDB", "indexing", "transaction management", "data modeling", "query optimization"]
            },
            {
                "course_id": "CS103",
                "title": "Operating Systems",
                "description": "Explore the inner workings of modern operating systems, including process scheduling, memory management, file systems, and concurrency. Topics include multi-threading, virtualization, security mechanisms, and real-time OS. Practical implementations will be covered using Linux and Windows environments.",
                "keywords": ["OS", "Linux", "Windows", "multi-threading", "scheduling", "memory management", "concurrency", "file systems", "virtualization"]
            },
            {
                "course_id": "CS104",
                "title": "Computer Networks",
                "description": "Dive into the fundamentals of computer networking, covering TCP/IP, network security, routing algorithms, firewalls, VPNs, wireless networks, and cloud networking. Students will learn how data is transmitted, secured, and optimized over various network topologies.",
                "keywords": ["networking", "TCP/IP", "routing", "firewall", "VPN", "cloud networking", "cybersecurity", "network security", "wireless networks"]
            },
            {
                "course_id": "CS105",
                "title": "Machine Learning",
                "description": "A comprehensive introduction to supervised and unsupervised learning, covering regression, classification, clustering, neural networks, deep learning, feature engineering, and model evaluation. This course is ideal for data scientists and AI researchers.",
                "keywords": ["machine learning", "AI", "deep learning", "regression", "clustering", "neural networks", "feature engineering", "model evaluation", "Python"]
            },
            {
                "course_id": "CS106",
                "title": "Web Development",
                "description": "Master full-stack web development, including HTML, CSS, JavaScript, React, Node.js, and databases. Learn about RESTful APIs, authentication, web security, server-side rendering, and modern frontend frameworks. This course is ideal for aspiring web developers.",
                "keywords": ["web development", "frontend", "backend", "React", "Node.js", "authentication", "RESTful APIs", "JavaScript", "CSS", "MongoDB"]
            },
            {
                "course_id": "CS107",
                "title": "Cybersecurity",
                "description": "Learn about ethical hacking, encryption, network security, penetration testing, and cybersecurity threats. Topics include cryptographic algorithms, secure coding practices, malware analysis, and risk assessment. This course prepares students for security analyst roles.",
                "keywords": ["cybersecurity", "encryption", "hacking", "penetration testing", "malware", "firewalls", "risk assessment", "secure coding", "network security"]
            },
            {
                "course_id": "CS108",
                "title": "Software Engineering",
                "description": "This course covers software development methodologies, including Agile, DevOps, software testing, design patterns, and software lifecycle management. Learn how to build scalable, maintainable software solutions using best engineering practices.",
                "keywords": ["software engineering", "SDLC", "Agile", "DevOps", "testing", "design patterns", "scalability", "system design", "software maintenance"]
            }
        ])

    def prepare_model(self, filtered_df):
        """Prepares the KNN model for selected courses"""
        filtered_df['content'] = filtered_df.apply(
            lambda row: ' '.join([
                row['title'], 
                row['description'], 
                ' '.join(row['keywords'] * 3)  # Weight keywords higher
            ]).lower(), 
            axis=1
        )

        # TF-IDF Vectorization
        content_matrix = self.tfidf_vectorizer.fit_transform(filtered_df['content'])

        # Fit KNN model with dynamic neighbors
        self.knn_model.set_params(n_neighbors=min(3, len(filtered_df)))
        self.knn_model.fit(content_matrix)

        return filtered_df, content_matrix

    def recommend_elective(self, selected_course_ids, skills, area_of_interest, future_career_paths, previous_fav_subjects):
        """Recommends the best elective from selected courses"""
        selected_courses_df = self.courses_df[self.courses_df['course_id'].isin(selected_course_ids)]
        if selected_courses_df.empty:
            return {"error": "Invalid course IDs"}

        # Train KNN only on selected courses
        filtered_df, content_matrix = self.prepare_model(selected_courses_df)

        # Create user profile from multiple factors
        user_profile = (
            ' '.join(filtered_df['content'].values) + " " +
            ' '.join(skills * 2) +  
            ' '.join(area_of_interest * 2) +
            ' '.join(future_career_paths * 3) +  
            ' '.join(previous_fav_subjects * 3)
        )
        user_vector = self.tfidf_vectorizer.transform([user_profile])

        # Find best elective among selected courses
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=3)

        recommended_courses = [filtered_df.iloc[idx] for idx in indices.flatten()]
        best_elective = recommended_courses[1] if len(recommended_courses) > 1 else recommended_courses[0]

        return {
            "course_id": best_elective["course_id"],
            "title": best_elective["title"],
            "description": best_elective["description"],
            "keywords": best_elective["keywords"],
            "match_score": round(float((1 - distances.flatten()[1]) * 100), 2)
        }

# Initialize recommender
recommender = KNNElectiveRecommender()

@app.before_request
def initialize():
    """Loads courses before the first request"""
    if recommender.courses_df is None:
        recommender.load_sample_data()

@app.route('/api/recommend', methods=['POST'])
def recommend_elective():
    """API to get the best elective from selected courses"""
    data = request.json
    return jsonify(recommender.recommend_elective(**data))

if __name__ == '__main__':
    app.run(debug=True)
