import numpy as np
import pandas as pd
import pickle
import os
import threading
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class KNNElectiveRecommender:
    def __init__(self):
        # Modified TF-IDF parameters for small document counts
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,     # Changed to 1 to handle small document counts
            max_df=1.0    # Changed to 1.0 to handle small document counts
        )
        
        self.knn_model = NearestNeighbors(
            n_neighbors=3,
            algorithm='auto',
            metric='cosine',
            n_jobs=-1
        )
        
        self.courses_df = None
        self.content_matrix = None
        self._lock = threading.Lock()

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
        """Enhanced model preparation with weighted features"""
        try:
            with self._lock:
                filtered_df['content'] = filtered_df.apply(
                    lambda row: ' '.join([
                        row['title'] * 3,
                        row['description'],
                        ' '.join(row['keywords'] * 4)
                    ]).lower(),
                    axis=1
                )

                # Apply TF-IDF
                self.content_matrix = self.tfidf_vectorizer.fit_transform(filtered_df['content'])
                
                # Dynamic neighbor selection based on dataset size
                n_neighbors = min(3, len(filtered_df))
                self.knn_model.set_params(n_neighbors=n_neighbors)
                self.knn_model.fit(self.content_matrix)

                return filtered_df, self.content_matrix
        except Exception as e:
            logger.error(f"Error in prepare_model: {str(e)}")
            raise

    def recommend_elective(self, selected_course_ids, skills, area_of_interest, 
                         future_career_paths, **kwargs):
        """Enhanced recommendation logic with better error handling"""
        try:
            selected_courses_df = self.courses_df[self.courses_df['course_id'].isin(selected_course_ids)]
            if selected_courses_df.empty:
                return {"error": "Invalid course IDs"}

            # Case 1: Single course selected
            if len(selected_courses_df) == 1:
                course = selected_courses_df.iloc[0]
                return {
                    "course_id": course["course_id"],
                    "title": course["title"],
                    "description": course["description"],
                    "keywords": course["keywords"],
                    "match_score": 100.0,
                    "note": "Single course selection: returning the selected course"
                }

            # Case 2: Multiple courses
            filtered_df, content_matrix = self.prepare_model(selected_courses_df)

            # Create user profile without previous subjects
            user_profile_components = [
                ' '.join(skills) * 3,
                ' '.join(area_of_interest) * 2,
                ' '.join(future_career_paths) * 4
            ]
            
            user_profile = ' '.join(component.lower() for component in user_profile_components if component)
            user_vector = self.tfidf_vectorizer.transform([user_profile])

            # Get recommendations
            distances, indices = self.knn_model.kneighbors(user_vector)
            similarity_scores = [round(float((1 - dist) * 100), 2) for dist in distances.flatten()]

            # Get recommendations with scores
            recommendations = []
            for idx, score in zip(indices.flatten(), similarity_scores):
                course = filtered_df.iloc[idx]
                if course["course_id"] not in selected_course_ids:
                    recommendations.append({
                        "course_id": course["course_id"],
                        "title": course["title"],
                        "description": course["description"],
                        "keywords": course["keywords"],
                        "match_score": score
                    })
                    break

            # If no non-selected course found, return the best matching course
            if not recommendations:
                best_idx = indices.flatten()[0]
                best_course = filtered_df.iloc[best_idx]
                return {
                    "course_id": best_course["course_id"],
                    "title": best_course["title"],
                    "description": best_course["description"],
                    "keywords": best_course["keywords"],
                    "match_score": similarity_scores[0],
                    "note": "Best matching course from selected courses"
                }

            return recommendations[0]

        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            # Last resort: Return the first course from selected courses
            course = selected_courses_df.iloc[0]
            return {
                "course_id": course["course_id"],
                "title": course["title"],
                "description": course["description"],
                "keywords": course["keywords"],
                "match_score": 100.0,
                "note": "Fallback recommendation due to processing error"
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
    """API endpoint with better error handling"""
    try:
        data = request.json
        required_fields = ['selected_course_ids', 'skills', 'area_of_interest', 'future_career_paths']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        result = recommender.recommend_elective(**data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
