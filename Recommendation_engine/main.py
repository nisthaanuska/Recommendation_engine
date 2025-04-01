import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random
from datasets import generate_academic_datasets

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Path constants
MODEL_PATH = 'knn_recommender_model.pkl'
SCORE_MODEL_PATH = 'score_predictor_model.pkl'

class SimpleRecommender:
    def __init__(self):
        # TF-IDF for converting course content to vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # KNN for finding similar courses
        self.model = NearestNeighbors(
            n_neighbors=5,
            algorithm='auto',
            metric='cosine'
        )
        
        self.courses_df = None
        self.ready = False
    
    def load_courses(self):
        """Load elective course data"""
        self.courses_df = pd.DataFrame([
            {
                "course_id": "CS301",
                "title": "Advanced Algorithms",
                "description": "Advanced study of algorithm design and analysis, including complex data structures, dynamic programming, graph algorithms, and algorithmic paradigms.",
                "keywords": ["algorithms", "data structures", "dynamic programming", "graph algorithms", "complexity analysis", "optimization", "problem solving", "computational complexity", "algorithmic paradigms", "divide and conquer"]
            },
            {
                "course_id": "CS302",
                "title": "Advanced Computer Networks",
                "description": "In-depth study of modern computer networks, protocols, network security, and emerging networking technologies.",
                "keywords": ["computer networks", "protocols", "TCP/IP", "network security", "routing", "switching", "SDN", "network architecture", "network protocols", "wireless networks"]
            },
            {
                "course_id": "CS303",
                "title": "Agile Software Development",
                "description": "Modern agile methodologies, practices, and tools for efficient software development and project management.",
                "keywords": ["agile", "scrum", "kanban", "sprint planning", "software development", "project management", "continuous integration", "DevOps", "test-driven development", "team collaboration"]
            },
            {
                "course_id": "CS304",
                "title": "Big Data Analytics",
                "description": "Analysis and processing of large-scale data sets using modern big data tools and technologies.",
                "keywords": ["big data", "hadoop", "spark", "data analytics", "distributed computing", "data processing", "NoSQL", "data visualization", "MapReduce", "data mining"]
            },
            {
                "course_id": "CS305",
                "title": "Blockchain Technology",
                "description": "Fundamentals of blockchain, cryptocurrencies, smart contracts, and decentralized applications.",
                "keywords": ["blockchain", "cryptocurrency", "smart contracts", "distributed ledger", "consensus algorithms", "cryptography", "decentralized systems", "web3", "ethereum", "bitcoin"]
            },
            {
                "course_id": "CS306",
                "title": "Computer Graphics and Animation",
                "description": "Principles and techniques of computer graphics, 3D modeling, animation, and visualization.",
                "keywords": ["computer graphics", "3D modeling", "animation", "rendering", "OpenGL", "visualization", "game graphics", "shaders", "texture mapping", "computer animation"]
            },
            {
                "course_id": "CS307",
                "title": "Computer Vision and Image Processing",
                "description": "Advanced techniques in computer vision, image processing, and visual understanding systems.",
                "keywords": ["computer vision", "image processing", "object detection", "feature extraction", "machine learning", "OpenCV", "deep learning", "pattern recognition", "image analysis", "visual computing"]
            },
            {
                "course_id": "CS308",
                "title": "Cyber-Physical Systems",
                "description": "Integration of computation, networking, and physical processes in modern embedded systems.",
                "keywords": ["cyber-physical systems", "IoT", "embedded systems", "real-time systems", "sensor networks", "control systems", "automation", "industrial IoT", "smart systems", "robotics"]
            },
            {
                "course_id": "CS309",
                "title": "Data Mining and Data Warehousing",
                "description": "Techniques and tools for data mining, knowledge discovery, and data warehouse design.",
                "keywords": ["data mining", "data warehousing", "ETL", "OLAP", "business intelligence", "predictive analytics", "clustering", "association rules", "data modeling", "statistical analysis"]
            },
            {
                "course_id": "CS310",
                "title": "Data Science and Machine Learning",
                "description": "Comprehensive coverage of data science methodologies and machine learning algorithms.",
                "keywords": ["data science", "machine learning", "statistical analysis", "predictive modeling", "data visualization", "Python", "R", "scikit-learn", "pandas", "data analytics"]
            },
            {
                "course_id": "CS311",
                "title": "Deep Learning and Neural Networks",
                "description": "Advanced neural network architectures and deep learning techniques for AI applications.",
                "keywords": ["deep learning", "neural networks", "CNN", "RNN", "LSTM", "tensorflow", "pytorch", "AI", "backpropagation", "GPU computing"]
            },
            {
                "course_id": "CS312",
                "title": "Distributed Computing",
                "description": "Principles and practices of distributed systems and parallel computing.",
                "keywords": ["distributed systems", "parallel computing", "distributed algorithms", "cloud computing", "scalability", "fault tolerance", "distributed databases", "microservices", "containerization", "cluster computing"]
            },
            {
                "course_id": "CS313",
                "title": "Distributed Operating Systems",
                "description": "Advanced concepts in distributed operating systems and distributed system design.",
                "keywords": ["operating systems", "distributed systems", "process management", "distributed file systems", "synchronization", "distributed algorithms", "virtualization", "system architecture", "resource management", "networking"]
            },
            {
                "course_id": "CS314",
                "title": "Edge Computing",
                "description": "Computing paradigms and technologies for edge and fog computing environments.",
                "keywords": ["edge computing", "fog computing", "IoT", "distributed systems", "real-time processing", "mobile computing", "edge analytics", "5G", "cloud computing", "network optimization"]
            },
            {
                "course_id": "CS315",
                "title": "Embedded Systems",
                "description": "Design and development of embedded systems and real-time applications.",
                "keywords": ["embedded systems", "microcontrollers", "real-time systems", "firmware", "IoT", "hardware interfaces", "RTOS", "embedded software", "digital electronics", "system programming"]
            },
            {
                "course_id": "CS316",
                "title": "Game Development and Design",
                "description": "Principles and practices of game design, development, and interactive entertainment.",
                "keywords": ["game development", "Unity", "Unreal Engine", "game design", "3D graphics", "game physics", "game AI", "game programming", "interactive design", "game mechanics"]
            },
            {
                "course_id": "CS317",
                "title": "High-Performance Computing",
                "description": "Advanced concepts in parallel processing and high-performance computing systems.",
                "keywords": ["HPC", "parallel computing", "GPU computing", "cluster computing", "supercomputing", "parallel algorithms", "MPI", "OpenMP", "performance optimization", "scientific computing"]
            },
            {
                "course_id": "CS318",
                "title": "Image Processing and Pattern Recognition",
                "description": "Advanced techniques for processing images and recognizing patterns in visual data.",
                "keywords": ["image processing", "pattern recognition", "computer vision", "feature extraction", "image enhancement", "image segmentation", "object recognition", "machine learning", "digital image processing", "image filtering"]
            },
            {
                "course_id": "CS319",
                "title": "Immersive Technologies",
                "description": "Development and applications of AR, VR, and mixed reality technologies.",
                "keywords": ["AR", "VR", "mixed reality", "Unity", "3D modeling", "interaction design", "spatial computing", "computer graphics", "XR development", "immersive experiences"]
            },
            {
                "course_id": "CS320",
                "title": "Information and Web Security",
                "description": "Security principles, cryptography, and protection mechanisms for web and information systems.",
                "keywords": ["cybersecurity", "web security", "cryptography", "network security", "ethical hacking", "penetration testing", "security protocols", "authentication", "authorization", "secure coding"]
            },
            {
                "course_id": "CS321",
                "title": "Information Retrieval Systems",
                "description": "Design and implementation of systems for searching and retrieving information from large data collections.",
                "keywords": ["information retrieval", "search engines", "text mining", "indexing", "ranking algorithms", "query processing", "web search", "document classification", "semantic search", "retrieval models"]
            },
            {
                "course_id": "CS322",
                "title": "Internet and Web Technologies",
                "description": "Advanced concepts in web development, protocols, and modern web technologies.",
                "keywords": ["web technologies", "HTTP", "RESTful APIs", "HTML5", "CSS3", "JavaScript frameworks", "web services", "responsive design", "progressive web apps", "web protocols"]
            },
            {
                "course_id": "CS323",
                "title": "Mobile App Development",
                "description": "Design and development of mobile applications for various platforms using modern frameworks.",
                "keywords": ["mobile development", "iOS", "Android", "React Native", "Flutter", "mobile UI/UX", "app architecture", "mobile APIs", "cross-platform development", "mobile testing"]
            },
            {
                "course_id": "CS324",
                "title": "Multimedia Computing",
                "description": "Processing, analysis, and presentation of multimedia content including audio, video, and images.",
                "keywords": ["multimedia", "audio processing", "video processing", "compression algorithms", "streaming technologies", "media encoding", "multimedia databases", "content-based retrieval", "interactive media", "digital media"]
            },
            {
                "course_id": "CS325",
                "title": "Natural Language Processing",
                "description": "Processing and understanding human language using computational techniques and AI.",
                "keywords": ["NLP", "text mining", "machine learning", "language models", "BERT", "transformers", "text analysis", "sentiment analysis", "information extraction", "computational linguistics"]
            },
            {
                "course_id": "CS326",
                "title": "Next-Generation Wireless Networks",
                "description": "Advanced wireless technologies including 5G, 6G, and future network architectures.",
                "keywords": ["5G", "6G", "wireless networks", "mobile communications", "network protocols", "IoT networking", "spectrum management", "network slicing", "edge computing", "wireless technologies"]
            },
            {
                "course_id": "CS327",
                "title": "Parallel and Distributed Systems",
                "description": "Design and implementation of parallel algorithms and distributed computing systems.",
                "keywords": ["parallel computing", "distributed systems", "concurrent programming", "parallel algorithms", "scalability", "multiprocessing", "distributed databases", "cluster computing", "performance optimization", "GPGPU"]
            },
            {
                "course_id": "CS328",
                "title": "Quantum Computing",
                "description": "Principles of quantum computation and quantum algorithms.",
                "keywords": ["quantum computing", "quantum algorithms", "quantum mechanics", "quantum circuits", "qubits", "quantum gates", "quantum programming", "quantum cryptography", "quantum simulation", "quantum information"]
            },
            {
                "course_id": "CS329",
                "title": "Fog Computing",
                "description": "Computing paradigm that extends cloud computing to the edge of the network, allowing data processing closer to the source.",
                "keywords": ["fog computing", "edge computing", "IoT", "distributed systems", "network architecture", "cloud integration", "real-time processing", "data localization", "network optimization", "sensor networks"]
            },
            {
                "course_id": "CS330",
                "title": "Computational Intelligence",
                "description": "Study of nature-inspired computational approaches and intelligent systems.",
                "keywords": ["computational intelligence", "neural networks", "fuzzy logic", "evolutionary computation", "swarm intelligence", "genetic algorithms", "machine learning", "pattern recognition", "optimization", "adaptive systems"]
            },
            {
                "course_id": "CS331",
                "title": "Compiler Design",
                "description": "Principles and techniques of programming language implementation and compiler construction.",
                "keywords": ["compiler design", "programming languages", "lexical analysis", "parsing", "code generation", "optimization", "language processing", "syntax analysis", "semantic analysis", "intermediate code"]
            }
        ])
        
        # Ensure all courses have keywords as lists
        for i, row in self.courses_df.iterrows():
            if isinstance(row['keywords'], str):
                self.courses_df.at[i, 'keywords'] = row['keywords'].split(',')
    
    def prepare_model(self):
        """Prepare the recommendation model"""
        # Create content field for each course by combining title, description, and keywords
        self.courses_df['content'] = self.courses_df.apply(
            lambda row: ' '.join([
                row['title'] * 3,  # Repeat title for higher weight
                row['description'],
                ' '.join(row['keywords'] * 2)  # Repeat keywords for higher weight
            ]).lower(),
            axis=1
        )
        
        # Transform text to TF-IDF vectors
        content_matrix = self.vectorizer.fit_transform(self.courses_df['content'])
        
        # Fit KNN model
        self.model.fit(content_matrix)
        self.ready = True
        
        return content_matrix
    
    def get_recommendations(self, selected_courses, skills, interests, career_goals, num_recommendations=1):
        """Get the best course from selected courses based on user preferences"""
        if not self.ready:
            self.prepare_model()
            
        # Verify selected courses exist
        selected_courses_df = self.courses_df[self.courses_df['title'].isin(selected_courses)]
        if len(selected_courses) == 0:
            return {
                "status": "error",
                "message": "No courses selected. Please select at least one course."
            }
            
        if selected_courses_df.empty:
            return {
                "status": "error", 
                "message": "None of the selected courses were found in our database."
            }
            
        # If only one course is selected, return it directly
        if len(selected_courses_df) == 1:
            course = selected_courses_df.iloc[0]
            return {
                "status": "success",
                "count": 1,
                "recommendations": [{
                    "course_id": course["course_id"],
                    "title": course["title"],
                    "description": course["description"],
                    "keywords": course["keywords"],
                    "match_score": 100.0,
                    "explanation": "This is your only selected course."
                }]
            }
        
        # Create user profile based on skills, interests, and career goals
        user_profile = ' '.join([
            ' '.join(skills) * 3,
            ' '.join(interests) * 3,
            ' '.join(career_goals) * 3
        ]).lower()
        
        # If user profile is empty, recommend the first selected course
        if not user_profile.strip():
            course = selected_courses_df.iloc[0]
            return {
                "status": "success",
                "count": 1,
                "recommendations": [{
                    "course_id": course["course_id"],
                    "title": course["title"],
                    "description": course["description"],
                    "keywords": course["keywords"],
                    "match_score": 100.0,
                    "explanation": "Recommended as default since no preferences were provided."
                }]
            }
        
        # Prepare the model if content column doesn't exist
        if 'content' not in selected_courses_df.columns:
            # Create content field for selected courses
            selected_courses_df['content'] = selected_courses_df.apply(
                lambda row: ' '.join([
                    row['title'] * 3,
                    row['description'],
                    ' '.join(row['keywords'] * 2)
                ]).lower(),
                axis=1
            )
            
        # Transform selected courses content to vectors
        selected_content = selected_courses_df['content'].tolist()
        # Fit the vectorizer on selected courses
        content_matrix = self.vectorizer.fit_transform(selected_content)
        
        # Transform user profile to vector using the same vectorizer
        user_vector = self.vectorizer.transform([user_profile])
        
        # Calculate cosine similarity between user profile and each course
        similarities = cosine_similarity(user_vector, content_matrix).flatten()
        
        # Get indices of courses sorted by similarity (highest first)
        sorted_indices = similarities.argsort()[::-1]
        
        # Create recommendations
        recommendations = []
        for idx in sorted_indices[:num_recommendations]:
            course = selected_courses_df.iloc[idx]
            similarity = float(similarities[idx]) * 100  # Convert to percentage
            
            # Create explanation
            explanation = self._create_explanation(course, skills, interests, career_goals)
            if not explanation.startswith("Recommended"):
                explanation = f"Recommended from your selected courses: {explanation}"
            
            recommendations.append({
                "course_id": course["course_id"],
                "title": course["title"],
                "description": course["description"],
                "keywords": course["keywords"],
                "match_score": round(similarity, 1),
                "explanation": explanation
            })
        
        return {
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        }
    
    def _create_explanation(self, course, skills, interests, career_goals):
        """Create a simple explanation for recommendation"""
        explanation_parts = []
        
        # Check for keyword matches
        course_keywords = set([k.lower() for k in course['keywords']])
        user_skills = set([s.lower() for s in skills])
        user_interests = set([i.lower() for i in interests])
        user_goals = set([g.lower() for g in career_goals])
        
        # Find overlaps
        skill_matches = course_keywords.intersection(user_skills)
        interest_matches = course_keywords.intersection(user_interests)
        goal_matches = course_keywords.intersection(user_goals)
        
        if skill_matches:
            explanation_parts.append(f"Matches your skills in {', '.join(skill_matches)}.")
            
        if interest_matches:
            explanation_parts.append(f"Aligns with your interests in {', '.join(interest_matches)}.")
            
        if goal_matches:
            explanation_parts.append(f"Supports your career goals in {', '.join(goal_matches)}.")
            
        if not explanation_parts:
            # Default explanation if no specific matches
            explanation_parts.append("Recommended based on your overall profile.")
            
        return " ".join(explanation_parts)
    
    def save_model(self):
        """Save the model to disk"""
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'courses_df': self.courses_df
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load the model from disk"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.model = data['model']
                    self.courses_df = data['courses_df']
                    self.ready = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class ScorePredictor:
    def __init__(self):
        self.model = None
        self.poly = PolynomialFeatures(degree=2)
        self.ready = False
        self.datasets = {}
        self.load_datasets()
        
    def load_datasets(self):
        """Load all academic datasets"""
        self.datasets = generate_academic_datasets()
        # Set the general dataset as the default
        self.data = self.datasets["general"]
        
    def generate_sample_data(self):
        """Generate 100 hardcoded academic records"""
        # This method is kept for backward compatibility
        # The actual data is now loaded from datasets.py
        self.load_datasets()
        
    def train_model(self, dataset_name="general"):
        """Train the score prediction model on selected dataset"""
        # Set the active dataset
        if dataset_name in self.datasets:
            self.data = self.datasets[dataset_name]
        else:
            # Default to general dataset if requested one doesn't exist
            self.data = self.datasets["general"]
            
        # We'll train multiple models for different semester predictions
        self.models = {}
        
        # Train model for 1st semester (based on 10th, 12th)
        X_1st = self.data[['10th', '12th']]
        y_1st = self.data['1st']
        X_1st_poly = self.poly.fit_transform(X_1st)
        model_1st = LinearRegression()
        model_1st.fit(X_1st_poly, y_1st)
        self.models['1st'] = model_1st
        
        # Train models for each subsequent semester
        for i, sem in enumerate(['2nd', '3rd', '4th', '5th', '6th', '7th', '8th']):
            # Features are all previous grades
            prev_sems = ['10th', '12th'] + [f'{j}st' if j == 1 else f'{j}nd' if j == 2 
                                          else f'{j}rd' if j == 3 else f'{j}th' 
                                          for j in range(1, i+2)]
            
            X = self.data[prev_sems]
            y = self.data[sem]
            
            # Apply polynomial features for better prediction
            X_poly = self.poly.fit_transform(X)
            
            # Train linear model
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Store model
            self.models[sem] = model
            
        self.ready = True
        return self.models
    
    def predict_next_semester(self, academic_record, dataset_name=None):
        """Predict the next semester's score based on previous scores"""
        # If dataset was specified, make sure the model is trained on it
        if dataset_name and (not self.ready or dataset_name not in self.datasets):
            self.train_model(dataset_name)
        # Make sure model is trained
        elif not self.ready:
            self.train_model()
            
        # Determine which semester to predict
        provided_sems = set(academic_record.keys())
        
        # Define all possible semesters in order
        all_sems = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
        
        # Find the last provided semester
        last_provided = None
        for sem in reversed(all_sems):
            if sem in provided_sems:
                last_provided = sem
                break
        
        # Find the next semester to predict
        next_sem = None
        if last_provided:
            try:
                last_idx = all_sems.index(last_provided)
                if last_idx < len(all_sems) - 1:
                    next_sem = all_sems[last_idx + 1]
            except ValueError:
                next_sem = None
        
        # If we can't determine the next semester
        if not next_sem:
            return {
                "status": "error",
                "message": "Cannot determine which semester to predict. Please provide proper previous semester scores."
            }
        
        # If trying to predict 1st semester, need 10th and 12th
        if next_sem == '1st' and ('10th' not in provided_sems or '12th' not in provided_sems):
            return {
                "status": "error",
                "message": "To predict 1st semester, both 10th and 12th scores are required."
            }
        
        # Check if we have a model for this prediction
        if next_sem not in self.models:
            return {
                "status": "error",
                "message": f"No prediction model available for {next_sem} semester."
            }
        
        # Determine required previous semesters for prediction
        if next_sem == '1st':
            req_sems = ['10th', '12th']
        else:
            sem_idx = all_sems.index(next_sem)
            req_sems = all_sems[:sem_idx]
        
        # Check if all required previous semesters are provided
        missing_sems = [sem for sem in req_sems if sem not in provided_sems]
        if missing_sems:
            return {
                "status": "error",
                "message": f"Missing scores for {', '.join(missing_sems)}. These are required to predict {next_sem} semester."
            }
        
        # Create feature vector for prediction
        X = [[academic_record[sem] for sem in req_sems]]
        X_poly = self.poly.fit_transform(X)
        
        # Make prediction
        predicted_score = float(self.models[next_sem].predict(X_poly)[0])
        
        # Ensure prediction is within valid GPA range
        predicted_score = min(10.0, max(6.0, predicted_score))
        
        # Round to 2 decimal places
        predicted_score = round(predicted_score, 2)
        
        # Prepare response with explanation
        performance_trend = self._analyze_performance_trend(academic_record, next_sem, predicted_score)
        
        return {
            "status": "success",
            "predicted_semester": next_sem,
            "predicted_score": predicted_score,
            "performance_analysis": performance_trend
        }
    
    def _analyze_performance_trend(self, academic_record, next_sem, predicted_score):
        """Analyze the student's performance trend to provide insights"""
        # Get previous semester scores in chronological order
        all_sems = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
        prev_scores = []
        
        for sem in all_sems:
            if sem in academic_record:
                prev_scores.append(academic_record[sem])
        
        # Basic analysis based on last few semesters
        if len(prev_scores) >= 2:
            recent_trend = prev_scores[-1] - prev_scores[-2]
            
            if predicted_score > prev_scores[-1]:
                if recent_trend > 0:
                    return "Your performance shows a consistent upward trend. Keep up the good work!"
                else:
                    return "Our prediction shows an improvement over your last semester, which is a positive sign."
            elif predicted_score < prev_scores[-1]:
                if recent_trend < 0:
                    return "Your scores show a downward trend. Consider focusing more on your studies."
                else:
                    return "While your recent performance improved, our model predicts a slight decrease. Maintain your study habits."
            else:
                return "Your performance is predicted to remain stable."
        
        # If not enough previous scores, provide a generic message
        return "Based on your academic history, this is our predicted score."
    
    def save_model(self):
        """Save the score prediction models to disk"""
        try:
            with open(SCORE_MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'data': self.data,
                    'datasets': self.datasets,
                    'poly': self.poly
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving score prediction model: {str(e)}")
            return False
    
    def load_model(self):
        """Load score prediction models from disk"""
        try:
            if os.path.exists(SCORE_MODEL_PATH):
                with open(SCORE_MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.models = data['models']
                    self.data = data['data']
                    if 'datasets' in data:
                        self.datasets = data['datasets']
                    else:
                        self.load_datasets()
                    self.poly = data['poly']
                    self.ready = True
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading score prediction model: {str(e)}")
            return False

# Initialize the recommender and score predictor
recommender = SimpleRecommender()
score_predictor = ScorePredictor()

# API routes
@app.before_request
def initialize():
    """Initialize models before first request"""
    if recommender.courses_df is None:
        if not recommender.load_model():
            recommender.load_courses()
            recommender.prepare_model()
            recommender.save_model()
            
    if not score_predictor.ready:
        if not score_predictor.load_model():
            score_predictor.train_model()
            score_predictor.save_model()

@app.route('/api/courses', methods=['GET'])
def get_courses():
    """Get all available courses"""
    courses = recommender.courses_df['title'].tolist()
    return jsonify({
        "status": "success",
        "count": len(courses),
        "courses": courses
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_courses():
    """Get course recommendations"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({
                "error": "No data provided",
                "example": {
                    "selected_courses": ["Data Structures", "Machine Learning", "Web Development"],
                    "skills": ["python", "algorithms"],
                    "interests": ["artificial intelligence"],
                    "career_goals": ["software engineer"]
                }
            }), 400
            
        # Extract parameters with defaults
        selected_courses = data.get('selected_courses', [])
        skills = data.get('skills', [])
        interests = data.get('interests', [])
        career_goals = data.get('career_goals', [])
        num_recommendations = min(int(data.get('num_recommendations', 1)), len(selected_courses))
        
        # Ensure at least one selected course
        if not selected_courses:
            return jsonify({
                "error": "No courses selected",
                "message": "Please select at least one course"
            }), 400
            
        # Get recommendations
        result = recommender.get_recommendations(
            selected_courses, 
            skills, 
            interests, 
            career_goals, 
            num_recommendations
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({
            "error": "Failed to generate recommendations",
            "message": str(e)
        }), 500

@app.route('/api/course/<course_id>', methods=['GET'])
def get_course(course_id):
    """Get details for a specific course"""
    course = recommender.courses_df[recommender.courses_df['course_id'] == course_id]
    
    if course.empty:
        return jsonify({
            "error": "Course not found"
        }), 404
        
    return jsonify({
        "status": "success",
        "course": course.iloc[0].drop('content', errors='ignore').to_dict()
    })

@app.route('/api/predict-score', methods=['POST'])
def predict_score():
    """Predict next semester's score based on academic history"""
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({
                "error": "No data provided",
                "example": {
                    "10th": 9.12,
                    "12th": 8.75,
                    "1st": 9.43,
                    "2nd": 9.33,
                    "3rd": 9.23,
                    "dataset": "engineering"  # Optional dataset name
                },
                "note": "Provide your academic scores in order. For 1st semester prediction, provide 10th and 12th scores."
            }), 400
            
        # Extract dataset name if provided
        dataset_name = data.pop('dataset', None)
        
        # Ensure we have at least one academic score
        if not any(key in data for key in ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th']):
            return jsonify({
                "error": "No valid academic scores provided",
                "message": "Please provide at least your 10th and 12th scores."
            }), 400
            
        # Get prediction
        result = score_predictor.predict_next_semester(data, dataset_name)
        
        # Add dataset info to response
        if dataset_name:
            result['dataset'] = dataset_name
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in score prediction: {str(e)}")
        return jsonify({
            "error": "Failed to generate score prediction",
            "message": str(e)
        }), 500

@app.route('/api/academic-records', methods=['GET'])
def get_academic_records():
    """Get sample academic records from the dataset"""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=10, type=int)
        offset = request.args.get('offset', default=0, type=int)
        pattern = request.args.get('pattern', default=None, type=int)
        dataset = request.args.get('dataset', default='general')
        
        # Ensure models are initialized
        if not score_predictor.ready:
            if not score_predictor.load_model():
                score_predictor.train_model()
        
        # Get the appropriate dataset
        if dataset in score_predictor.datasets:
            data_df = score_predictor.datasets[dataset]
        else:
            data_df = score_predictor.data
            dataset = 'general'
        
        # Get data based on parameters
        if pattern is not None and 0 <= pattern <= 9:
            # Get variations of a specific pattern
            pattern_indices = [pattern] + [10 + pattern + (i * 10) for i in range(9)]
            records = data_df.iloc[pattern_indices].reset_index(drop=True)
        else:
            # Get records with pagination
            records = data_df.iloc[offset:offset+limit].reset_index(drop=True)
        
        # Convert to dict for JSON response
        records_dict = records.to_dict(orient='records')
        
        # Add descriptions for the first 10 patterns (general patterns)
        pattern_descriptions = [
            "Example provided in the prompt",
            "High performer with consistent scores",
            "Average performer with gradual improvement",
            "Strong start but declining performance",
            "Weak start but significant improvement",
            "Fluctuating performance",
            "Consistent average performer",
            "U-shaped performance (starts high, dips, recovers)",
            "Inverted U-shape (starts low, peaks, declines)",
            "Late bloomer (mediocre start, strong finish)"
        ]
        
        # Add descriptions to the first 10 records if they're included
        for i, record in enumerate(records_dict):
            if i < 10 and (pattern is None or pattern == i):
                record['description'] = pattern_descriptions[i % 10]
        
        return jsonify({
            "status": "success",
            "dataset": dataset,
            "count": len(records_dict),
            "total_records": len(data_df),
            "records": records_dict,
            "patterns": {
                f"pattern_{i}": pattern_descriptions[i] for i in range(10)
            },
            "available_datasets": list(score_predictor.datasets.keys())
        })
        
    except Exception as e:
        logger.error(f"Error getting academic records: {str(e)}")
        return jsonify({
            "error": "Failed to get academic records",
            "message": str(e)
        }), 500

@app.route('/api/datasets', methods=['GET'])
def get_available_datasets():
    """Get list of all available academic datasets"""
    try:
        # Ensure score predictor is initialized
        if not score_predictor.ready:
            if not score_predictor.load_model():
                score_predictor.train_model()
                
        # Get dataset information
        dataset_info = {
            "general": "General academic patterns across disciplines",
            "engineering": "Engineering student performance patterns",
            "medical": "Medical school student performance patterns",
            "business": "Business and management student performance patterns",
            "arts": "Arts and humanities student performance patterns"
        }
        
        # Count records in each dataset
        dataset_counts = {name: len(df) for name, df in score_predictor.datasets.items()}
        
        return jsonify({
            "status": "success",
            "available_datasets": list(score_predictor.datasets.keys()),
            "dataset_info": dataset_info,
            "record_counts": dataset_counts,
            "usage": "Specify dataset in POST body when using /api/predict-score, or as query parameter for /api/academic-records"
        })
        
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}")
        return jsonify({
            "error": "Failed to get datasets",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "name": "Academic Recommendation API",
        "endpoints": {
            "/api/courses": "GET - List all courses",
            "/api/course/<course_id>": "GET - Get course details",
            "/api/recommend": "POST - Get course recommendations",
            "/api/predict-score": "POST - Predict next semester's score",
            "/api/academic-records": "GET - View sample academic records",
            "/api/datasets": "GET - List all available academic datasets"
        },
        "examples": {
            "recommend": {
                "url": "/api/recommend",
                "method": "POST",
                "body": {
                    "selected_courses": ["Data Structures", "Machine Learning", "Web Development"],
                    "skills": ["python", "algorithms"],
                    "interests": ["artificial intelligence"],
                    "career_goals": ["software engineer"]
                }
            },
            "predict-score": {
                "url": "/api/predict-score",
                "method": "POST",
                "body": {
                    "10th": 9.12,
                    "12th": 8.75,
                    "1st": 9.43,
                    "2nd": 9.33,
                    "3rd": 9.23,
                    "dataset": "engineering"  # Optional - specify dataset
                }
            },
            "academic-records": {
                "url": "/api/academic-records?limit=10&offset=0&dataset=medical",
                "method": "GET",
                "description": "Get sample academic records. Use pattern=0-9 to view specific patterns."
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True) 