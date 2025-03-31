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
        """Load sample course data"""
        self.courses_df = pd.DataFrame([
            {
                "course_id": "CS101",
                "title": "Data Structures",
                "description": "This course covers fundamental data structures, including arrays, linked lists, stacks, queues, trees, and graphs. Students will learn efficient sorting and searching algorithms, hash tables, and dynamic programming.",
                "keywords": ["data structures", "algorithms", "trees", "graphs", "sorting"]
            },
            {
                "course_id": "CS102",
                "title": "Database Management",
                "description": "Learn the foundations of relational database systems, covering SQL, normalization, indexing, transaction management, and distributed databases.",
                "keywords": ["database", "SQL", "indexing", "data modeling"]
            },
            {
                "course_id": "CS103",
                "title": "Operating Systems",
                "description": "Explore the inner workings of modern operating systems, including process scheduling, memory management, file systems, and concurrency.",
                "keywords": ["OS", "Linux", "Windows", "concurrency", "memory management"]
            },
            {
                "course_id": "CS104",
                "title": "Computer Networks",
                "description": "Dive into the fundamentals of computer networking, covering TCP/IP, network security, routing algorithms, firewalls, and VPNs.",
                "keywords": ["networking", "TCP/IP", "routing", "network security"]
            },
            {
                "course_id": "CS105",
                "title": "Machine Learning",
                "description": "A comprehensive introduction to supervised and unsupervised learning, covering regression, classification, clustering, neural networks, and deep learning.",
                "keywords": ["machine learning", "AI", "deep learning", "neural networks"]
            },
            {
                "course_id": "CS106",
                "title": "Web Development",
                "description": "Master full-stack web development, including HTML, CSS, JavaScript, React, Node.js, and databases.",
                "keywords": ["web development", "frontend", "backend", "JavaScript"]
            },
            {
                "course_id": "CS107",
                "title": "Cybersecurity",
                "description": "Learn about ethical hacking, encryption, network security, penetration testing, and cybersecurity threats.",
                "keywords": ["cybersecurity", "encryption", "security", "penetration testing"]
            },
            {
                "course_id": "CS108",
                "title": "Software Engineering",
                "description": "This course covers software development methodologies, including Agile, DevOps, software testing, and design patterns.",
                "keywords": ["software engineering", "Agile", "testing", "design patterns"]
            },
            {
                "course_id": "DS101",
                "title": "Data Science and Machine Learning",
                "description": "Comprehensive coverage of data science methodologies and machine learning algorithms including statistics and predictive modeling.",
                "keywords": ["data science", "machine learning", "statistics", "predictive modeling"]
            },
            {
                "course_id": "CS201",
                "title": "Deep Learning",
                "description": "Advanced neural network architectures and deep learning techniques including CNNs, RNNs, and GANs.",
                "keywords": ["deep learning", "neural networks", "CNN", "RNN", "AI"]
            },
            {
                "course_id": "CS202", 
                "title": "Artificial Intelligence",
                "description": "Fundamentals of AI including search algorithms, knowledge representation, planning, and intelligent agents.",
                "keywords": ["AI", "search algorithms", "knowledge representation", "planning"]
            },
            {
                "course_id": "CS203",
                "title": "Mobile App Development",
                "description": "Design and develop mobile applications for iOS and Android platforms using modern frameworks.",
                "keywords": ["mobile", "app development", "iOS", "Android", "React Native"]
            },
            {
                "course_id": "CS204",
                "title": "Cloud Computing",
                "description": "Learn about cloud services, virtualization, containers, and distributed systems in the cloud.",
                "keywords": ["cloud", "AWS", "Azure", "virtualization", "containers"]
            },
            {
                "course_id": "CS205",
                "title": "Blockchain Technology",
                "description": "Understand blockchain principles, cryptocurrencies, smart contracts, and decentralized applications.",
                "keywords": ["blockchain", "cryptocurrency", "smart contracts", "decentralized"]
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