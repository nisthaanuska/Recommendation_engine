# Academic Recommendation System

A comprehensive system that helps students with course selection and academic performance prediction.

## Features

### Course Recommendation
    - ** Smart Selection **: Recommends the best course from your selected options based on:
- Your skills
    - Your interests
        - Your career goals
            - ** Clear Explanations **: Understand why a specific course is recommended

### Score Prediction
    - ** Next Semester Prediction **: Predicts your next semester's score based on your academic history
        - ** Performance Analysis **: Provides insights about your academic performance trend
            - ** Progression Aware **: Understands the sequence of semesters and requires appropriate previous scores

## Installation

1. Clone the repository
2. Install dependencies:
```
   pip install -r requirements.txt
   ```
3. Run the server:
```
   python main.py
   ```

## API Usage

### Course Recommendation API

#### List All Courses
    ```
GET /api/courses
```

#### Get Course Details
    ```
GET /api/course/<course_id>
```

#### Get Recommendation
    ```
POST /api/recommend
```

Request body:
```json
{
  "selected_courses": ["Data Structures", "Machine Learning", "Web Development"],
  "skills": ["python", "algorithms"],
  "interests": ["artificial intelligence"],
  "career_goals": ["software engineer"]
}
```

Example response:
```json
{
  "status": "success",
  "count": 1,
  "recommendations": [
    {
      "course_id": "CS105",
      "title": "Machine Learning",
      "description": "A comprehensive introduction to supervised and unsupervised learning...",
      "keywords": ["machine learning", "AI", "deep learning", "neural networks"],
      "match_score": 92.5,
      "explanation": "Recommended from your selected courses: Matches your skills in algorithms. Aligns with your interests in artificial intelligence."
    }
  ]
}
```

### Score Prediction API

#### Predict Next Semester's Score
    ```
POST /api/predict-score
```

Request body:
```json
{
  "10th": 9.12,
  "12th": 8.75,
  "1st": 9.43,
  "2nd": 9.33,
  "3rd": 9.23
}
```

Example response:
```json
{
  "status": "success",
  "predicted_semester": "4th",
  "predicted_score": 9.15,
  "performance_analysis": "Your performance shows a consistent downward trend. Consider focusing more on your studies."
}
```

## How It Works

### Course Recommendation
1. ** Course Selection **: You provide multiple courses you're considering
2. ** Preference Analysis **: Your skills, interests, and career goals are analyzed
3. ** Matching **: System finds which of your selected courses best aligns with your preferences
4. ** Recommendation **: Returns the most suitable course with an explanation

### Score Prediction
1. ** Academic History **: System analyzes your previous semester scores
2. ** Pattern Recognition **: Identifies patterns in your performance over time
3. ** Prediction Model **: Uses polynomial regression to predict your next semester's score
4. ** Performance Analysis **: Provides insights about your academic trajectory

## Score Prediction Rules
    - To predict 1st semester, both 10th and 12th scores are required
        - To predict any semester, all previous semester scores are required
            - System automatically determines which semester to predict based on provided scores

## License

MIT

## Author

This recommendation engine was developed as a course project for elective course recommendation. 