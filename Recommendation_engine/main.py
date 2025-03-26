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
            },
            {
                "title": "Data Mining and Data Warehousing",
                "description": "Techniques for discovering patterns in large datasets and organizing data for analysis. Covers data preprocessing, mining algorithms, warehouse design, and ETL processes.",
                "keywords": ["data mining", "warehousing", "ETL", "pattern recognition", "data analysis", "business intelligence"]
            },
            {
                "title": "Data Science and Machine Learning",
                "description": "Comprehensive coverage of data science methodologies and machine learning algorithms. Includes statistical analysis, predictive modeling, and practical applications.",
                "keywords": ["data science", "machine learning", "statistics", "predictive modeling", "analytics", "python", "R"]
            },
            {
                "title": "Deep Learning and Neural Networks",
                "description": "Advanced neural network architectures and deep learning techniques. Covers CNNs, RNNs, GANs, transformers, and their applications in various domains.",
                "keywords": ["deep learning", "neural networks", "CNN", "RNN", "GAN", "AI", "pytorch", "tensorflow"]
            },
            {
                "title": "Digital Forensics",
                "description": "Investigation of digital artifacts and cyber incidents. Covers forensic tools, techniques, incident response, and legal aspects of digital investigations.",
                "keywords": ["forensics", "cybersecurity", "investigation", "incident response", "legal", "security"]
            },
            {
                "title": "Digital Marketing and Analytics",
                "description": "Digital marketing strategies and analytics tools for measuring campaign effectiveness. Covers SEO, social media marketing, and web analytics.",
                "keywords": ["digital marketing", "analytics", "SEO", "social media", "web analytics", "marketing strategy"]
            },
            {
                "title": "Distributed Computing",
                "description": "Principles and practices of distributed systems and computing. Covers distributed algorithms, fault tolerance, consistency models, and distributed databases.",
                "keywords": ["distributed systems", "parallel computing", "fault tolerance", "scalability", "consistency", "algorithms"]
            },
            {
                "title": "Edge Computing",
                "description": "Computing paradigm that brings computation closer to data sources. Covers edge architecture, IoT integration, and real-time processing at the network edge.",
                "keywords": ["edge computing", "IoT", "distributed systems", "real-time processing", "network architecture"]
            },
            {
                "title": "Embedded Systems",
                "description": "Design and implementation of embedded computing systems. Covers microcontrollers, real-time operating systems, and embedded software development.",
                "keywords": ["embedded systems", "microcontrollers", "RTOS", "firmware", "IoT", "hardware"]
            },
            {
                "title": "Ethical Hacking",
                "description": "Security testing and vulnerability assessment techniques. Covers penetration testing, security auditing, and ethical aspects of security testing.",
                "keywords": ["ethical hacking", "penetration testing", "security", "vulnerability assessment", "cybersecurity"]
            },
            {
                "title": "Evolutionary Computation",
                "description": "Nature-inspired algorithms for optimization and search. Covers genetic algorithms, evolutionary strategies, and swarm intelligence.",
                "keywords": ["evolutionary algorithms", "genetic programming", "optimization", "swarm intelligence", "AI"]
            },
            {
                "title": "Fuzzy Logic and Soft Computing",
                "description": "Principles of fuzzy logic and its applications in computing. Covers fuzzy systems, soft computing techniques, and approximate reasoning.",
                "keywords": ["fuzzy logic", "soft computing", "AI", "computational intelligence", "decision making"]
            },
            {
                "title": "Game Development and Design",
                "description": "Principles and practices of video game development. Covers game engines, graphics programming, game physics, and interactive design.",
                "keywords": ["game development", "Unity", "Unreal Engine", "graphics", "game design", "programming"]
            },
            {
                "title": "Geographic Information Systems",
                "description": "Analysis and visualization of geographic data. Covers spatial analysis, mapping technologies, and GIS applications.",
                "keywords": ["GIS", "spatial analysis", "mapping", "remote sensing", "geospatial", "data analysis"]
            },
            {
                "title": "Green Computing",
                "description": "Environmentally sustainable computing practices. Covers energy-efficient systems, sustainable IT, and green data center design.",
                "keywords": ["green computing", "sustainability", "energy efficiency", "environmental IT", "data centers"]
            },
            {
                "title": "High-Performance Computing",
                "description": "Advanced computing architectures and parallel programming. Covers supercomputing, parallel algorithms, and performance optimization.",
                "keywords": ["HPC", "parallel computing", "supercomputing", "optimization", "parallel programming"]
            },
            {
                "title": "Human-Centered AI",
                "description": "Design and development of AI systems focused on human needs and interaction. Covers user experience, ethical AI, and human-AI collaboration.",
                "keywords": ["AI", "human-centered design", "UX", "ethical AI", "human-computer interaction"]
            },
            {
                "title": "Humanoid Robotics",
                "description": "Design and control of humanoid robots. Covers robot kinematics, dynamics, control systems, and human-robot interaction.",
                "keywords": ["robotics", "humanoid", "control systems", "AI", "mechanical engineering"]
            },
            {
                "title": "Image Processing and Pattern Recognition",
                "description": "Digital image processing techniques and pattern recognition algorithms. Covers image enhancement, feature extraction, and classification.",
                "keywords": ["image processing", "pattern recognition", "computer vision", "machine learning", "AI"]
            },
            {
                "title": "Immersive Technologies",
                "description": "Development of AR, VR, and mixed reality applications. Covers 3D modeling, interaction design, and immersive user experiences.",
                "keywords": ["AR", "VR", "mixed reality", "3D", "interaction design", "Unity"]
            },
            {
                "title": "Information and Web Security",
                "description": "Security principles for web applications and information systems. Covers web vulnerabilities, security protocols, and defensive programming.",
                "keywords": ["web security", "cybersecurity", "information security", "secure coding", "protocols"]
            },
            {
                "title": "Next-Generation Wireless Networks",
                "description": "Advanced study of 5G/6G technologies, network architectures, and protocols. Covers mmWave, network slicing, and emerging wireless technologies.",
                "keywords": ["5G", "6G", "wireless", "networking", "telecommunications", "protocols", "mobile computing"]
            },
            {
                "title": "Parallel and Distributed Systems",
                "description": "Design and implementation of parallel and distributed computing systems. Covers parallel algorithms, distributed architectures, and system performance.",
                "keywords": ["parallel computing", "distributed systems", "algorithms", "system architecture", "performance"]
            },
            {
                "title": "Post-Quantum Cryptography",
                "description": "Cryptographic systems resistant to quantum computing attacks. Covers quantum-resistant algorithms, lattice-based cryptography, and security protocols.",
                "keywords": ["cryptography", "quantum computing", "security", "algorithms", "quantum resistance"]
            },
            {
                "title": "Quantum Computing",
                "description": "Principles of quantum computation and quantum information. Covers quantum algorithms, quantum circuits, and quantum programming frameworks.",
                "keywords": ["quantum computing", "quantum algorithms", "quantum circuits", "quantum information", "physics"]
            },
            {
                "title": "Recommender Systems",
                "description": "Design and implementation of recommendation algorithms. Covers collaborative filtering, content-based systems, and hybrid approaches.",
                "keywords": ["recommender systems", "machine learning", "data mining", "personalization", "algorithms"]
            },
            {
                "title": "Robotics and Intelligent Systems",
                "description": "Fundamentals of robotics and intelligent system design. Covers robot control, perception, planning, and artificial intelligence integration.",
                "keywords": ["robotics", "AI", "control systems", "perception", "planning", "automation"]
            },
            {
                "title": "Security in IoT",
                "description": "Security challenges and solutions in Internet of Things ecosystems. Covers device security, network protocols, and threat mitigation.",
                "keywords": ["IoT security", "cybersecurity", "embedded systems", "network security", "protocols"]
            },
            {
                "title": "Self-Adaptive Software Systems",
                "description": "Design of software systems that can modify their behavior at runtime. Covers adaptive algorithms, monitoring, and self-healing systems.",
                "keywords": ["adaptive systems", "software engineering", "autonomous systems", "self-healing", "monitoring"]
            },
            {
                "title": "Semantic Web and Ontology Engineering",
                "description": "Technologies and principles of the semantic web. Covers ontology design, RDF, OWL, and knowledge representation.",
                "keywords": ["semantic web", "ontology", "RDF", "knowledge representation", "linked data"]
            },
            {
                "title": "Sensor Networks and Pervasive Computing",
                "description": "Design and implementation of sensor networks and ubiquitous computing systems. Covers wireless sensors, data collection, and analysis.",
                "keywords": ["sensor networks", "pervasive computing", "IoT", "wireless", "data collection"]
            },
            {
                "title": "Social Network Analysis",
                "description": "Analysis of social networks and online social behavior. Covers graph theory, network metrics, and social media analytics.",
                "keywords": ["social networks", "graph analysis", "network science", "analytics", "data mining"]
            },
            {
                "title": "Software Defined Networking",
                "description": "Principles and practices of SDN and network virtualization. Covers OpenFlow, network controllers, and programmable networks.",
                "keywords": ["SDN", "networking", "virtualization", "OpenFlow", "network programming"]
            },
            {
                "title": "Software Testing and Quality Assurance",
                "description": "Comprehensive software testing methodologies and quality practices. Covers test automation, quality metrics, and testing frameworks.",
                "keywords": ["software testing", "QA", "test automation", "quality assurance", "testing frameworks"]
            },
            {
                "title": "Speech and Audio Processing",
                "description": "Digital processing of speech and audio signals. Covers speech recognition, synthesis, and audio analysis techniques.",
                "keywords": ["speech processing", "audio analysis", "signal processing", "recognition", "synthesis"]
            },
            {
                "title": "Swarm Intelligence",
                "description": "Study of decentralized, self-organizing systems. Covers swarm algorithms, collective behavior, and distributed optimization.",
                "keywords": ["swarm intelligence", "optimization", "collective behavior", "algorithms", "AI"]
            },
            {
                "title": "Trust and Privacy in AI",
                "description": "Ethical considerations and privacy aspects in AI systems. Covers privacy-preserving ML, trustworthy AI, and ethical guidelines.",
                "keywords": ["AI ethics", "privacy", "trust", "machine learning", "data protection"]
            },
            {
                "title": "Ubiquitous Computing",
                "description": "Design of computing systems that integrate seamlessly into everyday life. Covers ambient intelligence, context-aware systems, and smart environments.",
                "keywords": ["ubiquitous computing", "ambient intelligence", "IoT", "context-aware", "smart systems"]
            },
            {
                "title": "Vehicular Ad Hoc Networks",
                "description": "Study of communication networks for vehicles. Covers VANET protocols, vehicular communication, and intelligent transportation systems.",
                "keywords": ["VANET", "vehicular networks", "networking", "transportation", "communication"]
            },
            {
                "title": "Virtual Reality and Augmented Reality",
                "description": "Development of VR and AR applications. Covers 3D modeling, interaction design, and immersive experience creation.",
                "keywords": ["VR", "AR", "3D modeling", "interaction design", "immersive technology"]
            },
            {
                "title": "Wearable Computing",
                "description": "Design and development of wearable technology. Covers sensor integration, mobile computing, and human-centered design.",
                "keywords": ["wearable technology", "mobile computing", "sensors", "IoT", "human-centered design"]
            },
            {
                "title": "Web 3.0 and Decentralized Applications",
                "description": "Development of decentralized web applications. Covers blockchain, smart contracts, and distributed systems.",
                "keywords": ["Web3", "blockchain", "DApps", "smart contracts", "decentralized systems"]
            },
            {
                "title": "Wireless Sensor Networks and IoT Security",
                "description": "Security aspects of wireless sensor networks and IoT systems. Covers security protocols, threat analysis, and protection mechanisms.",
                "keywords": ["WSN", "IoT security", "network security", "sensors", "protocols"]
            },
            {
                "title": "Artificial Intelligence Ethics and Regulations",
                "description": "Study of ethical implications and regulatory frameworks for AI systems. Covers AI governance, compliance, and responsible AI development.",
                "keywords": ["AI ethics", "regulations", "compliance", "governance", "responsible AI"]
            },
            {
                "title": "Bio-Medical Data Processing",
                "description": "Processing and analysis of biomedical data. Covers medical imaging, health informatics, and clinical data analysis.",
                "keywords": ["biomedical", "health informatics", "medical imaging", "data analysis", "healthcare"]
            },
            {
                "title": "Computational Advertising",
                "description": "Algorithmic approaches to online advertising. Covers ad targeting, recommendation systems, and marketing analytics.",
                "keywords": ["advertising", "algorithms", "targeting", "analytics", "marketing"]
            },
            {
                "title": "Computational Finance",
                "description": "Application of computational methods in finance. Covers financial modeling, risk analysis, and algorithmic trading.",
                "keywords": ["finance", "algorithms", "modeling", "risk analysis", "trading"]
            },
            {
                "title": "Computational Geometry",
                "description": "Study of algorithms for geometric problems. Covers geometric data structures, spatial algorithms, and computational topology.",
                "keywords": ["geometry", "algorithms", "spatial computing", "computational topology", "mathematics"]
            },
            {
                "title": "Cyber Threat Intelligence",
                "description": "Analysis and response to cyber threats. Covers threat hunting, intelligence gathering, and security operations.",
                "keywords": ["cybersecurity", "threat intelligence", "security operations", "analysis", "incident response"]
            },
            {
                "title": "Data Visualization and Storytelling",
                "description": "Techniques for effective data visualization and communication. Covers visual design, interactive visualization, and data narrative.",
                "keywords": ["visualization", "data communication", "visual design", "storytelling", "analytics"]
            },
            {
                "title": "Explainable AI",
                "description": "Methods for making AI systems interpretable and explainable. Covers model interpretation, transparency, and accountability.",
                "keywords": ["XAI", "AI interpretability", "transparency", "machine learning", "ethics"]
            },
            {
                "title": "Fog Computing",
                "description": "Computing paradigm between edge and cloud. Covers fog architecture, distributed processing, and IoT integration.",
                "keywords": ["fog computing", "edge computing", "distributed systems", "IoT", "cloud computing"]
            },
            {
                "title": "Genetic Algorithms",
                "description": "Evolutionary computation techniques for optimization. Covers genetic programming, evolutionary strategies, and applications.",
                "keywords": ["genetic algorithms", "evolutionary computation", "optimization", "AI", "machine learning"]
            },
            {
                "title": "Graph Theory in Computer Science",
                "description": "Applications of graph theory in computing. Covers graph algorithms, network analysis, and computational problems.",
                "keywords": ["graph theory", "algorithms", "networks", "discrete mathematics", "computation"]
            },
            {
                "title": "Haptics and Human-Computer Interaction",
                "description": "Study of touch-based interaction in computing systems. Covers haptic interfaces, interaction design, and user experience.",
                "keywords": ["haptics", "HCI", "interaction design", "user experience", "interfaces"]
            },
            {
                "title": "Intelligent Transportation Systems",
                "description": "Smart transportation technologies and systems. Covers traffic management, autonomous vehicles, and transportation analytics.",
                "keywords": ["ITS", "transportation", "autonomous vehicles", "traffic management", "smart cities"]
            },
            {
                "title": "Internet Law and Cyber Regulations",
                "description": "Legal aspects of internet and technology use. Covers cyberlaws, digital rights, and regulatory compliance.",
                "keywords": ["cyber law", "regulations", "compliance", "digital rights", "internet governance"]
            },
            {
                "title": "Knowledge Engineering",
                "description": "Development of knowledge-based systems. Covers expert systems, knowledge representation, and reasoning systems.",
                "keywords": ["knowledge engineering", "expert systems", "AI", "knowledge representation", "reasoning"]
            },
            {
                "title": "Malware Analysis",
                "description": "Techniques for analyzing malicious software. Covers reverse engineering, threat analysis, and malware detection.",
                "keywords": ["malware", "security", "reverse engineering", "threat analysis", "cybersecurity"]
            },
            {
                "title": "Mobile and Wireless Security",
                "description": "Security aspects of mobile and wireless systems. Covers mobile security, wireless protocols, and threat mitigation.",
                "keywords": ["mobile security", "wireless security", "cybersecurity", "protocols", "networking"]
            },
            {
                "title": "Multimodal Interaction Systems",
                "description": "Design of systems using multiple input/output modalities. Covers speech, gesture, and multimodal interfaces.",
                "keywords": ["multimodal", "HCI", "interfaces", "interaction design", "user experience"]
            },
            {
                "title": "Neuro-Symbolic AI",
                "description": "Integration of neural networks and symbolic reasoning. Covers hybrid AI systems, knowledge integration, and reasoning.",
                "keywords": ["neuro-symbolic", "AI", "neural networks", "symbolic reasoning", "machine learning"]
            },
            {
                "title": "Privacy-Preserving Computing",
                "description": "Techniques for computing while preserving privacy. Covers cryptographic protocols, secure computation, and privacy technologies.",
                "keywords": ["privacy", "security", "cryptography", "secure computation", "data protection"]
            },
            {
                "title": "Software Reengineering",
                "description": "Methods for modernizing and improving existing software. Covers code analysis, refactoring, and system modernization.",
                "keywords": ["reengineering", "software maintenance", "refactoring", "legacy systems", "modernization"]
            },
            {
                "title": "Smart Contracts and Cryptoeconomics",
                "description": "Development and analysis of blockchain-based smart contracts. Covers contract programming, tokenomics, and blockchain economics.",
                "keywords": ["smart contracts", "blockchain", "cryptoeconomics", "DeFi", "programming"]
            },
            {
                "title": "Software Reliability",
                "description": "Methods for developing reliable software systems. Covers fault tolerance, reliability engineering, and quality assurance.",
                "keywords": ["reliability", "fault tolerance", "quality assurance", "testing", "software engineering"]
            },
            {
                "title": "Surveillance Technologies",
                "description": "Study of modern surveillance systems and privacy protection. Covers monitoring technologies, privacy preservation, and ethical considerations.",
                "keywords": ["surveillance", "privacy", "security", "ethics", "monitoring"]
            },
            {
                "title": "Synthetic Data Generation",
                "description": "Techniques for generating synthetic datasets. Covers data simulation, augmentation, and privacy-preserving data generation.",
                "keywords": ["synthetic data", "data generation", "simulation", "privacy", "machine learning"]
            },
            {
                "title": "Traffic Analysis in Cybersecurity",
                "description": "Analysis of network traffic for security purposes. Covers traffic monitoring, anomaly detection, and network forensics.",
                "keywords": ["traffic analysis", "network security", "monitoring", "forensics", "cybersecurity"]
            },
            {
                "title": "Web Personalization",
                "description": "Techniques for personalizing web experiences. Covers user modeling, recommendation systems, and adaptive interfaces.",
                "keywords": ["personalization", "UX", "web design", "user modeling", "adaptive systems"]
            },
            {
                "title": "Zero Trust Security",
                "description": "Implementation of zero trust security architectures. Covers security models, access control, and network segmentation.",
                "keywords": ["zero trust", "security", "access control", "network security", "cybersecurity"]
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
