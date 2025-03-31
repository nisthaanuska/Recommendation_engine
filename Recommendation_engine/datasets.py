import pandas as pd
import numpy as np

def generate_academic_datasets():
    """Generate multiple specialized academic datasets"""
    datasets = {
        "general": generate_general_dataset(),
        "engineering": generate_engineering_dataset(),
        "medical": generate_medical_dataset(),
        "business": generate_business_dataset(),
        "arts": generate_arts_dataset()
    }
    return datasets

def generate_general_dataset():
    """Generate the general academic dataset with 100 records"""
    # Create columns for the dataframe
    columns = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    
    # Hardcoded data with realistic academic patterns
    records = [
        # Example provided in the prompt
        {'10th': 9.12, '12th': 8.75, '1st': 9.43, '2nd': 9.33, '3rd': 9.23, '4th': 8.66, '5th': 8.78, '6th': 9.1, '7th': 9.4, '8th': 10.0},
        
        # High performer with consistent scores
        {'10th': 9.50, '12th': 9.45, '1st': 9.60, '2nd': 9.58, '3rd': 9.70, '4th': 9.65, '5th': 9.75, '6th': 9.80, '7th': 9.85, '8th': 9.90},
        
        # Average performer with gradual improvement
        {'10th': 7.80, '12th': 8.10, '1st': 8.20, '2nd': 8.35, '3rd': 8.50, '4th': 8.65, '5th': 8.80, '6th': 8.95, '7th': 9.10, '8th': 9.25},
        
        # Strong start but declining performance
        {'10th': 9.20, '12th': 9.10, '1st': 8.90, '2nd': 8.70, '3rd': 8.50, '4th': 8.30, '5th': 8.10, '6th': 7.90, '7th': 7.70, '8th': 7.50},
        
        # Weak start but significant improvement
        {'10th': 6.80, '12th': 7.20, '1st': 7.60, '2nd': 8.00, '3rd': 8.40, '4th': 8.80, '5th': 9.20, '6th': 9.40, '7th': 9.60, '8th': 9.80},
        
        # Fluctuating performance
        {'10th': 8.50, '12th': 7.90, '1st': 8.60, '2nd': 7.80, '3rd': 8.70, '4th': 7.70, '5th': 8.80, '6th': 7.60, '7th': 8.90, '8th': 7.50},
        
        # Consistent average performer
        {'10th': 8.00, '12th': 8.10, '1st': 8.05, '2nd': 8.15, '3rd': 8.10, '4th': 8.20, '5th': 8.15, '6th': 8.25, '7th': 8.20, '8th': 8.30},
        
        # U-shaped performance (starts high, dips, recovers)
        {'10th': 9.00, '12th': 8.70, '1st': 8.20, '2nd': 7.80, '3rd': 7.50, '4th': 7.80, '5th': 8.20, '6th': 8.60, '7th': 9.00, '8th': 9.40},
        
        # Inverted U-shape (starts low, peaks, declines)
        {'10th': 7.50, '12th': 8.00, '1st': 8.50, '2nd': 9.00, '3rd': 9.50, '4th': 9.00, '5th': 8.50, '6th': 8.00, '7th': 7.50, '8th': 7.00},
        
        # Late bloomer (mediocre start, strong finish)
        {'10th': 7.00, '12th': 7.20, '1st': 7.50, '2nd': 7.80, '3rd': 8.10, '4th': 8.50, '5th': 8.90, '6th': 9.20, '7th': 9.50, '8th': 9.80}
    ]
    
    # Complete the dataset to reach 100 records by adding variations of the above patterns
    for i in range(90):
        base_record = records[i % 10].copy()  # Cycle through the 10 patterns
        variation = i // 10 + 1  # Create 9 variations of each pattern
        
        # Apply small variations to each pattern
        for sem in columns:
            # Adjust scores with small variations (keeping within 6.0-10.0 range)
            base_record[sem] = min(10.0, max(6.0, base_record[sem] + (variation * 0.1 - 0.5)))
        
        records.append(base_record)
    
    # Create dataframe
    return pd.DataFrame(records)

def generate_engineering_dataset():
    """Generate dataset specific to engineering students (100 records)"""
    columns = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    
    # Base patterns for engineering students
    records = [
        # High performer in math/science subjects
        {'10th': 9.80, '12th': 9.70, '1st': 9.50, '2nd': 9.60, '3rd': 9.70, '4th': 9.80, '5th': 9.85, '6th': 9.90, '7th': 9.95, '8th': 10.0},
        
        # Strong in basics, struggle in advanced courses, then recover
        {'10th': 9.50, '12th': 9.40, '1st': 9.30, '2nd': 8.80, '3rd': 8.40, '4th': 8.20, '5th': 8.60, '6th': 9.00, '7th': 9.30, '8th': 9.50},
        
        # Steady performer with slight dip in middle years (complex subjects)
        {'10th': 8.70, '12th': 8.90, '1st': 8.85, '2nd': 8.75, '3rd': 8.50, '4th': 8.40, '5th': 8.60, '6th': 8.80, '7th': 9.00, '8th': 9.10},
        
        # Average start with significant improvement (adaptation to engineering)
        {'10th': 7.80, '12th': 8.00, '1st': 7.60, '2nd': 7.80, '3rd': 8.20, '4th': 8.60, '5th': 8.90, '6th': 9.10, '7th': 9.30, '8th': 9.50},
        
        # Struggling with transition from school to engineering college
        {'10th': 9.00, '12th': 8.90, '1st': 7.80, '2nd': 7.50, '3rd': 7.80, '4th': 8.20, '5th': 8.50, '6th': 8.70, '7th': 8.90, '8th': 9.10},
        
        # Strong in theory, struggles in practical/project semesters
        {'10th': 9.20, '12th': 9.30, '1st': 9.40, '2nd': 8.90, '3rd': 9.30, '4th': 8.80, '5th': 9.20, '6th': 8.80, '7th': 9.30, '8th': 9.50},
        
        # Practical-oriented student (better in project semesters)
        {'10th': 8.20, '12th': 8.30, '1st': 8.10, '2nd': 8.60, '3rd': 8.20, '4th': 8.70, '5th': 8.30, '6th': 8.90, '7th': 8.60, '8th': 9.20},
        
        # Consistent performer throughout
        {'10th': 8.50, '12th': 8.60, '1st': 8.55, '2nd': 8.65, '3rd': 8.60, '4th': 8.70, '5th': 8.65, '6th': 8.75, '7th': 8.70, '8th': 8.80},
        
        # Specialization pattern (improved after choosing specialization in 5th sem)
        {'10th': 8.20, '12th': 8.30, '1st': 8.10, '2nd': 8.00, '3rd': 8.20, '4th': 8.10, '5th': 8.50, '6th': 8.80, '7th': 9.10, '8th': 9.40},
        
        # Research-oriented pattern (improved in final year research project)
        {'10th': 8.40, '12th': 8.50, '1st': 8.30, '2nd': 8.40, '3rd': 8.50, '4th': 8.60, '5th': 8.70, '6th': 8.80, '7th': 9.20, '8th': 9.60}
    ]
    
    # Generate variations to complete 100 records
    for i in range(90):
        base_record = records[i % 10].copy()
        variation = i // 10 + 1
        
        for sem in columns:
            base_record[sem] = min(10.0, max(6.0, base_record[sem] + (variation * 0.08 - 0.4)))
        
        records.append(base_record)
    
    return pd.DataFrame(records)

def generate_medical_dataset():
    """Generate dataset specific to medical students (100 records)"""
    columns = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    
    # Base patterns for medical students
    records = [
        # High achiever throughout medical school
        {'10th': 9.80, '12th': 9.85, '1st': 9.70, '2nd': 9.75, '3rd': 9.80, '4th': 9.85, '5th': 9.90, '6th': 9.92, '7th': 9.95, '8th': 9.98},
        
        # Strong in theory, slightly lower in clinical years
        {'10th': 9.60, '12th': 9.70, '1st': 9.50, '2nd': 9.60, '3rd': 9.65, '4th': 9.40, '5th': 9.30, '6th': 9.35, '7th': 9.40, '8th': 9.45},
        
        # Struggles with basic sciences, excels in clinical work
        {'10th': 8.80, '12th': 8.90, '1st': 8.40, '2nd': 8.30, '3rd': 8.50, '4th': 8.90, '5th': 9.20, '6th': 9.40, '7th': 9.50, '8th': 9.60},
        
        # Consistent strong performer
        {'10th': 9.20, '12th': 9.30, '1st': 9.25, '2nd': 9.35, '3rd': 9.30, '4th': 9.40, '5th': 9.35, '6th': 9.45, '7th': 9.40, '8th': 9.50},
        
        # Initial adjustment period then strong performance
        {'10th': 9.40, '12th': 9.50, '1st': 8.80, '2nd': 8.90, '3rd': 9.20, '4th': 9.40, '5th': 9.60, '6th': 9.65, '7th': 9.70, '8th': 9.75},
        
        # Fluctuating performance based on subject interest
        {'10th': 9.10, '12th': 9.20, '1st': 9.50, '2nd': 8.90, '3rd': 9.40, '4th': 8.80, '5th': 9.30, '6th': 8.90, '7th': 9.50, '8th': 9.20},
        
        # Steady improvement throughout medical school
        {'10th': 8.50, '12th': 8.70, '1st': 8.80, '2nd': 9.00, '3rd': 9.10, '4th': 9.20, '5th': 9.30, '6th': 9.40, '7th': 9.50, '8th': 9.60},
        
        # High achiever with minor dip during intense periods
        {'10th': 9.70, '12th': 9.80, '1st': 9.75, '2nd': 9.50, '3rd': 9.60, '4th': 9.30, '5th': 9.40, '6th': 9.70, '7th': 9.80, '8th': 9.90},
        
        # Research-focused student (specialized early)
        {'10th': 9.30, '12th': 9.40, '1st': 9.35, '2nd': 9.45, '3rd': 9.55, '4th': 9.65, '5th': 9.75, '6th': 9.80, '7th': 9.85, '8th': 9.90},
        
        # Balanced student with consistent good performance
        {'10th': 9.00, '12th': 9.10, '1st': 9.05, '2nd': 9.15, '3rd': 9.10, '4th': 9.20, '5th': 9.15, '6th': 9.25, '7th': 9.20, '8th': 9.30}
    ]
    
    # Generate variations to complete 100 records
    for i in range(90):
        base_record = records[i % 10].copy()
        variation = i // 10 + 1
        
        for sem in columns:
            base_record[sem] = min(10.0, max(6.0, base_record[sem] + (variation * 0.07 - 0.35)))
        
        records.append(base_record)
    
    return pd.DataFrame(records)

def generate_business_dataset():
    """Generate dataset specific to business students (100 records)"""
    columns = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    
    # Base patterns for business students
    records = [
        # All-rounder with consistent performance
        {'10th': 9.20, '12th': 9.30, '1st': 9.25, '2nd': 9.35, '3rd': 9.30, '4th': 9.40, '5th': 9.35, '6th': 9.45, '7th': 9.40, '8th': 9.50},
        
        # Entrepreneurial type (improves in practical/project semesters)
        {'10th': 8.30, '12th': 8.40, '1st': 8.20, '2nd': 8.60, '3rd': 8.30, '4th': 8.70, '5th': 8.50, '6th': 8.90, '7th': 8.70, '8th': 9.20},
        
        # Finance specialist (improves after specialization)
        {'10th': 8.70, '12th': 8.80, '1st': 8.75, '2nd': 8.85, '3rd': 8.90, '4th': 9.00, '5th': 9.20, '6th': 9.30, '7th': 9.45, '8th': 9.60},
        
        # Marketing specialist (project-oriented improvement)
        {'10th': 8.50, '12th': 8.60, '1st': 8.40, '2nd': 8.70, '3rd': 8.50, '4th': 8.80, '5th': 8.90, '6th': 9.10, '7th': 9.20, '8th': 9.40},
        
        # Theory vs practical gap (better in application than theory)
        {'10th': 8.20, '12th': 8.30, '1st': 7.90, '2nd': 8.40, '3rd': 8.00, '4th': 8.50, '5th': 8.20, '6th': 8.70, '7th': 8.40, '8th': 8.90},
        
        # Internship impact pattern (improvement after internship period)
        {'10th': 8.40, '12th': 8.50, '1st': 8.45, '2nd': 8.55, '3rd': 8.50, '4th': 8.60, '5th': 9.00, '6th': 9.10, '7th': 9.20, '8th': 9.30},
        
        # Steady improvement throughout
        {'10th': 7.80, '12th': 8.00, '1st': 8.10, '2nd': 8.20, '3rd': 8.40, '4th': 8.60, '5th': 8.80, '6th': 9.00, '7th': 9.20, '8th': 9.40},
        
        # Networking-oriented student (improves with collaboration)
        {'10th': 8.00, '12th': 8.10, '1st': 8.00, '2nd': 8.30, '3rd': 8.10, '4th': 8.40, '5th': 8.60, '6th': 8.80, '7th': 9.00, '8th': 9.30},
        
        # Analytics specialist (strong in quantitative courses)
        {'10th': 9.20, '12th': 9.30, '1st': 9.10, '2nd': 9.40, '3rd': 9.20, '4th': 9.50, '5th': 9.30, '6th': 9.60, '7th': 9.40, '8th': 9.70},
        
        # Management generalist (consistent across subjects)
        {'10th': 8.60, '12th': 8.70, '1st': 8.65, '2nd': 8.75, '3rd': 8.70, '4th': 8.80, '5th': 8.75, '6th': 8.85, '7th': 8.80, '8th': 8.90}
    ]
    
    # Generate variations to complete 100 records
    for i in range(90):
        base_record = records[i % 10].copy()
        variation = i // 10 + 1
        
        for sem in columns:
            base_record[sem] = min(10.0, max(6.0, base_record[sem] + (variation * 0.09 - 0.45)))
        
        records.append(base_record)
    
    return pd.DataFrame(records)

def generate_arts_dataset():
    """Generate dataset specific to arts and humanities students (100 records)"""
    columns = ['10th', '12th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
    
    # Base patterns for arts students
    records = [
        # Creative specialist with steady improvement
        {'10th': 8.40, '12th': 8.50, '1st': 8.60, '2nd': 8.80, '3rd': 9.00, '4th': 9.20, '5th': 9.30, '6th': 9.40, '7th': 9.50, '8th': 9.60},
        
        # Literature/language specialist
        {'10th': 9.20, '12th': 9.30, '1st': 9.40, '2nd': 9.50, '3rd': 9.60, '4th': 9.70, '5th': 9.75, '6th': 9.80, '7th': 9.85, '8th': 9.90},
        
        # Multi-disciplinary performance (varies by subject)
        {'10th': 8.70, '12th': 8.80, '1st': 9.20, '2nd': 8.60, '3rd': 9.10, '4th': 8.70, '5th': 9.30, '6th': 8.80, '7th': 9.40, '8th': 9.00},
        
        # Project-oriented performer (better in practical/creative semesters)
        {'10th': 8.30, '12th': 8.40, '1st': 8.20, '2nd': 8.70, '3rd': 8.30, '4th': 8.80, '5th': 8.40, '6th': 8.90, '7th': 8.50, '8th': 9.00},
        
        # Theory-oriented student (better in analytical courses)
        {'10th': 8.80, '12th': 8.90, '1st': 9.20, '2nd': 8.70, '3rd': 9.10, '4th': 8.60, '5th': 9.00, '6th': 8.50, '7th': 8.90, '8th': 8.40},
        
        # Specialization impact (improvement after finding focus)
        {'10th': 7.80, '12th': 8.00, '1st': 8.10, '2nd': 8.20, '3rd': 8.30, '4th': 8.80, '5th': 9.20, '6th': 9.40, '7th': 9.50, '8th': 9.60},
        
        # Consistent performer across subjects
        {'10th': 8.50, '12th': 8.60, '1st': 8.55, '2nd': 8.65, '3rd': 8.60, '4th': 8.70, '5th': 8.65, '6th': 8.75, '7th': 8.70, '8th': 8.80},
        
        # Research-oriented humanities student
        {'10th': 8.90, '12th': 9.00, '1st': 8.80, '2nd': 8.90, '3rd': 9.10, '4th': 9.20, '5th': 9.30, '6th': 9.40, '7th': 9.60, '8th': 9.80},
        
        # Performance arts specialist (better in studio/performance courses)
        {'10th': 8.20, '12th': 8.30, '1st': 8.10, '2nd': 8.50, '3rd': 8.20, '4th': 8.60, '5th': 8.30, '6th': 8.70, '7th': 8.80, '8th': 9.20},
        
        # Late bloomer in specialized area
        {'10th': 7.60, '12th': 7.80, '1st': 7.90, '2nd': 8.00, '3rd': 8.10, '4th': 8.30, '5th': 8.60, '6th': 8.90, '7th': 9.20, '8th': 9.50}
    ]
    
    # Generate variations to complete 100 records
    for i in range(90):
        base_record = records[i % 10].copy()
        variation = i // 10 + 1
        
        for sem in columns:
            base_record[sem] = min(10.0, max(6.0, base_record[sem] + (variation * 0.085 - 0.425)))
        
        records.append(base_record)
    
    return pd.DataFrame(records) 