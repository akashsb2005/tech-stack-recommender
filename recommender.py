"""
Tech Stack Recommender: Content-Based Filtering Recommendation Engine
This script demonstrates the IPO (Input-Process-Output) model for mapping 
user-specified skill profiles to primary career paths/job roles.

Mathematical Mechanics:
1. TF-IDF weighting is executed to highlight specialized terms and discount common ones.
2. Similarity scores are computed via Cosine Similarity (directional alignment of high-dimensional vectors).
3. Fallback bypass logic triggers a trend-based Cold Start pathway if user profile remains empty or unrecognized.
"""

import csv
import math
import re
from typing import List, Dict, Tuple, Set

# Phase 0: Vocabulary Normalization Map (Alias Mapping)
VOCABULARY_NORMALIZATION_MAP = {
    "web design": "Frontend Development",
    "web development": "Frontend Development",
    "ui design": "Web Design",
    "ux design": "Web Design",
    "coding": "Software Development",
    "ml": "Machine Learning",
    "ai": "Machine Learning",
    "neural networks": "Deep Learning",
    "cloud computing": "AWS",
    "cloud": "Cloud Computing",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "js": "JavaScript",
    "ts": "TypeScript",
    "automation": "CI/CD",
    "github actions": "CI/CD",
    "databases": "SQL",
    "systems architect": "System Architecture"
}

def normalize_skill(skill_name: str) -> str:
    """
    Standardize raw skill string inputs to match our database vocabulary.
    Performs lowercasing, whitespace stripping, and alias dictionary checking.
    """
    cleaned = re.sub(r'\s+', ' ', skill_name.strip()).lower()
    if cleaned in VOCABULARY_NORMALIZATION_MAP:
        return VOCABULARY_NORMALIZATION_MAP[cleaned]
    # Return capitalized title as a reasonable default fallback if not in normalizer
    return skill_name.strip()

class ContentBasedRecommender:
    def __init__(self, raw_skills_path: str, job_roles_path: str):
        self.skills_vocab: Dict[int, str] = {}
        self.job_roles: List[Dict] = []
        self.vocabulary: List[str] = []
        self.idf_matrix: Dict[str, float] = {}
        
        self.load_data(raw_skills_path, job_roles_path)
        self.build_vocabulary()
        self.calculate_idf()
        
    def load_data(self, raw_skills_path: str, job_roles_path: str):
        """Loads skills dictionary and item profiles from CSVs."""
        # Load Raw Skills Vocab
        try:
            with open(raw_skills_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s_id = int(row['skill_id'])
                    s_name = row['skill_name'].strip()
                    self.skills_vocab[s_id] = s_name
        except FileNotFoundError:
            # Fallback mock dataset for standalone runs without files
            self.skills_vocab = {
                1: "Python", 2: "JavaScript", 3: "TypeScript", 4: "GoLang",
                15: "Docker", 16: "Kubernetes", 17: "AWS", 19: "CI/CD",
                21: "PostgreSQL", 24: "SQL", 25: "Machine Learning", 26: "Deep Learning"
            }
            
        # Load Job Roles
        try:
            with open(job_roles_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skills_list = [s.strip() for s in row['associated_skills'].split(',')]
                    self.job_roles.append({
                        'role_id': int(row['role_id']),
                        'role_name': row['role_name'],
                        'description': row['description'],
                        'skills': skills_list,
                        'popularity_score': float(row['popularity_score'])
                    })
        except FileNotFoundError:
            # Fallback mock job roles database
            self.job_roles = [
                {
                    'role_id': 1, 'role_name': 'Data Scientist',
                    'description': 'Builds ML models and analyzes complex statistics.',
                    'skills': ["Python", "SQL", "Machine Learning", "Deep Learning"],
                    'popularity_score': 95.0
                },
                {
                    'role_id': 2, 'role_name': 'DevOps Engineer',
                    'description': 'Automates cloud infrastructure and CI/CD pipelines.',
                    'skills': ["Docker", "Kubernetes", "AWS", "CI/CD"],
                    'popularity_score': 88.0
                },
                {
                    'role_id': 3, 'role_name': 'Fullstack Engineer',
                    'description': 'Coordinates full lifecycle web development.',
                    'skills': ["JavaScript", "TypeScript", "React", "NodeJS", "PostgreSQL"],
                    'popularity_score': 96.0
                }
            ]

    def build_vocabulary(self):
        """Construct the universal feature token collection from all Job Role records."""
        all_skills: Set[str] = set()
        for role in self.job_roles:
            for skill in role['skills']:
                all_skills.add(skill.lower())
        self.vocabulary = sorted(list(all_skills))

    def calculate_idf(self):
        """
        Compute Inverse Document Frequency (IDF) for all tokens.
        IDF = log( 1 + (Total Documents) / (Documents containing term t) )
        We utilize Laplace-like smoothing (the '+ 1' inside the log) to secure mathematical stability.
        """
        num_docs = len(self.job_roles)
        for term in self.vocabulary:
            doc_count = sum(1 for role in self.job_roles if any(skill.lower() == term for skill in role['skills']))
            # Applying logarithmic scaling to curb explosive linear penalty effects
            self.idf_matrix[term] = math.log10(1 + (num_docs / (doc_count if doc_count > 0 else 1)))

    def vectorize(self, skills: List[str]) -> Dict[str, float]:
        """
        Transform a raw skill list into a weighted TF-IDF sparse vector.
        TF = (Count of skill in prompt) / (Total terms in prompt)
        TF-IDF = TF * IDF
        """
        vector: Dict[str, float] = {term: 0.0 for term in self.vocabulary}
        if not skills:
            return vector
            
        # Count frequency of normalized user tokens matching our vector dimensions
        counts: Dict[str, int] = {}
        total_tokens = len(skills)
        for s in skills:
            normalized = s.lower()
            if normalized in vector:
                counts[normalized] = counts.get(normalized, 0) + 1
                
        # Calculate TF-IDF weight for each dimensional coordinate
        for term, freq in counts.items():
            tf = freq / total_tokens
            idf = self.idf_matrix.get(term, 0.0)
            vector[term] = tf * idf
            
        return vector

    def calculate_cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """
        Compute Cosine Similarity from scratch.
        Equation: cos(θ) = (A ⋅ B) / (||A|| * ||B||)
        Provides vector lengths scale invariance, focusing strictly on directional orientation.
        """
        # Dot product
        dot_product = sum(vec_a[term] * vec_b[term] for term in self.vocabulary)
        
        # Vector Magnitudes (Euclidean L2-norm)
        magnitude_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))
        magnitude_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))
        
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0
            
        return dot_product / (magnitude_a * magnitude_b)

    def recommend(self, user_skills: List[str], top_n: int = 3) -> Dict:
        """
        Generates personalized career domain recommendations based on skill alignments.
        Implements a Cold Start bypass to popular items if zero features map.
        """
        # Normalize and map incoming input criteria
        normalized_inputs = [normalize_skill(s) for s in user_skills if s.strip()]
        
        # Check if we triggered the User Cold Start threshold (no known tags)
        valid_terms = [s.lower() for s in normalized_inputs if s.lower() in self.vocabulary]
        
        # If no recognized traits found, deploy the Cold Start Bypass (Global Popularity Fallback)
        if not valid_terms:
            sorted_by_pop = sorted(self.job_roles, key=lambda x: x['popularity_score'], reverse=True)
            cold_start_recs = []
            for role in sorted_by_pop[:top_n]:
                cold_start_recs.append({
                    'role_name': role['role_name'],
                    'description': role['description'],
                    'score': 0.0,
                    'triggered_tags': [],
                    'match_percentage': 0.0,
                    'cold_start_active': True,
                    'popularity_score': role['popularity_score']
                })
            return {
                'recs': cold_start_recs,
                'mode': 'Cold Start Bypass (Popularity-Based)',
                'normalized_inputs': normalized_inputs
            }
            
        # Standard Ingestion & Vectorization (Normal Pipeline)
        user_vector = self.vectorize(normalized_inputs)
        
        scored_recs = []
        for role in self.job_roles:
            # Build an explicit vector representation for each candidate Job Role
            # For a job role document, terms that are listed in the role description are treated as present (TF=1/Length)
            role_vector = self.vectorize(role['skills'])
            
            # Compute angular intersection
            similarity = self.calculate_cosine_similarity(user_vector, role_vector)
            
            # Identify exact tags triggering this score match
            triggered_tags = [
                skill for skill in role['skills'] 
                if skill.lower() in [inp.lower() for inp in normalized_inputs]
            ]
            
            # Convert decimal score to visual match percentage with smoothing factor
            match_percentage = min(100.0, round(similarity * 100, 1))
            
            scored_recs.append({
                'role_name': role['role_name'],
                'description': role['description'],
                'score': similarity,
                'triggered_tags': triggered_tags,
                'match_percentage': match_percentage,
                'cold_start_active': False,
                'popularity_score': role['popularity_score']
            })
            
        # Phase 3: Sort & Filter Top-N items (descending by score, ties broken via global popularity)
        sorted_recs = sorted(
            scored_recs, 
            key=lambda x: (x['score'], x['popularity_score']), 
            reverse=True
        )
        
        return {
            'recs': sorted_recs[:top_n],
            'mode': 'Algorithmic Content-Based (Angular Alignment)',
            'normalized_inputs': normalized_inputs
        }

if __name__ == "__main__":
    print("=" * 60)
    print("   AI Engineering Portfolio: Tech Stack Recommender Project")
    print("=" * 60)
    
    # Initialize Engine with absolute path declarations (supports mock fallback natively)
    recommender = ContentBasedRecommender("raw_skills.csv", "job_roles.csv")
    
    # Showcase Normal Pipeline (Demo 1)
    user_inputs = ["Python", "Machine Learning", "deep learning", "databases"]
    print(f"\n[Client Input Ingested]: {user_inputs}")
    
    results = recommender.recommend(user_inputs, top_n=3)
    print(f"[Operation Pipeline Mode]: {results['mode']}")
    print(f"[Normalized Vocab Results]: {results['normalized_inputs']}\n")
    
    print("-" * 50)
    for idx, rec in enumerate(results['recs'], 1):
        print(f"Rank {idx}: {rec['role_name']} ({rec['match_percentage']}% Match)")
        print(f"  Description: {rec['description']}")
        print(f"  Triggering Intersections: {rec['triggered_tags']}")
        print(f"  Implicit Vector Cosine Score: {rec['score']:.4f}")
        print("-" * 50)
        
    # Showcase Cold Start Problem Resolver (Demo 2)
    print("\n" + "=" * 60)
    print("   Cold Start Bypass Demo (Simulating brand-new user with unknown queries)")
    print("=" * 60)
    unknown_inputs = ["Cooking", "Sleeping", "Painting"]
    print(f"\n[Client Input Ingested]: {unknown_inputs}")
    
    cs_results = recommender.recommend(unknown_inputs, top_n=3)
    print(f"[Operation Pipeline Mode]: {cs_results['mode']}\n")
    
    for idx, rec in enumerate(cs_results['recs'], 1):
        print(f"Fallback Rank {idx}: {rec['role_name']}")
        print(f"  Description: {rec['description']}")
        print(f"  Global Registry Popularity Score: {rec['popularity_score']}/100")
        print("-" * 50)
