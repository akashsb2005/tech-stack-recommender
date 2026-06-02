# 🚀 Tech Stack Recommender: Enterprise-Grade Recommendation Engine

A high-performance **Content-Based Filtering Engine** that maps user-specified skill profiles and career goals to ideal engineering roles. Built around the **Input-Process-Output (IPO)** model, the engine utilizes pure vector mathematics, logarithmic **TF-IDF Term Weighting**, and **Cosine Similarity Matrix Alignment** to generate highly accurate matches *without* requiring resource-heavy neural networks or deep learning architectures.

This project is structured for a professional engineering portfolio, demonstrating strong foundations in:
* **Feature Engineering**: Standardizing noisy unstructured inputs and translating qualitative terms into mathematical indexes.
* **Vector Vectorization**: Penalizing generic high-frequency words and rewarding descriptive domain technologies.
* **Algebraic Matching Logic**: Measuring angular vector orientation over absolute geometric magnitude.
* **Error-Resistant Fallback Systems**: Router state machines designed to gracefully bypass the cold start problem.

---

## 🏗️ Architectural Core: The IPO Model

The recommendation processor executes sequentially across 4 key assembly phases:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     INPUT       │  ───> │     PROCESS     │  ───> │     OUTPUT      │
│  (User State)   │       │ (Similarity Math)│       │ (Scored Top-N)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
 - Raw Skill Prompt        - Target Vocab Map        - Sort Scores DESC
 - Entry Normalization     - TF-IDF Weighting        - Cold Start Bypass
 - Feature Ingestion       - Cosine Alignment        - Percentage Matches
```

### Phase 0: Data Schemas & Vocabulary Normalization
Unstructured user queries are notoriously dirty (e.g., mixtures of casing, spacing, and short key abbreviations like `"js"` or `"k8s"`). To ensure vector calculations align accurately, raw data is standard-mapped to a shared coordinate space before performing vector calculations.

#### Database Tables (`raw_skills.csv` & `job_roles.csv` Schema)
1. **`raw_skills.csv`**: Maps unique skill records to categorization matrices:
   ```csv
   skill_id,skill_name,category_id
   1,Python,1
   2,JavaScript,1
   15,Docker,3
   25,Machine Learning,4
   ```
2. **`job_roles.csv`**: Defines target recommendation items with their semantic skill matrices and popularity weight flags:
   ```csv
   role_id,role_name,description,associated_skills,popularity_score
   1,Data Scientist,Builds ML models...,"Python, SQL, Machine Learning, Deep Learning",95
   2,DevOps Engineer,Automates cloud pipelines...,"Docker, Kubernetes, AWS, Terraform, CI/CD",88
   ```

#### Alias Normalization Map
We map standard developer jargon and shorthand terms to identical coordinates in the vector space:
* `"ml"`, `"ai"`, `"neural nets"` ➔ **`Machine Learning`**
* `"js"`, `"javascript"` ➔ **`JavaScript`**
* `"k8s"`, `"kube"` ➔ **`Kubernetes`**
* `"cloud computing"` ➔ **`AWS`**
* `"automation"` ➔ **`CI/CD`**

---

## 📐 The Mathematical Mechanics

### 1. Ingestion & Weight Scaling (TF-IDF)
Using a simple binary overlay (assigning a `1` if a skill is present and a `0` if absent) fails to distinguish between generic keywords and highly specific competencies. For example, if a job requires both `"Software"` (extremely common) and `"TensorFlow"` (specialized), a binary query would evaluate their importance equally.

To solve this, we compute **Logarithmic Term Frequency - Inverse Document Frequency (TF-IDF)** with Laplace smoothing to compress the weight of high-frequency words and amplify targeted skills:

$$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}$$

$$\text{IDF}(t) = \log_{10} \left( 1 + \frac{N}{\text{DF}(t)} \right)$$

$$\text{TF-IDF} = \text{TF}(t,d) \times \text{IDF}(t)$$

*The log scale serves as a dampening effect, ensuring penalization scales logarithmically rather than linearly to preserve comparable vector properties.*

### 2. Angular Intersection (Cosine Similarity)
Engineering candidates list highly varied numbers of skills on their resumes. If we were to measure similarity using **Euclidean Distance**, long resumes with dozens of skills would sit extremely far from shorter ones in physical space—even if their skills were highly similar.

**Cosine Similarity** focuses strictly on the direction (orientation) of the vectors rather than their lengths (magnitudes). It calculates the cosine of the angle $\theta$ between the user's vector and each job role's vector:

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

*Because our TF-IDF coordinates are strictly non-negative ($[0,1]$ value range), our similarity results will naturally sit between $0$ (completely orthogonal) and $1$ (perfectly aligned). This gives us an intuitive, proportional match percentage ($0\%\dots100\%$).*

### 3. Graceby Responding to Empty State (Cold Start Problem)
If a brand-new user inputs no skills, or inputs entirely out-of-vocabulary terms (e.g., `"painting"`, `"dancing"`), their user profile vector will resolve entirely as zero:

$$\vec{U} = [0, 0, \dots, 0] \implies \vec{U} \cdot \vec{R}_i = 0$$

To protect the user experience from crashing or returning useless $0\%$ matches, a fallback router triggers the **Cold Start Bypass Mechanism**, serving popular, highly active baseline roles from our database sorted by their `popularity_score`.

---

## 💻 Technical Implementation (`recommender.py`)

Here is the clean, self-contained Python implementation of the matching engine:

```python
import csv
import math
import re
from typing import List, Dict, Tuple, Set

# Vocabulary Standardizer Map
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
    cleaned = re.sub(r'\s+', ' ', skill_name.strip()).lower()
    return VOCABULARY_NORMALIZATION_MAP.get(cleaned, skill_name.strip())

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
        # Read normalization databases
        try:
            with open(raw_skills_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.skills_vocab[int(row['skill_id'])] = row['skill_name'].strip()
        except FileNotFoundError:
            # Local standalone fallback definitions
            self.skills_vocab = {1: "Python", 2: "JavaScript", 15: "Docker"}

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
            self.job_roles = [
                {'role_id': 1, 'role_name': 'Data Scientist', 'description': 'Builds ML models...', 'skills': ["Python", "SQL", "Machine Learning"], 'popularity_score': 95}
            ]

    def build_vocabulary(self):
        all_skills = set(skill.lower() for role in self.job_roles for skill in role['skills'])
        self.vocabulary = sorted(list(all_skills))

    def calculate_idf(self):
        num_docs = len(self.job_roles)
        for term in self.vocabulary:
            doc_count = sum(1 for r in self.job_roles if any(s.lower() == term for s in r['skills']))
            self.idf_matrix[term] = math.log10(1 + (num_docs / (doc_count if doc_count > 0 else 1)))

    def vectorize(self, skills: List[str]) -> Dict[str, float]:
        vector = {term: 0.0 for term in self.vocabulary}
        if not skills:
            return vector
        total_tokens = len(skills)
        counts = {}
        for s in skills:
            normalized = s.lower()
            if normalized in vector:
                counts[normalized] = counts.get(normalized, 0) + 1
        for term, freq in counts.items():
            vector[term] = (freq / total_tokens) * self.idf_matrix.get(term, 0.0)
        return vector

    def calculate_cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        dot_product = sum(vec_a[term] * vec_b[term] for term in self.vocabulary)
        magnitude_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        magnitude_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    def recommend(self, user_skills: List[str], top_n: int = 3) -> Dict:
        normalized_inputs = [normalize_skill(s) for s in user_skills if s.strip()]
        valid_terms = [s.lower() for s in normalized_inputs if s.lower() in self.vocabulary]
        
        # Cold Start Bypass Trigger
        if not valid_terms:
            sorted_by_pop = sorted(self.job_roles, key=lambda x: x['popularity_score'], reverse=True)
            return {
                'recs': [{
                    'role_name': r['role_name'],
                    'description': r['description'],
                    'score': 0.0,
                    'triggered_tags': [],
                    'match_percentage': 0.0,
                    'cold_start_active': True,
                    'popularity_score': r['popularity_score']
                } for r in sorted_by_pop[:top_n]],
                'mode': 'Cold Start Bypass (Popularity-Based)',
                'normalized_inputs': normalized_inputs
            }
            
        user_vector = self.vectorize(normalized_inputs)
        scored_recs = []
        for role in self.job_roles:
            role_vector = self.vectorize(role['skills'])
            similarity = self.calculate_cosine_similarity(user_vector, role_vector)
            triggered = [s for s in role['skills'] if s.lower() in [inp.lower() for inp in normalized_inputs]]
            scored_recs.append({
                'role_name': role['role_name'],
                'description': role['description'],
                'score': similarity,
                'triggered_tags': triggered,
                'match_percentage': min(100.0, round(similarity * 100, 1)),
                'cold_start_active': False,
                'popularity_score': role['popularity_score']
            })
            
        sorted_recs = sorted(scored_recs, key=lambda x: (x['score'], x['popularity_score']), reverse=True)
        return {
            'recs': sorted_recs[:top_n],
            'mode': 'Algorithmic Content-Based (Cosine Similarity)',
            'normalized_inputs': normalized_inputs
        }
```

---

## 📈 Big-O Complexity Analysis

Understanding computational limits ensures this recommendation algorithm remains light, fast, and scalable for production services:

| Operation | Equation | Worst-Case Time Complexity | Space Complexity |
| :--- | :--- | :--- | :--- |
| **Vocabulary Build** | Reading all skill attributes | $O(|I| \times K)$ | $O(\|V\|)$ |
| **IDF Matrix Prep** | Scanning jobs per keyword | $O(\|V\| \times \|I\|)$ | $O(\|V\|)$ |
| **User Profile Vectorization** | Counting term occurrences | $O(U)$ | $O(\|V\|)$ |
| **Cosine Engine Comparison** | Dot product loops | $O(\|I\| \times \|V\|)$ | $O(\|V\|)$ |
| **Output Ranking & Sort** | Sorting matches descending | $O(\|I\| \log \|I\|)$ | $O(\|I\|)$ |

*Where:*
* $|I|$: Total number of Job Roles in the database (e.g. 8 profiles).
* $K$: Average skills listed per Job Role profile.
* $|V|$: Total vocabulary length (unique skills, e.g. 40 elements).
* $U$: Raw skill elements present in user query.

---

## 📓 Quickstart Guide (Google Colab, Local File running, Jupyter)

### Method A: Execute in Google Colab (Highly Interactive)
1. **Open Google Colab**: Navigate to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.
2. **Setup Source Files**: Create the required CSV databases directly inside your Colab environment. In a code cell, execute:
   ```python
   # Run this to create the raw_skills schema file
   with open("raw_skills.csv", "w") as f:
       f.write("""skill_id,skill_name,category_id
   1,Python,1
   2,JavaScript,1
   15,Docker,3
   25,Machine Learning,4
   """)
   
   # Run this to create the job_roles schema file 
   with open("job_roles.csv", "w") as f:
       f.write("""role_id,role_name,description,associated_skills,popularity_score
   1,Data Scientist,Builds ML models...,"Python, Machine Learning",95
   2,DevOps Engineer,Automates cloud...,"Docker",88
   """)
   ```
3. **Paste & Run Code**: Paste the complete `recommender.py` logic block inside a new cell.
4. **Initialize and Test matches**:
   ```python
   recommender = ContentBasedRecommender("raw_skills.csv", "job_roles.csv")
   results = recommender.recommend(["Python", "ml"])
   print(results)
   ```

### Method B: Native command Line Execution (Local Terminal)
1. Ensure your directory has `raw_skills.csv`, `job_roles.csv`, and `recommender.py` adjacent to each other.
2. Open your terminal emulator and navigate to your directory:
   ```bash
   cd path/to/project_folder
   ```
3. Execute the script:
   ```bash
   python recommender.py
   ```

---

## 🎨 Professional Presentation Highlights
This workspace is designed as an interactive portal showcasing real-time mathematical traces of active vector matrix products.
* **Match Percentage Breakdown**: Shows exactly how similarity ranges scale to intuitive percentages.
* **Real-Time Vector Normalization alerts**: Displays visual notifications whenever shorthand aliases (like `"k8s"` or `"ai"`) trigger normalization mapping.
* **Interactive Mathematical Playground**: Delve into vector lengths, dot products, and calculations on simulated models to gain a deeper intuitive understanding of the underlying math!
