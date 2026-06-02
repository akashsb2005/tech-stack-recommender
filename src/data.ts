// Types & Mock Data for the React application

export interface Skill {
  skill_id: number;
  skill_name: string;
  category_id: number;
}

export interface JobRole {
  role_id: number;
  role_name: string;
  description: string;
  associated_skills: string[];
  popularity_score: number;
}

export interface RecommendationResult {
  role_name: string;
  description: string;
  score: number;
  triggered_tags: string[];
  match_percentage: number;
  cold_start_active: boolean;
  popularity_score: number;
}

export interface PipelineResult {
  recs: RecommendationResult[];
  mode: string;
  normalized_inputs: string[];
  user_vector: Record<string, number>;
  role_vectors: Record<string, Record<string, number>>;
  idf_matrix: Record<string, number>;
  vocabulary: string[];
}

export const SKILLS_DICTIONARY: Skill[] = [
  { skill_id: 1, skill_name: "Python", category_id: 1 },
  { skill_id: 2, skill_name: "JavaScript", category_id: 1 },
  { skill_id: 3, skill_name: "TypeScript", category_id: 1 },
  { skill_id: 4, skill_name: "GoLang", category_id: 1 },
  { skill_id: 5, skill_name: "Rust", category_id: 1 },
  { skill_id: 6, skill_name: "Java", category_id: 1 },
  { skill_id: 7, skill_name: "C++", category_id: 1 },
  { skill_id: 8, skill_name: "React", category_id: 2 },
  { skill_id: 9, skill_name: "Vue", category_id: 2 },
  { skill_id: 10, skill_name: "Angular", category_id: 2 },
  { skill_id: 11, skill_name: "NextJS", category_id: 2 },
  { skill_id: 12, skill_name: "NodeJS", category_id: 2 },
  { skill_id: 13, skill_name: "Django", category_id: 2 },
  { skill_id: 14, skill_name: "FastAPI", category_id: 2 },
  { skill_id: 15, skill_name: "Docker", category_id: 3 },
  { skill_id: 16, skill_name: "Kubernetes", category_id: 3 },
  { skill_id: 17, skill_name: "AWS", category_id: 3 },
  { skill_id: 18, skill_name: "Terraform", category_id: 3 },
  { skill_id: 19, skill_name: "CI/CD", category_id: 3 },
  { skill_id: 20, skill_name: "Linux", category_id: 3 },
  { skill_id: 21, skill_name: "PostgreSQL", category_id: 4 },
  { skill_id: 22, skill_name: "MongoDB", category_id: 4 },
  { skill_id: 23, skill_name: "Redis", category_id: 4 },
  { skill_id: 24, skill_name: "SQL", category_id: 4 },
  { skill_id: 25, skill_name: "Machine Learning", category_id: 4 },
  { skill_id: 26, skill_name: "Deep Learning", category_id: 4 },
  { skill_id: 27, skill_name: "TensorFlow", category_id: 4 },
  { skill_id: 28, skill_name: "PyTorch", category_id: 4 },
  { skill_id: 29, skill_name: "Scikit-Learn", category_id: 4 },
  { skill_id: 30, skill_name: "Flutter", category_id: 5 },
  { skill_id: 31, skill_name: "Swift", category_id: 5 },
  { skill_id: 32, skill_name: "Kotlin", category_id: 5 },
  { skill_id: 33, skill_name: "iOS", category_id: 5 },
  { skill_id: 34, skill_name: "Android", category_id: 5 },
  { skill_id: 35, skill_name: "Web Design", category_id: 2 },
  { skill_id: 36, skill_name: "Frontend Development", category_id: 2 },
  { skill_id: 37, skill_name: "API Development", category_id: 2 },
  { skill_id: 38, skill_name: "Database Modeling", category_id: 4 },
  { skill_id: 39, skill_name: "Cloud Computing", category_id: 3 },
  { skill_id: 40, skill_name: "System Architecture", category_id: 3 }
];

export const JOB_ROLES: JobRole[] = [
  {
    role_id: 1,
    role_name: "Data Scientist",
    description: "Builds ML models and analyzes complex statistics to extract business value.",
    associated_skills: ["Python", "SQL", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Scikit-Learn"],
    popularity_score: 95
  },
  {
    role_id: 2,
    role_name: "DevOps Engineer",
    description: "Automates cloud infrastructure, build pipelines, and system reliability.",
    associated_skills: ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux"],
    popularity_score: 88
  },
  {
    role_id: 3,
    role_name: "Backend Developer",
    description: "Designs scalable APIs, server systems, and robust database architectures.",
    associated_skills: ["Python", "Django", "FastAPI", "NodeJS", "PostgreSQL", "MongoDB", "Redis", "SQL", "API Development"],
    popularity_score: 92
  },
  {
    role_id: 4,
    role_name: "Frontend Developer",
    description: "Builds rich, interactive customer-facing web applications.",
    associated_skills: ["JavaScript", "TypeScript", "React", "NextJS", "Web Design", "Frontend Development"],
    popularity_score: 90
  },
  {
    role_id: 5,
    role_name: "Fullstack Engineer",
    description: "Coordinates full lifecycle web development from UI elements to data storage.",
    associated_skills: ["JavaScript", "TypeScript", "React", "NextJS", "NodeJS", "PostgreSQL", "MongoDB", "API Development"],
    popularity_score: 96
  },
  {
    role_id: 6,
    role_name: "Mobile Developer",
    description: "Constructs performance-first native or cross-platform handset applications.",
    associated_skills: ["TypeScript", "Flutter", "Swift", "Kotlin", "iOS", "Android"],
    popularity_score: 81
  },
  {
    role_id: 7,
    role_name: "Cloud Architect",
    description: "Designs and scales secure distributed enterprise architectures on public clouds.",
    associated_skills: ["AWS", "Docker", "Kubernetes", "Linux", "Cloud Computing", "System Architecture"],
    popularity_score: 85
  },
  {
    role_id: 8,
    role_name: "Cybersecurity Analyst",
    description: "Safeguards enterprise assets against digital intrusions and malicious active exploits.",
    associated_skills: ["Linux", "Python", "System Architecture", "GoLang", "Rust"],
    popularity_score: 78
  }
];

export const VOCABULARY_NORMALIZATION_MAP: Record<string, string> = {
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
};

export const ALIASES_DESCRIPTIONS: Record<string, string> = {
  "web design": "Maps to Frontend Development to align display standards",
  "js": "Expands abbreviation to formal library key 'JavaScript'",
  "ts": "Expands abbreviation to compiled framework key 'TypeScript'",
  "ml": "Elevates shortkey domain to high frequency 'Machine Learning'",
  "ai": "Translates general term directly into 'Machine Learning' cluster",
  "k8s": "Decodes standard infrastructure acronym to 'Kubernetes'",
  "cloud computing": "Dumps broad cloud text onto targeted 'AWS' dimension"
};

export function normalizeSkill(skillName: string): string {
  const cleaned = skillName.trim().replace(/\s+/g, ' ').toLowerCase();
  if (VOCABULARY_NORMALIZATION_MAP[cleaned]) {
    return VOCABULARY_NORMALIZATION_MAP[cleaned];
  }
  // Title-case fallback simulation
  return skillName.trim();
}

// 1. Build universal features vocabulary from our standard roles database
export function getVocabulary(): string[] {
  const allSkills = new Set<string>();
  JOB_ROLES.forEach((role) => {
    role.associated_skills.forEach((skill) => {
      allSkills.add(skill.toLowerCase());
    });
  });
  return Array.from(allSkills).sort();
}

// 2. Compute Inverse Document Frequency (IDF) Matrix
// IDF(t) = log10( 1 + N / DF(t) )
export function calculateIDF(vocab: string[]): Record<string, number> {
  const idf: Record<string, number> = {};
  const totalDocs = JOB_ROLES.length;
  
  vocab.forEach((term) => {
    const docCount = JOB_ROLES.filter((role) => 
      role.associated_skills.some((s) => s.toLowerCase() === term)
    ).length;
    
    // Laplace-style smoothed logarithmic inverse document frequency
    idf[term] = Math.log10(1 + (totalDocs / (docCount > 0 ? docCount : 1)));
  });
  
  return idf;
}

// 3. Convert explicit skills list to TF-IDF weights vector
export function vectorize(skills: string[], vocab: string[], idf: Record<string, number>): Record<string, number> {
  const vector: Record<string, number> = {};
  vocab.forEach((term) => {
    vector[term] = 0;
  });
  
  if (skills.length === 0) return vector;
  
  const counts: Record<string, number> = {};
  const totalTokens = skills.length;
  
  skills.forEach((s) => {
    const normalized = s.toLowerCase();
    if (termExistsInVocab(normalized, vocab)) {
      counts[normalized] = (counts[normalized] || 0) + 1;
    }
  });
  
  Object.keys(counts).forEach((term) => {
    const tf = counts[term] / totalTokens;
    const termIdf = idf[term] || 0;
    vector[term] = tf * termIdf;
  });
  
  return vector;
}

function termExistsInVocab(term: string, vocab: string[]): boolean {
  return vocab.includes(term);
}

// 4. Compute Cosine Similarity between two vectors
export function calculateCosineSimilarity(
  vecA: Record<string, number>,
  vecB: Record<string, number>,
  vocab: string[]
): { score: number; dotProduct: number; magnitudeA: number; magnitudeB: number } {
  let dotProduct = 0;
  let sumSqA = 0;
  let sumSqB = 0;
  
  vocab.forEach((term) => {
    const valA = vecA[term] || 0;
    const valB = vecB[term] || 0;
    dotProduct += valA * valB;
    sumSqA += valA * valA;
    sumSqB += valB * valB;
  });
  
  const magnitudeA = Math.sqrt(sumSqA);
  const magnitudeB = Math.sqrt(sumSqB);
  
  if (magnitudeA === 0 || magnitudeB === 0) {
    return { score: 0, dotProduct, magnitudeA, magnitudeB };
  }
  
  return {
    score: dotProduct / (magnitudeA * magnitudeB),
    dotProduct,
    magnitudeA,
    magnitudeB
  };
}

// 5. Entire Recommendation Engine Pipeline
export function runRecommendationPipeline(userSkills: string[], topN: number = 3): PipelineResult {
  const vocab = getVocabulary();
  const idf = calculateIDF(vocab);
  
  // Normalise and map
  const normalizedInputs = userSkills
    .map((s) => normalizeSkill(s))
    .filter((s) => s.trim().length > 0);
    
  // Check if any recognized terms are present
  const validTerms = normalizedInputs.filter((s) => vocab.includes(s.toLowerCase()));
  
  // COLD START BYPASS TRIGGER
  if (validTerms.length === 0) {
    // Sort roles by popularity_score
    const sortedByPop = [...JOB_ROLES].sort((a, b) => b.popularity_score - a.popularity_score);
    const recs: RecommendationResult[] = sortedByPop.slice(0, topN).map((role) => ({
      role_name: role.role_name,
      description: role.description,
      score: 0.0,
      triggered_tags: [],
      match_percentage: 0.0,
      cold_start_active: true,
      popularity_score: role.popularity_score
    }));
    
    return {
      recs,
      mode: "Cold Start Bypass Mode (Global Popularity Fallback Activated - No matching features found)",
      normalized_inputs: normalizedInputs,
      user_vector: {},
      role_vectors: {},
      idf_matrix: idf,
      vocabulary: vocab
    };
  }
  
  // Pipeline Scoring Mode
  const userVector = vectorize(normalizedInputs, vocab, idf);
  const roleVectors: Record<string, Record<string, number>> = {};
  
  const scoredRecs: RecommendationResult[] = JOB_ROLES.map((role) => {
    const roleVector = vectorize(role.associated_skills, vocab, idf);
    roleVectors[role.role_name] = roleVector;
    
    const { score } = calculateCosineSimilarity(userVector, roleVector, vocab);
    
    // Find matching tags
    const triggeredTags = role.associated_skills.filter((skill) =>
      normalizedInputs.some((inp) => inp.toLowerCase() === skill.toLowerCase())
    );
    
    const matchPercentage = Math.min(100.0, Math.round(score * 100 * 10) / 10);
    
    return {
      role_name: role.role_name,
      description: role.description,
      score,
      triggered_tags: triggeredTags,
      match_percentage: matchPercentage,
      cold_start_active: false,
      popularity_score: role.popularity_score
    };
  });
  
  // Sort recommendations by Cosine Score descending, breaking ties with Popularity Score
  const sortedRecs = [...scoredRecs].sort((a, b) => {
    if (Math.abs(a.score - b.score) < 0.0001) {
      return b.popularity_score - a.popularity_score;
    }
    return b.score - a.score;
  });
  
  return {
    recs: sortedRecs.slice(0, topN),
    mode: "Algorithmic Ingress Pipeline (TF-IDF Weighting & Angular Vector Matching Enabled)",
    normalized_inputs: normalizedInputs,
    user_vector: userVector,
    role_vectors: roleVectors,
    idf_matrix: idf,
    vocabulary: vocab
  };
}
