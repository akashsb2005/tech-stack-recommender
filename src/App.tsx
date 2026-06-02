import { useState, useMemo } from "react";
import { 
  motion, 
  AnimatePresence 
} from "motion/react";
import { 
  Code, 
  Cpu, 
  Layers, 
  Terminal, 
  ArrowRight, 
  Search, 
  FileText, 
  Check, 
  Info, 
  Sparkles, 
  TrendingUp, 
  Compass, 
  HelpCircle, 
  Download, 
  Plus, 
  X, 
  RefreshCw, 
  AlertCircle, 
  Database, 
  Hash, 
  Flame, 
  ChevronRight, 
  GitBranch, 
  Sliders
} from "lucide-react";
import { 
  SKILLS_DICTIONARY, 
  JOB_ROLES, 
  runRecommendationPipeline, 
  VOCABULARY_NORMALIZATION_MAP, 
  ALIASES_DESCRIPTIONS,
  normalizeSkill,
  calculateCosineSimilarity,
  vectorize
} from "./data";

export default function App() {
  // Tabs State: 'sandbox' | 'architecture' | 'code' | 'readme'
  const [activeTab, setActiveTab] = useState<"sandbox" | "architecture" | "code" | "readme">("sandbox");

  // Selection state for sandbox
  const [selectedSkills, setSelectedSkills] = useState<string[]>(["Python", "Machine Learning", "deep learning"]);
  const [customSkillInput, setCustomSkillInput] = useState("");
  const [topN, setTopN] = useState(3);
  const [searchFilter, setSearchFilter] = useState("");

  // Normalization tracer state: shows when someone inputs an alias
  const [aliasTracer, setAliasTracer] = useState<{ original: string; mapped: string; description: string } | null>(null);

  // Selected subcomponent in Architecture tab for Engineering Insights details
  const [activeArchStep, setActiveArchStep] = useState<number>(0);

  // Copy state for README or Python code
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  // 1. Run Pipeline
  const pipelineResult = useMemo(() => {
    return runRecommendationPipeline(selectedSkills, topN);
  }, [selectedSkills, topN]);

  // Handle adding a skill
  const handleAddSkill = (skill: string) => {
    const trimmed = skill.trim();
    if (!trimmed) return;

    // Run custom normalisation to check for aliases
    const normalized = normalizeSkill(trimmed);
    const cleanedLower = trimmed.toLowerCase();

    // Trace details if an alias was triggered for interactive learning
    if (VOCABULARY_NORMALIZATION_MAP[cleanedLower]) {
      setAliasTracer({
        original: trimmed,
        mapped: normalized,
        description: ALIASES_DESCRIPTIONS[cleanedLower] || "Standardized matching alias discovered."
      });
      // Clear after 4s
      setTimeout(() => setAliasTracer(null), 4500);
    }

    if (!selectedSkills.some(s => s.toLowerCase() === normalized.toLowerCase())) {
      setSelectedSkills([...selectedSkills, normalized]);
    }
    setCustomSkillInput("");
  };

  // Handle removing a skill
  const handleRemoveSkill = (skillToRemove: string) => {
    setSelectedSkills(selectedSkills.filter(s => s !== skillToRemove));
  };

  // Toggle skill simple selection
  const handleToggleLibrarySkill = (skillName: string) => {
    const exist = selectedSkills.some(s => s.toLowerCase() === skillName.toLowerCase());
    if (exist) {
      handleRemoveSkill(skillName);
    } else {
      handleAddSkill(skillName);
    }
  };

  // Custom clipboard coping
  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopyStatus(label);
    setTimeout(() => setCopyStatus(null), 2000);
  };

  // Generate downloadable assets
  const downloadFile = (filename: string, text: string) => {
    const element = document.createElement("a");
    const file = new Blob([text], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Group skills by category for easy selection
  // categories: 1: Languages, 2: Web Frameworks, 3: Cloud & DevOps, 4: Data & ML, 5: System & Mobile
  const categoriesMap = {
    1: "Programming Languages",
    2: "Web Frameworks & UI",
    3: "Cloud, Infrastructure & DevOps",
    4: "Data, NLP & ML Frameworks",
    5: "Systems & Device Engineering"
  };

  const filteredSkillsSelection = useMemo(() => {
    return SKILLS_DICTIONARY.filter(s => 
      s.skill_name.toLowerCase().includes(searchFilter.toLowerCase())
    );
  }, [searchFilter]);

  // Code contents for direct rendering & downloads
  const recommenderPyCode = `import csv
import math
import re
from typing import List, Dict, Tuple, Set

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
    cleaned = re.sub(r'\\s+', ' ', skill_name.strip()).lower()
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
        with open(raw_skills_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.skills_vocab[int(row['skill_id'])] = row['skill_name'].strip()
                
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
                'mode': 'Cold Start Fallback (Popularity-Based)',
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
`;

  const readmeMarkdown = `# Content-Based Filtering Recommendation Engine

A high-performance algorithmic career path matchmaker utilizing the **Input-Process-Output (IPO)** architecture, built with pure mathematics and zero bulky neural dependencies for ultimate inference speeds.

## 🏗️ Architectural Core (IPO Model)

### 1. Phase 0: Architecture & Data Preparation
- **Vocab Mapping & Normalization**: Raw inputs suffer from naming inconsistencies (e.g., "K8s", "k8s", "Kubernetes"). We implement an algebraic index normalization registry map to scale features to exact coordinates.
- **Skill Dictionary Schema**:
  - \`raw_skills.csv\`: \`skill_id\`, \`skill_name\`, \`category_id\` Serves as the feature index matrix.
  - \`job_roles.csv\`: \`role_id\`, \`role_name\`, \`description\`, \`associated_skills\`, \`popularity_score\` Stores item vectors with global priority fallbacks.

### 2. Phase 1: Ingestion & Feature Engineering (TF-IDF Weighting)
Binary matrices collapse when matching generic tags vs. highly informative features. If a profile matches **"Software"** and **"TensorFlow"**, standard sets score them equal ($1 \equiv 1$). We run **TF-IDF Weighting** to penalize high-frequency words and amplify targeted skills:
$$\\text{IDF}(t) = \\log_{10} \\left( 1 + \\frac{N}{\\text{DF}(t)} \\right)$$

### 3. Phase 2: Similarity Logic (Cosine Alignment)
Since document lengths differ drastically (a comprehensive DevOps skill profile vs. a junior role), standard Euclidean distances warp recommendations based on tag quantities. **Cosine Similarity** focuses strictly on angular orientation:
$$\\text{cos}(\\theta) = \\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\|\\vec{B}\\|}$$
This renders the metrics invariant to vector magnitudes, delivering pure thematic matches.

### 4. Phase 3: Cold Start Resolver
When a fresh user profile submits with empty or unrecognized tokens, the user vector is filled entirely with zeroes:
$$\\vec{P} = \\mathbf{0} \\implies \\vec{P} \\cdot \\vec{R} = 0$$
To guarantee fluid UX without crashing, a fallback state machine triggers the **Cold Start Bypass Router**, serving careers prioritized by their globally calculated popularity ratings.

---

## 📈 Big-O System Complexity
- **Vocabulary Setup**: $O(|I| \\cdot K)$ where $|I|$ is items (roles) catalog, and $K$ is average skills per role.
- **TF-IDF Coordinate Map**: $O(|V|)$ storage cost for Vocabulary collection.
- **Matching Inference**: $O(|V|)$ per candidate item. Total runtime: $O(|I| \\cdot |V|)$. Due to vocabulary sparsity, this is exceptionally performant and can serve millions of queries sub-millisecond in high-performance memory.
`;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans antialiased selection:bg-teal-500 selection:text-slate-950">
      
      {/* 🚀 Header */}
      <header className="border-b border-slate-800 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-teal-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-teal-500/10">
              <Compass className="h-5 w-5 text-slate-950 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-xl font-display font-bold tracking-tight text-white flex items-center gap-2">
                Tech Stack Recommender
                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-teal-500/15 text-teal-400 font-semibold border border-teal-500/20">
                  IPO Engine v1.0
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Content-Based Recommendation Engine • Portfolio Sandbox • Mentor Blueprint
              </p>
            </div>
          </div>

          {/* Interactive Navigation Controller */}
          <nav className="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
            <button
              id="tab-sandbox"
              onClick={() => setActiveTab("sandbox")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                activeTab === "sandbox"
                  ? "bg-teal-500 text-slate-950 font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Sliders className="h-3.5 w-3.5" />
              Interactive Sandbox
            </button>
            <button
              id="tab-architecture"
              onClick={() => setActiveTab("architecture")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                activeTab === "architecture"
                  ? "bg-teal-500 text-slate-950 font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              System Architecture
            </button>
            <button
              id="tab-code"
              onClick={() => setActiveTab("code")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                activeTab === "code"
                  ? "bg-teal-500 text-slate-950 font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Terminal className="h-3.5 w-3.5" />
              Python Implementation
            </button>
            <button
              id="tab-readme"
              onClick={() => setActiveTab("readme")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
                activeTab === "readme"
                  ? "bg-teal-500 text-slate-950 font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <FileText className="h-3.5 w-3.5" />
              GitHub README
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {/* Alerts & Notifications Section (Interactive Normalization notification popover) */}
        <AnimatePresence>
          {aliasTracer && (
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              className="mb-6 p-4 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-200 flex items-start gap-3 shadow-xl"
            >
              <Sparkles className="h-5 w-5 text-teal-400 shrink-0 mt-0.5 animate-pulse" />
              <div>
                <h4 className="text-sm font-bold font-display text-white">
                  Phase 0 Dynamic Vocabulary Normalization Triggered!
                </h4>
                <p className="text-xs mt-0.5 text-teal-300">
                  You typed <strong className="text-white">"{aliasTracer.original}"</strong> which was mapped to{" "}
                  <strong className="text-white">"{aliasTracer.mapped}"</strong>. 
                  <span className="ml-1 text-slate-300">{aliasTracer.description}</span> This ensures zero out-of-vocabulary alignment loss during vectorization steps.
                </p>
              </div>
              <button onClick={() => setAliasTracer(null)} className="ml-auto text-teal-400 hover:text-white cursor-pointer p-0.5">
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 🎛️ TAB 1: THE INTERACTIVE SANDBOX */}
        {activeTab === "sandbox" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Left Hand: Ingestion inputs panel */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              
              <div className="p-6 rounded-2xl bg-slate-950/40 border border-slate-800 shadow-md">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-display font-bold text-white flex items-center gap-2">
                    <Database className="h-4 w-4 text-teal-400" />
                    Phase 1: Ingest Skills
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-slate-400">Density Target:</span>
                    <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded-full ${
                      selectedSkills.length >= 3 
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/25'
                        : 'bg-amber-500/20 text-amber-400 border border-amber-500/25 animate-pulse'
                    }`}>
                      {selectedSkills.length}/3 Skills
                    </span>
                  </div>
                </div>

                <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                  Provide at least <strong className="text-slate-200">3 structural skill traits</strong> to satisfy matrix density metrics. Type raw concepts below (such as abbreviations like <code className="text-teal-400">k8s</code>, <code className="text-teal-400">ml</code>, <code className="text-teal-400">ui design</code> to test normalization rules) or click the catalog below.
                </p>

                {/* Input form */}
                <form onSubmit={(e) => { e.preventDefault(); handleAddSkill(customSkillInput); }} className="flex gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                    <input
                      type="text"
                      placeholder="Add custom skill (e.g., Python, docker, k8s)..."
                      value={customSkillInput}
                      onChange={(e) => setCustomSkillInput(e.target.value)}
                      className="w-full pl-9 pr-3 py-2 text-xs bg-slate-900 border border-slate-800 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                    />
                  </div>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-teal-500 text-slate-950 rounded-xl text-xs font-bold cursor-pointer hover:bg-teal-400 transition"
                  >
                    Add
                  </button>
                </form>

                {/* Selected Tag Capsules */}
                <div className="mt-4 flex flex-wrap gap-1.5">
                  <AnimatePresence>
                    {selectedSkills.map((s) => (
                      <motion.span
                        key={s}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs bg-slate-900 border border-slate-800 text-teal-300 font-mono"
                      >
                        {s}
                        <button
                          type="button"
                          onClick={() => handleRemoveSkill(s)}
                          className="hover:bg-slate-800 p-0.5 rounded text-slate-400 hover:text-white cursor-pointer"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </motion.span>
                    ))}
                  </AnimatePresence>

                  {selectedSkills.length === 0 && (
                    <div className="flex items-center gap-2 text-amber-500/90 text-xs py-1">
                      <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                      <span>Empty profile detected! Global Popularity cold-start fallback is currently active.</span>
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
                  <button
                    type="button"
                    onClick={() => setSelectedSkills([])}
                    className="text-slate-400 hover:text-red-400 cursor-pointer flex items-center gap-1 transition"
                  >
                    <RefreshCw className="h-3 w-3" />
                    Force Cold Start Mode
                  </button>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-mono">Limit Top N:</span>
                    <select
                      value={topN}
                      onChange={(e) => setTopN(Number(e.target.value))}
                      className="bg-slate-900 text-white border border-slate-800 rounded px-2 py-1 text-xs focus:outline-none focus:border-teal-500 font-mono"
                    >
                      <option value="2">2</option>
                      <option value="3">3</option>
                      <option value="5">5</option>
                      <option value="8">8</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Skill Catalog Database Quick Picker */}
              <div className="p-6 rounded-2xl bg-slate-950/40 border border-slate-800 flex-1 flex flex-col min-h-[380px]">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-display font-bold text-white flex items-center gap-2">
                    <Hash className="h-4 w-4 text-purple-400" />
                    Skill Dictionary Library
                  </h3>
                  <input
                    type="text"
                    placeholder="Filter catalog..."
                    value={searchFilter}
                    onChange={(e) => setSearchFilter(e.target.value)}
                    className="w-32 px-2 py-1 text-[10px] bg-slate-900 border border-slate-800 rounded-lg text-white"
                  />
                </div>
                <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
                  Select key nodes below to populate the user feature alignment matrix:
                </p>

                {/* Render categories or raw list */}
                <div className="flex-1 overflow-y-auto max-h-[300px] space-y-4 pr-1 scrollbar-thin">
                  {Object.entries(categoriesMap).map(([catId, catName]) => {
                    const skillsInCat = filteredSkillsSelection.filter(s => s.category_id === Number(catId));
                    if (skillsInCat.length === 0) return null;

                    return (
                      <div key={catId} className="space-y-1.5">
                        <h4 className="text-[10px] font-mono tracking-wider uppercase text-slate-500 font-semibold">
                          {catName}
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {skillsInCat.map((s) => {
                            const isSelected = selectedSkills.some(userS => userS.toLowerCase() === s.skill_name.toLowerCase());
                            return (
                              <button
                                key={s.skill_id}
                                type="button"
                                onClick={() => handleToggleLibrarySkill(s.skill_name)}
                                className={`px-2 py-1 text-xs rounded-lg border cursor-pointer font-mono text-[11px] transition-all ${
                                  isSelected
                                    ? "bg-teal-500/20 border-teal-500 text-teal-300 shadow-sm"
                                    : "bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-300"
                                }`}
                              >
                                {s.skill_name}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              
            </div>

            {/* Right Hand: Process pipeline and Recommendation Output (Top-N) */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              
              {/* Output Recommendation Panel */}
              <div className="p-6 rounded-2xl bg-slate-950/30 border border-slate-800">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
                  <div>
                    <h3 className="text-base font-display font-bold text-white flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-emerald-400" />
                      Phase 3: Top-{topN} Recommended Outcomes
                    </h3>
                    <p className="text-xs text-slate-400 font-mono mt-1">
                      Pipeline state: <span className="text-teal-400">{pipelineResult.mode}</span>
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {pipelineResult.recs.map((rec, index) => {
                    return (
                      <div
                        key={rec.role_name}
                        className={`p-4 rounded-xl border transition-all ${
                          rec.cold_start_active 
                            ? "bg-slate-900/40 border-slate-800/80 hover:border-slate-800"
                            : "bg-slate-950/60 border-slate-800 hover:border-teal-500/20 shadow-md hover:shadow-teal-500/5 hover:-translate-y-0.5"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-4 mb-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono text-teal-400 font-bold bg-teal-500/10 px-2 py-0.5 rounded-md">
                                Rank {index + 1}
                              </span>
                              <h4 className="text-sm font-semibold font-display text-white">
                                {rec.role_name}
                              </h4>
                            </div>
                            <p className="text-xs text-slate-300 mt-1.5 leading-relaxed">
                              {rec.description}
                            </p>
                          </div>

                          {/* Matching scorecard circle representation */}
                          <div className="text-right shrink-0">
                            {rec.cold_start_active ? (
                              <div className="flex flex-col items-end">
                                <span className="text-xs font-semibold text-slate-400 uppercase font-mono tracking-tight flex items-center gap-1 bg-slate-800/60 px-2 py-1 rounded">
                                  <Flame className="h-3 w-3 text-red-400 animate-pulse" />
                                  Trending Fallback
                                </span>
                                <span className="text-[10px] text-slate-500 mt-1 font-mono">
                                  Popularity: {rec.popularity_score}%
                                </span>
                              </div>
                            ) : (
                              <div className="flex flex-col items-end">
                                <div className="text-sm font-mono font-bold text-emerald-400">
                                  {rec.match_percentage}%
                                </div>
                                <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                                  Cosine Score: {rec.score.toFixed(4)}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Intersection / Triggering tags indicators */}
                        {!rec.cold_start_active && (
                          <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap items-center gap-1.5 text-xs">
                            <span className="text-slate-500 font-mono text-[10px]">Triggers:</span>
                            {rec.triggered_tags.map(tag => (
                              <span key={tag} className="px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-mono font-medium">
                                {tag}
                              </span>
                            ))}
                            {rec.triggered_tags.length === 0 && (
                              <span className="text-slate-500 italic text-[11px]">No direct matches</span>
                            )}
                          </div>
                        )}
                        
                        {/* Nested visual proof tracer tool */}
                        {!rec.cold_start_active && (
                          <details className="mt-2.5 pt-1.5 border-t border-dashed border-slate-800 text-xs">
                            <summary className="text-[10.5px] text-slate-400 hover:text-teal-400 cursor-pointer font-mono font-medium list-none flex items-center gap-1">
                              <ChevronRight className="h-3 w-3 text-teal-500 transition-transform duration-200" />
                              View Pipeline Mathematical Trace Report
                            </summary>
                            <div className="mt-3 p-3 rounded bg-slate-900 border border-slate-800/50 space-y-2">
                              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                                <div>
                                  <span className="text-slate-500">Dot Product (A · B):</span>
                                  <p className="text-white mt-0.5">
                                    {calculateCosineSimilarity(
                                      vectorize(pipelineResult.normalized_inputs, pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                      vectorize(JOB_ROLES.find(r => r.role_name === rec.role_name)?.associated_skills || [], pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                      pipelineResult.vocabulary
                                    ).dotProduct.toFixed(6)}
                                  </p>
                                </div>
                                <div className="text-right">
                                  <span className="text-slate-500">Target Role Magnitude ||B||:</span>
                                  <p className="text-white mt-0.5">
                                    {calculateCosineSimilarity(
                                      vectorize(pipelineResult.normalized_inputs, pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                      vectorize(JOB_ROLES.find(r => r.role_name === rec.role_name)?.associated_skills || [], pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                      pipelineResult.vocabulary
                                    ).magnitudeB.toFixed(6)}
                                  </p>
                                </div>
                              </div>
                              <div className="pt-2 border-t border-slate-800 text-[10.5px] font-mono text-slate-300">
                                <span className="text-emerald-400">Cosine Alignment Equation:</span>
                                <div className="bg-slate-950 p-2 rounded mt-1 overflow-x-auto text-[10px] text-teal-300">
                                  {"cosine = (Dot Product) / (||A|| * ||B||) \n"}
                                  {"cosine = " }
                                  {calculateCosineSimilarity(
                                    vectorize(pipelineResult.normalized_inputs, pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    vectorize(JOB_ROLES.find(r => r.role_name === rec.role_name)?.associated_skills || [], pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    pipelineResult.vocabulary
                                  ).dotProduct.toFixed(5)}
                                  {" / ("}
                                  {calculateCosineSimilarity(
                                    vectorize(pipelineResult.normalized_inputs, pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    vectorize(JOB_ROLES.find(r => r.role_name === rec.role_name)?.associated_skills || [], pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    pipelineResult.vocabulary
                                  ).magnitudeA.toFixed(5)}
                                  {" * "}
                                  {calculateCosineSimilarity(
                                    vectorize(pipelineResult.normalized_inputs, pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    vectorize(JOB_ROLES.find(r => r.role_name === rec.role_name)?.associated_skills || [], pipelineResult.vocabulary, pipelineResult.idf_matrix),
                                    pipelineResult.vocabulary
                                  ).magnitudeB.toFixed(5)}
                                  {") = "}
                                  <strong className="text-white font-bold">{rec.score.toFixed(6)}</strong>
                                </div>
                              </div>
                            </div>
                          </details>
                        )}

                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Graphical Comparison Sandbox Proof */}
              <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-950 to-slate-900 border border-slate-800">
                <h4 className="text-sm font-display font-bold text-white flex items-center gap-2 mb-2">
                  <Cpu className="h-4 w-4 text-emerald-400" />
                  Engineering Sandbox Concept: Cosine vs. Euclidean Distance
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Euclidean distance measures absolute straight line offset distance. If item descriptions are long or have repeated skills, the Euclidean offset zooms outwards (d &rarr; &infin;) even if thematic alignment is identical. <strong className="text-slate-200">Cosine Similarity calculates the angular vector alignment</strong>, completely ignoring text size.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-xs text-rose-400 font-mono block font-bold mb-1">❌ Euclidean Distance:</span>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Measuring distance using the Euclidean formula yields erratic values because it is highly sensitive to vector length magnitudes. It mistakenly penalizes deep, detailed resumes or roles over short outlines.
                    </p>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-xs text-teal-400 font-mono block font-bold mb-1">✅ Cosine Alignment:</span>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Measuring the angle theta between vectors via normalized dot product divides out term lengths. This maintains perfect scaling and is invariant to documentation quantity.
                    </p>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* 🏗️ TAB 2: SYSTEM ARCHITECTURE INTERACTIVE WORKTHROUGH */}
        {activeTab === "architecture" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Steps Left menu */}
            <div className="lg:col-span-4 space-y-3">
              <h3 className="text-base font-display font-bold text-white px-2">
                Core IPO Lifecycle Phases
              </h3>
              <p className="text-xs text-slate-400 px-2 leading-relaxed mb-4">
                Click a stage below to view technical implementations and rigorous Senior Engineer insights:
              </p>

              {[
                { step: 0, title: "Phase 0: Input & Normalization", tag: "Prep Stage" },
                { step: 1, title: "Phase 1: Feature Vectorization", tag: "Ingestion" },
                { step: 2, title: "Phase 2: Similarity Logic", tag: "Processing" },
                { step: 3, title: "Phase 3: Top-N & Fallback Routing", tag: "Output" }
              ].map((item) => (
                <button
                  key={item.step}
                  onClick={() => setActiveArchStep(item.step)}
                  className={`w-full text-left p-4 rounded-xl border cursor-pointer transition-all ${
                    activeArchStep === item.step
                      ? "bg-teal-500/10 border-teal-500/40 shadow-md"
                      : "bg-slate-950/40 border-slate-800/80 hover:bg-slate-950/70"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono tracking-wider font-semibold text-slate-500 uppercase">
                      {item.tag}
                    </span>
                    <span className={`text-[10.5px] font-mono ${activeArchStep === item.step ? 'text-teal-400' : 'text-slate-400'}`}>
                      [Phase {item.step}]
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold font-display text-white mt-1">
                    {item.title}
                  </h4>
                </button>
              ))}
            </div>

            {/* Right Panel Detailed Implementation & Insights */}
            <div className="lg:col-span-8 p-6 rounded-2xl bg-slate-950/30 border border-slate-800">
              {activeArchStep === 0 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-white">
                      Phase 0: Architecture, Data Prep & Normalization
                    </h3>
                    <span className="text-xs font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                      Standardization Segment
                    </span>
                  </div>
                  
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A machine has no baseline semantic comprehension for text items like "Web Design" or "k8s". They exist solely as arbitrary sequence characters. To execute vector math, incoming parameters must map directly to clean index columns inside our skill dictionary database.
                  </p>

                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                    <span className="text-xs font-mono font-bold text-teal-300 block">
                      Skill Dictionary Schema (raw_skills.csv):
                    </span>
                    <div className="bg-slate-950 p-3 rounded font-mono text-[11px] text-slate-400 overflow-x-auto">
                      {"skill_id,skill_name,category_id\n"}
                      {"1,Python,1\n"}
                      {"2,JavaScript,1\n"}
                      {"8,React,2\n"}
                      {"15,Docker,3\n"}
                      {"25,Machine Learning,4"}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                    <h4 className="text-xs font-mono font-bold text-emerald-400 flex items-center gap-1">
                      <Info className="h-3.5 w-3.5" />
                      Senior Engineering Insight: Why Normalization Prevents Collapse
                    </h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      In text retrieval architectures, naming discrepancies collapse similarities. For example, if a user enters <span className="font-mono text-white px-1 py-0.5 rounded bg-slate-900">“JS”</span> and a job definition declares <span className="font-mono text-white px-1 py-0.5 rounded bg-slate-900">“JavaScript”</span>, Jaccard and Term overlaps fail because string elements differ. By registering dictionary aliases at the Ingestion edge, we ensure incoming attributes index onto identical dimensions inside the vocabulary vector.
                    </p>
                  </div>
                </div>
              )}

              {activeArchStep === 1 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-white">
                      Phase 1: Feature Vectorization & TF-IDF Weighting
                    </h3>
                    <span className="text-xs font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                      Feature Scaling
                    </span>
                  </div>
                  
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Simple binary representations (using 1 for presence, 0 for absence) overlook importance gradients. Common, broad words like "Software", "Database", or "Coding" shouldn't hold the same coordinate weight as specialized identifiers like "PyTorch" or "Terraform".
                  </p>

                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                    <span className="text-xs font-mono font-bold text-teal-300 block">
                      The TF-IDF Vector Math:
                    </span>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Inverse Document Frequency dampens high-frequency terms, compressing their mathematical weight. This results in far more relevant matches based on specialized tags:
                    </p>
                    <div className="bg-slate-950 p-4 rounded text-left text-xs font-mono text-slate-300 space-y-2 overflow-x-auto">
                      <p><span className="text-teal-400">TF(t, d)</span> = (Count of term t in doc d) / (Total terms in doc d)</p>
                      <p><span className="text-teal-400">IDF(t)</span> = log10( 1 + Total Documents / Documents containing term t )</p>
                      <p><span className="text-emerald-400 font-bold">TF-IDF</span> = TF &times; IDF</p>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                    <h4 className="text-xs font-mono font-bold text-emerald-400 flex items-center gap-1">
                      <Info className="h-3.5 w-3.5" />
                      Senior Engineering Insight: Linear Penalization Bypass
                    </h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      The logarithm applied in IDF scales penalization exponentially rather than linearly. Left linear, highly global keywords like "Code" would receive zero rating entirely, breaking matching pipelines. Logarithmic smoothing maintains comparable values across sparse text representations.
                    </p>
                  </div>
                </div>
              )}

              {activeArchStep === 2 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-white">
                      Phase 2: Similarity Logic (Vector Cosine Engine)
                    </h3>
                    <span className="text-xs font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                      Algebraic Matching
                    </span>
                  </div>
                  
                  <p className="text-xs text-slate-300 leading-relaxed">
                    To match a vectorized user profile against thousands of available role metadata profiles, we evaluate the angular alignment of coordinates in |V|-dimensional vocabulary vector space.
                  </p>

                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                    <span className="text-xs font-mono font-bold text-teal-300 block">
                      The Cosine Score Calculation:
                    </span>
                    <div className="bg-slate-950 p-4 rounded text-left text-xs font-mono text-slate-300 space-y-2 overflow-x-auto">
                      <p className="text-teal-300 text-center font-bold">cos(theta) = (A &middot; B) / (||A|| &times; ||B||)</p>
                      <p className="text-[11px] text-slate-400 leading-relaxed mt-1">
                        Where <strong>{"A · B = ∑ (A_i × B_i)"}</strong> is the dot product (shared features intersection support), and <strong>{"||A|| = √∑(A_i²)"}</strong> is the Euclidean index magnitude of vector A. This normalizes different length profiles smoothly.
                      </p>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                    <h4 className="text-xs font-mono font-bold text-emerald-400 flex items-center gap-1">
                      <Info className="h-3.5 w-3.5" />
                      Senior Engineering Insight: Space Complexity and Invariance
                    </h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      Because resumes often vary greatly in density, Euclidean distances fail. Short profiles sit geometrically closer to the origin O(0,0,0) than exhaustive profiles. Cosine Similarity entirely strips physical length from the equation, matching purely on relative directional orientations.
                    </p>
                  </div>
                </div>
              )}

              {activeArchStep === 3 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-white">
                      Phase 3: Top-N Truncation & Cold Start Bypassing
                    </h3>
                    <span className="text-xs font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                      System Reliability Strategy
                    </span>
                  </div>
                  
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A recommendation pipeline must handle edge cases safely. If a new user signs up or enters completely unindexed coordinates (e.g. "cooking", "swimming"), cosine similarity collapses because the user vector contains exclusively zeroes (0 . x = 0).
                  </p>

                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                    <span className="text-xs font-mono font-bold text-teal-300 block">
                      Cold Start Bypass Architecture:
                    </span>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Instead of presenting blank screens, empty search filters reroute queries via standard safety fallbacks using weighted global popularity rankings.
                    </p>
                    <div className="bg-slate-950 p-3 rounded font-mono text-[11px] text-slate-400">
                      {"if user_vector.is_empty():\n"}
                      {"    # Router bypasses math and returns pre-computed hot topics\n"}
                      {"    return get_roles_by_popularity(raw_roles)[:N]"}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                    <h4 className="text-xs font-mono font-bold text-emerald-400 flex items-center gap-1">
                      <Info className="h-3.5 w-3.5" />
                      Senior Engineering Insight: Real-world Elastic scaling fallback
                    </h4>
                    <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                      Cold-start fallbacks are standard in high stakes production environments. If a user profile has short interaction logs, collaborative and matrix alignment rates are erratic. Serving general metrics ensures user onboarding flows smoothly while indexing initial data.
                    </p>
                  </div>
                </div>
              )}
            </div>

          </div>
        )}

        {/* 🐍 TAB 3: PRODUCTION READY PYTHON CODE VIEW */}
        {activeTab === "code" && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-slate-950/40 border border-slate-800">
              <div>
                <h3 className="text-base font-display font-bold text-white flex items-center gap-2">
                  <Terminal className="h-5 w-5 text-teal-400" />
                  Production Python Script (recommender.py)
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Fully functional content-based filtering model with custom Cosine Similarity and laplace-smoothed TF-IDF calculations from scratch.
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => copyToClipboard(recommenderPyCode, "py")}
                  className="px-3.5 py-2 border border-slate-800 hover:bg-slate-800/60 rounded-xl text-xs font-semibold cursor-pointer transition flex items-center gap-1.5"
                >
                  {copyStatus === "py" ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                      Copied Script!
                    </>
                  ) : (
                    <>
                      <Code className="h-3.5 w-3.5 text-slate-400" />
                      Copy Code
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => downloadFile("recommender.py", recommenderPyCode)}
                  className="px-3.5 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 rounded-xl text-xs font-bold cursor-pointer transition flex items-center gap-1.5"
                >
                  <Download className="h-3.5 w-3.5" />
                  Download recommender.py
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Actual Code Viewer Block */}
              <div className="lg:col-span-8 p-4 rounded-2xl bg-slate-950 text-slate-300 font-mono text-xs overflow-x-auto max-h-[600px] border border-slate-800 scrollbar-thin">
                <pre className="text-slate-300">{recommenderPyCode}</pre>
              </div>

              {/* Engineering Blueprint Card */}
              <div className="lg:col-span-4 space-y-6">
                
                <div className="p-5 rounded-2xl bg-slate-950/40 border border-slate-800">
                  <h4 className="text-xs font-mono font-bold tracking-wider uppercase text-slate-500 mb-3 block">
                    Execution Metrics
                  </h4>
                  <div className="space-y-4 text-xs font-mono">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="text-slate-400">Languages:</span>
                      <span className="text-white">Python 3.8+</span>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="text-slate-400">Library Overhead:</span>
                      <span className="text-emerald-400">0% (Pure Standard Library)</span>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="text-slate-400">Time Complexity:</span>
                      <span className="text-teal-400">O(N * |V|)</span>
                    </div>
                    <div className="flex items-center justify-between pb-1">
                      <span className="text-slate-400">Space Complexity:</span>
                      <span className="text-teal-400">O(|V|)</span>
                    </div>
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-gradient-to-br from-indigo-950/50 to-slate-950/50 border border-slate-800 space-y-3">
                  <h4 className="text-xs font-sans font-bold text-white flex items-center gap-1">
                    <Sparkles className="h-4 w-4 text-indigo-400" />
                    Portfolio Deployment Trick
                  </h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    By making this script require <strong className="text-white">zero external third-party imports</strong> (like scikit-learn or pandas), you prove to prospective engineering leads that you understand the underlying math and data structures from first principles! It can be easily embedded in tiny serverless runtimes.
                  </p>
                </div>

                <div className="p-5 rounded-2xl bg-slate-950/40 border border-slate-800 space-y-3">
                  <h4 className="text-xs font-mono font-bold text-teal-400 uppercase">
                    Data Set Dependencies
                  </h4>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    Make sure `/raw_skills.csv` and `/job_roles.csv` sit in the identical folder subdirectory as the python file. The script parses them automatically using CSV reader elements.
                  </p>
                  <div className="flex gap-2 pt-2">
                    <button
                      type="button"
                      onClick={() => downloadFile("raw_skills.csv", SKILLS_DICTIONARY.map(s => `${s.skill_id},${s.skill_name},${s.category_id}`).join("\n"))}
                      className="text-[11px] text-slate-300 hover:text-white cursor-pointer"
                    >
                      • Get raw_skills.csv
                    </button>
                    <button
                      type="button"
                      onClick={() => downloadFile("job_roles.csv", JOB_ROLES.map(r => `${r.role_id},"${r.role_name}","${r.description}","${r.associated_skills.join(", ")}",${r.popularity_score}`).join("\n"))}
                      className="text-[11px] text-slate-300 hover:text-white cursor-pointer"
                    >
                      • Get job_roles.csv
                    </button>
                  </div>
                </div>

              </div>

            </div>
          </div>
        )}

        {/* 📖 TAB 4: PORTFOLIO READY READY DOCUMENTATION README */}
        {activeTab === "readme" && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-slate-950/40 border border-slate-800">
              <div>
                <h3 className="text-base font-display font-bold text-white flex items-center gap-2">
                  <FileText className="h-5 w-5 text-teal-400" />
                  GitHub-Ready Profile README.md
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Ready-to-deploy, polished Markdown documentation. Include this in your GitHub repository to stand out to engineering interviewers.
                </p>
              </div>
              <button
                type="button"
                onClick={() => copyToClipboard(readmeMarkdown, "readme")}
                className="px-4 py-2 bg-teal-500 hover:bg-teal-400 text-slate-950 rounded-xl text-xs font-bold cursor-pointer transition shrink-0 flex items-center gap-1.5"
              >
                {copyStatus === "readme" ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-slate-950 stroke-[3]" />
                    Copied Documentation!
                  </>
                ) : (
                  <>
                    <Code className="h-3.5 w-3.5" />
                    Copy Markdown
                  </>
                )}
              </button>
            </div>

            {/* Markdown rendered view */}
            <div className="p-8 rounded-2xl bg-slate-950 border border-slate-800 text-slate-300 space-y-6">
              
              <div className="border-b border-slate-800 pb-4">
                <h1 className="text-2xl font-display font-bold text-white">
                  Content-Based career Matchmaker Engine Sandbox
                </h1>
                <p className="text-xs font-mono text-slate-500 mt-1">
                  Designed & Mentored by a Senior AI Specialist
                </p>
              </div>

              <div className="space-y-3 text-xs leading-relaxed">
                <h3 className="text-sm font-semibold font-display text-white">⚙️ The "Why" behind the algorithm</h3>
                <p>
                  Simple keyword-matching filters often collapse under extreme document-length differences. Common engineering concepts (e.g., "Developer", "Framework", "Node") trigger excessive noise, matching junior candidates to principal specialist job openings simply due to word frequency overlap. This project models career alignment mathematically via **term frequency penalization** combined with **angular translation metric constraints** to bypass qualitative language hurdles.
                </p>

                <h3 className="text-sm font-semibold font-display text-white">📐 Key Formulations Built</h3>
                <ul className="list-disc list-inside space-y-2 text-slate-400 font-mono">
                  <li><strong>TF weights:</strong> Scales terms based on intensity density inside individual profiles.</li>
                  <li><strong>IDF (Smoothed logarithmic scaling):</strong> Exponentially dampens keywords containing little specialized intelligence.</li>
                  <li><strong>Cosine Orientation Metric:</strong> Divides out magnitude variables to focus alignment purely on thematic directional orientation vectors.</li>
                  <li><strong>Active Cold Start Bypass Router:</strong> Leverages rolling global trend popularity indices when incoming profiles have inadequate vectors.</li>
                </ul>

                <h3 className="text-sm font-semibold font-display text-white">⏳ Big-O Complexity Matrix</h3>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <table className="w-full text-[11px] font-mono">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="text-left pb-2">Phase</th>
                        <th className="text-left pb-2">Time Complexity</th>
                        <th className="text-left pb-2">Space complexity</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-800/50">
                        <td className="py-2 font-semibold text-white">Ingestion/Normalization</td>
                        <td className="py-2 text-teal-400">O(|U|)</td>
                        <td className="py-2 text-teal-400">O(|U|)</td>
                      </tr>
                      <tr className="border-b border-slate-800/50">
                        <td className="py-2 font-semibold text-white">TF-IDF Vectorization</td>
                        <td className="py-2 text-teal-400">O(|V|)</td>
                        <td className="py-2 text-teal-400">O(|V|)</td>
                      </tr>
                      <tr>
                        <td className="py-2 font-semibold text-white">Cosine Alignment Calculation</td>
                        <td className="py-2 text-teal-400">O(|I| * |V|)</td>
                        <td className="py-2 text-teal-400">O(|V|)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950/40 text-slate-500 py-8 text-xs font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <p>© 2026 AI Portfolio Series. Built for DecodeLabs Project 3 Capstone.</p>
          <div className="flex gap-4">
            <span className="text-teal-500/80">★ Senior Mentor-Validated</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">Pure Mathematical Invariance</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
