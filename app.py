from flask import Flask, render_template, request, jsonify
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from functools import lru_cache
import threading
import os
from flask import redirect, url_for
import fitz  # PyMuPDF

# Logging ayarları
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global değişkenler
search_engine = None
SIMILARITY_THRESHOLD = 0.35  # Minimum bedzerlik eşiği

# Add this function to process PDF files
def pdf_to_text(pdf_path):
    """Convert PDF to text using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        # Use the "blocks" option for better word spacing
        blocks = page.get_text("blocks")
        for block in blocks:
            # block[4] contains the text
            text += block[4] + "\n"  # Add newline after each block
            
    return text

def setup_database():
    """Neo4j database setup"""
    try:
        # Neo4j connection
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
        
        # Optional database cleanup
        graph.delete_all()
        
        # Dictionaries to store academics and projects
        academics = {}
        projects = []
        
        # Check if there's existing data in the database
        existing_projects = list(graph.run("MATCH (p:Project) RETURN p").data())
        for item in existing_projects:
            projects.append(item['p'])
            
        existing_academics = list(graph.run("MATCH (a:Academic) RETURN a").data())
        for item in existing_academics:
            academics[item['a']['name']] = item['a']
        
        # Create calls (example calls for testing)
        calls = []
        
        # Only create calls if they don't already exist
        existing_calls = list(graph.run("MATCH (c:Call) RETURN c").data())
        if not existing_calls:
            call1 = Node(
                "Call", 
                id=100, 
                title="H2025 Research Grant on AI", 
                description="Research call focusing on Natural Language Processing, machine learning, and sustainability applications. We're looking for innovative approaches to solve environmental challenges using AI."
            )
            calls.append(call1)
            
            call2 = Node(
                "Call", 
                id=101, 
                title="EU Graph Data Initiative", 
                description="Call for projects on graph mining, knowledge graphs, and network analysis with applications in social network analysis and infrastructure optimization."
            )
            calls.append(call2)
            
            graph.create(call1 | call2)
            
            # Add keywords for calls
            for call in calls:
                description = call["description"].lower()
                # Define potential keywords in call
                potential_keywords = ["ai", "nlp", "machine learning", "sustainability", 
                                    "graph mining", "knowledge graphs", "network analysis"]
                                    
                for keyword in potential_keywords:
                    if keyword in description:
                        # Find or create keyword node
                        keyword_node = Node("Keyword", name=keyword)
                        try:
                            graph.merge(keyword_node, "Keyword", "name")
                            # Call-Keyword relationship
                            has_theme_rel = Relationship(call, "HAS_THEME", keyword_node)
                            graph.create(has_theme_rel)
                        except Exception as e:
                            logger.warning(f"Call-Keyword relationship error: {e}")
                            continue
        else:
            for item in existing_calls:
                calls.append(item['c'])
        
        logger.info(f"Database initialized with {len(projects)} projects, {len(academics)} academics, and {len(calls)} calls")
        
        return graph, academics, projects, calls

    except Exception as e:
        logger.error(f"Database setup error: {e}", exc_info=True)
        return None, {}, [], []



class SemanticSearchEngine:
    def __init__(self, graph, academics, projects, calls):
        """Semantik arama motoru kurulumu"""
        self.graph = graph
        self.academics = academics
        self.projects = projects
        self.calls = calls
        self.matcher = NodeMatcher(graph)
        
        # Model yükleme işleminin durum değişkeni
        self.model_loaded = False
        self.lock = threading.Lock()
        
        # Modeli arka planda yükle
        self.load_model_thread = threading.Thread(target=self._load_model)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
    
    def _load_model(self):
        """Modeli arka planda yükle"""
        try:
            logger.info("NLP modeli yükleniyor...")
            # Daha iyi bir model kullanın - Türkçe desteği daha iyi olan model
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            
            # Vektör temsillerini hesapla
            self.precompute_embeddings()
            
            with self.lock:
                self.model_loaded = True
                
            logger.info("NLP modeli ve vektör temsilleri hazır.")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}", exc_info=True)
    
    def wait_for_model(self):
        """Model yüklenene kadar bekle"""
        start_time = time.time()
        while not self.model_loaded:
            if time.time() - start_time > 60:  # 60 saniye timeout
                raise TimeoutError("Model yükleme zaman aşımına uğradı")
            time.sleep(0.5)
    
    def precompute_embeddings(self):
        """Proje ve çağrı içeriklerinin vektör temsillerini önceden hesapla"""
        logger.info("Vektör temsilleri hesaplanıyor...")
        
        # Projeler için vektör temsilleri
        self.project_texts = []
        self.project_nodes = []
        
        batch_size = 16  # Bellek kullanımını optimize etmek için batch işleme
        
        # Projeleri toplu olarak işle
        for project in self.projects:
            # Proje başlığı, özet ve anahtar kelimeleri birleştir
            content = f"{project['title']} {project['abstract']} {project['keywords']}"
            self.project_texts.append(content)
            self.project_nodes.append(project)
        
        # Projeler için vektör temsilleri hesapla
        if self.project_texts:
            self.project_embeddings = self.model.encode(
                self.project_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            self.project_embeddings = np.array([])
        
        # Çağrılar için vektör temsilleri
        self.call_texts = []
        self.call_nodes = []
        
        for call in self.calls:
            content = f"{call['title']} {call['description']}"
            self.call_texts.append(content)
            self.call_nodes.append(call)
        
        # Çağrılar için vektör temsilleri hesapla
        if self.call_texts:
            self.call_embeddings = self.model.encode(
                self.call_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            self.call_embeddings = np.array([])
        
        # Akademisyenler için vektör temsilleri
        self.academic_texts = []
        self.academic_nodes = []
        
        for name, academic in self.academics.items():
            # Akademisyenin adı ve uzmanlık alanı
            content = f"{name} {academic['keywords']}"
            
            # Akademisyenin projelerinden özet bilgi ekle (tüm projelerin bilgilerini değil)
            cypher_query = """
            MATCH (a:Academic {name: $name})-[r:OWNS]->(p:Project)
            RETURN p.title, p.keywords
            LIMIT 5
            """
            
            results = self.graph.run(cypher_query, name=name).data()
            
            for result in results:
                if result.get('p.title'):
                    content += f" {result.get('p.title')}"
                if result.get('p.keywords'):
                    content += f" {result.get('p.keywords')}"
                    
            # Uzman olduğu anahtar kelimeleri ekle
            cypher_query = """
            MATCH (a:Academic {name: $name})-[r:EXPERT_IN]->(k:Keyword)
            RETURN k.name
            """
            
            results = self.graph.run(cypher_query, name=name).data()
            for result in results:
                if result.get('k.name'):
                    content += f" {result.get('k.name')}"
            
            self.academic_texts.append(content)
            self.academic_nodes.append(academic)
        
        # Akademisyenler için vektör temsilleri hesapla
        if self.academic_texts:
            self.academic_embeddings = self.model.encode(
                self.academic_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        else:
            self.academic_embeddings = np.array([])
        
        logger.info("Vektör temsilleri hesaplandı.")
    
    @lru_cache(maxsize=100)
    def get_query_embedding(self, query):
        """Sorgu vektörünü hesapla ve önbelleğe al"""
        return self.model.encode([query])[0]
    
    def _graph_based_search(self, query, top_k=5):
        """Neo4j graph veritabanı üzerinden anahtar kelime tabanlı arama"""
        # Sorguyu temizle ve küçük harfe dönüştür
        cleaned_query = query.lower().strip()
        
        # Cypher sorgusu ile doğrudan Neo4j üzerinden arama
        project_query = """
        MATCH (p:Project)
        WHERE toLower(p.title) CONTAINS $query OR toLower(p.abstract) CONTAINS $query OR toLower(p.keywords) CONTAINS $query
        RETURN p, ID(p) as id
        LIMIT $limit
        """
        
        call_query = """
        MATCH (c:Call)
        WHERE toLower(c.title) CONTAINS $query OR toLower(c.description) CONTAINS $query
        RETURN c, ID(c) as id
        LIMIT $limit
        """
        
        academic_query = """
        MATCH (a:Academic)
        WHERE toLower(a.name) CONTAINS $query OR toLower(a.keywords) CONTAINS $query
        RETURN a, ID(a) as id
        LIMIT $limit
        """
        
        # Anahtar kelime tabanlı sorgu
        keyword_project_query = """
        MATCH (k:Keyword)-[r:HAS_KEYWORD]-(p:Project)
        WHERE toLower(k.name) CONTAINS $query
        RETURN p, ID(p) as id
        LIMIT $limit
        """
        
        keyword_academic_query = """
        MATCH (k:Keyword)-[r:EXPERT_IN]-(a:Academic)
        WHERE toLower(k.name) CONTAINS $query
        RETURN a, ID(a) as id
        LIMIT $limit
        """
        
        keyword_call_query = """
        MATCH (k:Keyword)-[r:HAS_THEME]-(c:Call)
        WHERE toLower(k.name) CONTAINS $query
        RETURN c, ID(c) as id
        LIMIT $limit
        """
        
        # Sorguları çalıştır
        graph_projects = list(self.graph.run(project_query, query=cleaned_query, limit=top_k).data())
        graph_projects += list(self.graph.run(keyword_project_query, query=cleaned_query, limit=top_k).data())
        
        graph_calls = list(self.graph.run(call_query, query=cleaned_query, limit=top_k).data())
        graph_calls += list(self.graph.run(keyword_call_query, query=cleaned_query, limit=top_k).data())
        
        graph_academics = list(self.graph.run(academic_query, query=cleaned_query, limit=top_k).data())
        graph_academics += list(self.graph.run(keyword_academic_query, query=cleaned_query, limit=top_k).data())
        
        # Sonuçları düzenle
        graph_results = {
            "projects": [],
            "calls": [],
            "academics": []
        }
        
        # Projeleri işle
        seen_projects = set()
        for result in graph_projects:
            if result['id'] not in seen_projects:
                seen_projects.add(result['id'])
                project = result['p']
                proj_dict = {
                    "id": project["id"],
                    "title": project["title"],
                    "year": project["year"],
                    "keywords": project["keywords"],
                    "similarity": 1.0  # Graph tabanlı aramada similarity değeri yok
                }
                
                if "abstract" in project:
                    ab = project["abstract"]
                    proj_dict["abstract"] = (ab[:150] + "...") if len(ab) > 150 else ab
                    
                graph_results["projects"].append(proj_dict)
        
        # Çağrıları işle
        seen_calls = set()
        for result in graph_calls:
            if result['id'] not in seen_calls:
                seen_calls.add(result['id'])
                call = result['c']
                call_dict = {
                    "id": call["id"],
                    "title": call["title"],
                    "similarity": 1.0  # Graph tabanlı aramada similarity değeri yok
                }
                
                if "description" in call:
                    desc = call["description"]
                    call_dict["description"] = (desc[:150] + "...") if len(desc) > 150 else desc
                    
                graph_results["calls"].append(call_dict)
        
        # Akademisyenleri işle
        seen_academics = set()
        for result in graph_academics:
            if result['id'] not in seen_academics:
                seen_academics.add(result['id'])
                academic = result['a']
                academic_dict = {
                    "name": academic["name"],
                    "keywords": academic["keywords"],
                    "similarity": 1.0  # Graph tabanlı aramada similarity değeri yok
                }
                
                graph_results["academics"].append(academic_dict)
        
        return graph_results
    
    def search(self, query, top_k=5):
        """Verilen sorgu için en alakalı projeleri, çağrıları ve akademisyenleri bul"""
        logger.info(f"Sorgu: '{query}' için arama yapılıyor...")
        
        # Model yüklenmesini bekle
        self.wait_for_model()
        
        # Graph tabanlı arama sonuçları
        graph_results = self._graph_based_search(query, top_k)
        
        # Semantik arama sonuçları
        semantic_results = {
            "projects": [],
            "calls": [],
            "academics": []
        }
        
        # 1) Sorgu vektörünü hesapla
        query_embedding = self.get_query_embedding(query)
        
        # 2) Projeler için benzerlik hesapla
        if len(self.project_embeddings) > 0:
            sims = cosine_similarity([query_embedding], self.project_embeddings)[0]
            top_idxs = np.argsort(sims)[::-1][:top_k]
            
            for idx in top_idxs:
                sim = float(sims[idx])
                if sim > SIMILARITY_THRESHOLD:
                    proj_node = self.project_nodes[idx]
                    proj_dict = {
                        "id": proj_node["id"],
                        "title": proj_node["title"],
                        "year": proj_node["year"],
                        "keywords": proj_node["keywords"],
                        "similarity": sim
                    }
                    
                    # abstract varsa kısaltarak ekleyelim
                    if "abstract" in proj_node:
                        ab = proj_node["abstract"]
                        proj_dict["abstract"] = (ab[:150] + "...") if len(ab) > 150 else ab
                        
                    semantic_results["projects"].append(proj_dict)
        
        # 3) Çağrılar için benzerlik hesapla
        if len(self.call_embeddings) > 0:
            sims = cosine_similarity([query_embedding], self.call_embeddings)[0]
            top_idxs = np.argsort(sims)[::-1][:top_k]
            
            for idx in top_idxs:
                sim = float(sims[idx])
                if sim > SIMILARITY_THRESHOLD:
                    call_node = self.call_nodes[idx]
                    call_dict = {
                        "id": call_node["id"],
                        "title": call_node["title"],
                        "similarity": sim
                    }
                    
                    # description varsa kısaltarak ekle
                    if "description" in call_node:
                        desc = call_node["description"]
                        call_dict["description"] = (desc[:150] + "...") if len(desc) > 150 else desc
                        
                    semantic_results["calls"].append(call_dict)
        
        # 4) Akademisyenler için benzerlik hesapla
        if len(self.academic_embeddings) > 0:
            sims = cosine_similarity([query_embedding], self.academic_embeddings)[0]
            top_idxs = np.argsort(sims)[::-1][:top_k]
            
            for idx in top_idxs:
                sim = float(sims[idx])
                if sim > SIMILARITY_THRESHOLD:
                    academic_node = self.academic_nodes[idx]
                    academic_dict = {
                        "name": academic_node["name"],
                        "keywords": academic_node["keywords"],
                        "similarity": sim
                    }
                    
                    semantic_results["academics"].append(academic_dict)
        
        # 5) Projelere bağlı akademisyenleri ekle (projelerden bulunamayan akademisyenler için)
        authors = {}
        
        for proj in semantic_results["projects"]:
            # İlgili project node objesini alalım
            cypher_query = """
            MATCH (a:Academic)-[r:OWNS]->(p:Project {id: $project_id})
            RETURN a.name as name, a.keywords as keywords
            """
            
            results = self.graph.run(cypher_query, project_id=proj["id"]).data()
            
            for result in results:
                name = result.get("name")
                if name and name not in authors:
                    authors[name] = {
                        "name": name,
                        "keywords": result.get("keywords", ""),
                        "similarity": proj["similarity"] * 0.9  # Projeden bulunan akademisyenlere daha düşük benzerlik skoru
                    }
        
        # 6) Semantik ve graph tabanlı sonuçları birleştir
        combined_results = {
            "projects": semantic_results["projects"] + graph_results["projects"],
            "calls": semantic_results["calls"] + graph_results["calls"],
            "academics": semantic_results["academics"] + graph_results["academics"] + list(authors.values())
        }
        
        # Tekilleştirme işlemi
        unique_results = {
            "projects": {},
            "calls": {},
            "academics": {}
        }
        
        # Projeleri tekilleştir
        for proj in combined_results["projects"]:
            proj_id = proj["id"]
            if proj_id not in unique_results["projects"] or proj["similarity"] > unique_results["projects"][proj_id]["similarity"]:
                unique_results["projects"][proj_id] = proj
        
        # Çağrıları tekilleştir
        for call in combined_results["calls"]:
            call_id = call["id"]
            if call_id not in unique_results["calls"] or call["similarity"] > unique_results["calls"][call_id]["similarity"]:
                unique_results["calls"][call_id] = call
        
        # Akademisyenleri tekilleştir
        for academic in combined_results["academics"]:
            name = academic["name"]
            if name not in unique_results["academics"] or academic["similarity"] > unique_results["academics"][name]["similarity"]:
                unique_results["academics"][name] = academic

        # 8) Sonuçları listeler halinde yeniden düzenle
        final_results = {
            "projects": list(unique_results["projects"].values()),
            "calls": list(unique_results["calls"].values()),
            "academics": list(unique_results["academics"].values())
        }
        
        # 9) Sonuçları verimlilik açısından sınırla
        final_results["projects"] = final_results["projects"][:top_k]
        final_results["calls"] = final_results["calls"][:top_k]
        final_results["academics"] = final_results["academics"][:top_k]
        
        # 10) SIMILARITY_THRESHOLD değerinin altındaki sonuçları filtrele
        final_results["projects"] = [p for p in final_results["projects"] if p["similarity"] >= SIMILARITY_THRESHOLD]
        final_results["calls"] = [c for c in final_results["calls"] if c["similarity"] >= SIMILARITY_THRESHOLD]
        final_results["academics"] = [a for a in final_results["academics"] if a["similarity"] >= SIMILARITY_THRESHOLD]
        
        logger.info(f"Arama tamamlandı. Bulunan sonuçlar: {len(final_results['projects'])} proje, {len(final_results['calls'])} çağrı, {len(final_results['academics'])} akademisyen")
        
        return final_results


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Arama sorgusu boş olamaz'}), 400
        
        if not search_engine:
            return jsonify({'error': 'Arama motoru başlatılamadı'}), 500
        
        # Arama motoruna sorguyu gönder
        results = search_engine.search(query, top_k=5)
        
        # Sonuçları toplam benzerlik puanına göre sırala
        for category in results:
            results[category] = sorted(results[category], key=lambda x: x['similarity'], reverse=True)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Arama sırasında hata: {e}", exc_info=True)
        return jsonify({'error': f'Arama işlemi başarısız: {str(e)}'}), 500
    

# Add a route for the upload page
@app.route('/upload', methods=['GET'])
def upload_form():
    return render_template('upload.html')

# Add a route to handle the form submission
@app.route('/submit_article', methods=['POST'])
def submit_article():
    try:
        # Extract form data
        title = request.form.get('title', '')
        abstract = request.form.get('abstract', '')
        keywords = request.form.get('keywords', '')
        year = request.form.get('year', '')
        academic = request.form.get('academic', '')
        
        # Handle PDF file upload
        pdf_file = request.files.get('pdf_file')
        content = ""
        
        if pdf_file and pdf_file.filename:
            # Create uploads directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
                
            # Save the PDF temporarily
            pdf_path = os.path.join('uploads', pdf_file.filename)
            pdf_file.save(pdf_path)
            
            # Extract text from the PDF
            content = pdf_to_text(pdf_path)
            
            # Clean up - optionally remove the file after processing
            # os.remove(pdf_path)
        
        # Generate a new article ID
        # Here we're using timestamp as a simple solution
        import time
        article_id = int(time.time())
        
        # Create the project node in Neo4j
        project = Node(
            "Project",
            id=article_id,
            title=title.strip(),
            abstract=abstract.strip(),
            keywords=keywords.strip(),
            year=int(year) if year.isdigit() else 0,
            content=content  # Add the PDF content here
        )
        
        # Get the graph connection
        graph = search_engine.graph
        graph.create(project)
        
        # Process academics
        academic_names = [name.strip() for name in academic.split(',') if name.strip()]
        
        for name in academic_names:
            # Check if academic exists
            academic_node = search_engine.matcher.match("Academic", name=name).first()
            
            if not academic_node:
                # Create keyword summary for the academic
                keyword_samples = []
                if keywords and keywords.strip():
                    keyword_samples = [kw.strip() for kw in keywords.split(',')[:3] if kw.strip()]
                
                keyword_str = f"Expert in {', '.join(keyword_samples)}" if keyword_samples else "Researcher"
                
                academic_node = Node(
                    "Academic",
                    name=name,
                    keywords=keyword_str
                )
                graph.create(academic_node)
                search_engine.academics[name] = academic_node
            
            # Create relationship between academic and project
            owns_rel = Relationship(academic_node, "OWNS", project)
            graph.create(owns_rel)
            
            # Process keywords
            keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            for keyword in keywords_list:
                keyword_node = Node("Keyword", name=keyword.lower())
                try:
                    graph.merge(keyword_node, "Keyword", "name")
                    
                    # Project-Keyword relationship
                    has_keyword_rel = Relationship(project, "HAS_KEYWORD", keyword_node)
                    graph.create(has_keyword_rel)
                    
                    # Academic-Keyword relationship
                    expert_in_rel = Relationship(academic_node, "EXPERT_IN", keyword_node)
                    graph.create(expert_in_rel)
                except Exception as e:
                    logger.warning(f"Keyword relationship error: {e}")
                    continue
        
        # Add the new project to the search engine's project list
        search_engine.projects.append(project)
        
        # Update embeddings if the model is already loaded
        if search_engine.model_loaded:
            search_engine.precompute_embeddings()
        
        return jsonify({"success": True, "message": "Article added successfully", "article_id": article_id})
        
    except Exception as e:
        logger.error(f"Error while submitting article: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Sistem sağlık kontrolü"""
    try:
        if not search_engine:
            return jsonify({'status': 'error', 'message': 'Arama motoru yüklenmedi'}), 500
        
        model_status = "loaded" if search_engine.model_loaded else "loading"
        
        return jsonify({
            'status': 'ok',
            'model_status': model_status,
            'database_connection': bool(search_engine.graph),
            'projects_count': len(search_engine.projects),
            'academics_count': len(search_engine.academics),
            'calls_count': len(search_engine.calls)
        })
        
    except Exception as e:
        logger.error(f"Sağlık kontrolü sırasında hata: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    logger.info("Uygulama başlatılıyor...")
    logger.info("Veritabanı kuruluyor...")
    graph, academics, projects, calls = setup_database()
    
    if not graph:
        logger.error("Veritabanı kurulumu başarısız oldu!")
    else:
        logger.info("Semantik arama motoru başlatılıyor...")
        search_engine = SemanticSearchEngine(graph, academics, projects, calls)
        logger.info("Web sunucusu başlatılıyor...")
        app.run(debug=True, port=5001)