from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
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
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pdfplumber
import tempfile
import uuid # For unique filenames
from werkzeug.utils import secure_filename # For securing filenames
import datetime # For default year

# Logging ayarları
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global değişkenler
search_engine = None
SIMILARITY_THRESHOLD = 0.35  # Minimum bedzerlik eşiği

UPLOAD_FOLDER = 'uploads' # Define your upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
                project_node = result['p'] # Get the node directly
                proj_dict = {
                    "id": project_node.get("id"),
                    "title": project_node.get("title"),
                    "year": project_node.get("year"),
                    "keywords": project_node.get("keywords"),
                    "similarity": 1.0,
                    "url": project_node.get("url"),  # <<< ADDED
                    "pdf_filename": project_node.get("pdf_filename"),  # <<< ADDED
                    "abstract": (project_node.get("abstract", "")[:150] + "...") if len(project_node.get("abstract", "")) > 150 else project_node.get("abstract", "")
                }
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
                        "id": proj_node.get("id"),
                        "title": proj_node.get("title"),
                        "year": proj_node.get("year"),
                        "keywords": proj_node.get("keywords"),
                        "similarity": sim,
                        "url": proj_node.get("url"),  # <<< ADDED
                        "pdf_filename": proj_node.get("pdf_filename"),  # <<< ADDED
                        "abstract": (proj_node.get("abstract", "")[:150] + "...") if len(proj_node.get("abstract", "")) > 150 else proj_node.get("abstract", "")
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

@app.route('/serve_pdf/<filename>')
def serve_pdf(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        logger.error(f"PDF file not found: {filename}")
        return jsonify({"error": "File not found"}), 404
    
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

# Add a route to handle the form submission
@app.route('/submit_article', methods=['POST'])
def submit_article():
    try:
        # Extract form data
        title = request.form.get('title', '').strip()
        abstract = request.form.get('abstract', '').strip()
        keywords = request.form.get('keywords', '').strip()
        year_str = request.form.get('year', '')
        academic = request.form.get('academic', '').strip()
        article_link = request.form.get('article_link', '').strip()

        
        # Handle PDF file upload
        pdf_file = request.files.get('pdf_file')
        content_from_pdf = ""
        stored_pdf_filename = None
        project_url = article_link # Prioritize provided external link

        if pdf_file and pdf_file.filename:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            original_filename = secure_filename(pdf_file.filename)
            unique_id = str(uuid.uuid4().hex) # Generate a unique ID for the filename
            stored_pdf_filename = f"{unique_id}_{original_filename}"
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_pdf_filename)
            pdf_file.save(pdf_path)
            content_from_pdf = pdf_to_text(pdf_path)

            if not project_url: # If no external link, use the link to our served PDF
                project_url = url_for('serve_pdf', filename=stored_pdf_filename, _external=False) # Relative URL

        article_id = int(time.time())
        year = int(year_str) if year_str.isdigit() else datetime.datetime.now().year
        
        project_props = {
            "id": article_id,
            "title": title,
            "abstract": abstract if abstract else content_from_pdf[:500], # Use PDF content for abstract if empty
            "keywords": keywords,
            "year": year,
            "content": content_from_pdf, # Full content from PDF
            "url": project_url if project_url else None, # Store the determined URL
        }
        if stored_pdf_filename:
            project_props["pdf_filename"] = stored_pdf_filename
        
        project = Node("Project", **project_props)

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

@app.route('/system_health')
def system_health():
    return render_template('health.html')


@app.route('/health', methods=['GET'])
def health():
    """Sistem sağlık kontrolü"""
    try:
        if not search_engine:
            return jsonify({'status': 'error', 'message': 'Arama motoru yüklenmedi'}), 500
        
        model_status = "loaded" if search_engine.model_loaded else "loading"
        
        # Veritabanı bağlantısını aktif olarak test et
        db_connected = False
        try:
            # Neo4j bağlantısını test etmek için basit bir sorgu çalıştır
            result = search_engine.graph.run("MATCH (n) RETURN count(n) as count LIMIT 1").data()
            db_connected = len(result) > 0
        except Exception as e:
            logger.error(f"Veritabanı bağlantı testi hatası: {e}")
            db_connected = False
        
        return jsonify({
            'status': 'ok',
            'model_status': model_status,
            'database_connection': db_connected,
            'projects_count': len(search_engine.projects),
            'academics_count': len(search_engine.academics),
            'calls_count': len(search_engine.calls)
        })
        
    except Exception as e:
        logger.error(f"Sağlık kontrolü sırasında hata: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Çağrı sayfası için route
@app.route('/add_call', methods=['GET'])
def add_call_form():
    return render_template('add_call.html')


# Çağrı formunu işleme endpointi
@app.route('/submit_call', methods=['POST'])
def submit_call():
    try:
        # Form verilerini al
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        deadline = request.form.get('deadline', '')
        related_academics = request.form.get('related_academics', '')
        
        # Çağrı ID'si oluştur
        import time
        call_id = int(time.time())
        
        # Çağrı node'unu Neo4j'de oluştur
        call = Node(
            "Call",
            id=call_id,
            title=title.strip(),
            description=description.strip(),
            deadline=deadline.strip()
        )
        
        # Graph bağlantısını al
        graph = search_engine.graph
        graph.create(call)
        
        # İlgili akademisyenleri işle
        academic_names = [name.strip() for name in related_academics.split(',') if name.strip()]
        
        for name in academic_names:
            # Akademisyen var mı diye kontrol et
            academic_node = search_engine.matcher.match("Academic", name=name).first()
            
            if not academic_node:
                # Akademisyen yoksa oluştur
                academic_node = Node(
                    "Academic",
                    name=name,
                    keywords="Researcher"
                )
                graph.create(academic_node)
                search_engine.academics[name] = academic_node
            
            # Akademisyen ve çağrı arasında ilişki oluştur
            related_to_rel = Relationship(academic_node, "RELATED_TO", call)
            graph.create(related_to_rel)
        
        # Çağrıdan keywords çıkar ve bunları ekle
        description_words = description.lower().split()
        potential_keywords = ["ai", "nlp", "machine learning", "sustainability", 
                        "graph mining", "knowledge graphs", "network analysis",
                        "research", "grant", "funding", "scholarship"]
                        
        for keyword in potential_keywords:
            if keyword in description.lower():
                # Keyword node'unu bul veya oluştur
                keyword_node = Node("Keyword", name=keyword)
                try:
                    graph.merge(keyword_node, "Keyword", "name")
                    # Çağrı-Anahtar Kelime ilişkisi
                    has_theme_rel = Relationship(call, "HAS_THEME", keyword_node)
                    graph.create(has_theme_rel)
                except Exception as e:
                    logger.warning(f"Call-Keyword relationship error: {e}")
                    continue
        
        # Yeni çağrıyı search engine'in call listesine ekle
        search_engine.calls.append(call)
        
        # Model yüklüyse embeddinglari güncelle
        if search_engine.model_loaded:
            search_engine.precompute_embeddings()
        
        return jsonify({"success": True, "message": "Call added successfully", "call_id": call_id})
        
    except Exception as e:
        logger.error(f"Error while submitting call: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# Add this function to extract metadata from IEEE Xplore and similar URLs
def extract_article_metadata_from_url(url):
    """Extract metadata from academic paper URLs like IEEE Xplore"""
    metadata = {
        "title": "",
        "authors": "",
        "year": "",
        "abstract": "",
        "keywords": "",
        "content": ""
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Try with requests first
    try:
        logger.info(f"Trying to fetch metadata from URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_source = response.text
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Requests access failed: {e}. Trying with Selenium...")
        
        # Fall back to Selenium for JavaScript-heavy sites
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to load
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            driver.quit()
        except Exception as e:
            logger.error(f"Selenium access also failed: {e}")
            return metadata
    
    # Handle IEEE Xplore
    if "ieeexplore.ieee.org" in url:
        try:
            # Look for xplGlobal.document.metadata script
            script_tag = soup.find("script", string=re.compile("xplGlobal.document.metadata"))
            
            if script_tag:
                script_content = script_tag.string
                match = re.search(r'xplGlobal.document.metadata\s*=\s*({.*?});', script_content, re.DOTALL)
                
                if match:
                    ieee_metadata = json.loads(match.group(1))
                    
                    # Extract data
                    metadata["title"] = ieee_metadata.get("title", "")
                    
                    # Authors
                    authors_list = ieee_metadata.get("authors", [])
                    metadata["authors"] = ", ".join([author["name"] for author in authors_list]) if authors_list else ""
                    
                    # Publication year
                    pub_date = ieee_metadata.get("publicationDate", "")
                    if pub_date:
                        year_match = re.search(r'\d{4}', pub_date)
                        if year_match:
                            metadata["year"] = year_match.group(0)
                    
                    # Abstract
                    metadata["abstract"] = ieee_metadata.get("abstract", "")
                    
                    # Keywords
                    keywords_list = ieee_metadata.get("keywords", [])
                    keywords = []
                    for keyword_group in keywords_list:
                        keywords.extend(keyword_group.get("kwd", []))
                    metadata["keywords"] = ", ".join(keywords)
                    
                    # Content extraction can remain empty as we'll use PDF for full content
                    logger.info(f"Successfully extracted metadata from IEEE URL: {url}")
                    return metadata
            
            logger.warning("IEEE metadata script not found or unable to parse")
            
            # Fallback to general extraction
            metadata["title"] = soup.find("meta", property="og:title")["content"] if soup.find("meta", property="og:title") else ""
            metadata["abstract"] = soup.find("div", class_="abstract-text") and soup.find("div", class_="abstract-text").text.strip()
            
            # Try to find authors
            authors_section = soup.find("div", class_="authors-info-container")
            if authors_section:
                authors = authors_section.find_all("a", class_="author-name")
                metadata["authors"] = ", ".join([author.text.strip() for author in authors])
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing IEEE page: {e}", exc_info=True)
            return metadata
    
    # Generic extraction for other academic sites
    try:
        # Try common meta tags for title
        title_tag = soup.find("meta", property="og:title") or soup.find("meta", name="citation_title")
        if title_tag and title_tag.get("content"):
            metadata["title"] = title_tag["content"]
        else:
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.text
        
        # Try common meta tags for authors
        author_tags = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_tags:
            metadata["authors"] = ", ".join([tag["content"] for tag in author_tags])
        
        # Try to find publication year
        year_tag = soup.find("meta", attrs={"name": "citation_publication_date"})
        if year_tag and year_tag.get("content"):
            year_match = re.search(r'\d{4}', year_tag["content"])
            if year_match:
                metadata["year"] = year_match.group(0)
        
        # Try to find abstract
        abstract_tag = soup.find("meta", attrs={"name": "citation_abstract"})
        if abstract_tag and abstract_tag.get("content"):
            metadata["abstract"] = abstract_tag["content"]
        else:
            # Try common abstract containers
            abstract_containers = [
                soup.find("div", class_="abstract"),
                soup.find("section", class_="abstract"),
                soup.find("p", class_="abstract"),
                soup.find("div", id="abstract")
            ]
            
            for container in abstract_containers:
                if container:
                    metadata["abstract"] = container.text.strip()
                    break
        
        # Try to find keywords
        keywords_containers = [
            soup.find("div", class_="keywords"),
            soup.find("div", class_="kwd-group"),
            soup.find("ul", class_="keywords")
        ]
        
        for container in keywords_containers:
            if container:
                keywords = container.find_all(["li", "a", "span"])
                if keywords:
                    metadata["keywords"] = ", ".join([k.text.strip() for k in keywords])
                    break
                else:
                    metadata["keywords"] = container.text.strip()
                break
        
        logger.info(f"Extracted metadata from URL: {url}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from URL: {e}", exc_info=True)
        return metadata

def extract_content_from_pdf(pdf_file):
    """Extract textual content from an uploaded PDF file, including full content and
    the section between introduction and references/conclusion"""
    content = ""
    
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Extract text using PyMuPDF (faster approach)
        try:
            doc = fitz.open(temp_path)
            for page in doc:
                content += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}. Trying pdfplumber...")
            
            # Fallback to pdfplumber for more robust extraction
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
        
        # Extract main content (between introduction and references/conclusion)
        main_content_pattern = re.compile(
            r"(?i)(\b1\.?\s*introduction\b.*?)(\b(?:references|bibliography|conclusion)\b)", 
            re.DOTALL
        )
        main_content_match = main_content_pattern.search(content)
        
        main_content = ""
        if main_content_match:
            main_content = main_content_match.group(1).strip()
        
        # Try to extract introduction section if available
        intro_pattern = re.compile(
            r"(?i)(\b1\.?\s*introduction\b.*?)(\b\d+\.?\s*\w+)", 
            re.DOTALL
        )
        intro_match = intro_pattern.search(content)
        
        intro_content = ""
        if intro_match:
            intro_content = intro_match.group(1).strip()
        
        # Delete temporary file
        os.unlink(temp_path)
        
        return {
            'full_content': content, 
            'intro': intro_content,
            'main_content': main_content  # Add the main content between intro and references
        }
        
    except Exception as e:
        logger.error(f"Error extracting content from PDF: {e}", exc_info=True)
        return {'full_content': '', 'intro': '', 'main_content': ''}

# Add new route to handle URL-based article submission
@app.route('/submit_article_url', methods=['POST'])
def submit_article_url():
    try:
        # Get the article URL
        article_url = request.form.get('article_url', '')
        
        if not article_url:
            return jsonify({"success": False, "error": "Article URL is required"}), 400
        
        # Extract metadata from the URL
        metadata = extract_article_metadata_from_url(article_url)
        
        # Check if we have a PDF file
        pdf_file = request.files.get('pdf_file')
        content_from_pdf = ""
        stored_pdf_filename = None
        
        if pdf_file and pdf_file.filename:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            original_filename = secure_filename(pdf_file.filename)
            unique_id = str(uuid.uuid4().hex)
            stored_pdf_filename = f"{unique_id}_{original_filename}"
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_pdf_filename)
            pdf_file.save(pdf_path)
            
            extraction_result = extract_content_from_pdf(pdf_file) # Use your existing function
            content_from_pdf = extraction_result.get('full_content', '')
            
            if not metadata.get('abstract') and extraction_result.get('intro'):
                metadata['abstract'] = extraction_result['intro'][:500]
        
        # Generate a new article ID
        article_id = int(time.time())
        year_val = int(metadata['year']) if metadata.get('year', '').isdigit() else datetime.datetime.now().year

        # Create the project node in Neo4j
        project_props = {
            "id": article_id,
            "title": metadata.get('title', 'N/A').strip(),
            "abstract": metadata.get('abstract', '').strip(),
            "keywords": metadata.get('keywords', '').strip(),
            "year": year_val,
            "content": content_from_pdf, # Full content from PDF if provided
            "url": article_url # The external URL is the primary link
        }
        if stored_pdf_filename:
            project_props["pdf_filename"] = stored_pdf_filename
            if not content_from_pdf and not metadata.get('abstract'): # if no abstract from meta and no content from pdf yet
                 project_props["abstract"] = pdf_to_text(os.path.join(app.config['UPLOAD_FOLDER'], stored_pdf_filename))[:500]

        project = Node("Project", **project_props)
        # Get the graph connection
        graph = search_engine.graph
        graph.create(project)
        
        # Process academics (authors)
        academic_names = [name.strip() for name in metadata['authors'].split(',') if name.strip()]
        
        for name in academic_names:
            # Check if academic exists
            academic_node = search_engine.matcher.match("Academic", name=name).first()
            
            if not academic_node:
                # Create keyword summary for the academic
                keyword_samples = []
                if metadata['keywords'] and metadata['keywords'].strip():
                    keyword_samples = [kw.strip() for kw in metadata['keywords'].split(',')[:3] if kw.strip()]
                
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
            keywords_list = [kw.strip() for kw in metadata['keywords'].split(',') if kw.strip()]
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
        
        return jsonify({
            "success": True, 
            "message": "Article added successfully from URL", 
            "article_id": article_id,
            "metadata": metadata  # Return the extracted metadata for confirmation
        })
        
    except Exception as e:
        logger.error(f"Error while submitting article from URL: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/upload', methods=['GET'])
def upload_form():
    # Create the uploads folder if it doesn't exist when the upload page is accessed
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
        except OSError as e:
            logger.error(f"Could not create upload folder: {app.config['UPLOAD_FOLDER']}. Error: {e}")
    return render_template('upload.html')


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