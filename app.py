from flask import Flask, render_template, request, jsonify, send_from_directory
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
import threading
import os
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import json
import re
import tempfile
import uuid
from werkzeug.utils import secure_filename
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# PyKEEN imports
import torch
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
import pickle

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
search_engine = None


SIMILARITY_THRESHOLD = 0.35
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def setup_database():
    """Neo4j database setup with sample data"""
    try:
        return Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        return None

class SemanticSearchEngine:
    def __init__(self, graph):
        self.graph = graph
        self.matcher = NodeMatcher(graph)
        self.current_model = "sentence_transformer_multilingual"  # Default to multilingual
        self.model_loaded = False
        self.lock = threading.Lock()
        
        # Sentence Transformer models
        self.st_model_multilingual = None
        self.st_model_minilm = None # New attribute for MiniLM
        self.project_embeddings = np.array([])
        self.call_embeddings = np.array([])
        self.academic_embeddings = np.array([])
        
        # PyKEEN model
        self.pykeen_model = None
        self.triples_factory = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.entity_embeddings = None
        
        # Load data
        self.load_data()
        
        # Initialize with sentence transformer (multilingual as default)
        self.change_model("sentence_transformer_multilingual")
    
    def load_data(self):
        """Load data from Neo4j"""
        self.projects = list(self.graph.run("MATCH (p:Project) RETURN p").data())
        self.projects = [item['p'] for item in self.projects]
        
        self.calls = list(self.graph.run("MATCH (c:Call) RETURN c").data())
        self.calls = [item['c'] for item in self.calls]
        
        self.academics = {}
        academic_data = list(self.graph.run("MATCH (a:Academic) RETURN a").data())
        for item in academic_data:
            academic = item['a']
            self.academics[academic['name']] = academic
    
    def change_model(self, model_type):
        logger.info(f"Changing model to: {model_type}")

        with self.lock:
            self.model_loaded = False

            # Önceki model referanslarını temizle (bellek boşaltmaya yardımcı olur)
            if self.st_model_multilingual:
                del self.st_model_multilingual
                self.st_model_multilingual = None
            if self.st_model_minilm:
                del self.st_model_minilm
                self.st_model_minilm = None
            if self.pykeen_model:
                del self.pykeen_model
                self.pykeen_model = None

            self.current_model = model_type

            if model_type == "sentence_transformer_multilingual":
                success = self._load_sentence_transformer_multilingual()
            elif model_type == "sentence_transformer_minilm":
                success = self._load_sentence_transformer_minilm()
            elif model_type == "pykeen":
                success = self._load_pykeen_model()
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False

            self.model_loaded = success
            return success
    
    def _load_sentence_transformer_multilingual(self):
        """Load paraphrase-multilingual-mpnet-base-v2 model"""
        try:
            logger.info("Loading Sentence Transformer (Multilingual) model...")
            self.st_model_multilingual = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            self._compute_embeddings(self.st_model_multilingual) # Pass the model to compute_embeddings
            logger.info("Sentence Transformer (Multilingual) model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer (Multilingual): {e}")
            return False

    def _load_sentence_transformer_minilm(self): # New method for MiniLM
        """Load all-MiniLM-L12-v2 model"""
        try:
            logger.info("Loading Sentence Transformer (MiniLM) model...")
            self.st_model_minilm = SentenceTransformer('all-MiniLM-L12-v2')
            self._compute_embeddings(self.st_model_minilm) # Pass the model to compute_embeddings
            logger.info("Sentence Transformer (MiniLM) model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer (MiniLM): {e}")
            return False
    
    def _compute_embeddings(self, model): # Modified to accept a model
        """Compute embeddings for all data using the given model"""
        try:
            # Project embeddings
            project_texts = []
            for project in self.projects:
            # Content kısmını da dahil et
                content_text = project.get('content', '')[:2000]  # İlk 2000 karakter
                text = f"{project.get('title', '')} {project.get('abstract', '')} {project.get('keywords', '')} {content_text}"
                project_texts.append(text)
        
            if project_texts:
                # REMOVE 'convert_to_numpy=True'
                self.project_embeddings = model.encode(project_texts) 
            
            # Call embeddings
            call_texts = []
            for call in self.calls:
                text = f"{call.get('title', '')} {call.get('description', '')}"
                call_texts.append(text)
            
            if call_texts:
                # REMOVE 'convert_to_numpy=True'
                self.call_embeddings = model.encode(call_texts)
            
            # Academic embeddings
            academic_texts = []
            for name, academic in self.academics.items():
                text = f"{name} {academic.get('keywords', '')}"
                academic_texts.append(text)
            
            if academic_texts:
                # REMOVE 'convert_to_numpy=True'
                self.academic_embeddings = model.encode(academic_texts)
            
            logger.info("Embeddings computed successfully")
            
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise
    
    def _load_pykeen_model(self):
        """Load and train PyKEEN model"""
        try:
            logger.info("Preparing PyKEEN model...")
            
            # Prepare triples
            triples = []
            entities = set()
            relations = set()
            
            # Academic-Project relationships
            for project in self.projects:
                project_id = f"project_{project['id']}"
                entities.add(project_id)
                
                # Get academics for this project
                query = "MATCH (a:Academic)-[:OWNS]->(p:Project {id: $id}) RETURN a.name"
                try:
                    results = self.graph.run(query, id=project['id']).data()
                    
                    for result in results:
                        academic_name = f"academic_{str(result['a.name']).replace(' ', '_')}"
                        entities.add(academic_name)
                        triples.append((academic_name, "OWNS", project_id))
                        relations.add("OWNS")
                except Exception as e:
                    logger.warning(f"Error getting academics for project {project['id']}: {e}")
                
                # Project-Keyword relationships
                if 'keywords' in project and project['keywords']:
                    keywords_list = str(project['keywords']).split(',')
                    for kw in keywords_list:
                        kw_clean = kw.strip().lower().replace(' ', '_')
                        if kw_clean:
                            kw_name = f"keyword_{kw_clean}"
                            entities.add(kw_name)
                            triples.append((project_id, "HAS_KEYWORD", kw_name))
                            relations.add("HAS_KEYWORD")
            
            # Academic-Keyword relationships
            for name, academic in self.academics.items():
                academic_name = f"academic_{str(name).replace(' ', '_')}"
                entities.add(academic_name)
                
                if 'keywords' in academic and academic['keywords']:
                    keywords_list = str(academic['keywords']).split(',')
                    for kw in keywords_list:
                        kw_clean = kw.strip().lower().replace(' ', '_')
                        if kw_clean:
                            kw_name = f"keyword_{kw_clean}"
                            entities.add(kw_name)
                            triples.append((academic_name, "EXPERT_IN", kw_name))
                            relations.add("EXPERT_IN")
            
            # Call-Keyword relationships
            for call in self.calls:
                call_id = f"call_{call['id']}"
                entities.add(call_id)
                
                # Extract keywords from description
                desc = str(call.get('description', '')).lower()
                keywords = ["ai", "nlp", "machine learning", "deep learning", "computer vision", "research", "grant", "funding"]
                for kw in keywords:
                    if kw in desc:
                        kw_name = f"keyword_{kw.replace(' ', '_')}"
                        entities.add(kw_name)
                        triples.append((call_id, "HAS_THEME", kw_name))
                        relations.add("HAS_THEME")
            
            if not triples:
                logger.error("No triples generated for PyKEEN")
                return False
            
            # Create mappings - TÜM DEĞERLERİN STRING OLDUĞUNDAN EMİN OLUN
            entities_list = sorted([str(entity) for entity in entities])
            relations_list = sorted([str(relation) for relation in relations])
            
            # String-based mappings oluştur
            self.entity_to_id = {}
            self.id_to_entity = {}
            
            for idx, entity in enumerate(entities_list):
                entity_str = str(entity)
                self.entity_to_id[entity_str] = idx
                self.id_to_entity[idx] = entity_str
            
            relation_to_id = {}
            for idx, relation in enumerate(relations_list):
                relation_to_id[str(relation)] = idx
            
            # Convert triples to IDs
            mapped_triples = []
            for head, relation, tail in triples:
                head_str = str(head)
                tail_str = str(tail)
                relation_str = str(relation)
                
                if head_str in self.entity_to_id and tail_str in self.entity_to_id and relation_str in relation_to_id:
                    mapped_triples.append([
                        self.entity_to_id[head_str],
                        relation_to_id[relation_str],
                        self.entity_to_id[tail_str]
                    ])
            
            if not mapped_triples:
                logger.error("No mapped triples generated for PyKEEN")
                return False
            
            # Create triples factory
            self.triples_factory = TriplesFactory.from_labeled_triples(
                triples=np.array(triples),
                entity_to_id=self.entity_to_id,
                relation_to_id=relation_to_id
            )
            
            # Train model
            self.pykeen_model = TransE(
                triples_factory=self.triples_factory,
                embedding_dim=100,
                random_seed=42
            )
            
            training_loop = SLCWATrainingLoop(
                model=self.pykeen_model,
                triples_factory=self.triples_factory,
                optimizer='Adam'
            )
            
            training_loop.train(triples_factory=self.triples_factory, num_epochs=50)
            
            # Get entity embeddings
            self.entity_embeddings = self.pykeen_model.entity_representations[0]().detach().cpu().numpy()
            
            logger.info("PyKEEN model loaded and trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PyKEEN model: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search using the current model"""
        if not self.model_loaded:
            return {"projects": [], "calls": [], "academics": []}
        
        if self.current_model == "sentence_transformer_multilingual":
            return self._search_sentence_transformer(query, top_k, self.st_model_multilingual)
        elif self.current_model == "sentence_transformer_minilm": # New model type
            return self._search_sentence_transformer(query, top_k, self.st_model_minilm)
        elif self.current_model == "pykeen":
            return self._search_pykeen(query, top_k)
        else:
            return {"projects": [], "calls": [], "academics": []}
    
    def _search_sentence_transformer(self, query, top_k, model): # Modified to accept a model
        """Search using sentence transformer"""
        try:
            query_embedding = model.encode([query])[0]
            results = {"projects": [], "calls": [], "academics": []}
            
            # Graph tabanlı arama sonuçları
            graph_results = self._graph_based_search(query, top_k)
            
            # Semantik arama sonuçları
            semantic_results = {"projects": [], "calls": [], "academics": []}
            
            # Search projects
            if len(self.project_embeddings) > 0:
                similarities = cosine_similarity([query_embedding], self.project_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    sim = float(similarities[idx])
                    if sim > SIMILARITY_THRESHOLD:
                        project = self.projects[idx]
                        semantic_results["projects"].append({
                            "id": project.get("id"),
                            "title": project.get("title"),
                            "abstract": project.get("abstract", "")[:150] + "..." if len(project.get("abstract", "")) > 150 else project.get("abstract", ""),
                            "keywords": project.get("keywords"),
                            "year": project.get("year"),
                            "similarity": sim,
                            "url": project.get("url"),
                            "pdf_filename": project.get("pdf_filename")
                        })
            
            # Search calls
            if len(self.call_embeddings) > 0:
                similarities = cosine_similarity([query_embedding], self.call_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    sim = float(similarities[idx])
                    if sim > SIMILARITY_THRESHOLD:
                        call = self.calls[idx]
                        semantic_results["calls"].append({
                            "id": call["id"],
                            "title": call["title"],
                            "description": call.get("description", "")[:150] + "..." if len(call.get("description", "")) > 150 else call.get("description", ""),
                            "similarity": sim
                        })
            
            # Search academics
            if len(self.academic_embeddings) > 0:
                academic_list = list(self.academics.values())
                similarities = cosine_similarity([query_embedding], self.academic_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    sim = float(similarities[idx])
                    if sim > SIMILARITY_THRESHOLD:
                        academic = academic_list[idx]
                        semantic_results["academics"].append({
                            "name": academic["name"],
                            "keywords": academic["keywords"],
                            "similarity": sim
                        })
            
            # Projelere bağlı akademisyenleri ekle
            authors = {}
            for proj in semantic_results["projects"] + graph_results["projects"]:
                cypher_query = """
                MATCH (a:Academic)-[:OWNS]->(p:Project {id: $project_id})
                RETURN a.name as name, a.keywords as keywords
                """
                try:
                    results_query = self.graph.run(cypher_query, project_id=proj["id"]).data()
                    for result in results_query:
                        name = result.get("name")
                        if name and name not in authors:
                            authors[name] = {
                                "name": name,
                                "keywords": result.get("keywords", ""),
                                "similarity": proj["similarity"] * 0.9
                            }
                except Exception as e:
                    logger.warning(f"Error getting authors for project {proj['id']}: {e}")
            
            # Akademisyenlere bağlı projeleri ekle
            related_projects = {}
            for academic in semantic_results["academics"] + graph_results["academics"]:
                cypher_query = """
                MATCH (a:Academic {name: $name})-[:OWNS]->(p:Project)
                RETURN p.id as id, p.title as title, p.year as year, p.keywords as keywords, p.abstract as abstract, p.url as url, p.pdf_filename as pdf_filename
                """
                try:
                    results_query = self.graph.run(cypher_query, name=academic["name"]).data()
                    for result in results_query:
                        proj_id = result.get("id")
                        if proj_id and proj_id not in related_projects:
                            related_projects[proj_id] = {
                                "id": proj_id,
                                "title": result.get("title"),
                                "year": result.get("year"),
                                "keywords": result.get("keywords"),
                                "abstract": (result.get("abstract", "")[:150] + "...") if len(result.get("abstract", "")) > 150 else result.get("abstract", ""),
                                "similarity": academic["similarity"] * 0.8,
                                "url": result.get("url"),
                                "pdf_filename": result.get("pdf_filename")
                            }
                except Exception as e:
                    logger.warning(f"Error getting projects for academic {academic['name']}: {e}")
            
            # Sonuçları birleştir
            combined_results = {
                "projects": semantic_results["projects"] + graph_results["projects"] + list(related_projects.values()),
                "calls": semantic_results["calls"] + graph_results["calls"],
                "academics": semantic_results["academics"] + graph_results["academics"] + list(authors.values())
            }
            
            # Tekilleştirme
            for category in combined_results:
                seen = set()
                unique_list = []
                for item in combined_results[category]:
                    key = item.get("id") if "id" in item else item.get("name")
                    if key is not None and key not in seen:
                        seen.add(key)
                        unique_list.append(item)
                    elif key in seen:
                        # Daha yüksek similarity skorunu tut
                        existing_item = next((res for res in unique_list if (res.get("id") if "id" in res else res.get("name")) == key), None)
                        if existing_item and item["similarity"] > existing_item["similarity"]:
                            existing_item.update(item)
                
                combined_results[category] = sorted(unique_list, key=lambda x: x["similarity"], reverse=True)[:top_k]
            
            # Similarity threshold uygula
            final_results = {
                "projects": [p for p in combined_results["projects"] if p["similarity"] >= SIMILARITY_THRESHOLD],
                "calls": [c for c in combined_results["calls"] if c["similarity"] >= SIMILARITY_THRESHOLD],
                "academics": [a for a in combined_results["academics"] if a["similarity"] >= SIMILARITY_THRESHOLD]
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in sentence transformer search: {e}")
            return {"projects": [], "calls": [], "academics": []}
    
    def _search_pykeen(self, query, top_k):
        """Search using PyKEEN embeddings"""
        try:
            if self.entity_embeddings is None:
                return {"projects": [], "calls": [], "academics": []}
            
            query_terms = query.lower().split()
            potential_entities = []
            
            # Find matching entities - TİP KONTROLÜ EKLE
            for entity_name in self.entity_to_id.keys():
                entity_name_str = str(entity_name).lower()
                if any(term in entity_name_str for term in query_terms):
                    potential_entities.append(str(entity_name))
            
            if not potential_entities:
                return {"projects": [], "calls": [], "academics": []}
            
            # Get embeddings for query entities
            query_embeddings = []
            for entity_name in potential_entities:
                entity_name_str = str(entity_name)
                if entity_name_str in self.entity_to_id:
                    query_embeddings.append(self.entity_embeddings[self.entity_to_id[entity_name_str]])
            
            if not query_embeddings:
                return {"projects": [], "calls": [], "academics": []}
            
            # Average query embedding
            avg_query_embedding = np.mean(query_embeddings, axis=0)
            
            # Calculate similarities
            similarities = cosine_similarity([avg_query_embedding], self.entity_embeddings)[0]
            
            results = {"projects": [], "calls": [], "academics": []}
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            
            for idx in sorted_indices:
                sim = similarities[idx]
                if sim < SIMILARITY_THRESHOLD:
                    continue
                
                # GÜVENLE STRING DÖNÜŞÜMÜ
                entity_name = str(self.id_to_entity.get(int(idx), ""))
                
                if not entity_name:
                    continue
                
                if entity_name.startswith('project_'):
                    try:
                        project_id = int(entity_name.replace('project_', ''))
                        project = next((p for p in self.projects if p.get('id') == project_id), None)
                        if project and len(results["projects"]) < top_k:
                            results["projects"].append({
                                "id": project.get("id"),
                                "title": project.get("title"),
                                "abstract": project.get("abstract", "")[:150] + "..." if len(project.get("abstract", "")) > 150 else project.get("abstract", ""),
                                "keywords": project.get("keywords"),
                                "year": project.get("year"),
                                "similarity": float(sim),
                                "url": project.get("url"),
                                "pdf_filename": project.get("pdf_filename")
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing project entity {entity_name}: {e}")
                        continue
                
                elif entity_name.startswith('call_'):
                    try:
                        call_id = int(entity_name.replace('call_', ''))
                        call = next((c for c in self.calls if c.get('id') == call_id), None)
                        if call and len(results["calls"]) < top_k:
                            results["calls"].append({
                                "id": call.get("id"),
                                "title": call.get("title"),
                                "description": call.get("description", "")[:150] + "..." if len(call.get("description", "")) > 150 else call.get("description", ""),
                                "similarity": float(sim)
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing call entity {entity_name}: {e}")
                        continue
                
                elif entity_name.startswith('academic_'):
                    try:
                        academic_name = entity_name.replace('academic_', '').replace('_', ' ')
                        if academic_name in self.academics and len(results["academics"]) < top_k:
                            academic = self.academics[academic_name]
                            results["academics"].append({
                                "name": academic["name"],
                                "keywords": academic["keywords"],
                                "similarity": float(sim)
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing academic entity {entity_name}: {e}")
                        continue
        
            # Graph tabanlı arama sonuçlarını da ekle
            graph_results = self._graph_based_search(query, top_k)
            
            # Sonuçları birleştir
            combined_results = {
                "projects": results["projects"] + graph_results["projects"],
                "calls": results["calls"] + graph_results["calls"],
                "academics": results["academics"] + graph_results["academics"]
            }
            
            # Tekilleştir
            final_results = {"projects": [], "calls": [], "academics": []}
            
            for category in combined_results:
                seen = set()
                for item in combined_results[category]:
                    key = item.get("id") if "id" in item else item.get("name")
                    if key is not None and key not in seen:
                        seen.add(key)
                        final_results[category].append(item)
                    elif key in seen:
                        # Daha yüksek similarity skorunu tut
                        existing_item = next((res for res in final_results[category] if (res.get("id") if "id" in res else res.get("name")) == key), None)
                        if existing_item and item["similarity"] > existing_item["similarity"]:
                            existing_item.update(item)
                
                # Similarity threshold uygula ve sırala
                final_results[category] = sorted([item for item in final_results[category] if item["similarity"] >= SIMILARITY_THRESHOLD], 
                                               key=lambda x: x["similarity"], reverse=True)[:top_k]
            
            return final_results


            
        except Exception as e:
            logger.error(f"Error in PyKEEN search: {e}")
            return {"projects": [], "calls": [], "academics": []}
        
    def _graph_based_search(self, query, top_k=5):
        """Neo4j graph veritabanı üzerinden anahtar kelime tabanlı arama"""
        try:
            cleaned_query = query.lower().strip()
            
            # Projeler için Cypher sorguları
            project_query = """
            MATCH (p:Project)
            WHERE toLower(p.title) CONTAINS $query OR toLower(p.abstract) CONTAINS $query OR toLower(p.keywords) CONTAINS $query OR toLower(p.content) CONTAINS $query
            RETURN p, ID(p) as id
            LIMIT $limit
            """
            
            # Çağrılar için
            call_query = """
            MATCH (c:Call)
            WHERE toLower(c.title) CONTAINS $query OR toLower(c.description) CONTAINS $query
            RETURN c, ID(c) as id
            LIMIT $limit
            """
            
            # Akademisyenler için
            academic_query = """
            MATCH (a:Academic)
            WHERE toLower(a.name) CONTAINS $query OR toLower(a.keywords) CONTAINS $query
            RETURN a, ID(a) as id
            LIMIT $limit
            """
            
            # Anahtar kelime tabanlı sorgular
            keyword_project_query = """
            MATCH (k:Keyword)-[:HAS_KEYWORD]-(p:Project)
            WHERE toLower(k.name) CONTAINS $query
            RETURN p, ID(p) as id
            LIMIT $limit
            """
            
            keyword_academic_query = """
            MATCH (k:Keyword)-[:EXPERT_IN]-(a:Academic)
            WHERE toLower(k.name) CONTAINS $query
            RETURN a, ID(a) as id
            LIMIT $limit
            """
            
            keyword_call_query = """
            MATCH (k:Keyword)-[:HAS_THEME]-(c:Call)
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
            results = {"projects": [], "calls": [], "academics": []}
            
            # Projeleri işle
            seen_projects = set()
            for result in graph_projects:
                if result['id'] not in seen_projects:
                    seen_projects.add(result['id'])
                    project_node = result['p']
                    proj_dict = {
                        "id": project_node.get("id"),
                        "title": project_node.get("title"),
                        "year": project_node.get("year"),
                        "keywords": project_node.get("keywords"),
                        "similarity": 1.0,  # Graph tabanlı aramada similarity yok
                        "url": project_node.get("url"),
                        "pdf_filename": project_node.get("pdf_filename"),
                        "abstract": (project_node.get("abstract", "")[:150] + "...") if len(project_node.get("abstract", "")) > 150 else project_node.get("abstract", "")
                    }
                    results["projects"].append(proj_dict)
            
            # Çağrıları işle
            seen_calls = set()
            for result in graph_calls:
                if result['id'] not in seen_calls:
                    seen_calls.add(result['id'])
                    call = result['c']
                    call_dict = {
                        "id": call["id"],
                        "title": call["title"],
                        "similarity": 1.0
                    }
                    if "description" in call:
                        desc = call["description"]
                        call_dict["description"] = (desc[:150] + "...") if len(desc) > 150 else desc
                    results["calls"].append(call_dict)
            
            # Akademisyenleri işle
            seen_academics = set()
            for result in graph_academics:
                if result['id'] not in seen_academics:
                    seen_academics.add(result['id'])
                    academic = result['a']
                    academic_dict = {
                        "name": academic["name"],
                        "keywords": academic.get("keywords", ""),
                        "similarity": 1.0
                    }
                    results["academics"].append(academic_dict)


            # İLIŞKISEL SONUÇLARI EKLE

        except Exception as e:
            pass


        # Bulunan projelerin yazarlarını ekle
        for project in results["projects"]:
            cypher_query = """
            MATCH (a:Academic)-[:OWNS]->(p:Project {id: $project_id})
            RETURN a.name as name, a.keywords as keywords
            """
            try:
                academic_results = self.graph.run(cypher_query, project_id=project["id"]).data()
                for result in academic_results:
                    name = result.get("name")
                    if name and not any(a["name"] == name for a in results["academics"]):
                        results["academics"].append({
                            "name": name,
                            "keywords": result.get("keywords", ""),
                            "similarity": 0.9  # İlişkisel sonuç olarak düşük skor
                        })
            except Exception as e:
                logger.warning(f"Error getting authors for project {project['id']}: {e}")
        
        # Bulunan akademisyenlerin projelerini ekle
        for academic in results["academics"]:
            cypher_query = """
            MATCH (a:Academic {name: $name})-[:OWNS]->(p:Project)
            RETURN p.id as id, p.title as title, p.year as year, p.keywords as keywords, 
                   p.abstract as abstract, p.url as url, p.pdf_filename as pdf_filename
            """
            try:
                project_results = self.graph.run(cypher_query, name=academic["name"]).data()
                for result in project_results:
                    proj_id = result.get("id")
                    if proj_id and not any(p["id"] == proj_id for p in results["projects"]):
                        results["projects"].append({
                            "id": proj_id,
                            "title": result.get("title"),
                            "year": result.get("year"),
                            "keywords": result.get("keywords"),
                            "abstract": (result.get("abstract", "")[:150] + "...") if len(result.get("abstract", "")) > 150 else result.get("abstract", ""),
                            "similarity": 0.9,
                            "url": result.get("url"),
                            "pdf_filename": result.get("pdf_filename")
                        })
            except Exception as e:
                logger.warning(f"Error getting projects for academic {academic['name']}: {e}")
            
            # Akademisyenin çağrılarını ekle
            cypher_query = """
            MATCH (a:Academic {name: $name})-[:OWNS]->(c:Call)
            RETURN c.id as id, c.title as title, c.description as description
            """
            try:
                call_results = self.graph.run(cypher_query, name=academic["name"]).data()
                for result in call_results:
                    call_id = result.get("id")
                    if call_id and not any(c["id"] == call_id for c in results["calls"]):
                        results["calls"].append({
                            "id": call_id,
                            "title": result.get("title"),
                            "description": (result.get("description", "")[:150] + "...") if len(result.get("description", "")) > 150 else result.get("description", ""),
                            "similarity": 0.9
                        })
            except Exception as e:
                logger.warning(f"Error getting calls for academic {academic['name']}: {e}")
            
            return results
            

def pdf_to_text(pdf_path):
    """Convert PDF to text using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

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




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    global search_engine
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if search_engine is None:
            return jsonify({'error': 'Search engine not initialized'}), 500
        
        results = search_engine.search(query, top_k=5)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500
def extract_arxiv_metadata(soup, metadata):
    """ArXiv specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1.title, .title')
        if title_elem:
            title_text = title_elem.get_text().replace('Title:', '').strip()
            metadata["title"] = title_text
        
        # Authors
        authors_elem = soup.select_one('.authors')
        if authors_elem:
            author_links = authors_elem.find_all('a')
            authors = [link.get_text().strip() for link in author_links if link.get_text().strip()]
            metadata["authors"] = ", ".join(authors)
        
        # Year from submission date
        dateline = soup.select_one('.dateline')
        if dateline:
            year_match = re.search(r'(20\d{2})', dateline.get_text())
            if year_match:
                metadata["year"] = year_match.group(1)
        
        # Abstract
        abstract_elem = soup.select_one('.abstract')
        if abstract_elem:
            abstract_text = abstract_elem.get_text().replace('Abstract:', '').strip()
            metadata["abstract"] = abstract_text
        
        # Keywords from subjects
        subjects_elem = soup.select_one('.subjects')
        if subjects_elem:
            subjects = subjects_elem.get_text().replace('Subjects:', '').strip()
            metadata["keywords"] = subjects
        
    except Exception as e:
        logger.error(f"Error in ArXiv metadata extraction: {e}")
    
    return metadata

def extract_acm_metadata(soup, metadata):
    """ACM Digital Library specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1.citation__title, .hlFld-Title')
        if title_elem:
            metadata["title"] = title_elem.get_text().strip()
        
        # Authors
        author_elems = soup.select('.loa__author-name, .citation__authors .author')
        if author_elems:
            authors = [elem.get_text().strip() for elem in author_elems]
            metadata["authors"] = ", ".join(authors)
        
        # Year
        year_elem = soup.select_one('.epub-section__date, .citation__date')
        if year_elem:
            year_match = re.search(r'(20\d{2})', year_elem.get_text())
            if year_match:
                metadata["year"] = year_match.group(1)
        
        # Abstract
        abstract_elem = soup.select_one('.abstractSection p, .citation__abstract')
        if abstract_elem:
            metadata["abstract"] = abstract_elem.get_text().strip()
        
        # Keywords
        keyword_elems = soup.select('.keywords-section .keyword, .citation__keywords .keyword')
        if keyword_elems:
            keywords = [elem.get_text().strip() for elem in keyword_elems]
            metadata["keywords"] = ", ".join(keywords)
        
    except Exception as e:
        logger.error(f"Error in ACM metadata extraction: {e}")
    
    return metadata

def extract_springer_metadata(soup, metadata):
    """Springer specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1.c-article-title, .ArticleTitle')
        if title_elem:
            metadata["title"] = title_elem.get_text().strip()
        
        # Authors
        author_elems = soup.select('.c-article-author-list .c-article-author, .AuthorList .Author')
        if author_elems:
            authors = []
            for elem in author_elems:
                author_name = elem.get_text().strip()
                if author_name:
                    authors.append(author_name)
            metadata["authors"] = ", ".join(authors)
        
        # Year
        year_elem = soup.select_one('.c-article-identifiers time, .ArticleCitation time')
        if year_elem:
            year_match = re.search(r'(20\d{2})', year_elem.get('datetime', '') or year_elem.get_text())
            if year_match:
                metadata["year"] = year_match.group(1)
        
        # Abstract
        abstract_elem = soup.select_one('#Abs1-content, .AbstractSection')
        if abstract_elem:
            metadata["abstract"] = abstract_elem.get_text().strip()
        
        # Keywords
        keyword_elems = soup.select('.c-article-subject-list li, .KeywordGroup .Keyword')
        if keyword_elems:
            keywords = [elem.get_text().strip() for elem in keyword_elems]
            metadata["keywords"] = ", ".join(keywords)
        
    except Exception as e:
        logger.error(f"Error in Springer metadata extraction: {e}")
    
    return metadata

def extract_sciencedirect_metadata(soup, metadata):
    """ScienceDirect specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1.title-text, .article-header h1')
        if title_elem:
            metadata["title"] = title_elem.get_text().strip()
        
        # Authors
        author_elems = soup.select('.author-group .author, .article-header .author')
        if author_elems:
            authors = []
            for elem in author_elems:
                # Extract just the author name, skip affiliations
                given_name = elem.select_one('.given-name, .text.given-name')
                surname = elem.select_one('.surname, .text.surname')
                if given_name and surname:
                    full_name = f"{given_name.get_text().strip()} {surname.get_text().strip()}"
                    authors.append(full_name)
                else:
                    author_text = elem.get_text().strip()
                    if author_text and len(author_text) < 100:  # Avoid long affiliation texts
                        authors.append(author_text)
            metadata["authors"] = ", ".join(authors)
        
        # Year
        year_elem = soup.select_one('.publication-volume .vol-issue, .article-info .date')
        if year_elem:
            year_match = re.search(r'(20\d{2})', year_elem.get_text())
            if year_match:
                metadata["year"] = year_match.group(1)
        
        # Abstract
        abstract_elem = soup.select_one('#abstracts .abstract, .abstract-content')
        if abstract_elem:
            metadata["abstract"] = abstract_elem.get_text().strip()
        
        # Keywords
        keyword_elems = soup.select('.keywords-section .keyword, .author-keywords .keyword')
        if keyword_elems:
            keywords = [elem.get_text().strip() for elem in keyword_elems]
            metadata["keywords"] = ", ".join(keywords)
        
    except Exception as e:
        logger.error(f"Error in ScienceDirect metadata extraction: {e}")
    
    return metadata

def extract_researchgate_metadata(soup, metadata):
    """ResearchGate specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1.research-detail-header-section__title')
        if title_elem:
            metadata["title"] = title_elem.get_text().strip()
        
        # Authors
        author_elems = soup.select('.research-detail-authors .nova-legacy-e-text--color-inherit')
        if author_elems:
            authors = [elem.get_text().strip() for elem in author_elems]
            metadata["authors"] = ", ".join(authors)
        
        # Year
        date_elem = soup.select_one('.research-detail-header-section__metadata .nova-legacy-e-text')
        if date_elem:
            year_match = re.search(r'(20\d{2})', date_elem.get_text())
            if year_match:
                metadata["year"] = year_match.group(1)
        
        # Abstract
        abstract_elem = soup.select_one('.research-detail-middle-section__abstract')
        if abstract_elem:
            metadata["abstract"] = abstract_elem.get_text().strip()
        
    except Exception as e:
        logger.error(f"Error in ResearchGate metadata extraction: {e}")
    
    return metadata

def extract_semantic_scholar_metadata(soup, metadata):
    """Semantic Scholar specific metadata extraction"""
    try:
        # Title
        title_elem = soup.select_one('h1[data-selenium-selector="paper-detail-title"]')
        if title_elem:
            metadata["title"] = title_elem.get_text().strip()
        
        # Authors
        author_elems = soup.select('.author-list .author-tile')
        if author_elems:
            authors = []
            for elem in author_elems:
                author_name = elem.select_one('.cl-paper-authors__author-list__name')
                if author_name:
                    authors.append(author_name.get_text().strip())
            metadata["authors"] = ", ".join(authors)
        
        # Year
        year_elem = soup.select_one('.paper-meta .paper-meta__year')
        if year_elem:
            metadata["year"] = year_elem.get_text().strip()
        
        # Abstract
        abstract_elem = soup.select_one('.tldr-section__text, .paper-detail-page__abstract')
        if abstract_elem:
            metadata["abstract"] = abstract_elem.get_text().strip()
        
    except Exception as e:
        logger.error(f"Error in Semantic Scholar metadata extraction: {e}")
    
    return metadata

def extract_generic_metadata(soup, metadata, fallback=False):
    """Generic metadata extraction for unknown sites"""
    try:
        # Only proceed if we don't have the data or if this is a fallback
        if not metadata["title"] or fallback:
            # Title extraction
            title_selectors = [
                ("meta", {"property": "og:title"}),
                ("meta", {"name": "citation_title"}),
                ("meta", {"name": "dc.title"}),
                ("meta", {"name": "title"}),
                ("meta", {"property": "twitter:title"}),
                ("h1", {"class": re.compile(r"title|headline", re.I)}),
                ("h1", {"id": re.compile(r"title|headline", re.I)}),
                ("h1", {}),
                ("title", {})
            ]
            
            for tag_name, attrs in title_selectors:
                if tag_name == "title":
                    tag = soup.find(tag_name)
                    if tag and tag.get_text().strip():
                        title_text = tag.get_text().strip()
                        title_text = re.sub(r'\s*[-|]\s*[^-|]*$', '', title_text)
                        if title_text and len(title_text) > 5:
                            metadata["title"] = title_text[:200]
                            break
                else:
                    tag = soup.find(tag_name, attrs)
                    if tag:
                        title_text = tag.get("content") if tag.get("content") else tag.get_text()
                        if title_text and title_text.strip() and len(title_text.strip()) > 5:
                            metadata["title"] = title_text.strip()[:200]
                            break
        
        if not metadata["authors"] or fallback:
            # Author extraction
            author_selectors = [
                ("meta", {"name": "citation_author"}),
                ("meta", {"name": "dc.creator"}),
                ("meta", {"name": "author"}),
                ("meta", {"property": "article:author"}),
            ]
            
            authors = []
            for tag_name, attrs in author_selectors:
                if tag_name == "meta":
                    tags = soup.find_all(tag_name, attrs)
                    for tag in tags:
                        if tag.get("content"):
                            author_text = tag.get("content").strip()
                            if author_text and len(author_text) > 2 and len(author_text) < 100:
                                clean_author = re.sub(r'\s+', ' ', author_text)
                                if clean_author not in authors:
                                    authors.append(clean_author)
                
                if authors:
                    break
            
            # If no meta authors found, try CSS selectors
            if not authors:
                css_selectors = [
                    "[class*='author']",
                    "[id*='author']",
                    ".author-name",
                    ".byline",
                    ".author"
                ]
                
                for selector in css_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        author_text = element.get_text().strip()
                        if author_text and len(author_text) > 2 and len(author_text) < 100:
                            clean_author = re.sub(r'\s+', ' ', author_text)
                            clean_author = re.sub(r'\b(By|Author:|Authors:)\b', '', clean_author, flags=re.IGNORECASE).strip()
                            if clean_author and clean_author not in authors:
                                authors.append(clean_author)
                    
                    if authors:
                        break
            
            metadata["authors"] = ", ".join(authors[:10])
        
        # Continue with year, abstract, keywords, and content extraction...
        # (Rest of the generic extraction code remains the same)
        
    except Exception as e:
        logger.error(f"Error in generic metadata extraction: {e}")
    
    return metadata
@app.route('/change_model', methods=['POST'])
def change_model():
    global search_engine
    
    try:
        data = request.get_json()
        model_type = data.get('model_type', '')
        
        # Updated allowed model types
        if model_type not in ['sentence_transformer_multilingual', 'sentence_transformer_minilm', 'pykeen']:
            return jsonify({'success': False, 'error': 'Invalid model type'}), 400
        
        if not search_engine:
            return jsonify({'success': False, 'error': 'Search engine not initialized'}), 500
        
        # Change model in background thread
        def change_model_bg():
            search_engine.change_model(model_type)
        
        thread = threading.Thread(target=change_model_bg)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': f'Model change initiated: {model_type}'})
        
    except Exception as e:
        logger.error(f"Model change error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit_article', methods=['POST'])
def submit_article():
    global search_engine
    
    try:
        title = request.form.get('title', '').strip()
        abstract = request.form.get('abstract', '').strip()
        keywords = request.form.get('keywords', '').strip()
        year_str = request.form.get('year', '')
        academic_str = request.form.get('academic', '').strip()
        article_link = request.form.get('article_link', '').strip()
        
        # Handle PDF upload
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
            content_from_pdf = pdf_to_text(pdf_path)
        
        article_id = int(time.time())
        year = int(year_str) if year_str.isdigit() else datetime.datetime.now().year
        
        # Create project node
        project_props = {
            "id": article_id,
            "title": title,
            "abstract": abstract if abstract else content_from_pdf[:500] if content_from_pdf else "No abstract provided.",
            "keywords": keywords,
            "year": year,
            "content": content_from_pdf,
            "url": article_link if article_link else None,
        }
        if stored_pdf_filename:
            project_props["pdf_filename"] = stored_pdf_filename
        
        project = Node("Project", **project_props)
        search_engine.graph.create(project)
        
        # Process academics
        academic_names = [name.strip() for name in academic_str.split(',') if name.strip()]
        
        for name in academic_names:
            academic_node = search_engine.matcher.match("Academic", name=name).first()
            
            if not academic_node:
                keyword_samples = [kw.strip() for kw in keywords.split(',')[:3] if kw.strip()]
                keyword_str = f"Expert in {', '.join(keyword_samples)}" if keyword_samples else "Researcher"
                
                academic_node = Node("Academic", name=name, keywords=keyword_str)
                search_engine.graph.create(academic_node)
                search_engine.academics[name] = academic_node
            
            # Create relationships
            owns_rel = Relationship(academic_node, "OWNS", project)
            search_engine.graph.create(owns_rel)
            
            # Process keywords
            keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            for keyword in keywords_list:
                keyword_node = Node("Keyword", name=keyword.lower())
                search_engine.graph.merge(keyword_node, "Keyword", "name")
                
                has_keyword_rel = Relationship(project, "HAS_KEYWORD", keyword_node)
                search_engine.graph.create(has_keyword_rel)
                
                expert_in_rel = Relationship(academic_node, "EXPERT_IN", keyword_node)
                search_engine.graph.create(expert_in_rel)
        
        # Update search engine data
        search_engine.load_data()
        if search_engine.model_loaded and (search_engine.current_model == "sentence_transformer_multilingual" or search_engine.current_model == "sentence_transformer_minilm"):
            # Recompute embeddings with the currently active Sentence Transformer model
            if search_engine.current_model == "sentence_transformer_multilingual" and search_engine.st_model_multilingual:
                search_engine._compute_embeddings(search_engine.st_model_multilingual)
            elif search_engine.current_model == "sentence_transformer_minilm" and search_engine.st_model_minilm:
                search_engine._compute_embeddings(search_engine.st_model_minilm)
        
        return jsonify({"success": True, "message": "Article added successfully", "article_id": article_id})
        
    except Exception as e:
        logger.error(f"Error submitting article: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/submit_article_url', methods=['POST'])
@app.route('/submit_article_url', methods=['POST'])
def submit_article_url():
    global search_engine
    
    try:
        article_url = request.form.get('article_url', '')
        
        if not article_url:
            return jsonify({"success": False, "error": "Article URL is required"}), 400
        
        logger.info(f"Extracting metadata from URL: {article_url}")
        
        # Extract metadata from URL
        metadata = extract_article_metadata_from_url(article_url)
        
        # Handle PDF file if provided
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
            content_from_pdf = pdf_to_text(pdf_path)
            
            # If PDF content is available, prioritize it over web content
            if content_from_pdf and len(content_from_pdf.strip()) > 100:
                metadata['content'] = content_from_pdf
                
                # If no abstract from web but PDF content available, extract from PDF
                if not metadata.get('abstract') and content_from_pdf:
                    sentences = re.split(r'[.!?]+', content_from_pdf)
                    abstract_sentences = []
                    for sentence in sentences[:5]:  # First 5 sentences
                        if len(sentence.strip()) > 20:
                            abstract_sentences.append(sentence.strip())
                    if abstract_sentences:
                        metadata['abstract'] = '. '.join(abstract_sentences)[:1000] + "..."
        
        article_id = int(time.time())
        year_val = int(metadata['year']) if metadata.get('year', '').isdigit() else datetime.datetime.now().year
        
        title_to_use = metadata.get('title', '').strip()
        print(f"Extracted title: {title_to_use}")
        if not title_to_use:
            title_to_use = f"Article from URL {article_url}"
        
        # Use web content if available, otherwise use PDF content
        final_content = metadata.get('content', '') or content_from_pdf
        final_abstract = metadata.get('abstract', '') or 'No abstract available.'
        final_keywords = metadata.get('keywords', '') or 'No keywords available.'
        
        # DEBUG: Log the extracted data
        logger.info(f"Title: {title_to_use}")
        logger.info(f"Abstract length: {len(final_abstract)}")
        logger.info(f"Keywords: {final_keywords}")
        logger.info(f"Content length: {len(final_content)}")
        logger.info(f"Authors: {metadata.get('authors', 'None')}")
        
        # Create project node with all metadata - FIX: Ensure all fields are properly added
        project_props = {
            "id": article_id,
            "title": title_to_use,
            "abstract": final_abstract.strip() if final_abstract.strip() != 'No abstract available.' else "",
            "keywords": final_keywords.strip() if final_keywords.strip() != 'No keywords available.' else "",
            "year": year_val,
            "content": final_content[:5000] if final_content else "",  # Limit content length
            "url": article_url
        }
        if stored_pdf_filename:
            project_props["pdf_filename"] = stored_pdf_filename
        
        # FIX: Create project node and add to search_engine.projects list
        project = Node("Project", **project_props)
        search_engine.graph.create(project)
        
        # FIX: Add the new project to the local projects list immediately
        search_engine.projects.append(project_props)
        
        logger.info(f"Created project node with ID: {article_id}")
        
        # Process authors - FIX: Improve author parsing
        academic_names = []
        if metadata.get('authors'):
            authors_text = metadata['authors']
            # Better author splitting logic
            author_parts = re.split(r'[,;&]|\band\b|\bve\b', authors_text)
            for part in author_parts:
                clean_name = part.strip()
                # Remove common academic titles and affiliations
                clean_name = re.sub(r'\b(Dr\.?|Prof\.?|PhD\.?|M\.?D\.?|MSc\.?|BSc\.?)\b', '', clean_name, flags=re.IGNORECASE)
                clean_name = re.sub(r'\([^)]*\)', '', clean_name)  # Remove parentheses content
                clean_name = re.sub(r'\d+', '', clean_name)  # Remove numbers
                clean_name = clean_name.strip()
                
                # Filter valid names
                if (clean_name and 
                    len(clean_name) > 2 and 
                    len(clean_name) < 100 and
                    not clean_name.lower() in ['et', 'al', 'and', 'the', 'al.'] and
                    ' ' in clean_name):  # Ensure it looks like a real name
                    academic_names.append(clean_name)
        
        logger.info(f"Found {len(academic_names)} authors: {academic_names}")
        
        # FIX: Create academic nodes and relationships with better error handling
        created_academics = []
        for name in academic_names:
            try:
                academic_node = search_engine.matcher.match("Academic", name=name).first()
                
                if not academic_node:
                    keyword_samples = []
                    if final_keywords and final_keywords != 'No keywords available.':
                        keyword_parts = re.split(r'[,;]', final_keywords)
                        keyword_samples = [kw.strip() for kw in keyword_parts[:3] if kw.strip() and len(kw.strip()) > 2]
                    
                    keyword_str = f"Expert in {', '.join(keyword_samples)}" if keyword_samples else "Researcher"
                    
                    academic_node = Node("Academic", name=name, keywords=keyword_str)
                    search_engine.graph.create(academic_node)
                    search_engine.academics[name] = {"name": name, "keywords": keyword_str}
                    created_academics.append(name)
                    logger.info(f"Created new academic: {name}")
                
                # Create ownership relationship
                owns_rel = Relationship(academic_node, "OWNS", project)
                search_engine.graph.create(owns_rel)
                logger.info(f"Created OWNS relationship: {name} -> Project {article_id}")
                
            except Exception as e:
                logger.error(f"Error creating academic {name}: {e}")
                continue
        
        # FIX: Process keywords with better validation and create keyword nodes
        created_keywords = []
        if final_keywords and final_keywords != 'No keywords available.':
            # Split keywords by common separators
            keyword_parts = re.split(r'[,;]', final_keywords)
            keywords_list = []
            for kw in keyword_parts:
                clean_kw = kw.strip().lower()
                # Remove common stop words and very short keywords
                if (clean_kw and 
                    len(clean_kw) > 2 and 
                    len(clean_kw) < 50 and  # Reasonable keyword length
                    clean_kw not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an']):
                    keywords_list.append(clean_kw)
            
            logger.info(f"Processing {len(keywords_list)} keywords: {keywords_list}")
            
            for keyword in keywords_list[:15]:  # Limit to 15 keywords
                try:
                    keyword_node = Node("Keyword", name=keyword)
                    search_engine.graph.merge(keyword_node, "Keyword", "name")
                    created_keywords.append(keyword)
                    
                    # Create project-keyword relationship
                    has_keyword_rel = Relationship(project, "HAS_KEYWORD", keyword_node)
                    search_engine.graph.create(has_keyword_rel)
                    logger.info(f"Created HAS_KEYWORD relationship: Project {article_id} -> {keyword}")
                    
                    # Create academic-keyword relationships
                    for name in academic_names:
                        try:
                            academic_node = search_engine.matcher.match("Academic", name=name).first()
                            if academic_node:
                                expert_in_rel = Relationship(academic_node, "EXPERT_IN", keyword_node)
                                search_engine.graph.create(expert_in_rel)
                                logger.info(f"Created EXPERT_IN relationship: {name} -> {keyword}")
                        except Exception as e:
                            logger.error(f"Error creating EXPERT_IN relationship for {name} -> {keyword}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error creating keyword {keyword}: {e}")
                    continue
        
        # FIX: Update search engine data and recompute embeddings
        logger.info("Reloading search engine data...")
        search_engine.load_data()
        
        # Recompute embeddings if sentence transformer model is active
        if search_engine.model_loaded:
            try:
                if search_engine.current_model == "sentence_transformer_multilingual" and search_engine.st_model_multilingual:
                    logger.info("Recomputing embeddings with multilingual model...")
                    search_engine._compute_embeddings(search_engine.st_model_multilingual)
                elif search_engine.current_model == "sentence_transformer_minilm" and search_engine.st_model_minilm:
                    logger.info("Recomputing embeddings with MiniLM model...")
                    search_engine._compute_embeddings(search_engine.st_model_minilm)
                logger.info("Embeddings recomputed successfully")
            except Exception as e:
                logger.error(f"Error recomputing embeddings: {e}")
        
        logger.info(f"Successfully added article from URL: {article_url}")
        
        # FIX: Verify the data was actually added by checking the database
        verification_query = "MATCH (p:Project {id: $id}) RETURN p.title, p.abstract, p.keywords, p.content"
        verification_result = search_engine.graph.run(verification_query, id=article_id).data()
        
        if verification_result:
            logger.info(f"Verification successful. Project found in database: {verification_result[0]}")
        else:
            logger.error(f"Verification failed. Project {article_id} not found in database")
        
        return jsonify({
            "success": True,
            "message": "Article added successfully from URL",
            "article_id": article_id,
            "metadata": {
                "title": title_to_use,
                "authors": metadata.get('authors', ''),
                "year": str(year_val),
                "abstract_length": len(final_abstract),
                "keywords_count": len(final_keywords.split(',')) if final_keywords != 'No keywords available.' else 0,
                "content_length": len(final_content),
                "created_academics": created_academics,
                "created_keywords": created_keywords,
                "verification_passed": bool(verification_result),
                "extracted_info": {
                    "has_title": bool(metadata.get('title')),
                    "has_authors": bool(metadata.get('authors')),
                    "has_year": bool(metadata.get('year')),
                    "has_abstract": bool(metadata.get('abstract')),
                    "has_keywords": bool(metadata.get('keywords')),
                    "has_content": bool(metadata.get('content')),
                    "has_pdf": bool(stored_pdf_filename)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error submitting article from URL: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/serve_pdf/<filename>')
def serve_pdf(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/upload')
def upload_form():
    return render_template('upload.html')

@app.route('/add_call')
def add_call_form():
    return render_template('add_call.html')

@app.route('/health')
def health():
    """System health check"""
    # AJAX isteğini doğru şekilde tanı
    if request.headers.get('Accept') == 'application/json' or 'application/json' in request.headers.get('Accept', ''):
        try:
            if search_engine is None:
                return jsonify({'status': 'error', 'message': 'Search engine not initialized'}), 500
            
            model_status = "loaded" if search_engine.model_loaded else "loading"
            
            # Test database connection
            db_connected = False
            try:
                result = search_engine.graph.run("MATCH (n) RETURN count(n) as count LIMIT 1").data()
                db_connected = True
            except Exception as e:
                logger.error(f"Database connection test failed: {e}")
                db_connected = False
            
            return jsonify({
                'status': 'ok',
                'model_status': model_status,
                'current_model': search_engine.current_model,
                'database_connection': db_connected,
                'projects_count': len(search_engine.projects),
                'academics_count': len(search_engine.academics),
                'calls_count': len(search_engine.calls)
            })
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # Normal browser isteği ise HTML döndür
    return render_template('health.html')


@app.route('/model_status', methods=['GET'])
def model_status():
    """Model yükleme durumunu kontrol et"""
    try:
        if search_engine is None:
            return jsonify({'loaded': False, 'current_model': None})
        
        return jsonify({
            'loaded': search_engine.model_loaded,
            'current_model': search_engine.current_model
        })
        
    except Exception as e:
        logger.error(f"Model status check error: {e}")
        return jsonify({'loaded': False, 'current_model': None})
@app.route('/list')
def list_page():
    """Veri yönetim sayfasını render eder."""
    return render_template('list.html')

@app.route('/api/list-all', methods=['GET'])
def get_all_items():
    """Veritabanındaki tüm projeleri ve çağrıları listeler."""
    global search_engine
    if not search_engine:
        return jsonify({"error": "Arama motoru başlatılmadı."}), 500
    try:
        # Projeleri al
        project_query = "MATCH (p:Project) RETURN p.id as id, p.title as title ORDER BY p.id DESC"
        project_results = search_engine.graph.run(project_query).data()

        # Çağrıları al
        call_query = "MATCH (c:Call) RETURN c.id as id, c.title as title ORDER BY c.id DESC"
        call_results = search_engine.graph.run(call_query).data()

        # Akademisyenleri al (YENİ)
        academic_query = "MATCH (a:Academic) RETURN a.name as name ORDER BY a.name"
        academic_results = search_engine.graph.run(academic_query).data()

        return jsonify({'projects': project_results, 'calls': call_results, 'academics': academic_results})
    except Exception as e:
        logger.error(f"Tüm verileri listeleme hatası: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete-item', methods=['POST'])
def delete_item():
    """Belirtilen ID ve tipe sahip öğeyi ve ilişkili verileri veritabanından siler."""
    global search_engine
    if not search_engine:
        return jsonify({"success": False, "error": "Arama motoru başlatılmadı."}), 500

    try:
        data = request.get_json()
        entity_type = data.get('type')  # 'project', 'call', veya 'academic'
        entity_id = data.get('id')
        
        graph = search_engine.graph

        if entity_type == 'project' or entity_type == 'call':
            # Proje veya Çağrı silme mantığı (mevcut kodunuzdaki gibi)
            node_label = entity_type.capitalize()
            entity_id = int(entity_id)

            # Silmeden önce potansiyel yetim kalacak keywordleri bul
            keyword_query = f"""
            MATCH (:{node_label} {{id: $id}})-[:HAS_KEYWORD|HAS_THEME]->(k:Keyword)
            RETURN DISTINCT k.name as name
            """
            keyword_names = [r['name'] for r in graph.run(keyword_query, id=entity_id)]

            # Potansiyel yetim akademisyenleri bul
            academic_query = f"""
            MATCH (a:Academic)-[:OWNS]->(:{node_label} {{id: $id}})
            RETURN a.name as name
            """
            academic_names = [r['name'] for r in graph.run(academic_query, id=entity_id)]

            # Ana öğeyi sil
            delete_query = f"MATCH (n:{node_label} {{id: $id}}) DETACH DELETE n"
            graph.run(delete_query, id=entity_id)
            
            # Yetim kalan keyword'leri temizle
            for kw in keyword_names:
                rel_check = "MATCH (k:Keyword {name: $kw})<--() RETURN count(k) as count"
                count = graph.run(rel_check, kw=kw).evaluate()
                if count == 0:
                    graph.run("MATCH (k:Keyword {name: $kw}) DELETE k", kw=kw)
            
            # Yetim kalan akademisyenleri temizle
            for name in academic_names:
                rel_check = "MATCH (a:Academic {name: $name})-[:OWNS]->() RETURN count(a) as count"
                count = graph.run(rel_check, name=name).evaluate()
                if count == 0:
                    graph.run("MATCH (a:Academic {name: $name}) DETACH DELETE a", name=name)

        elif entity_type == 'academic':
            # YENİ AKADEMİSYEN SİLME MANTIĞI
            # entity_id burada akademisyenin ismidir (string)
            academic_name = entity_id

            # 1. Silinecek akademisyene bağlı projelerin/çağrıların keyword'lerini bul
            keyword_query = """
            MATCH (:Academic {name: $name})-[:OWNS]->(n)-[:HAS_KEYWORD|:HAS_THEME]->(k:Keyword)
            RETURN DISTINCT k.name as name
            """
            keyword_names = [r['name'] for r in graph.run(keyword_query, name=academic_name)]

            # 2. Akademisyeni ve sahip olduğu TÜM projeleri/çağrıları tek seferde sil
            #    DETACH DELETE ilişkileri de otomatik olarak kaldırır.
            delete_query = """
            MATCH (a:Academic {name: $name})
            OPTIONAL MATCH (a)-[:OWNS]->(owned_node)
            DETACH DELETE a, owned_node
            """
            graph.run(delete_query, name=academic_name)

            # 3. Potansiyel olarak yetim kalmış keyword'leri temizle
            for kw in keyword_names:
                rel_check = "MATCH (k:Keyword {name: $kw})<--() RETURN count(k) as count"
                count = graph.run(rel_check, kw=kw).evaluate()
                if count == 0:
                    graph.run("MATCH (k:Keyword {name: $kw}) DELETE k", kw=kw)
        else:
            return jsonify({"success": False, "error": "Geçersiz öğe tipi."}), 400

        # Veritabanı değiştiği için arama motoru verilerini yeniden yükle
        search_engine.load_data()
        if search_engine.model_loaded and (search_engine.current_model == "sentence_transformer_multilingual" or search_engine.current_model == "sentence_transformer_minilm"):
            # Recompute embeddings with the currently active Sentence Transformer model
            if search_engine.current_model == "sentence_transformer_multilingual" and search_engine.st_model_multilingual:
                search_engine._compute_embeddings(search_engine.st_model_multilingual)
            elif search_engine.current_model == "sentence_transformer_minilm" and search_engine.st_model_minilm:
                search_engine._compute_embeddings(search_engine.st_model_minilm)

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Öğe silme hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
if __name__ == '__main__':
    logger.info("Starting application...")
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    graph = setup_database()
    
    if not graph:
        logger.error("Database setup failed!")
    else:
        logger.info("Starting search engine...")
        search_engine = SemanticSearchEngine(graph)
        logger.info("Starting web server...")
        app.run(host='0.0.0.0', port=5001, debug=False)