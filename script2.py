#!/usr/bin/env python3
"""
Sentence Transformers Performance Evaluation Script for Academic Search Engine
Bu script, akademik arama motorundaki Sentence Transformers modellerinin performansını değerlendirir.

Metrics:
- Precision@K, Recall@K, F1@K
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Response Time Performance
- Hit Rate@K
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statistics
import random
import threading

# Ana uygulama kodunu import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import SemanticSearchEngine, setup_database
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure app.py is in the same directory and required packages are installed")
    sys.exit(1)

@dataclass
class EvaluationResult:
    """Evaluation sonuçlarını saklamak için dataclass"""
    model_name: str
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    hit_rate_at_k: Dict[int, float]
    avg_response_time: float
    total_queries: int
    successful_queries: int

class SentenceTransformersEvaluator:
    """Sentence Transformers modellerinin performansını değerlendiren sınıf"""
    
    def __init__(self, search_engine: SemanticSearchEngine, log_level=logging.INFO):
        self.search_engine = search_engine
        self.test_queries = []
        self.ground_truth = {}
        self.results_cache = {}
        
        # Logging setup
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Evaluation parameters
        self.k_values = [1, 3, 5, 10]
        self.similarity_threshold = 0.35
        
        # Desteklenen Sentence Transformers modelleri
        self.supported_models = [
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        ]
        
    def generate_test_queries(self, num_queries: int = 30) -> List[Dict]:
        """
        Mevcut verilerden test sorguları oluştur
        """
        self.logger.info(f"Generating {num_queries} test queries...")
        
        test_queries = []
        
        # 1. Proje başlıklarından sorgu oluştur
        for i, project in enumerate(self.search_engine.projects[:num_queries//3]):
            if hasattr(project, 'get'):
                title = project.get('title', '')
                keywords = project.get('keywords', '')
                project_id = project.get('id')
            else:
                title = getattr(project, 'title', '')
                keywords = getattr(project, 'keywords', '')
                project_id = getattr(project, 'id', None)
            
            if title and project_id:
                # Başlığın bir kısmını sorgu olarak kullan
                title_words = title.split()
                if len(title_words) >= 2:
                    query = ' '.join(title_words[:3])  # İlk 3 kelime
                    test_queries.append({
                        'query': query,
                        'expected_project_ids': [project_id],
                        'expected_academics': [],
                        'expected_calls': [],
                        'type': 'project_title'
                    })
        
        # 2. Anahtar kelimelerden sorgu oluştur
        for i, project in enumerate(self.search_engine.projects[num_queries//3:2*num_queries//3]):
            if hasattr(project, 'get'):
                keywords = project.get('keywords', '')
                project_id = project.get('id')
            else:
                keywords = getattr(project, 'keywords', '')
                project_id = getattr(project, 'id', None)
            
            if keywords and project_id:
                keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                if keyword_list:
                    query = random.choice(keyword_list)  # Rastgele bir anahtar kelime
                    test_queries.append({
                        'query': query,
                        'expected_project_ids': [project_id],
                        'expected_academics': [],
                        'expected_calls': [],
                        'type': 'keyword'
                    })
        
        # 3. Akademisyen isimlerinden sorgu oluştur
        academic_names = list(self.search_engine.academics.keys())[:num_queries//3]
        for name in academic_names:
            if name:
                # İsmin bir kısmını sorgu olarak kullan
                name_parts = name.split()
                if len(name_parts) >= 2:
                    query = name_parts[-1]  # Soyadı
                    test_queries.append({
                        'query': query,
                        'expected_project_ids': [],
                        'expected_academics': [name],
                        'expected_calls': [],
                        'type': 'academic'
                    })
        
        # 4. Genel alan sorguları ekle
        domain_queries = [
            {'query': 'machine learning', 'type': 'domain'},
            {'query': 'artificial intelligence', 'type': 'domain'},
            {'query': 'deep learning', 'type': 'domain'},
            {'query': 'natural language processing', 'type': 'domain'},
            {'query': 'computer vision', 'type': 'domain'},
            {'query': 'data mining', 'type': 'domain'},
            {'query': 'neural networks', 'type': 'domain'},
            {'query': 'sustainability', 'type': 'domain'},
            {'query': 'climate change', 'type': 'domain'},
            {'query': 'bioinformatics', 'type': 'domain'}
        ]
        
        for domain_query in domain_queries[:num_queries//4]:
            domain_query.update({
                'expected_project_ids': [],
                'expected_academics': [],
                'expected_calls': []
            })
            test_queries.append(domain_query)
        
        self.test_queries = test_queries[:num_queries]
        self.logger.info(f"Generated {len(self.test_queries)} test queries")
        return self.test_queries
    
    def calculate_precision_recall_f1(self, retrieved_ids: List, expected_ids: List, k: int) -> Tuple[float, float, float]:
        """Precision@K, Recall@K ve F1@K hesapla"""
        if not expected_ids:
            return 0.0, 0.0, 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(expected_ids))
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(expected_ids) if expected_ids else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_ndcg(self, retrieved_similarities: List[float], expected_ids: List, retrieved_ids: List, k: int) -> float:
        """NDCG@K hesapla"""
        if not expected_ids or not retrieved_ids:
            return 0.0
        
        # Relevance scores: 1 if in expected_ids, 0 otherwise
        relevance_scores = []
        for i, retrieved_id in enumerate(retrieved_ids[:k]):
            if retrieved_id in expected_ids:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)
        
        if not any(relevance_scores):
            return 0.0
        
        # Ideal relevance scores (all relevant items first)
        ideal_scores = [1.0] * min(len(expected_ids), k) + [0.0] * max(0, k - len(expected_ids))
        
        try:
            # sklearn's ndcg_score expects 2D arrays
            ndcg = ndcg_score([ideal_scores], [relevance_scores], k=k)
            return ndcg
        except:
            return 0.0
    
    def calculate_mrr(self, retrieved_ids: List, expected_ids: List) -> float:
        """Mean Reciprocal Rank hesapla"""
        if not expected_ids:
            return 0.0
        
        for i, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id in expected_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_hit_rate(self, retrieved_ids: List, expected_ids: List, k: int) -> float:
        """Hit Rate@K hesapla"""
        if not expected_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        return 1.0 if any(rid in expected_ids for rid in retrieved_k) else 0.0
    
    def safe_load_model(self, model_name: str) -> bool:
        """CUDA sorunlarını çözerek modeli güvenli bir şekilde yükle"""
        try:
            self.logger.info(f"Loading model safely: {model_name}")
            
            with self.search_engine.lock:
                self.search_engine.model_loaded = False
                self.search_engine.current_model_name = model_name
                self.search_engine.is_pykeen_model = model_name.startswith('pykeen:')
                
                # CUDA sorununu çözmek için özel yükleme
                if torch.cuda.is_available():
                    try:
                        # İlk olarak CPU'da yükle
                        self.search_engine.model = SentenceTransformer(model_name, device='cpu')
                        self.logger.info("Model loaded on CPU, moving to CUDA...")
                        # Manuel olarak CUDA'ya taşı
                        self.search_engine.model = self.search_engine.model.to('cuda')
                        self.logger.info("Model successfully moved to CUDA")
                    except Exception as cuda_error:
                        self.logger.warning(f"CUDA transfer failed: {cuda_error}, staying on CPU...")
                        self.search_engine.model = SentenceTransformer(model_name, device='cpu')
                else:
                    self.logger.info("CUDA not available, using CPU")
                    self.search_engine.model = SentenceTransformer(model_name, device='cpu')
                
                # Embeddings'leri hesapla
                self.search_engine.precompute_embeddings()
                self.search_engine.model_loaded = True
                
            self.logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            return False
    
    def evaluate_single_query(self, query_data: Dict) -> Dict:
        """Tek bir sorgu için evaluasyon yap"""
        query = query_data['query']
        expected_project_ids = query_data.get('expected_project_ids', [])
        expected_academics = query_data.get('expected_academics', [])
        
        start_time = time.time()
        
        try:
            # Model durumunu kontrol et
            if not self.search_engine.model_loaded:
                self.logger.error(f"Model not loaded when evaluating query: {query}")
                return {
                    'response_time': time.time() - start_time,
                    'success': False,
                    'error': 'Model not loaded'
                }
            
            # Arama yap
            self.logger.debug(f"Searching for query: {query}")
            results = self.search_engine.search(query, top_k=10)
            response_time = time.time() - start_time
            
            if not results:
                self.logger.warning(f"No results returned for query: {query}")
                return {
                    'response_time': response_time,
                    'success': False,
                    'error': 'No results returned'
                }
            
            # Proje sonuçlarını değerlendir
            project_results = results.get('projects', [])
            retrieved_project_ids = [p['id'] for p in project_results if 'id' in p]
            retrieved_similarities = [p['similarity'] for p in project_results if 'similarity' in p]
            
            # Akademisyen sonuçlarını değerlendir
            academic_results = results.get('academics', [])
            retrieved_academic_names = [a['name'] for a in academic_results if 'name' in a]
            
            self.logger.debug(f"Query '{query}' returned {len(project_results)} projects, {len(academic_results)} academics")
            
            # Metrikleri hesapla
            metrics = {}
            
            # Proje metrikleri
            for k in self.k_values:
                precision, recall, f1 = self.calculate_precision_recall_f1(
                    retrieved_project_ids, expected_project_ids, k
                )
                ndcg = self.calculate_ndcg(
                    retrieved_similarities, expected_project_ids, retrieved_project_ids, k
                )
                hit_rate = self.calculate_hit_rate(retrieved_project_ids, expected_project_ids, k)
                
                metrics[f'precision_at_{k}'] = precision
                metrics[f'recall_at_{k}'] = recall
                metrics[f'f1_at_{k}'] = f1
                metrics[f'ndcg_at_{k}'] = ndcg
                metrics[f'hit_rate_at_{k}'] = hit_rate
            
            # MRR hesapla
            mrr_projects = self.calculate_mrr(retrieved_project_ids, expected_project_ids)
            mrr_academics = self.calculate_mrr(retrieved_academic_names, expected_academics)
            metrics['mrr'] = max(mrr_projects, mrr_academics)
            
            metrics['response_time'] = response_time
            metrics['success'] = True
            metrics['num_results'] = len(project_results) + len(academic_results)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating query '{query}': {e}", exc_info=True)
            return {
                'response_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_model(self, model_name: str) -> EvaluationResult:
        """Belirli bir model için tam evaluasyon yap"""
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Modeli güvenli bir şekilde yükle
        if not self.safe_load_model(model_name):
            self.logger.error(f"Failed to load model: {model_name}")
            return None
        
        # Test sorguları oluştur (eğer henüz oluşturulmamışsa)
        if not self.test_queries:
            self.generate_test_queries()
        
        # Her sorgu için evaluasyon yap
        all_metrics = []
        successful_queries = 0
        
        for i, query_data in enumerate(self.test_queries):
            self.logger.info(f"Evaluating query {i+1}/{len(self.test_queries)}: {query_data['query']}")
            
            metrics = self.evaluate_single_query(query_data)
            all_metrics.append(metrics)
            
            if metrics.get('success', False):
                successful_queries += 1
        
        # Ortalama metrikleri hesapla
        avg_metrics = {}
        
        # Her metrik için ortalama hesapla
        metric_names = [
            'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10',
            'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
            'f1_at_1', 'f1_at_3', 'f1_at_5', 'f1_at_10',
            'ndcg_at_1', 'ndcg_at_3', 'ndcg_at_5', 'ndcg_at_10',
            'hit_rate_at_1', 'hit_rate_at_3', 'hit_rate_at_5', 'hit_rate_at_10',
            'mrr', 'response_time'
        ]
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in all_metrics if m.get('success', False)]
            avg_metrics[metric_name] = np.mean(values) if values else 0.0
        
        # Sonuçları organize et
        result = EvaluationResult(
            model_name=model_name,
            precision_at_k={k: avg_metrics[f'precision_at_{k}'] for k in self.k_values},
            recall_at_k={k: avg_metrics[f'recall_at_{k}'] for k in self.k_values},
            f1_at_k={k: avg_metrics[f'f1_at_{k}'] for k in self.k_values},
            ndcg_at_k={k: avg_metrics[f'ndcg_at_{k}'] for k in self.k_values},
            mrr=avg_metrics['mrr'],
            hit_rate_at_k={k: avg_metrics[f'hit_rate_at_{k}'] for k in self.k_values},
            avg_response_time=avg_metrics['response_time'],
            total_queries=len(self.test_queries),
            successful_queries=successful_queries
        )
        
        self.logger.info(f"Completed evaluation for {model_name}")
        return result
    
    def compare_models(self, models: List[str] = None) -> Dict[str, EvaluationResult]:
        """Birden fazla modeli karşılaştır"""
        if models is None:
            models = self.supported_models
        
        results = {}
        
        for model in models:
            self.logger.info(f"Starting evaluation for model: {model}")
            result = self.evaluate_model(model)
            if result:
                results[model] = result
            else:
                self.logger.warning(f"Skipping model {model} due to evaluation failure")
        
        return results
    
    def generate_report(self, results: Dict[str, EvaluationResult], output_dir: str = "evaluation_results"):
        """Sonuçları içeren detaylı rapor oluştur"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON raporu
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'precision_at_k': result.precision_at_k,
                'recall_at_k': result.recall_at_k,
                'f1_at_k': result.f1_at_k,
                'ndcg_at_k': result.ndcg_at_k,
                'mrr': result.mrr,
                'hit_rate_at_k': result.hit_rate_at_k,
                'avg_response_time': result.avg_response_time,
                'total_queries': result.total_queries,
                'successful_queries': result.successful_queries
            }
        
        json_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # CSV raporu
        csv_data = []
        for model_name, result in results.items():
            row = {'model': model_name}
            row.update({f'precision@{k}': v for k, v in result.precision_at_k.items()})
            row.update({f'recall@{k}': v for k, v in result.recall_at_k.items()})
            row.update({f'f1@{k}': v for k, v in result.f1_at_k.items()})
            row.update({f'ndcg@{k}': v for k, v in result.ndcg_at_k.items()})
            row.update({f'hit_rate@{k}': v for k, v in result.hit_rate_at_k.items()})
            row['mrr'] = result.mrr
            row['avg_response_time'] = result.avg_response_time
            row['success_rate'] = result.successful_queries / result.total_queries
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        # Görselleştirmeler
        self._create_visualizations(results, output_dir, timestamp)
        
        # Metin raporu
        text_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("Sentence Transformers Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Queries: {list(results.values())[0].total_queries}\n\n")
            
            for model_name, result in results.items():
                f.write(f"\nModel: {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Success Rate: {result.successful_queries}/{result.total_queries} "
                       f"({result.successful_queries/result.total_queries*100:.1f}%)\n")
                f.write(f"Average Response Time: {result.avg_response_time:.3f}s\n")
                f.write(f"MRR: {result.mrr:.3f}\n\n")
                
                f.write("Precision@K:\n")
                for k, v in result.precision_at_k.items():
                    f.write(f"  @{k}: {v:.3f}\n")
                
                f.write("\nRecall@K:\n")
                for k, v in result.recall_at_k.items():
                    f.write(f"  @{k}: {v:.3f}\n")
                
                f.write("\nF1@K:\n")
                for k, v in result.f1_at_k.items():
                    f.write(f"  @{k}: {v:.3f}\n")
                
                f.write("\nNDCG@K:\n")
                for k, v in result.ndcg_at_k.items():
                    f.write(f"  @{k}: {v:.3f}\n")
                
                f.write("\nHit Rate@K:\n")
                for k, v in result.hit_rate_at_k.items():
                    f.write(f"  @{k}: {v:.3f}\n")
                f.write("\n")
        
        self.logger.info(f"Evaluation report generated in {output_dir}")
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        print(f"Files generated:")
        print(f"  - {json_file}")
        print(f"  - {csv_file}")
        print(f"  - {text_file}")
    
    def _create_visualizations(self, results: Dict[str, EvaluationResult], output_dir: str, timestamp: str):
        """Performans görselleştirmeleri oluştur"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # 1. Precision@K karşılaştırması
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision@K
        for model_name, result in results.items():
            axes[0, 0].plot(self.k_values, [result.precision_at_k[k] for k in self.k_values], 
                           marker='o', label=model_name.split('/')[-1])
        axes[0, 0].set_title('Precision@K')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Recall@K
        for model_name, result in results.items():
            axes[0, 1].plot(self.k_values, [result.recall_at_k[k] for k in self.k_values], 
                           marker='s', label=model_name.split('/')[-1])
        axes[0, 1].set_title('Recall@K')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1@K
        for model_name, result in results.items():
            axes[1, 0].plot(self.k_values, [result.f1_at_k[k] for k in self.k_values], 
                           marker='^', label=model_name.split('/')[-1])
        axes[1, 0].set_title('F1@K')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # NDCG@K
        for model_name, result in results.items():
            axes[1, 1].plot(self.k_values, [result.ndcg_at_k[k] for k in self.k_values], 
                           marker='d', label=model_name.split('/')[-1])
        axes[1, 1].set_title('NDCG@K')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('NDCG')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_comparison_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Response Time ve MRR karşılaştırması
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = [name.split('/')[-1] for name in results.keys()]
        response_times = [result.avg_response_time for result in results.values()]
        mrrs = [result.mrr for result in results.values()]
        
        ax1.bar(model_names, response_times)
        ax1.set_title('Average Response Time')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(model_names, mrrs)
        ax2.set_title('Mean Reciprocal Rank (MRR)')
        ax2.set_ylabel('MRR')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"performance_comparison_{timestamp}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def test_single_model(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """Tek bir modeli manuel olarak test et"""
    print(f"Testing single model: {model_name}")
    
    try:
        # CUDA sorununu test et
        if torch.cuda.is_available():
            try:
                # İlk olarak CPU'da yükle
                model = SentenceTransformer(model_name, device='cpu')
                print(f"✓ Model loaded on CPU")
                
                # CUDA'ya taşımayı dene
                model = model.to('cuda')
                print(f"✓ Model moved to CUDA successfully")
                
                # Test embedding
                test_text = "artificial intelligence machine learning"
                embedding = model.encode(test_text)
                print(f"✓ Embedding generated: shape {embedding.shape}")
                
            except Exception as cuda_error:
                print(f"⚠ CUDA failed: {cuda_error}")
                # CPU'da dene
                model = SentenceTransformer(model_name, device='cpu')
                print(f"✓ Model loaded on CPU (fallback)")
                
                test_text = "artificial intelligence machine learning"
                embedding = model.encode(test_text)
                print(f"✓ Embedding generated: shape {embedding.shape}")
        else:
            print("ℹ CUDA not available, using CPU")
            model = SentenceTransformer(model_name, device='cpu')
            print(f"✓ Model loaded on CPU")
            
            test_text = "artificial intelligence machine learning"
            embedding = model.encode(test_text)
            print(f"✓ Embedding generated: shape {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def main():
    """Ana evaluasyon scripti"""
    print("Sentence Transformers Model Performance Evaluation")
    print("=" * 50)
    
    # Database setup
    print("Setting up database connection...")
    graph, academics, projects, calls = setup_database()
    
    if not graph:
        print("Database setup failed! Exiting.")
        return
    
    print(f"Database ready: {len(projects)} projects, {len(academics)} academics, {len(calls)} calls")
    
    # Search engine setup
    print("Initializing search engine...")
    search_engine = SemanticSearchEngine(
        graph, academics, projects, calls,
        default_model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    )
    
    # Evaluator setup
    evaluator = SentenceTransformersEvaluator(search_engine, log_level=logging.INFO)
    
    # Test modelleri - doğru model isimlerini kullan
    models_to_test = [
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ]
    
    # Önce modellerin erişilebilirliğini test et
    print("\nTesting model availability...")
    available_models = []
    for model in models_to_test:
        try:
            print(f"Testing {model}...")
            if test_single_model(model):
                print(f"✓ {model} - Available")
                available_models.append(model)
            else:
                print(f"✗ {model} - Failed")
        except Exception as e:
            print(f"✗ {model} - Error: {e}")
    
    if not available_models:
        print("No valid models found! Exiting.")
        return
    
    print(f"\nTesting {len(available_models)} models...")
    print("This may take several minutes...")
    
    # Modelleri değerlendir
    results = evaluator.compare_models(available_models)
    
    # Rapor oluştur
    if results:
        evaluator.generate_report(results)
        
        # Özet
        print("\nEvaluation Summary:")
        print("-" * 30)
        for model_name, result in results.items():
            print(f"\nModel: {model_name.split('/')[-1]}")
            print(f"  MRR: {result.mrr:.3f}")
            print(f"  Precision@5: {result.precision_at_k[5]:.3f}")
            print(f"  Recall@5: {result.recall_at_k[5]:.3f}")
            print(f"  F1@5: {result.f1_at_k[5]:.3f}")
            print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
            print(f"  Success Rate: {result.successful_queries}/{result.total_queries}")
    else:
        print("No results obtained. Check logs for errors.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Sentence Transformers models')
    parser.add_argument('--test-model', type=str, help='Test a single model manually')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.test_model:
        # Manuel model testi
        success = test_single_model(args.test_model)
        sys.exit(0 if success else 1)
    else:
        # Ana evaluasyon
        main()