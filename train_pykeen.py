import os
import torch
import pandas as pd
from py2neo import Graph
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- YAPILANDIRMA ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
OUTPUT_DIRECTORY = "pykeen_model"

# PyKEEN modelinin eğitimi için kullanılacak ayarlar
PYKEEN_MODEL = 'TransE'
TRAINING_EPOCHS = 100
EMBEDDING_DIM = 50

def fetch_triples_from_neo4j():
    """Neo4j veritabanından tüm ilişkileri üçlü (triple) formatında çeker."""
    try:
        logger.info(f"Neo4j veritabanına bağlanılıyor: {NEO4J_URI}")
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        query = """
        MATCH (h)-[r]->(t)
        WITH
          CASE WHEN 'Academic' IN labels(h) THEN h.name WHEN 'Project' IN labels(h) THEN h.title WHEN 'Call' IN labels(h) THEN h.title WHEN 'Keyword' IN labels(h) THEN h.name ELSE NULL END AS head_label,
          type(r) AS relation_label,
          CASE WHEN 'Academic' IN labels(t) THEN t.name WHEN 'Project' IN labels(t) THEN t.title WHEN 'Call' IN labels(t) THEN t.title WHEN 'Keyword' IN labels(t) THEN t.name ELSE NULL END AS tail_label
        WHERE head_label IS NOT NULL AND tail_label IS NOT NULL
        RETURN head_label AS head, relation_label AS relation, tail_label AS tail
        """
        
        logger.info("İlişkiler (triples) Neo4j'den çekiliyor...")
        data = graph.run(query).to_data_frame()
        
        if data.empty:
            logger.warning("Veritabanından hiç ilişki çekilemedi.")
            return None
            
        logger.info(f"Toplam {len(data)} adet ilişki başarıyla çekildi.")
        return data

    except Exception as e:
        logger.error(f"Neo4j'den veri çekilirken hata oluştu: {e}", exc_info=True)
        return None

def train_and_save_model(triples_df):
    """Verilen üçlülerle bir PyKEEN modeli eğitir ve kaydeder."""
    logger.info("PyKEEN için TriplesFactory oluşturuluyor...")
    tf = TriplesFactory.from_labeled_triples(triples_df.values)
    
    logger.info(f"'{PYKEEN_MODEL}' modeli {TRAINING_EPOCHS} epoch ile eğitilmeye başlanıyor...")
    
    pipeline_result = pipeline(
        training=tf,
        testing=tf,
        model=PYKEEN_MODEL,
        model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
        training_kwargs=dict(num_epochs=TRAINING_EPOCHS, use_tqdm_batch=True),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    logger.info("Model eğitimi tamamlandı.")
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        
    # Modelin kendisini ve veri yapısını (TriplesFactory) kaydet
    pipeline_result.save_to_directory(OUTPUT_DIRECTORY)
    # ÖNEMLİ: TriplesFactory'i ayrıca ve daha basit bir yolla kaydediyoruz.
    tf.to_path_binary(os.path.join(OUTPUT_DIRECTORY, 'training_triples'))
    
    logger.info(f"Eğitilmiş model ve veriler başarıyla '{OUTPUT_DIRECTORY}' klasörüne kaydedildi.")


if __name__ == '__main__':
    triples_dataframe = fetch_triples_from_neo4j()
    if triples_dataframe is not None and not triples_dataframe.empty:
        train_and_save_model(triples_dataframe)
    else:
        logger.error("Model eğitimi için veri bulunamadığından işlem durduruldu.")